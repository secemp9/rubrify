"""Generation and refinement of rubrics from natural language."""

from __future__ import annotations

from typing import Any

from rubrify._meta_rubric import (
    COMPLIANCE_GENERATOR,
    DETECTION_GENERATOR,
    META_EVALUATOR,
    SCORING_GENERATOR,
)
from rubrify._mutations import RubricMutation
from rubrify._properties import validate
from rubrify._types import ICLExample
from rubrify.provenance import RefinementReport, RefinementStep, RubricProvenance
from rubrify.result import EvaluationResult
from rubrify.rubric import ConstraintRubric, Rubric

_GENERATORS = {
    "scoring": SCORING_GENERATOR,
    "detection": DETECTION_GENERATOR,
    "compliance": COMPLIANCE_GENERATOR,
}


def _extract_xml(text: str) -> str:
    """Extract XML from LLM output, stripping markdown fences and prose.

    Raises ValueError if no valid <LLM_JUDGE_SPEC> XML is found.
    """
    import xml.etree.ElementTree as ET

    # 1. Try direct parse
    try:
        root = ET.fromstring(text)
        if root.tag == "LLM_JUDGE_SPEC":
            return text
    except ET.ParseError:
        pass

    # 2. Extract from markdown fences
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    tokens = md.parse(text)

    for token in tokens:
        if token.type == "fence" and token.content:
            content = token.content.strip()
            try:
                root = ET.fromstring(content)
                if root.tag == "LLM_JUDGE_SPEC":
                    return content
            except ET.ParseError:
                continue

    raise ValueError(
        "LLM output does not contain valid <LLM_JUDGE_SPEC> XML. "
        f"Raw output (first 500 chars): {text[:500]}"
    )


def generate(
    source: str,
    *,
    client: Any,
    model: str,
    rubric_type: str = "scoring",
    name: str | None = None,
    evaluate: bool = False,
    min_meta_score: int | None = None,
    max_attempts: int = 1,
    repair_invalid_xml: bool = True,
    return_report: bool = False,
    **kwargs: Any,
) -> Rubric | tuple[Rubric, EvaluationResult] | tuple[Rubric, RefinementReport]:
    """Generate a rubric from source material using the any2rubric pipeline.

    Gen: Src -> Rub (the generation functor)

    Phase 4 adds a thresholded retry loop:

    * ``max_attempts`` caps the number of times the generator is invoked.
    * ``repair_invalid_xml`` decides whether to retry on parse/validation
      failures (``True``) or raise immediately (``False``).
    * ``min_meta_score`` asks for an optional meta-evaluator threshold; when
      set, every successful attempt is meta-evaluated and the loop only
      stops when the score meets the threshold or attempts are exhausted.
    * ``return_report=True`` returns ``(rubric, RefinementReport)`` in place
      of the rubric alone, exposing the ``stopped_reason`` and per-attempt
      trail. Guard rail 6 (no silent fallbacks) is enforced through the
      explicit ``stopped_reason`` values.

    Backwards compat: with the default arguments
    (``max_attempts=1``, ``min_meta_score=None``, ``return_report=False``)
    the behavior matches the pre-Phase-4 single-shot generator, including
    the legacy ``evaluate=True`` tuple return.
    """
    import rubrify

    generator = _GENERATORS.get(rubric_type)
    if generator is None:
        raise ValueError(f"Unknown rubric_type: {rubric_type!r}. Use: {list(_GENERATORS)}")

    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    steps: list[RefinementStep] = []
    last_rubric: Rubric | None = None
    last_meta_result: EvaluationResult | None = None
    last_score: int | None = None
    last_error: ValueError | None = None
    stopped_reason = "max_iters"

    for attempt in range(max_attempts):
        try:
            raw_output = str(generator.apply(source, client=client, model=model, **kwargs))
            raw_xml = _extract_xml(raw_output)
            rubric = rubrify.loads(raw_xml)
            if name:
                rubric.name = name
            validation = validate(rubric)
            if not validation.is_valid:
                error_msgs = [e.message for e in validation.errors]
                raise ValueError(f"Generated rubric failed validation: {error_msgs}")
        except ValueError as exc:
            last_error = exc
            steps.append(
                RefinementStep(
                    kind="generate",
                    reason=f"attempt {attempt + 1} failed: {exc}",
                    before_version="",
                    after_version="",
                    mutation_names=(),
                    meta_score=None,
                )
            )
            if not repair_invalid_xml:
                raise
            continue

        meta_result: EvaluationResult | None = None
        current_score: int | None = None
        if min_meta_score is not None or evaluate:
            meta_result = META_EVALUATOR.evaluate(raw_xml, client=client, model=model)
            current_score = meta_result.score

        steps.append(
            RefinementStep(
                kind="generate",
                reason=f"attempt {attempt + 1} succeeded",
                before_version="",
                after_version=rubric.version,
                mutation_names=(),
                meta_score=current_score,
            )
        )

        last_rubric = rubric
        last_meta_result = meta_result
        last_score = current_score

        if min_meta_score is None:
            stopped_reason = "target_met"
            break
        if current_score is not None and current_score >= min_meta_score:
            stopped_reason = "target_met"
            break
        stopped_reason = "max_iters"

    if last_rubric is None:
        # Every attempt failed with a parse/validation error.
        if last_error is not None:
            raise last_error
        raise ValueError("Generation failed: no rubric produced and no error captured")

    provenance = last_rubric.provenance or RubricProvenance()
    if not provenance.source_kind:
        provenance.source_kind = "concept"
    if not provenance.source_summary:
        provenance.source_summary = source[:200]
    if not provenance.generated_by_model:
        provenance.generated_by_model = model
    if last_meta_result is not None and not provenance.evaluated_by_model:
        provenance.evaluated_by_model = model
    for step in steps:
        provenance.add_step(step)
    last_rubric.provenance = provenance

    report = RefinementReport(
        iterations=len(steps),
        start_score=None,
        end_score=last_score,
        stopped_reason=stopped_reason,
        steps=tuple(steps),
    )

    if return_report:
        return last_rubric, report

    if evaluate and last_meta_result is not None:
        return last_rubric, last_meta_result

    return last_rubric


def refine(
    rubric: Rubric,
    *,
    client: Any,
    model: str,
    corpus: list[str] | None = None,
    target_score: int | None = None,
    max_iters: int = 1,
    stop_on_no_mutations: bool = True,
    return_report: bool = False,
) -> Rubric | tuple[Rubric, RefinementReport]:
    """Refine a rubric using the Refine monad.

    Refine: Rub -> Rub (the refinement endofunctor)

    Phase 4 adds a thresholded iteration loop around the original single
    pass. Meta-evaluate once to obtain ``start_score``; if ``target_score``
    is already met, stop immediately. Otherwise iterate up to ``max_iters``
    times, each iteration calling :func:`_suggest_mutations` to derive
    concrete structural fixes, applying them via :meth:`Rubric.evolve`, and
    re-evaluating. The loop exits with an explicit ``stopped_reason``
    (guard rail 6):

    * ``target_met`` — score met or exceeded ``target_score``.
    * ``no_mutations`` — no structural fixes suggested (and
      ``stop_on_no_mutations=True``).
    * ``score_regressed`` — the iteration made the score worse; the
      previous rubric is returned instead.
    * ``max_iters`` — the loop budget was exhausted without meeting the
      target.

    Backwards compat: ``max_iters=1``, ``target_score=None``,
    ``return_report=False`` produces a single-iteration refine that
    returns the rubric (unchanged when ``_suggest_mutations`` yields an
    empty list, matching the pre-Phase-4 behavior).
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")

    steps: list[RefinementStep] = []

    initial_meta = META_EVALUATOR.evaluate(rubric.to_xml(), client=client, model=model)
    start_score = initial_meta.score

    current_rubric: Rubric = rubric
    current_meta_result: EvaluationResult = initial_meta
    current_score: int | None = start_score

    if target_score is not None and current_score is not None and current_score >= target_score:
        stopped_reason = "target_met"
        _attach_refine_provenance(current_rubric, steps, evaluated_by_model=model)
        report = RefinementReport(
            iterations=0,
            start_score=start_score,
            end_score=current_score,
            stopped_reason=stopped_reason,
            steps=tuple(steps),
        )
        if return_report:
            return current_rubric, report
        return current_rubric

    stopped_reason = "max_iters"
    for iteration in range(max_iters):
        weak = [cid for cid, score in current_meta_result.subscores.items() if score < 3]
        mutations = _suggest_mutations(current_rubric, weak)

        if not mutations:
            steps.append(
                RefinementStep(
                    kind="refine_iter",
                    reason="no mutations suggested",
                    before_version=current_rubric.version,
                    after_version=current_rubric.version,
                    mutation_names=(),
                    meta_score=current_score,
                )
            )
            if stop_on_no_mutations:
                stopped_reason = "no_mutations"
                break
            continue

        new_rubric = current_rubric.evolve(mutations)
        new_meta_result = META_EVALUATOR.evaluate(new_rubric.to_xml(), client=client, model=model)
        new_score = new_meta_result.score

        step = RefinementStep(
            kind="refine_iter",
            reason=f"iteration {iteration + 1} applied suggested mutations",
            before_version=current_rubric.version,
            after_version=new_rubric.version,
            mutation_names=tuple(type(m).__name__ for m in mutations),
            meta_score=new_score,
        )
        steps.append(step)

        if current_score is not None and new_score is not None and new_score < current_score:
            # Regression: keep the previous rubric and stop.
            stopped_reason = "score_regressed"
            break

        # Advance state.
        current_rubric = new_rubric
        current_meta_result = new_meta_result
        current_score = new_score

        if target_score is not None and new_score is not None and new_score >= target_score:
            stopped_reason = "target_met"
            break

    _attach_refine_provenance(current_rubric, steps, evaluated_by_model=model)

    report = RefinementReport(
        iterations=len(steps),
        start_score=start_score,
        end_score=current_score,
        stopped_reason=stopped_reason,
        steps=tuple(steps),
    )

    if return_report:
        return current_rubric, report
    return current_rubric


def _attach_refine_provenance(
    rubric: Rubric, steps: list[RefinementStep], *, evaluated_by_model: str = ""
) -> None:
    """Append ``steps`` to ``rubric.provenance``, creating it if needed."""
    if rubric.provenance is None:
        rubric.provenance = RubricProvenance()
    if evaluated_by_model and not rubric.provenance.evaluated_by_model:
        rubric.provenance.evaluated_by_model = evaluated_by_model
    for step in steps:
        rubric.provenance.add_step(step)


def _suggest_mutations(rubric: Rubric, weak_properties: list[str]) -> list[RubricMutation]:
    """Map META_EVALUATOR weakness signals to concrete mutations.

    META_EVALUATOR criterion -> property predicate mapping:
      C1 -> p_anchored       (can't fix programmatically: needs content)
      C2 -> p_steering        (CAN fix: add default steering constraint)
      C3 -> p_mechanical      (can't fix programmatically: needs domain knowledge)
      C4 -> p_dq              (CAN fix: add schema-violation DQ)
      C5 -> p_economy         (can't fix: needs judgment about which criteria to cut)

    Returns mutations for the cases we CAN handle structurally.
    """
    from rubrify._mutations import AddDisqualifier, AddSteeringConstraint
    from rubrify._properties import p_dq, p_steering
    from rubrify._types import Disqualifier

    mutations: list[RubricMutation] = []

    for cid in weak_properties:
        if cid == "C2":
            if p_steering(rubric) == 0 and rubric.output_schema is not None:
                mutations.append(
                    AddSteeringConstraint(
                        key="rationale_anchor",
                        value="Begin with 'BECAUSE:' and end with '.'; exactly 35 words.",
                    )
                )

        elif cid == "C4" and p_dq(rubric) == 0:
            mutations.append(
                AddDisqualifier(
                    Disqualifier(
                        id="DQ1",
                        description="Output violates required schema format.",
                    )
                )
            )

    return mutations


# ---------------------------------------------------------------------------
# Phase 5: behavior-oriented generation helpers
# ---------------------------------------------------------------------------
#
# These helpers expose the rubric-behavior families named in PHILOSOPHY.md
# anchor 2 ("roleplaying == jailbreak == context following == rubrics") as
# concrete entry points. The split is deliberate and documented:
#
#   * ``generate_evaluator``, ``generate_detector``, ``generate_classifier``
#     are thin wrappers around :func:`generate` and DO call the LLM. They
#     route to the existing ``SCORING_GENERATOR`` / ``DETECTION_GENERATOR`` /
#     ``COMPLIANCE_GENERATOR`` profiles and return a full-fledged
#     :class:`rubrify.rubric.Rubric` (or the same tuple shapes that
#     :func:`generate` produces when ``return_report=True`` or
#     ``evaluate=True``).
#
#   * ``generate_constraint``, ``generate_transformer``, and
#     ``generate_from_examples`` are pure Python construction helpers. They
#     do NOT call the LLM. The "generation" here is building a
#     :class:`rubrify.rubric.ConstraintRubric` from user-supplied
#     instructions, examples, and templates. Guard rail 1 (no abstraction
#     without a reference) keeps these thin — they are the obvious
#     one-liners a user would otherwise copy into their own code.


def generate_evaluator(
    source: str,
    *,
    client: Any,
    model: str,
    **kwargs: Any,
) -> Rubric | tuple[Rubric, EvaluationResult] | tuple[Rubric, RefinementReport]:
    """Generate a scoring rubric (evaluator behavior).

    Thin wrapper around :func:`generate` with ``rubric_type="scoring"``.
    This helper DOES call the LLM: the scoring-generator meta-rubric is
    applied to ``source`` to author the XML ``<LLM_JUDGE_SPEC>``.
    """
    return generate(source, client=client, model=model, rubric_type="scoring", **kwargs)


def generate_detector(
    source: str,
    *,
    client: Any,
    model: str,
    **kwargs: Any,
) -> Rubric | tuple[Rubric, EvaluationResult] | tuple[Rubric, RefinementReport]:
    """Generate a detection rubric (detector behavior with inverted scoring).

    Thin wrapper around :func:`generate` with ``rubric_type="detection"``.
    This helper DOES call the LLM.
    """
    return generate(source, client=client, model=model, rubric_type="detection", **kwargs)


def generate_classifier(
    source: str,
    *,
    client: Any,
    model: str,
    **kwargs: Any,
) -> Rubric | tuple[Rubric, EvaluationResult] | tuple[Rubric, RefinementReport]:
    """Generate a compliance-style classification rubric (judge behavior).

    Thin wrapper around :func:`generate` with ``rubric_type="compliance"``.
    This helper DOES call the LLM.
    """
    return generate(source, client=client, model=model, rubric_type="compliance", **kwargs)


def generate_constraint(
    instructions: str,
    *,
    output_format: str = "",
    examples: list[ICLExample] | None = None,
    name: str = "",
    behaviors: frozenset[str] = frozenset({"force"}),
) -> ConstraintRubric:
    """Build a :class:`ConstraintRubric` from instructions and examples.

    This is a Python construction helper, NOT an LLM call. The instructions
    and examples are provided directly by the caller and wrapped into a
    :class:`ConstraintRubric`. ``behaviors`` is metadata-only per guard
    rail 3 and defaults to ``frozenset({"force"})`` because forcing is the
    canonical constraint behavior from
    ``references/main/rubrics/special_ones/completeness_rubric.md``.
    """
    return ConstraintRubric(
        name=name,
        instructions=instructions,
        output_format=output_format,
        examples=examples or [],
        behaviors=behaviors,
    )


def generate_transformer(
    instructions: str,
    *,
    template: str = "",
    placeholders: tuple[str, ...] = ("content",),
    examples: list[ICLExample] | None = None,
    name: str = "",
) -> ConstraintRubric:
    """Build a transformation :class:`ConstraintRubric` with a
    :class:`~rubrify.input_render.TemplateRenderer`.

    This is a Python construction helper, NOT an LLM call. When
    ``template`` is non-empty a ``TemplateRenderer`` is attached so the
    user message is built via ``{placeholder}`` substitution at evaluation
    time. ``behaviors`` is always ``frozenset({"transform"})``.
    """
    from rubrify.input_render import TemplateRenderer

    rubric = ConstraintRubric(
        name=name,
        instructions=instructions,
        output_format="",
        examples=examples or [],
        behaviors=frozenset({"transform"}),
    )
    if template:
        rubric.input_renderer = TemplateRenderer(
            template=template,
            placeholders=placeholders,
        )
    return rubric


def generate_from_examples(
    task_description: str,
    examples: list[ICLExample],
    *,
    name: str = "",
    behaviors: frozenset[str] = frozenset({"extract"}),
) -> ConstraintRubric:
    """Build a :class:`ConstraintRubric` driven primarily by ICL examples.

    This is a Python construction helper, NOT an LLM call. The
    ``task_description`` becomes ``instructions`` and the provided
    ``examples`` list is passed through unchanged. Default behavior tag is
    ``frozenset({"extract"})`` because extraction is the canonical
    example-driven rubric family.
    """
    return ConstraintRubric(
        name=name,
        instructions=task_description,
        output_format="",
        examples=examples,
        behaviors=behaviors,
    )
