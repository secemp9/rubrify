"""Generation and refinement of rubrics from natural language."""

from __future__ import annotations

from typing import Any

from rubrify._meta_rubric import (
    COMPLIANCE_GENERATOR,
    DETECTION_GENERATOR,
    META_EVALUATOR,
    SCORING_GENERATOR,
)
from rubrify._properties import validate
from rubrify.result import EvaluationResult
from rubrify.rubric import Rubric

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
    **kwargs: Any,
) -> Rubric | tuple[Rubric, EvaluationResult]:
    """Generate a rubric from source material using the any2rubric pipeline.

    Gen: Src -> Rub (the generation functor)
    """
    import rubrify

    generator = _GENERATORS.get(rubric_type)
    if generator is None:
        raise ValueError(f"Unknown rubric_type: {rubric_type!r}. Use: {list(_GENERATORS)}")

    # Apply the generation constraint (always returns raw XML string)
    raw_output = str(generator.apply(source, client=client, model=model, **kwargs))
    raw_xml = _extract_xml(raw_output)

    # Parse the generated XML into a Rubric object
    rubric = rubrify.loads(raw_xml)
    if name:
        rubric.name = name

    # Validate against property lattice
    validation = validate(rubric)
    if not validation.is_valid:
        error_msgs = [e.message for e in validation.errors]
        raise ValueError(f"Generated rubric failed validation: {error_msgs}")

    if evaluate:
        meta_result = META_EVALUATOR.evaluate(raw_xml, client=client, model=model)
        return rubric, meta_result

    return rubric


def refine(
    rubric: Rubric,
    *,
    client: Any,
    model: str,
    corpus: list[str] | None = None,
) -> Rubric:
    """Refine a rubric using the Refine monad.

    Refine: Rub -> Rub (the refinement endofunctor)
    Evaluates the rubric with META_EVALUATOR, identifies weaknesses,
    and returns an evolved rubric.
    """
    # Step 1: Meta-evaluate
    meta_result = META_EVALUATOR.evaluate(rubric.to_xml(), client=client, model=model)

    # Step 2: Identify weak properties
    weak = [cid for cid, score in meta_result.subscores.items() if score < 3]

    # Step 3: Generate mutations based on weaknesses
    from rubrify._mutations import RubricMutation

    mutations: list[RubricMutation] = _suggest_mutations(rubric, weak, meta_result)

    # Step 4: Evolve
    if mutations:
        return rubric.evolve(mutations)
    return rubric


def _suggest_mutations(
    rubric: Rubric, weak_properties: list[str], meta_result: EvaluationResult
) -> list[Any]:
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

    mutations: list[Any] = []

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
