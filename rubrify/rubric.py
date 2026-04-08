from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

from rubrify._types import (
    AdviceRule,
    Criterion,
    DecisionRule,
    Disqualifier,
    ICLExample,
    InputField,
    MappingExample,
    OutputSchema,
    PatternLibrary,
    Scoring,
    ValidationMust,
)
from rubrify.result import ConstraintResult, EvaluationResult, EvaluationTrace

if TYPE_CHECKING:
    from rubrify.input_render import InputRenderer
    from rubrify.provenance import RubricProvenance
    from rubrify.repair import RepairResult


def _infer_parser_kind(schema: OutputSchema | None) -> str:
    """Return the parser label used by ``EvaluationTrace``."""
    if schema is None:
        return "raw"
    if schema.constraints.get("must_be_json"):
        return "json"
    if schema.constraints.get("must_use_xml_tags"):
        return "xml"
    return "raw"


class Rubric:
    def __init__(
        self,
        name: str = "",
        version: str = "1.0",
        mission: str = "",
    ) -> None:
        self.name = name
        self.version = version
        self.mission = mission
        self.inputs: list[InputField] = []
        self.criteria: dict[str, Criterion] = {}
        self.disqualifiers: list[Disqualifier] = []
        self.output_schema: OutputSchema | None = None
        self.scoring: Scoring | None = None
        self.pattern_library: PatternLibrary | None = None
        self.decision_logic: list[DecisionRule] = []
        self.advice_rules: list[AdviceRule] = []
        self.mapping_examples: list[MappingExample] = []
        self.validation_musts: list[ValidationMust] = []
        self.definitions: dict[str, str] = {}  # For ComplianceJudge <definitions>
        self.what_to_judge: str = ""  # For ComplianceJudge <what_to_judge>
        self.scoring_guidance: str = ""  # For ComplianceJudge <scoring_guidance>
        self.input_renderer: InputRenderer | None = None  # Phase 1: optional user-msg renderer
        # Phase 4: optional lineage metadata. NEVER emitted in to_xml().
        self.provenance: RubricProvenance | None = None

    def add_criterion(self, criterion: Criterion) -> None:
        self.criteria[criterion.id] = criterion

    def add_disqualifier(self, dq: Disqualifier) -> None:
        self.disqualifiers.append(dq)

    def add_decision_rule(self, rule: DecisionRule) -> None:
        self.decision_logic.append(rule)

    def add_advice_rule(self, rule: AdviceRule) -> None:
        self.advice_rules.append(rule)

    def add_mapping_example(self, example: MappingExample) -> None:
        self.mapping_examples.append(example)

    @property
    def output_format(self) -> str:
        """Detect output format from output_schema constraints."""
        if self.output_schema is None:
            return "unknown"
        if self.output_schema.constraints.get("must_be_json"):
            return "json"
        if self.output_schema.constraints.get("must_use_xml_tags"):
            return "xml"
        return self.output_schema.format

    def genre_criteria(self, genre: str) -> list[Criterion]:
        """Return criteria matching a specific genre slug."""
        result = []
        for c in self.criteria.values():
            if c.genre and genre in c.genre.split(","):
                result.append(c)
        return result

    def copy(self) -> Rubric:
        return copy.deepcopy(self)

    def to_xml(self) -> str:
        from rubrify.xml_io import rubric_to_xml

        return rubric_to_xml(self)

    def to_json(self) -> str:
        """Serialize the rubric to a JSON string."""
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "mission": self.mission,
        }
        if self.inputs:
            data["inputs"] = [
                {"name": f.name, "required": f.required, "description": f.description}
                for f in self.inputs
            ]
        if self.criteria:
            data["criteria"] = {
                cid: {
                    "id": c.id,
                    "name": c.name,
                    "weight": c.weight,
                    "anchors": {str(k): v for k, v in c.anchors.items()},
                    "mechanical_rules": c.mechanical_rules,
                    "uses_patterns": c.uses_patterns,
                    "genre": c.genre,
                    "notes": c.notes,
                }
                for cid, c in self.criteria.items()
            }
        if self.disqualifiers:
            data["disqualifiers"] = [
                {"id": dq.id, "description": dq.description} for dq in self.disqualifiers
            ]
        if self.output_schema:
            data["output_schema"] = {
                "format": self.output_schema.format,
                "template": self.output_schema.template,
                "constraints": self.output_schema.constraints,
            }
        if self.scoring:
            data["scoring"] = {
                "formula": self.scoring.formula,
                "labels": {f"{lo}-{hi}": label for (lo, hi), label in self.scoring.labels.items()},
                "inverted": self.scoring.inverted,
            }
        if self.pattern_library:
            data["pattern_library"] = {
                "entries": self.pattern_library.entries,
                "flags": self.pattern_library.flags,
            }
        if self.decision_logic:
            data["decision_logic"] = [
                {"id": r.id, "condition": r.condition} for r in self.decision_logic
            ]
        if self.advice_rules:
            data["advice_rules"] = [{"when": r.when, "advice": r.advice} for r in self.advice_rules]
        if self.mapping_examples:
            data["mapping_examples"] = [
                {
                    "id": e.id,
                    "user": e.user,
                    "assistant": e.assistant,
                    "verdict": e.verdict,
                }
                for e in self.mapping_examples
            ]
        if self.validation_musts:
            data["validation_musts"] = [m.description for m in self.validation_musts]
        if self.definitions:
            data["definitions"] = self.definitions
        if self.what_to_judge:
            data["what_to_judge"] = self.what_to_judge
        if self.scoring_guidance:
            data["scoring_guidance"] = self.scoring_guidance
        return json.dumps(data, indent=2)

    def evaluate(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        repair: bool = False,
        observe: bool = False,
        warn_unsupported: bool = False,
        **kwargs: Any,
    ) -> EvaluationResult:
        import dataclasses

        from rubrify.input_render import CandidateTextRenderer, validate_payload
        from rubrify.parse import parse_response

        if warn_unsupported:
            from rubrify.model_policy import warn_unsupported as _warn_unsupported

            _warn_unsupported(model)

        system_msg = self.to_xml()

        payload: dict[str, Any] = {"text": text, **kwargs}
        # Legacy alias: the pre-Phase-1 implementation always wrapped ``text`` in
        # <candidate_text>, which means any rubric declaring ``candidate_text`` as
        # a required input expects ``text`` to satisfy it. Expose it explicitly
        # so ``validate_payload`` accepts the legacy call shape.
        if "candidate_text" not in payload:
            payload["candidate_text"] = text

        if self.inputs:
            validate_payload(payload, self.inputs)

        renderer = self.input_renderer or CandidateTextRenderer()
        user_msg = renderer.render(payload)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        start = time.monotonic()
        raw = client.chat(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
        )
        elapsed = time.monotonic() - start

        result = parse_response(raw, self.output_schema, repair=repair)

        if observe:
            trace = EvaluationTrace(
                system_prompt=system_msg,
                user_message=user_msg,
                model=model,
                parser=_infer_parser_kind(self.output_schema),
                repair_notes=result.repair_notes,
                elapsed_seconds=elapsed,
            )
            result = dataclasses.replace(result, trace=trace)

        return result

    def evaluate_batch(
        self,
        texts: list[str],
        *,
        client: Any,
        model: str,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate multiple texts sequentially."""
        return [self.evaluate(text, client=client, model=model, **kwargs) for text in texts]

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).write_text(self.to_xml(), encoding="utf-8")

    def export_provenance(self, path: str) -> None:
        """Write ``self.provenance`` to a sidecar JSON file at ``path``.

        Phase 4: provenance is runtime metadata and does not belong in
        canonical XML (guard rail 5). Callers who want to persist a
        rubric's lineage alongside its XML emit a sidecar JSON file via
        this method. Raises ``ValueError`` if no provenance has been
        attached — guard rail 6 forbids silent no-ops.
        """
        from pathlib import Path

        if self.provenance is None:
            raise ValueError(
                f"Rubric {self.name!r} has no provenance to export; "
                "attach a RubricProvenance before calling export_provenance"
            )
        payload = json.dumps(self.provenance.to_dict(), indent=2)
        Path(path).write_text(payload, encoding="utf-8")

    def __or__(self, other: Rubric) -> Rubric:
        """Criteria union: merge two rubrics. Second wins on ID conflicts."""
        result = self.copy()
        # Criteria union (second wins conflicts)
        for cid, crit in other.criteria.items():
            result.criteria[cid] = copy.deepcopy(crit)
        # DQ union
        existing_dq_ids = {dq.id for dq in result.disqualifiers}
        for dq in other.disqualifiers:
            if dq.id not in existing_dq_ids:
                result.disqualifiers.append(copy.deepcopy(dq))
        # Mission concat
        if other.mission:
            if result.mission:
                result.mission = result.mission + " " + other.mission
            else:
                result.mission = other.mission
        # Merge pattern libraries
        if other.pattern_library:
            if result.pattern_library is None:
                result.pattern_library = copy.deepcopy(other.pattern_library)
            else:
                for pid, pat in other.pattern_library.entries.items():
                    result.pattern_library.entries[pid] = pat
                    if pid in other.pattern_library._entry_types:
                        result.pattern_library._entry_types[pid] = (
                            other.pattern_library._entry_types[pid]
                        )
        return result

    def __and__(self, other: Rubric) -> ProductRubric:
        """Product: parallel evaluation against both rubrics."""
        return ProductRubric([self, other])

    def project(self, criterion_ids: set[str]) -> Rubric:
        """Project to a subset of criteria. IDs not found are silently ignored."""
        result = self.copy()
        result.criteria = {
            cid: crit for cid, crit in result.criteria.items() if cid in criterion_ids
        }
        return result

    def reweight(self, weights: dict[str, int]) -> Rubric:
        """Return a copy with updated criterion weights."""
        result = self.copy()
        for cid, new_weight in weights.items():
            if cid in result.criteria:
                result.criteria[cid].weight = new_weight
        return result

    def evolve(self, mutations: list[Any]) -> Rubric:
        """Apply a sequence of mutations and bump version."""
        from rubrify._mutations import _bump_version

        result = self.copy()
        for mutation in mutations:
            mutation.apply(result)
        result.version = _bump_version(result.version)
        return result

    def __repr__(self) -> str:
        return (
            f"Rubric(name={self.name!r}, version={self.version!r}, criteria={len(self.criteria)})"
        )


class ConstraintRubric:
    """Behavioral constraint rubric (no scoring). Uses instructions + ICL examples.

    Phase 2 enrichment: carries a ``behaviors`` frozenset of metadata tags
    (``judge``, ``score``, ``detect``, ``force``, ``transform``, ``extract``,
    ``calibrate``) describing what the rubric *claims* to do, and a list of
    local ``validators`` that can post-hoc check the raw model output.

    **``behaviors`` is metadata only.** Behaviors compose — a rubric may carry
    any subset of the seven canonical values (see ``rubrify._behaviors`` for
    the full taxonomy and reference citations). Guard rail 3 of
    ``PHILOSOPHY.md`` is binding: the runtime never branches on ``behaviors``.
    Two rubrics that differ only in their ``behaviors`` frozenset execute
    through identical code paths.
    """

    def __init__(
        self,
        name: str = "",
        instructions: str = "",
        output_format: str = "",
        examples: list[ICLExample] | None = None,
        *,
        behaviors: frozenset[str] = frozenset(),
        validators: list[Callable[[str], tuple[bool, str | None]]] | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.output_format = output_format
        self.examples: list[ICLExample] = examples or []
        self.input_renderer: InputRenderer | None = None  # Phase 1: optional renderer
        # Phase 2: metadata-only behavior tags; never drives dispatch.
        self.behaviors: frozenset[str] = behaviors
        # Phase 2: local structural validators run on the raw model output.
        self.validators: list[Callable[[str], tuple[bool, str | None]]] = (
            list(validators) if validators is not None else []
        )

    @overload
    def apply(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        parse_as: str | None = ...,
        repair: bool = ...,
        observe: Literal[False] = ...,
        warn_unsupported: bool = ...,
        **kwargs: Any,
    ) -> str | dict[str, Any]: ...

    @overload
    def apply(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        parse_as: str | None = ...,
        repair: bool = ...,
        observe: Literal[True],
        warn_unsupported: bool = ...,
        **kwargs: Any,
    ) -> tuple[str | dict[str, Any], EvaluationTrace]: ...

    def apply(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        parse_as: str | None = None,
        repair: bool = False,
        observe: bool = False,
        warn_unsupported: bool = False,
        **kwargs: Any,
    ) -> str | dict[str, Any] | tuple[str | dict[str, Any], EvaluationTrace]:
        if warn_unsupported:
            from rubrify.model_policy import warn_unsupported as _warn_unsupported

            _warn_unsupported(model)

        system_msg = self.to_xml()

        if self.input_renderer is not None:
            payload: dict[str, Any] = {"text": text, **kwargs}
            user_msg = self.input_renderer.render(payload)
        else:
            user_msg = text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        start = time.monotonic()
        raw: str = client.chat(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
        )
        elapsed = time.monotonic() - start

        repair_notes: tuple[str, ...] = ()
        parsed: str | dict[str, Any]

        if parse_as == "json":
            source = raw
            if repair:
                from rubrify.repair import extract_json_candidate

                repair_result = extract_json_candidate(raw)
                repair_notes = repair_result.notes
                if repair_result.repaired:
                    source = repair_result.text
            parsed = json.loads(source)
        else:
            parsed = raw

        if observe:
            parser_kind = "json" if parse_as == "json" else "raw"
            trace = EvaluationTrace(
                system_prompt=system_msg,
                user_message=user_msg,
                model=model,
                parser=parser_kind,
                repair_notes=repair_notes,
                elapsed_seconds=elapsed,
            )
            return parsed, trace

        return parsed

    def validate_output(self, output: str) -> tuple[bool, list[str]]:
        """Run every registered validator on ``output`` and aggregate violations.

        Each validator returns ``(is_valid, message | None)``. Returns
        ``(all_valid, violations)`` where ``violations`` is the list of
        non-``None`` messages collected from failing validators. If no
        validators are registered the call returns ``(True, [])``.
        """
        violations: list[str] = []
        for validator in self.validators:
            is_valid, message = validator(output)
            if not is_valid:
                violations.append(message if message is not None else "validation failed")
        return (len(violations) == 0, violations)

    def apply_and_validate(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        repair: bool = False,
        observe: bool = False,
        **kwargs: Any,
    ) -> ConstraintResult:
        """Apply the rubric and run ``self.validators`` on the raw model output.

        Returns a :class:`ConstraintResult` with ``output``, ``valid``,
        ``violations``, and (when ``observe=True``) ``trace`` populated.
        Validators operate on the raw string output; ``parse_as`` is therefore
        not accepted here. Callers who need parsed payloads should use
        :meth:`apply` directly.
        """
        if "parse_as" in kwargs:
            raise TypeError(
                "apply_and_validate does not accept parse_as; validators run on the raw output"
            )

        trace: EvaluationTrace | None = None
        if observe:
            applied = self.apply(
                text,
                client=client,
                model=model,
                repair=repair,
                observe=True,
                **kwargs,
            )
            output_value, trace = applied
        else:
            output_value = self.apply(
                text,
                client=client,
                model=model,
                repair=repair,
                observe=False,
                **kwargs,
            )

        if not isinstance(output_value, str):
            raise TypeError(
                "apply_and_validate expects ConstraintRubric.apply to return a string; "
                f"got {type(output_value).__name__}"
            )

        valid, violations = self.validate_output(output_value)
        return ConstraintResult(
            output=output_value,
            valid=valid,
            violations=tuple(violations),
            trace=trace,
        )

    def apply_with_repair(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        repair_fn: Callable[[str], RepairResult] | None = None,
        **kwargs: Any,
    ) -> ConstraintResult:
        """Apply, validate, and optionally run a *local* repair function on failure.

        If validation passes, the initial :class:`ConstraintResult` is returned
        unchanged. If validation fails and ``repair_fn`` is provided, the
        function is called with the raw output and must return a
        :class:`rubrify.repair.RepairResult`. The repaired text is revalidated
        and a new :class:`ConstraintResult` is returned with ``repaired`` and
        ``repair_notes`` populated from the repair result. ``repair_fn`` is
        strictly local — no model-assisted repair.
        """
        initial = self.apply_and_validate(text, client=client, model=model, **kwargs)
        if initial.valid or repair_fn is None:
            return initial

        repaired = repair_fn(initial.output)
        revalid, reviolations = self.validate_output(repaired.text)
        return ConstraintResult(
            output=repaired.text,
            valid=revalid,
            violations=tuple(reviolations),
            repaired=repaired.repaired,
            repair_notes=repaired.notes,
            trace=initial.trace,
        )

    def to_xml(self) -> str:
        """Serialize to XML system prompt format."""
        from rubrify.xml_io import constraint_rubric_to_xml

        return constraint_rubric_to_xml(self)

    def __repr__(self) -> str:
        return f"ConstraintRubric(name={self.name!r}, examples={len(self.examples)})"


class ProductRubric:
    """Parallel evaluation against multiple rubrics. Returns list of results."""

    def __init__(self, rubrics: list[Rubric]) -> None:
        self.rubrics = rubrics

    def evaluate(
        self, text: str, *, client: Any, model: str, **kwargs: Any
    ) -> list[EvaluationResult]:
        return [r.evaluate(text, client=client, model=model, **kwargs) for r in self.rubrics]


class CoproductRubric:
    """Conditional dispatch: selects a rubric based on input characteristics."""

    def __init__(
        self,
        rubrics: dict[str, Rubric],
        selector: Callable[..., str],
    ) -> None:
        self.rubrics = rubrics
        self.selector = selector

    def evaluate(self, text: str, *, client: Any, model: str, **kwargs: Any) -> EvaluationResult:
        key = self.selector(text, **kwargs)
        return self.rubrics[key].evaluate(text, client=client, model=model, **kwargs)
