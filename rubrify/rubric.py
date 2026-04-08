from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

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
from rubrify.result import EvaluationResult


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

    def evaluate(self, text: str, *, client: Any, model: str, **kwargs: Any) -> EvaluationResult:
        from xml.sax.saxutils import escape as xml_escape

        from rubrify.parse import parse_response

        system_msg = self.to_xml()
        parts = [f"<candidate_text>{xml_escape(text)}</candidate_text>"]
        for key in ("context", "genre", "goal", "audience"):
            if key in kwargs:
                parts.append(f"<{key}>{xml_escape(str(kwargs[key]))}</{key}>")
        user_msg = "\n".join(parts)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        raw = client.chat(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
        )
        return parse_response(raw, self.output_schema)

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
    """Behavioral constraint rubric (no scoring). Uses instructions + ICL examples."""

    def __init__(
        self,
        name: str = "",
        instructions: str = "",
        output_format: str = "",
        examples: list[ICLExample] | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.output_format = output_format
        self.examples: list[ICLExample] = examples or []

    def apply(
        self,
        text: str,
        *,
        client: Any,
        model: str,
        parse_as: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        system_msg = self.to_xml()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ]
        raw: str = client.chat(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
        )
        if parse_as == "json":
            return json.loads(raw)  # type: ignore[no-any-return]
        return raw

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
