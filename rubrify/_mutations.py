from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rubrify.rubric import Rubric

from rubrify._types import Criterion, Disqualifier, MappingExample


@dataclass(slots=True)
class AddCriterion:
    criterion: Criterion

    def apply(self, rubric: Rubric) -> Rubric:
        rubric.add_criterion(self.criterion)
        return rubric


@dataclass(slots=True)
class RemoveCriterion:
    criterion_id: str

    def apply(self, rubric: Rubric) -> Rubric:
        if self.criterion_id not in rubric.criteria:
            raise KeyError(f"Criterion {self.criterion_id!r} not found")
        del rubric.criteria[self.criterion_id]
        return rubric


@dataclass(slots=True)
class AdjustWeight:
    criterion_id: str
    new_weight: int

    def apply(self, rubric: Rubric) -> Rubric:
        if self.criterion_id not in rubric.criteria:
            raise KeyError(f"Criterion {self.criterion_id!r} not found")
        rubric.criteria[self.criterion_id].weight = self.new_weight
        return rubric


@dataclass(slots=True)
class AddPattern:
    pattern_id: str
    pattern: str

    def apply(self, rubric: Rubric) -> Rubric:
        from rubrify._types import PatternLibrary

        if rubric.pattern_library is None:
            rubric.pattern_library = PatternLibrary()
        rubric.pattern_library.add(self.pattern_id, self.pattern)
        return rubric


@dataclass(slots=True)
class AddDisqualifier:
    disqualifier: Disqualifier

    def apply(self, rubric: Rubric) -> Rubric:
        rubric.add_disqualifier(self.disqualifier)
        return rubric


@dataclass(slots=True)
class AddSteeringConstraint:
    """Add or update a constraint in the output schema."""

    key: str
    value: str

    def apply(self, rubric: Rubric) -> Rubric:
        from rubrify._types import OutputSchema

        if rubric.output_schema is None:
            rubric.output_schema = OutputSchema()
        rubric.output_schema.constraints[self.key] = self.value
        return rubric


@dataclass(slots=True)
class AddMappingExample:
    """Append a :class:`MappingExample` to a rubric's mapping examples list.

    Used by ``calibration_to_mutations`` as a conservative structural fix:
    when a compliance rubric fails an ``expected_verdict`` calibration case,
    the bridge suggests scaffolding a new mapping example for that case so a
    human or model can fill in the content. No content is invented.
    """

    example: MappingExample

    def apply(self, rubric: Rubric) -> Rubric:
        rubric.mapping_examples.append(self.example)
        return rubric


RubricMutation = (
    AddCriterion
    | RemoveCriterion
    | AdjustWeight
    | AddPattern
    | AddDisqualifier
    | AddSteeringConstraint
    | AddMappingExample
)


def _bump_version(version: str) -> str:
    """Bump minor version: '1.0' -> '2.0', '3.0' -> '4.0'."""
    parts = version.split(".")
    major = int(parts[0]) + 1
    return f"{major}.0"
