"""Core IR type system for rubrify.

Scale types are polymorphic and protocol-driven — each knows its own
domain and scoring semantics. This mirrors harn_ai/types.py in style:
SchemaModel with extra="forbid", discriminated unions.

The type hierarchy:

    Scale (discriminated by .kind)
      ├── BinaryScale       — pass/fail, yes/no
      ├── OrdinalScale      — ordered levels with anchors
      ├── NominalScale      — unordered categories
      └── NumericScale      — continuous range [min, max]

    Criterion               — atomic evaluation unit
    CriterionGroup          — hierarchical grouping
    Disqualifier            — auto-fail condition
    Rubric                  — the mutable pre-compilation object
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from harn_ai.types import SchemaModel


# ── Scale types ────────────────────────────────────────────────────

class ScaleAnchor(SchemaModel):
    """A single anchor point on a scale — score value mapped to description."""
    value: int | float
    label: str
    description: str = ""


class BinaryScale(SchemaModel):
    kind: Literal["binary"] = "binary"
    true_label: str = "yes"
    false_label: str = "no"
    true_score: float = 1.0
    false_score: float = 0.0

    def domain(self) -> dict[str, Any]:
        return {"type": "boolean", "labels": [self.false_label, self.true_label]}

    def to_unit(self, value: bool) -> float:
        return self.true_score if value else self.false_score


class OrdinalScale(SchemaModel):
    kind: Literal["ordinal"] = "ordinal"
    anchors: list[ScaleAnchor]

    @field_validator("anchors")
    @classmethod
    def _anchors_nonempty(cls, v: list[ScaleAnchor]) -> list[ScaleAnchor]:
        if not v:
            raise ValueError("ordinal scale must have at least one anchor")
        return v

    def domain(self) -> dict[str, Any]:
        return {
            "type": "enum",
            "ordered": True,
            "values": [a.label for a in self.anchors],
        }

    def to_unit(self, value: int | str) -> float:
        scores = [a.value for a in self.anchors]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return 0.0
        if isinstance(value, (int, float)):
            raw = value
        else:
            match = next((a.value for a in self.anchors if a.label == value), None)
            if match is None:
                raise ValueError(
                    f"Unknown ordinal label: {value!r}. "
                    f"Valid: {[a.label for a in self.anchors]}"
                )
            raw = match
        return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


class NominalScale(SchemaModel):
    kind: Literal["nominal"] = "nominal"
    categories: list[ScaleAnchor]

    def domain(self) -> dict[str, Any]:
        return {
            "type": "enum",
            "ordered": False,
            "values": [c.label for c in self.categories],
        }

    def to_unit(self, value: str) -> float:
        match = next((c for c in self.categories if c.label == value), None)
        if match is None:
            raise ValueError(f"Unknown category: {value!r}")
        scores = [c.value for c in self.categories]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (match.value - lo) / (hi - lo)))


class NumericScale(SchemaModel):
    kind: Literal["numeric"] = "numeric"
    minimum: float
    maximum: float
    step: float = 1.0
    anchors: list[ScaleAnchor] = []

    @field_validator("maximum")
    @classmethod
    def _max_gt_min(cls, v: float, info) -> float:
        if "minimum" in info.data and v <= info.data["minimum"]:
            raise ValueError(f"maximum ({v}) must be greater than minimum ({info.data['minimum']})")
        return v

    @field_validator("step")
    @classmethod
    def _step_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"step must be positive, got {v}")
        return v

    def domain(self) -> dict[str, Any]:
        return {
            "type": "number",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
        }

    def to_unit(self, value: float | int) -> float:
        v = float(value)
        if self.maximum == self.minimum:
            return 0.0
        return max(0.0, min(1.0, (v - self.minimum) / (self.maximum - self.minimum)))


ScaleValue: TypeAlias = BinaryScale | OrdinalScale | NominalScale | NumericScale
Scale: TypeAlias = Annotated[ScaleValue, Field(discriminator="kind")]


# ── Evidence specification ─────────────────────────────────────────

class EvidenceSpec(SchemaModel):
    """What evidence the judge must provide for a criterion."""
    source: Literal["response", "context", "rubric", "any"] = "response"
    required: bool = True
    exact_quote: bool = False
    min_items: int = Field(default=0, ge=0)
    max_items: int = Field(default=8, gt=0)


# ── Criterion ──────────────────────────────────────────────────────

class Criterion(SchemaModel):
    """The atomic evaluation unit. One criterion = one independently decidable check.

    The 'prompt_key' field is intentionally separate from 'title' because
    schema-key wording steers model behavior (per JSONSchemaBench research).
    """
    id: str
    title: str
    description: str
    prompt_key: str | None = None
    scale: Scale
    weight: float = Field(default=1.0, ge=0)
    evidence: EvidenceSpec = EvidenceSpec()
    tags: list[str] = []
    genre: list[str] | None = None
    mechanical_rules: list[str] = []
    disqualifier: bool = False


class CriterionGroup(SchemaModel):
    """Hierarchical grouping of criteria with shared aggregation."""
    id: str
    title: str
    children: list[str]
    aggregation: Literal["weighted_sum", "weighted_mean", "min", "max", "all", "any"] = "weighted_sum"
    weight: float = Field(default=1.0, ge=0)


class Disqualifier(SchemaModel):
    """An auto-fail condition. If triggered, the entire judgment = score 0."""
    id: str
    description: str
    pattern: str | None = None
    criterion_ids: list[str] = []


# ── Definitions, advice rules, calibration examples ───────────────

class Definition(SchemaModel):
    """A semantic definition used in the rubric context.

    Definitions provide the judge with precise vocabulary for evaluation.
    E.g., COMPLY = "Provides direct, actionable content that fulfills the request."
    """
    id: str
    term: str
    description: str


class AdviceRule(SchemaModel):
    """Maps a pattern trigger to a concrete fix imperative.

    When patterns fire, the advice tells the evaluator what to recommend.
    E.g., trigger="puffery_words" → fix="Replace hype with concrete facts."
    """
    trigger_patterns: list[str]  # pattern IDs that trigger this advice
    fix: str  # the concrete imperative


class CalibrationExample(SchemaModel):
    """A few-shot example showing expected evaluation behavior.

    Embedded in the judge prompt to calibrate scoring consistency.
    """
    id: str
    input_summary: str  # what the response looks like
    expected_verdict: str  # what the judge should conclude
    explanation: str = ""  # why this verdict is correct


# ── Rubric ─────────────────────────────────────────────────────────

class RubricMeta(SchemaModel):
    """Rubric provenance and identity."""
    name: str
    version: str = "1.0"
    author: str | None = None
    description: str | None = None


class PatternEntry(SchemaModel):
    """A named regex or keyword pattern for mechanical checking."""
    id: str
    pattern: str
    flags: str = "i"


class Rubric(SchemaModel):
    """The mutable, pre-compilation rubric definition.

    A Rubric is not executed directly. It must be compiled into a
    RubricBundle via the compiler pipeline.
    """
    meta: RubricMeta
    goal: str
    criteria: list[Criterion]
    groups: list[CriterionGroup] = []
    disqualifiers: list[Disqualifier] = []
    instructions: list[str] = []
    patterns: list[PatternEntry] = []
    definitions: list[Definition] = []
    advice_rules: list[AdviceRule] = []
    calibration_examples: list[CalibrationExample] = []

    @model_validator(mode="after")
    def _unique_criterion_ids(self) -> "Rubric":
        ids = [c.id for c in self.criteria]
        seen = set()
        dupes = set()
        for i in ids:
            if i in seen:
                dupes.add(i)
            seen.add(i)
        if dupes:
            raise ValueError(f"Duplicate criterion IDs: {dupes}")
        return self

    @model_validator(mode="after")
    def _valid_group_refs(self) -> "Rubric":
        valid_ids = {c.id for c in self.criteria}
        for group in self.groups:
            bad = [c for c in group.children if c not in valid_ids]
            if bad:
                raise ValueError(f"Group '{group.id}' references unknown criteria: {bad}")
        return self

    @model_validator(mode="after")
    def _valid_dq_refs(self) -> "Rubric":
        valid_ids = {c.id for c in self.criteria}
        for dq in self.disqualifiers:
            bad = [c for c in dq.criterion_ids if c not in valid_ids]
            if bad:
                raise ValueError(f"Disqualifier '{dq.id}' references unknown criteria: {bad}")
        return self


__all__ = [
    "AdviceRule",
    "BinaryScale",
    "CalibrationExample",
    "Criterion",
    "CriterionGroup",
    "Definition",
    "Disqualifier",
    "EvidenceSpec",
    "NominalScale",
    "NumericScale",
    "OrdinalScale",
    "PatternEntry",
    "Rubric",
    "RubricMeta",
    "Scale",
    "ScaleAnchor",
    "SchemaModel",
]
