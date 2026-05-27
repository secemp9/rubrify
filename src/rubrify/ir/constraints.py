"""ConstraintBinding, RitualConstraint, SurfacePolicy — the thesis in code.

The researcher's core insight: (roleplaying == jailbreak == context following) == rubrics.

A ConstraintBinding is the explicit triple connecting:
  1. A semantic criterion (what we evaluate)
  2. A surface-layer projection (how the LLM sees it — XML tag, JSON path)
  3. An output field (where the LLM writes its judgment)

RitualConstraint formalizes "useful weirdness" with TYPED fields — not
natural-language strings parsed by string manipulation.
"""

from __future__ import annotations

from typing import Literal

from rubrify.ir.types import SchemaModel


class SurfaceProjection(SchemaModel):
    """One surface-layer representation of a constraint."""
    codec: str
    path: str
    attributes: dict[str, str] = {}


class ConstraintBinding(SchemaModel):
    """The triple-layer alignment: criterion <-> surface tag <-> output field."""
    criterion_id: str
    output_field: str
    evidence_source: str = "response"
    authority: Literal["instruction", "data", "meta"] = "instruction"
    projections: list[SurfaceProjection] = []


class AuthorityBlock(SchemaModel):
    """Marks a prompt section as instruction vs. data."""
    id: str
    authority: Literal["instruction", "data", "meta"]
    kind: Literal[
        "system_prompt", "rubric_spec", "judge_instructions",
        "response_under_test", "context_document", "calibration_example",
    ]
    model_should_follow: bool = True


class RitualConstraint(SchemaModel):
    """A 'useful weirdness' constraint with TYPED enforcement parameters.

    The description field is the human/prompt-facing text. Enforcement
    operates on the structured fields directly — no string parsing.
    """
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    # Typed parameters — only the relevant ones are set per ritual
    prefix: str | None = None
    suffix: str | None = None
    token: str | None = None
    word_count: int | None = None
    word_count_mode: Literal["exactly", "max", "min"] | None = None


__all__ = [
    "AuthorityBlock",
    "ConstraintBinding",
    "RitualConstraint",
    "SurfaceProjection",
]
