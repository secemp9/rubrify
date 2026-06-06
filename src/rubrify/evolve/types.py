"""Evolve types: data structures for rubric evolution via GEPA.

AnnotatedExample: a single (response, human_annotation) pair.
JudgmentTrajectory: per-example trace for GEPA's reflective dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rubrify.engine.judgment import Judgment


@dataclass(slots=True)
class AnnotatedExample:
    """A single (response, human_annotation) pair for rubric validation."""
    id: str
    response_text: str
    context_text: str | None
    # Human annotations: criterion_id -> human-assigned score (on the criterion's native scale)
    human_scores: dict[str, float | int | str | bool]
    # Optional: overall human quality label (e.g., "good", "poor")
    human_label: str | None = None
    # Optional: genre tag for genre-conditional criteria
    genre: str | None = None


@dataclass(slots=True)
class JudgmentTrajectory:
    """Per-example trajectory capturing everything the reflection LM needs.

    Opaque to GEPA -- only consumed by RubricEvolverAdapter.make_reflective_dataset().
    """
    example: AnnotatedExample
    judgment: Judgment
    per_criterion_errors: dict[str, float]  # criterion_id -> |human - judge| / scale_range
    compilation_issues: list[str]


__all__ = [
    "AnnotatedExample",
    "JudgmentTrajectory",
]
