"""Evolve types: data structures for rubric evolution via GEPA.

AnnotatedExample: a single (response, human_annotation) pair.
JudgmentTrajectory: per-example trace for GEPA's reflective dataset.
RubricQualityScore: multi-dimensional rubric quality assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(slots=True)
class RubricQualityScore:
    """Multi-dimensional rubric quality assessment."""
    # Primary: agreement with human annotations (normalized, 0-1)
    agreement: float
    # Consistency: same response -> same score across N runs (1 - CV, 0-1)
    consistency: float
    # Discrimination: spread of scores across quality levels (normalized entropy, 0-1)
    discrimination: float
    # Composite: weighted combination for GEPA's scalar score
    composite: float
    # Per-criterion diagnostics
    per_criterion_agreement: dict[str, float] = field(default_factory=dict)
    per_criterion_consistency: dict[str, float] = field(default_factory=dict)
    # The raw judgments for trajectory capture
    judgments: list[Judgment] = field(default_factory=list)


__all__ = [
    "AnnotatedExample",
    "JudgmentTrajectory",
    "RubricQualityScore",
]
