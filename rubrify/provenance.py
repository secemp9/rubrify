"""Rubric provenance and refinement lineage.

Phase 4 deliverable. Provenance is the minimum viable lineage for a rubric:
*where it came from*, *who generated or evaluated it*, *how it was refined*,
and *which calibration suites it was checked against*. Provenance is runtime
metadata only — it NEVER appears in canonical XML (guard rail 5) and it is
NEVER used for runtime dispatch (guard rail 3). It may be exported to a
sidecar JSON file next to the rubric via :meth:`Rubric.export_provenance`.

Per guard rail 8 of ``PHILOSOPHY.md``, this module is deliberately minimal.
There is no metadata database, no event bus, no history graph. Each refine
or generate step appends one :class:`RefinementStep`; each stopped loop
surfaces a :class:`RefinementReport` with an explicit ``stopped_reason``
(guard rail 6 — no silent fallbacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RefinementStep:
    """A single entry in a rubric's refinement lineage.

    ``kind`` is one of ``"generate"``, ``"evolve"``, ``"refine_iter"``, or
    ``"calibration_fix"``. ``reason`` is a short human-readable explanation of
    why this step ran (e.g. ``"attempt 2 succeeded"``, ``"no mutations"``).
    ``before_version`` / ``after_version`` capture the rubric version string
    before and after the step (both may be empty strings for the initial
    generate step). ``mutation_names`` lists the class names of any
    :class:`rubrify._mutations.RubricMutation` applied. ``meta_score`` is the
    meta-evaluator score observed *after* the step, if one was measured.
    """

    kind: str
    reason: str
    before_version: str
    after_version: str
    mutation_names: tuple[str, ...] = ()
    meta_score: int | None = None


@dataclass(slots=True)
class RubricProvenance:
    """Accumulating lineage metadata attached to a :class:`rubrify.Rubric`.

    Not frozen: ``add_step`` mutates ``refinement_steps`` in place so
    iterative refinement can record its trail on the same object. Guard rail
    8 keeps this minimal — there is no tree of ancestors, no diff store, no
    refinement graph. Just enough to answer ``where did this rubric come
    from and what has happened to it?``
    """

    source_kind: str = ""
    source_summary: str = ""
    generated_by_model: str = ""
    evaluated_by_model: str = ""
    parent_name: str = ""
    parent_version: str = ""
    calibration_suites: list[str] = field(default_factory=list)
    refinement_steps: list[RefinementStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def add_step(self, step: RefinementStep) -> None:
        """Append a :class:`RefinementStep` to the lineage."""
        self.refinement_steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this provenance record."""
        return {
            "source_kind": self.source_kind,
            "source_summary": self.source_summary,
            "generated_by_model": self.generated_by_model,
            "evaluated_by_model": self.evaluated_by_model,
            "parent_name": self.parent_name,
            "parent_version": self.parent_version,
            "calibration_suites": list(self.calibration_suites),
            "refinement_steps": [
                {
                    "kind": step.kind,
                    "reason": step.reason,
                    "before_version": step.before_version,
                    "after_version": step.after_version,
                    "mutation_names": list(step.mutation_names),
                    "meta_score": step.meta_score,
                }
                for step in self.refinement_steps
            ],
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricProvenance:
        """Round-trip companion to :meth:`to_dict`."""
        steps_data = data.get("refinement_steps", [])
        steps = [
            RefinementStep(
                kind=str(raw.get("kind", "")),
                reason=str(raw.get("reason", "")),
                before_version=str(raw.get("before_version", "")),
                after_version=str(raw.get("after_version", "")),
                mutation_names=tuple(raw.get("mutation_names", ()) or ()),
                meta_score=raw.get("meta_score"),
            )
            for raw in steps_data
        ]
        return cls(
            source_kind=str(data.get("source_kind", "")),
            source_summary=str(data.get("source_summary", "")),
            generated_by_model=str(data.get("generated_by_model", "")),
            evaluated_by_model=str(data.get("evaluated_by_model", "")),
            parent_name=str(data.get("parent_name", "")),
            parent_version=str(data.get("parent_version", "")),
            calibration_suites=list(data.get("calibration_suites", []) or []),
            refinement_steps=steps,
            tags=list(data.get("tags", []) or []),
        )


@dataclass(frozen=True, slots=True)
class RefinementReport:
    """Outcome of a thresholded ``generate()`` or ``refine()`` loop.

    ``stopped_reason`` is always one of ``"target_met"``, ``"no_mutations"``,
    ``"max_iters"``, ``"score_regressed"``, or ``"invalid_generation"`` —
    guard rail 6 forbids silent fallbacks, so every exit from the loop must
    name itself. ``steps`` preserves the per-iteration trail for diagnostics.
    """

    iterations: int
    start_score: int | None
    end_score: int | None
    stopped_reason: str
    steps: tuple[RefinementStep, ...]

    @property
    def improved(self) -> bool:
        """``True`` iff both scores are known and ``end_score > start_score``."""
        return (
            self.start_score is not None
            and self.end_score is not None
            and self.end_score > self.start_score
        )
