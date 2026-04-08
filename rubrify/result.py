from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    """Opt-in observability record of a single ``evaluate`` / ``apply`` call.

    Phase 1 deliverable. Populated only when the caller passes ``observe=True``.
    The trace is attached to the result (or returned alongside it for
    ``ConstraintRubric.apply``) so the user can inspect exactly what was sent
    to the model, which parser ran, whether repair fired, and how long the
    round-trip took.
    """

    system_prompt: str
    user_message: str
    model: str
    parser: str  # "json" | "xml" | "raw"
    repair_notes: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    score: int | None = None  # 0-100 for scoring, 0-15 for detection
    label: str | None = None  # "Publish-ready", "Severe", etc.
    verdict: str | None = None  # "Yes"/"Somewhat"/"No" for compliance
    subscores: dict[str, int] = field(default_factory=dict)  # {"C1": 4, "C2": 2, ...}
    rationale: str = ""  # "BECAUSE: ..."
    evidence: list[str] = field(default_factory=list)
    actions: dict[str, Any] | None = None  # {"coaching": [...], "edits": [...]}
    diagnostics: dict[str, Any] | None = None  # {"hedges": 8, ...}
    violations: list[str] = field(default_factory=list)  # ["DQ1"]
    advice: list[str] | None = None  # ["Replace hype...", ...]
    risk: int | None = None  # For detection rubrics
    band: str | None = None  # "Severe", "Clean", etc.
    raw: str = ""  # Raw model output for debugging
    extras: dict[str, Any] = field(default_factory=dict)  # Unknown output fields passthrough
    repaired: bool = False
    repair_notes: tuple[str, ...] = ()
    trace: EvaluationTrace | None = None


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Outcome of a ``ConstraintRubric.apply_and_validate`` call.

    Phase 2 deliverable. Carries the raw model ``output``, a ``valid`` flag,
    the list of violation messages from failing validators, and optional
    repair / observability metadata. The trace is populated only when the
    caller requested ``observe=True``.
    """

    output: str
    valid: bool
    violations: tuple[str, ...] = ()
    repaired: bool = False
    repair_notes: tuple[str, ...] = ()
    trace: EvaluationTrace | None = None
