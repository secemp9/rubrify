from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
