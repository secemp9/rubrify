"""Judgment types: the output of a judge execution."""

from __future__ import annotations

from dataclasses import dataclass

from rubrify.ir.types import SchemaModel


class EvidenceQuote(SchemaModel):
    """A concrete piece of evidence extracted by the judge."""
    source: str = "response"
    text: str = ""


class CriterionJudgment(SchemaModel):
    """The judgment for a single criterion."""
    criterion_id: str
    value: bool | int | float | str | None = None
    unit_score: float = 0.0
    evidence: list[EvidenceQuote] = []
    rationale: str = ""
    confidence: float | None = None
    warnings: list[str] = []


class AggregatedScore(SchemaModel):
    """Result of typed aggregation over criterion judgments."""
    raw_score: float = 0.0
    normalized_score: float = 0.0
    method: str = "weighted_mean"
    group_scores: dict[str, float] = {}


@dataclass(slots=True)
class JudgeUsage:
    """Token usage tracking across all LLM calls in a judgment."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    def __iadd__(self, other: JudgeUsage) -> JudgeUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.api_calls += other.api_calls
        return self


class Judgment(SchemaModel):
    """The complete judgment output. Terminal object of a judge execution."""
    criterion_judgments: list[CriterionJudgment] = []
    aggregation: AggregatedScore = AggregatedScore()
    decision: str | None = None
    violations: list[str] = []
    ritual_warnings: list[str] = []
    pattern_hits: dict[str, list[str]] = {}
    usage: JudgeUsage = JudgeUsage()
    timestamp: int = 0


__all__ = [
    "AggregatedScore",
    "CriterionJudgment",
    "EvidenceQuote",
    "Judgment",
    "JudgeUsage",
]
