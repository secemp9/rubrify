"""Meta-metric: rubric quality measurement against human annotations.

Computes three orthogonal quality dimensions:
  - agreement: how well the rubric's judgments match human annotations
  - consistency: how stable scores are across repeated runs
  - discrimination: how well the rubric spreads scores across quality levels

These feed into GEPA's multi-objective Pareto front via objective_scores.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from rubrify.engine.judgment import Judgment
from rubrify.ir.types import (
    BinaryScale,
    Criterion,
    NumericScale,
    OrdinalScale,
)


def compute_agreement(
    judgments: list[Judgment],
    examples: list["AnnotatedExample"],
    criteria: list[Criterion],
) -> tuple[float, dict[str, float]]:
    """Compute agreement between rubric judgments and human annotations.

    Uses a normalized absolute-error metric scaled to [0, 1] where
    1.0 = perfect agreement. This is simpler and more robust than
    Cohen's kappa for ordinal scales with few annotators.

    For each criterion, agreement = 1 - (mean_absolute_error / scale_range).
    Overall agreement = weighted mean across criteria.
    """
    from rubrify.evolve.types import AnnotatedExample  # noqa: F811

    per_criterion: dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion in criteria:
        errors: list[float] = []
        scale = criterion.scale
        scale_range = _get_scale_range(scale)

        for judgment, example in zip(judgments, examples):
            if criterion.id not in example.human_scores:
                continue
            human_val = example.human_scores[criterion.id]
            cj = next(
                (cj for cj in judgment.criterion_judgments if cj.criterion_id == criterion.id),
                None,
            )
            if cj is None or cj.value is None:
                errors.append(1.0)  # Maximum disagreement if criterion not scored
                continue

            human_numeric = _to_numeric(human_val, scale)
            judge_numeric = _to_numeric(cj.value, scale)
            if scale_range > 0:
                errors.append(abs(human_numeric - judge_numeric) / scale_range)
            else:
                errors.append(0.0 if human_numeric == judge_numeric else 1.0)

        if errors:
            agreement = 1.0 - statistics.mean(errors)
        else:
            agreement = 0.0
        per_criterion[criterion.id] = agreement
        weighted_sum += agreement * criterion.weight
        total_weight += criterion.weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    return overall, per_criterion


def compute_consistency(
    judgment_runs: list[list[Judgment]],
    criteria: list[Criterion],
) -> tuple[float, dict[str, float]]:
    """Measure scoring consistency across repeated runs.

    For each (example, criterion), computes the coefficient of variation
    across N runs. Consistency = 1 - mean(CV), where lower variance
    means higher consistency.

    judgment_runs: list of N complete judgment lists (one per run).
    """
    per_criterion: dict[str, float] = {}
    all_cvs: list[float] = []

    for criterion in criteria:
        cvs: list[float] = []
        n_examples = len(judgment_runs[0]) if judgment_runs else 0

        for ex_idx in range(n_examples):
            scores_across_runs: list[float] = []
            for run_judgments in judgment_runs:
                cj = next(
                    (cj for cj in run_judgments[ex_idx].criterion_judgments
                     if cj.criterion_id == criterion.id),
                    None,
                )
                if cj is not None:
                    scores_across_runs.append(cj.unit_score)

            if len(scores_across_runs) >= 2:
                mean_val = statistics.mean(scores_across_runs)
                if mean_val > 0:
                    cv = statistics.stdev(scores_across_runs) / mean_val
                else:
                    cv = 0.0 if statistics.stdev(scores_across_runs) == 0 else 1.0
                cvs.append(cv)

        if cvs:
            avg_cv = statistics.mean(cvs)
            per_criterion[criterion.id] = max(0.0, 1.0 - avg_cv)
        else:
            per_criterion[criterion.id] = 1.0
        all_cvs.extend(cvs)

    overall = max(0.0, 1.0 - statistics.mean(all_cvs)) if all_cvs else 1.0
    return overall, per_criterion


def compute_discrimination(
    judgments: list[Judgment],
    criteria: list[Criterion],
) -> float:
    """Measure how well the rubric discriminates between quality levels.

    A good rubric spreads scores across its scale range. A rubric that
    assigns the same score to everything is useless. We use normalized
    entropy of the score distribution.

    Returns a value in [0, 1] where 1.0 = maximum discrimination.
    """
    all_unit_scores: list[float] = []
    for judgment in judgments:
        for cj in judgment.criterion_judgments:
            all_unit_scores.append(cj.unit_score)

    if not all_unit_scores:
        return 0.0

    # Bin scores into 10 buckets for entropy calculation
    n_bins = 10
    counts = [0] * n_bins
    for score in all_unit_scores:
        bin_idx = min(int(score * n_bins), n_bins - 1)
        counts[bin_idx] += 1

    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(n_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _get_scale_range(scale: Any) -> float:
    """Get the numeric range of a scale for error normalization."""
    if isinstance(scale, BinaryScale):
        return 1.0
    if isinstance(scale, NumericScale):
        return scale.maximum - scale.minimum
    if isinstance(scale, OrdinalScale):
        values = [a.value for a in scale.anchors]
        return max(values) - min(values) if values else 1.0
    return 1.0


def _to_numeric(value: Any, scale: Any) -> float:
    """Convert a scale value to a float for comparison."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str) and isinstance(scale, OrdinalScale):
        match = next((a.value for a in scale.anchors if a.label == value), None)
        if match is not None:
            return float(match)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


__all__ = [
    "compute_agreement",
    "compute_consistency",
    "compute_discrimination",
]
