"""Judge loop: criterion-by-criterion execution over a locked bundle.

Unlike harnify_agent's agent_loop which iterates on tool calls,
the judge_loop iterates over CRITERIA. Each criterion is a separate
LLM call. There is no tool-call/tool-result cycle.

Algorithm:
  1. Verify bundle is locked
  2. Resolve active criteria (genre filtering)
  3. For each criterion: execute via CriterionExecutor
  4. Verify evidence spans
  5. Check disqualifiers
  6. Aggregate scores
  7. Compute decision label
  8. Return Judgment
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from harnify_ai.types import Model

from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import ConstraintBinding, RitualConstraint
from rubrify.ir.types import Criterion
from rubrify.engine.judgment import (
    AggregatedScore,
    CriterionJudgment,
    Judgment,
    JudgeUsage,
)
from rubrify.engine.executor import execute_criterion


async def run_judge_loop(
    bundle: RubricBundle,
    response_text: str,
    model: Model,
    *,
    context_text: str | None = None,
    active_genre: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    parallel: bool = False,
    use_tool: bool = True,
    on_criterion_start: Callable[[str], None] | None = None,
    on_criterion_done: Callable[[str, CriterionJudgment], None] | None = None,
) -> Judgment:
    """Main judge loop. Evaluates all active criteria against a response.

    Returns a complete Judgment with per-criterion scores, aggregation,
    and decision label.
    """
    if not bundle.locked:
        raise RuntimeError("Cannot judge with an unlocked bundle. Compile first.")

    usage = JudgeUsage()

    # Step 1: Resolve active criteria and their bindings
    active_criteria = _resolve_active_criteria(bundle, active_genre)
    bindings_by_id = {b.criterion_id: b for b in bundle.bindings}

    # Step 2: Execute each criterion (binding-driven)
    if parallel:
        judgments = await _execute_parallel(
            active_criteria, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=usage,
            use_tool=use_tool,
            on_start=on_criterion_start, on_done=on_criterion_done,
        )
    else:
        judgments = await _execute_sequential(
            active_criteria, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=usage,
            use_tool=use_tool,
            on_start=on_criterion_start, on_done=on_criterion_done,
        )

    # Step 3: Check disqualifiers
    violations = _check_disqualifiers(bundle, judgments, response_text)

    # Step 3.5: Run mechanical pattern checks
    pattern_hits = _run_mechanical_checks(bundle, response_text)

    # Step 4: Verify evidence
    for cj in judgments:
        _verify_evidence(cj, response_text)

    # Step 4.5: Verify ritual constraints
    ritual_warnings = _verify_rituals(bundle, judgments)

    # Step 5: Aggregate
    aggregation = _aggregate(bundle, judgments, violations)

    # Step 6: Decision label
    decision = _compute_decision(aggregation, violations, thresholds=bundle.surface_policy.decision_thresholds)

    return Judgment(
        rubric_hash=bundle.hash,
        criterion_judgments=judgments,
        aggregation=aggregation,
        decision=decision,
        violations=violations,
        ritual_warnings=ritual_warnings,
        pattern_hits=pattern_hits,
        usage=usage,
        timestamp=int(time.time() * 1000),
    )


async def _execute_sequential(
    criteria: list[Criterion],
    bindings_by_id: dict[str, ConstraintBinding],
    bundle: RubricBundle,
    response_text: str,
    model: Model,
    *,
    context_text: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    usage: JudgeUsage,
    use_tool: bool = True,
    on_start: Callable[[str], None] | None,
    on_done: Callable[[str, CriterionJudgment], None] | None,
) -> list[CriterionJudgment]:
    results = []
    for criterion in criteria:
        if on_start:
            on_start(criterion.id)
        local_usage = JudgeUsage()
        try:
            cj = await execute_criterion(
                criterion, bundle, response_text, model,
                binding=bindings_by_id.get(criterion.id),
                context_text=context_text, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens, usage=local_usage,
                use_tool=use_tool,
            )
        except Exception as e:
            cj = CriterionJudgment(
                criterion_id=criterion.id,
                value=None,
                unit_score=0.0,
                rationale="",
                confidence=None,
                warnings=[f"Criterion evaluation failed: {e}"],
            )
        usage += local_usage
        results.append(cj)
        if on_done:
            on_done(criterion.id, cj)
    return results


async def _execute_parallel(
    criteria: list[Criterion],
    bindings_by_id: dict[str, ConstraintBinding],
    bundle: RubricBundle,
    response_text: str,
    model: Model,
    *,
    context_text: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    usage: JudgeUsage,
    use_tool: bool = True,
    on_start: Callable[[str], None] | None,
    on_done: Callable[[str, CriterionJudgment], None] | None,
) -> list[CriterionJudgment]:
    async def _run_one(criterion: Criterion) -> tuple[CriterionJudgment, JudgeUsage]:
        if on_start:
            on_start(criterion.id)
        local_usage = JudgeUsage()
        try:
            cj = await execute_criterion(
                criterion, bundle, response_text, model,
                binding=bindings_by_id.get(criterion.id),
                context_text=context_text, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens, usage=local_usage,
                use_tool=use_tool,
            )
        except Exception as e:
            cj = CriterionJudgment(
                criterion_id=criterion.id,
                value=None,
                unit_score=0.0,
                rationale="",
                confidence=None,
                warnings=[f"Criterion evaluation failed: {e}"],
            )
        if on_done:
            on_done(criterion.id, cj)
        return cj, local_usage

    pairs = list(await asyncio.gather(*[_run_one(c) for c in criteria]))
    for _cj, local_usage in pairs:
        usage += local_usage
    return [cj for cj, _ in pairs]


def _resolve_active_criteria(bundle: RubricBundle, active_genre: str | None) -> list[Criterion]:
    """Filter criteria based on genre-conditional activation."""
    active = []
    for criterion in bundle.rubric.criteria:
        if criterion.genre is None:
            active.append(criterion)
        elif active_genre and active_genre in criterion.genre:
            active.append(criterion)
    return active


def _check_disqualifiers(
    bundle: RubricBundle,
    judgments: list[CriterionJudgment],
    response_text: str = "",
) -> list[str]:
    """Check disqualifier conditions. No fallbacks — patterns must be pre-compiled."""
    violations = []
    scores_by_id = {cj.criterion_id: cj for cj in judgments}

    for dq in bundle.rubric.disqualifiers:
        # Pattern-based DQ — uses pre-compiled patterns from bundle
        if dq.pattern:
            compiled = bundle.compiled_patterns.get(f"dq_{dq.id}")
            if compiled is None:
                raise RuntimeError(
                    f"Disqualifier '{dq.id}' has pattern but no compiled regex in bundle. "
                    f"This indicates a compilation bug."
                )
            matched = any(compiled.search(cj.rationale) for cj in judgments)
            if not matched and response_text:
                matched = compiled.search(response_text) is not None
            if matched:
                violations.append(dq.id)

        # Criterion-linked DQ
        for cid in dq.criterion_ids:
            if cid in scores_by_id and scores_by_id[cid].unit_score == 0.0:
                if dq.id not in violations:
                    violations.append(dq.id)

    return violations


def _run_mechanical_checks(bundle: RubricBundle, response_text: str) -> dict[str, list[str]]:
    """Run compiled PatternEntry patterns against the response text.
    Returns {pattern_id: [matched_strings]}.

    Handles regex capture groups: findall returns tuples when the pattern
    has groups. We flatten to the full match string via finditer instead.
    """
    import re as _re
    results: dict[str, list[str]] = {}
    for p in bundle.rubric.patterns:
        compiled = bundle.compiled_patterns.get(p.id)
        if compiled:
            matches = [m.group(0) for m in compiled.finditer(response_text)]
            if matches:
                results[p.id] = matches
    return results


def _normalize_text(text: str) -> str:
    """Strip quotes, collapse whitespace, lowercase. No regex."""
    text = text.strip().strip('"\'\u201c\u201d\u2018\u2019')
    text = " ".join(text.split())
    return text.lower()


def _verify_evidence(cj: CriterionJudgment, response_text: str) -> None:
    """Verify evidence quotes exist in the response.

    Two levels only:
      1. Exact containment
      2. Normalized containment (strip quotes, collapse whitespace, lowercase)
    No loose subsequence matching — it is too imprecise to be meaningful.
    Unverified evidence is flagged AND warned.
    """
    for quote in cj.evidence:
        if quote.text and quote.source == "response":
            if quote.text in response_text:
                continue
            if _normalize_text(quote.text) in _normalize_text(response_text):
                continue
            quote.source = "unverified"
            cj.warnings.append(f"Evidence quote not found in response: '{quote.text[:60]}...'")


def _verify_rituals(bundle: RubricBundle, judgments: list[CriterionJudgment]) -> list[str]:
    """Verify ritual constraints against LLM output.

    Checks each RitualConstraint against the relevant field in the judgments.
    Returns a list of warning strings for violations.
    """
    warnings: list[str] = []
    if not bundle.rituals:
        return warnings

    for ritual in bundle.rituals:
        target_values = _find_ritual_targets(judgments, ritual.target_field)
        if not target_values:
            continue

        for target in target_values:
            violation = _check_single_ritual(ritual, target)
            if violation:
                warnings.append(violation)

    return warnings


def _find_ritual_targets(judgments: list[CriterionJudgment], target_field: str) -> list[str]:
    """Find the values to check a ritual against."""
    if target_field == "rationale":
        return [cj.rationale for cj in judgments if cj.rationale]
    if target_field == "output":
        return []  # output-level rituals checked during parsing
    return []


def _check_single_ritual(ritual: RitualConstraint, target: str) -> str | None:
    """Check one ritual constraint against one target value.

    Uses typed fields directly — no string parsing of natural language.
    """
    # Prefix check
    if ritual.prefix is not None:
        if not target.strip().startswith(ritual.prefix):
            return f"Ritual {ritual.id}: expected prefix '{ritual.prefix}', got '{target[:40]}...'"

    # Suffix check
    if ritual.suffix is not None:
        if not target.rstrip().endswith(ritual.suffix):
            return f"Ritual {ritual.id}: expected suffix '{ritual.suffix}', got '...{target[-20:]}'"

    # Fixed token check
    if ritual.token is not None:
        if ritual.token not in target:
            return f"Ritual {ritual.id}: expected token '{ritual.token}' not found"

    # Word count check
    if ritual.word_count is not None:
        actual = len(target.split())
        expected = ritual.word_count
        mode = ritual.word_count_mode or "exactly"
        if mode == "exactly" and actual != expected:
            return f"Ritual {ritual.id}: expected exactly {expected} words, got {actual}"
        if mode == "max" and actual > expected:
            return f"Ritual {ritual.id}: expected max {expected} words, got {actual}"
        if mode == "min" and actual < expected:
            return f"Ritual {ritual.id}: expected min {expected} words, got {actual}"

    return None


def _aggregate(
    bundle: RubricBundle,
    judgments: list[CriterionJudgment],
    violations: list[str],
) -> AggregatedScore:
    """Typed, deterministic weighted aggregation with optional group hierarchy."""
    if violations:
        return AggregatedScore(raw_score=0.0, normalized_score=0.0, method="disqualified")

    criteria_by_id = {c.id: c for c in bundle.rubric.criteria}
    judgments_by_id = {cj.criterion_id: cj for cj in judgments}

    # Validate all judgments reference known criteria
    for cj in judgments:
        if cj.criterion_id not in criteria_by_id:
            raise RuntimeError(
                f"CriterionJudgment references unknown criterion '{cj.criterion_id}'. "
                f"Valid IDs: {list(criteria_by_id.keys())}"
            )

    # --- No groups: flat weighted mean (the common case) ---
    if not bundle.rubric.groups:
        total_weight = 0.0
        weighted_sum = 0.0
        for cj in judgments:
            w = criteria_by_id[cj.criterion_id].weight
            total_weight += w
            weighted_sum += cj.unit_score * w

        if total_weight == 0.0:
            return AggregatedScore(raw_score=0.0, normalized_score=0.0, method="weighted_mean")

        normalized = (weighted_sum / total_weight) * 100.0
        return AggregatedScore(
            raw_score=weighted_sum,
            normalized_score=round(normalized, 2),
            method="weighted_mean",
        )

    # --- Groups exist: per-group aggregation ---
    group_scores: dict[str, float] = {}
    grouped_criterion_ids: set[str] = set()

    for group in bundle.rubric.groups:
        child_scores: list[tuple[float, float]] = []  # (unit_score, weight)
        for cid in group.children:
            cj = judgments_by_id.get(cid)
            if cj is None:
                continue  # criterion not active (genre-filtered out)
            child_scores.append((cj.unit_score, criteria_by_id[cid].weight))
            grouped_criterion_ids.add(cid)

        if not child_scores:
            group_scores[group.id] = 0.0
            continue

        agg = group.aggregation
        if agg in ("weighted_sum", "weighted_mean"):
            tw = sum(w for _, w in child_scores)
            if tw == 0.0:
                group_scores[group.id] = 0.0
            else:
                group_scores[group.id] = sum(s * w for s, w in child_scores) / tw
        elif agg == "min":
            group_scores[group.id] = min(s for s, _ in child_scores)
        elif agg == "max":
            group_scores[group.id] = max(s for s, _ in child_scores)
        elif agg == "all":
            group_scores[group.id] = 1.0 if all(s > 0 for s, _ in child_scores) else 0.0
        elif agg == "any":
            group_scores[group.id] = 1.0 if any(s > 0 for s, _ in child_scores) else 0.0
        else:
            raise RuntimeError(f"Unknown aggregation strategy '{agg}' on group '{group.id}'")

    # Final score: weighted mean of group scores + ungrouped criteria
    total_weight = 0.0
    weighted_sum = 0.0

    for group in bundle.rubric.groups:
        total_weight += group.weight
        weighted_sum += group_scores[group.id] * group.weight

    # Include ungrouped criteria directly
    for cj in judgments:
        if cj.criterion_id not in grouped_criterion_ids:
            w = criteria_by_id[cj.criterion_id].weight
            total_weight += w
            weighted_sum += cj.unit_score * w

    if total_weight == 0.0:
        return AggregatedScore(
            raw_score=0.0, normalized_score=0.0,
            method="grouped_weighted_mean", group_scores=group_scores,
        )

    normalized = (weighted_sum / total_weight) * 100.0
    return AggregatedScore(
        raw_score=weighted_sum,
        normalized_score=round(normalized, 2),
        method="grouped_weighted_mean",
        group_scores=group_scores,
    )


def _compute_decision(
    aggregation: AggregatedScore,
    violations: list[str],
    thresholds: list[tuple[float, str]] | None = None,
) -> str:
    """Map normalized score to a human-readable decision label."""
    if violations:
        return "Rejected"
    score = aggregation.normalized_score
    if thresholds is not None:
        sorted_thresholds = sorted(thresholds, key=lambda t: t[0], reverse=True)
        for min_score, label in sorted_thresholds:
            if score >= min_score:
                return label
        return sorted_thresholds[-1][1] if sorted_thresholds else "Unclassified"
    if score >= 90:
        return "Publish-ready"
    if score >= 75:
        return "Strong draft"
    if score >= 60:
        return "Workable draft"
    if score >= 40:
        return "Needs major revision"
    return "Fundamentally unclear"


__all__ = [
    "run_judge_loop",
]
