"""Judge loop: strategy-aware execution over a locked bundle.

Unlike harn_agent's agent_loop which iterates on tool calls,
the judge_loop iterates over CRITERIA. Depending on the execution
strategy, criteria may be evaluated individually, in groups, or all
at once in a single holistic call.

Algorithm:
  1. Verify bundle is locked
  2. Resolve active criteria (genre filtering)
  3. Partition criteria into call units based on execution_strategy:
     - "per_criterion": one call per criterion (maximum isolation)
     - "grouped": one call per CriterionGroup (ungrouped = individual)
     - "holistic": one call for ALL active criteria
  4. Execute each call unit via execute_criterion or execute_group
  5. Verify evidence spans
  6. Check disqualifiers
  7. Verify output constraints (respecting scope: call/criterion/judgment)
  8. Aggregate scores
  9. Compute decision label
  10. Return Judgment
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from harn_ai.types import Model

from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import ConstraintBinding, OutputConstraint
from rubrify.ir.types import Criterion
from rubrify.engine.judgment import (
    AggregatedScore,
    CriterionJudgment,
    Judgment,
    JudgeUsage,
)
from rubrify.engine.executor import execute_criterion, execute_group


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

    Dispatches based on execution_strategy from the bundle's surface policy:
      - "per_criterion": one LLM call per criterion (default, maximum isolation)
      - "grouped": one LLM call per CriterionGroup; ungrouped criteria individually
      - "holistic": one LLM call for ALL active criteria simultaneously

    Returns a complete Judgment with per-criterion scores, aggregation,
    and decision label.
    """
    if not bundle.locked:
        raise RuntimeError("Cannot judge with an unlocked bundle. Compile first.")

    usage = JudgeUsage()

    # Step 1: Resolve active criteria and their bindings
    active_criteria = _resolve_active_criteria(bundle, active_genre)
    bindings_by_id = {b.criterion_id: b for b in bundle.bindings}

    # Step 2: Determine execution strategy
    strategy = (
        bundle.surface_policy.execution_strategy
        if bundle.surface_policy
        else "per_criterion"
    )

    # Step 3: Partition active criteria into call units based on strategy
    call_units = _partition_call_units(active_criteria, bundle, strategy)

    # Step 4: Execute call units (sequential or parallel)
    if parallel:
        judgments = await _execute_call_units_parallel(
            call_units, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=usage,
            use_tool=use_tool,
            on_start=on_criterion_start, on_done=on_criterion_done,
        )
    else:
        judgments = await _execute_call_units_sequential(
            call_units, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=usage,
            use_tool=use_tool,
            on_start=on_criterion_start, on_done=on_criterion_done,
        )

    # Step 5: Check disqualifiers
    violations = _check_disqualifiers(bundle, judgments, response_text)

    # Step 5.5: Run mechanical pattern checks
    pattern_hits = _run_mechanical_checks(bundle, response_text)

    # Step 6: Verify evidence
    evidence_warnings: list[str] = []
    for cj in judgments:
        evidence_warnings.extend(_verify_evidence(cj, response_text))

    # Step 6.5: Verify output constraints (respecting scope)
    constraint_violations, constraint_warnings = _verify_output_constraints(
        bundle, judgments, call_units
    )
    violations.extend(constraint_violations)
    constraint_warnings.extend(evidence_warnings)

    # Step 7: Aggregate
    aggregation = _aggregate(bundle, judgments, violations)

    # Step 8: Decision label
    decision = _compute_decision(aggregation, violations, thresholds=bundle.surface_policy.decision_thresholds)

    return Judgment(
        criterion_judgments=judgments,
        aggregation=aggregation,
        decision=decision,
        violations=violations,
        constraint_warnings=constraint_warnings,
        pattern_hits=pattern_hits,
        usage=usage,
        timestamp=int(time.time() * 1000),
    )


def _partition_call_units(
    active_criteria: list[Criterion],
    bundle: RubricBundle,
    strategy: str,
) -> list[list[Criterion]]:
    """Partition active criteria into call units based on execution strategy.

    - "holistic": one call unit containing ALL active criteria
    - "grouped": one call unit per CriterionGroup (+ ungrouped individually)
    - "per_criterion": one call unit per criterion (current/default behavior)
    """
    if strategy == "holistic":
        # One call unit containing ALL active criteria
        return [active_criteria] if active_criteria else []

    elif strategy == "grouped":
        # One call unit per CriterionGroup, plus ungrouped criteria individually
        call_units: list[list[Criterion]] = []
        grouped_ids: set[str] = set()
        criteria_by_id = {c.id: c for c in active_criteria}

        for group in bundle.rubric.groups:
            group_criteria = [
                criteria_by_id[cid]
                for cid in group.children
                if cid in criteria_by_id
            ]
            if group_criteria:
                call_units.append(group_criteria)
                grouped_ids.update(c.id for c in group_criteria)

        # Ungrouped criteria get individual calls
        for c in active_criteria:
            if c.id not in grouped_ids:
                call_units.append([c])

        return call_units

    else:
        # "per_criterion" — one call unit per criterion (default)
        return [[c] for c in active_criteria]


async def _execute_call_units_sequential(
    call_units: list[list[Criterion]],
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
    """Execute call units sequentially."""
    results: list[CriterionJudgment] = []
    for call_unit in call_units:
        unit_results = await _execute_one_call_unit(
            call_unit, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=usage,
            use_tool=use_tool, on_start=on_start, on_done=on_done,
        )
        results.extend(unit_results)
    return results


async def _execute_call_units_parallel(
    call_units: list[list[Criterion]],
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
    """Execute all call units in parallel via asyncio.gather."""
    async def _run_unit(call_unit: list[Criterion]) -> tuple[list[CriterionJudgment], JudgeUsage]:
        local_usage = JudgeUsage()
        unit_results = await _execute_one_call_unit(
            call_unit, bindings_by_id, bundle, response_text, model,
            context_text=context_text, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, usage=local_usage,
            use_tool=use_tool, on_start=on_start, on_done=on_done,
        )
        return unit_results, local_usage

    pairs = list(await asyncio.gather(*[_run_unit(unit) for unit in call_units]))
    results: list[CriterionJudgment] = []
    for unit_results, local_usage in pairs:
        usage += local_usage
        results.extend(unit_results)
    return results


async def _execute_one_call_unit(
    call_unit: list[Criterion],
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
    """Execute a single call unit (one or many criteria).

    - len(call_unit) == 1: use execute_criterion (backwards compat, proven code)
    - len(call_unit) > 1: use execute_group (multi-criterion in one LLM call)
    """
    if len(call_unit) == 1:
        criterion = call_unit[0]
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
        if on_done:
            on_done(criterion.id, cj)
        return [cj]
    else:
        # Multi-criterion call unit — use execute_group
        for c in call_unit:
            if on_start:
                on_start(c.id)
        try:
            group_judgments, group_usage = await execute_group(
                call_unit, bundle, response_text, model,
                api_key=api_key,
                use_tool=use_tool,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            group_judgments = [
                CriterionJudgment(
                    criterion_id=c.id,
                    value=None,
                    unit_score=0.0,
                    rationale="",
                    confidence=None,
                    warnings=[f"Group evaluation failed: {e}"],
                )
                for c in call_unit
            ]
            group_usage = JudgeUsage()
        usage += group_usage
        for cj in group_judgments:
            if on_done:
                on_done(cj.criterion_id, cj)
        return group_judgments


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


def _verify_evidence(cj: CriterionJudgment, response_text: str) -> list[str]:
    """Verify evidence quotes exist in the response.

    Two levels only:
      1. Exact containment
      2. Normalized containment (strip quotes, collapse whitespace, lowercase)
    No loose subsequence matching — it is too imprecise to be meaningful.

    Returns a list of warning strings for quotes that could not be verified.
    Does NOT mutate the CriterionJudgment.
    """
    warnings: list[str] = []
    for quote in cj.evidence:
        if quote.text and quote.source == "response":
            if quote.text in response_text:
                continue
            if _normalize_text(quote.text) in _normalize_text(response_text):
                continue
            warnings.append(f"Evidence quote not found in response: '{quote.text[:60]}...'")
    return warnings


def _verify_output_constraints(
    bundle: RubricBundle,
    judgments: list[CriterionJudgment],
    call_units: list[list[Criterion]] | None = None,
) -> tuple[list[str], list[str]]:
    """Verify output constraints against LLM output, respecting scope.

    Constraint scope determines the granularity of checking:
      - scope == "call": check against targets from each call unit separately.
        For per_criterion strategy, this is one check per criterion.
        For grouped/holistic, this is one check per group's shared output.
      - scope == "criterion": check against each criterion's output individually,
        regardless of call structure.
      - scope == "judgment": check once against the entire judgment's combined
        outputs (all criteria concatenated/aggregated).

    Returns (hard_violations, soft_warnings):
      - hard_violations: constraint IDs for enforcement="hard" failures
        (these trigger disqualification and score=0)
      - soft_warnings: descriptive strings for enforcement="soft" failures
        (informational only)

    Hard violations are deduplicated by constraint ID: one constraint produces
    at most one violation entry, regardless of how many targets fail. This
    mirrors how disqualifiers work (DQ1 fires once, not once per match).
    Per-target failure details are preserved in soft_warnings for diagnostics.
    """
    hard_violations: list[str] = []
    soft_warnings: list[str] = []
    if not bundle.output_constraints:
        return hard_violations, soft_warnings

    # Build lookup: criterion_id -> CriterionJudgment
    judgments_by_id = {cj.criterion_id: cj for cj in judgments}

    for constraint in bundle.output_constraints:
        scope = getattr(constraint, "scope", "call")

        if scope == "judgment":
            # Check once against ALL judgments combined
            target_values = _find_constraint_targets(judgments, constraint.target_field)
            if not target_values:
                continue
            # For judgment scope, concatenate all targets into one value
            combined = " ".join(target_values)
            failure_msg = constraint.check(combined)
            if failure_msg:
                _record_constraint_failure(
                    constraint, failure_msg, hard_violations, soft_warnings
                )

        elif scope == "criterion":
            # Check against each criterion's output individually
            for cj in judgments:
                target_values = _find_constraint_targets([cj], constraint.target_field)
                for target in target_values:
                    failure_msg = constraint.check(target)
                    if failure_msg:
                        _record_constraint_failure(
                            constraint, failure_msg, hard_violations, soft_warnings
                        )

        else:
            # scope == "call" (default): check per call unit
            if call_units:
                for call_unit in call_units:
                    # Get judgments for this call unit
                    unit_judgments = [
                        judgments_by_id[c.id]
                        for c in call_unit
                        if c.id in judgments_by_id
                    ]
                    if not unit_judgments:
                        continue
                    target_values = _find_constraint_targets(
                        unit_judgments, constraint.target_field
                    )
                    # For call scope with grouped execution, deduplicate shared
                    # rationales (execute_group shares the same rationale across
                    # all CriterionJudgments in a group)
                    seen_targets: set[str] = set()
                    for target in target_values:
                        if target in seen_targets:
                            continue
                        seen_targets.add(target)
                        failure_msg = constraint.check(target)
                        if failure_msg:
                            _record_constraint_failure(
                                constraint, failure_msg, hard_violations, soft_warnings
                            )
            else:
                # Fallback: no call_units info, check all judgments (legacy compat)
                target_values = _find_constraint_targets(judgments, constraint.target_field)
                for target in target_values:
                    failure_msg = constraint.check(target)
                    if failure_msg:
                        _record_constraint_failure(
                            constraint, failure_msg, hard_violations, soft_warnings
                        )

    return hard_violations, soft_warnings


def _record_constraint_failure(
    constraint: OutputConstraint,
    failure_msg: str,
    hard_violations: list[str],
    soft_warnings: list[str],
) -> None:
    """Record a constraint failure into the appropriate list."""
    violation_id = f"constraint:{constraint.id}"
    if constraint.enforcement == "hard":
        if violation_id not in hard_violations:
            hard_violations.append(violation_id)
        soft_warnings.append(
            f"[HARD] Constraint {constraint.id}: {failure_msg}"
        )
    else:
        soft_warnings.append(
            f"[SOFT] Constraint {constraint.id}: {failure_msg}"
        )


def _find_constraint_targets(judgments: list[CriterionJudgment], target_field: str) -> list[str]:
    """Find the values to check a constraint against."""
    if target_field == "rationale":
        return [cj.rationale for cj in judgments if cj.rationale]
    if target_field == "evidence":
        return [
            quote.text
            for cj in judgments
            for quote in cj.evidence
            if quote.text
        ]
    if target_field == "score":
        return [str(cj.unit_score) for cj in judgments]
    if target_field == "output":
        return []  # output-level constraints checked during parsing
    return []


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
