"""Compiler passes: bind, audit.

Each pass transforms a Rubric (or its derivatives) and returns the result.
Passes are pure functions — no side effects, no LLM calls.
"""

from __future__ import annotations

from rubrify.ir.types import Rubric
from rubrify.ir.constraints import ConstraintBinding, OutputConstraint, SurfaceProjection
from rubrify.ir.roles import SurfacePolicy


def bind(rubric: Rubric, policy: SurfacePolicy) -> list[ConstraintBinding]:
    """Generate ConstraintBindings for each criterion.

    Each binding connects:
      - criterion_id (the criterion's ID)
      - output_field (the JSON path where the judge writes its score)
      - projections (surface-specific renderings: XML tag, JSON key)

    This is the triple-layer alignment formalized.
    """
    bindings: list[ConstraintBinding] = []

    for criterion in rubric.criteria:
        projections: list[SurfaceProjection] = []

        # XML projection
        if policy.input_codec == "xml":
            projections.append(SurfaceProjection(
                codec="xml",
                path=f'criterion[@id="{criterion.id}"]',
                attributes={
                    "id": criterion.id,
                    "name": criterion.title,
                    "weight": str(criterion.weight),
                },
            ))

        # JSON output projection
        projections.append(SurfaceProjection(
            codec="json",
            path=f"criterion_scores.{criterion.id}",
        ))

        bindings.append(ConstraintBinding(
            criterion_id=criterion.id,
            output_field=f"criterion_scores.{criterion.id}",
            evidence_source=criterion.evidence.source,
            authority="instruction",
            projections=projections,
        ))

    return bindings


def audit_coverage(rubric: Rubric, bindings: list[ConstraintBinding]) -> list[str]:
    """Verify all criteria have bindings. Return list of issues."""
    bound_ids = {b.criterion_id for b in bindings}
    issues = []
    for criterion in rubric.criteria:
        if criterion.id not in bound_ids:
            issues.append(f"Criterion {criterion.id} has no ConstraintBinding")
    return issues


def audit_projection_completeness(rubric: Rubric, bindings: list[ConstraintBinding], policy: SurfacePolicy) -> list[str]:
    """Verify that every binding has projections matching the policy's codecs."""
    issues = []
    for binding in bindings:
        if policy.input_codec == "xml":
            has_xml = any(p.codec == "xml" for p in binding.projections)
            if not has_xml:
                issues.append(f"Binding for '{binding.criterion_id}' missing XML projection (policy requires xml input)")
    return issues


def audit_scale_consistency(rubric: Rubric) -> list[str]:
    """Verify scale types are internally consistent."""
    issues = []
    for criterion in rubric.criteria:
        scale = criterion.scale
        if scale.kind == "ordinal" and not scale.anchors:
            issues.append(f"Criterion {criterion.id}: ordinal scale has no anchors")
        if scale.kind == "numeric" and scale.maximum <= scale.minimum:
            issues.append(f"Criterion {criterion.id}: numeric scale max <= min")
    return issues


def audit_output_constraints(
    output_constraints: list[OutputConstraint],
    rubric: Rubric,
    execution_strategy: str = "per_criterion",
) -> list[str]:
    """Validate output constraints for recognized fields, duplicate IDs, hard-enforcement safety, and scope-strategy mismatches."""
    issues: list[str] = []

    recognized_fields = {"rationale", "evidence", "score", "advice"}
    # Fields that may not always be present in judge output.
    optional_fields = {"evidence"}
    # If the rubric has no advice_rules, 'advice' is not expected.
    if not rubric.advice_rules:
        optional_fields.add("advice")

    # Check for duplicate IDs.
    seen_ids: dict[str, int] = {}
    for constraint in output_constraints:
        seen_ids[constraint.id] = seen_ids.get(constraint.id, 0) + 1

    for cid, count in seen_ids.items():
        if count > 1:
            issues.append(f"Duplicate constraint id {cid!r} appears {count} times")

    # Per-constraint checks.
    for constraint in output_constraints:
        if constraint.target_field not in recognized_fields:
            issues.append(
                f"Constraint {constraint.id!r}: target_field {constraint.target_field!r} "
                f"is not a recognized field (expected one of {sorted(recognized_fields)})"
            )

        if constraint.enforcement == "hard" and constraint.target_field in optional_fields:
            issues.append(
                f"Constraint {constraint.id!r}: enforcement is 'hard' but target_field "
                f"{constraint.target_field!r} may not always be present"
            )

        # Scope-strategy mismatch warnings.
        if execution_strategy == "per_criterion" and constraint.scope == "call" and constraint.enforcement == "hard":
            issues.append(
                f"Constraint {constraint.id!r}: scope='call' with enforcement='hard' "
                f"in per_criterion mode — each criterion is a separate call, so this "
                f"constraint fires per-criterion (may be stricter than intended)"
            )

        if execution_strategy == "holistic" and constraint.scope == "criterion":
            issues.append(
                f"Constraint {constraint.id!r}: scope='criterion' in holistic mode — "
                f"all criteria share one response, so per-criterion scope has limited meaning"
            )

    return issues


__all__ = [
    "audit_coverage",
    "audit_output_constraints",
    "audit_projection_completeness",
    "audit_scale_consistency",
    "bind",
]
