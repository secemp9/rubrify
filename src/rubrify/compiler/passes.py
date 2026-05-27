"""Compiler passes: normalize, bind, audit.

Each pass transforms a Rubric (or its derivatives) and returns the result.
Passes are pure functions — no side effects, no LLM calls.
"""

from __future__ import annotations

from rubrify.ir.types import Rubric
from rubrify.ir.constraints import ConstraintBinding, SurfaceProjection
from rubrify.ir.roles import SurfacePolicy


def normalize(rubric: Rubric) -> Rubric:
    """Normalize IDs and set default prompt_keys.

    Criterion IDs are already guaranteed unique by Rubric model validation.
    """
    normalized_criteria = []
    for criterion in rubric.criteria:
        prompt_key = criterion.prompt_key or criterion.id
        if prompt_key != criterion.prompt_key:
            normalized_criteria.append(criterion.model_copy(update={"prompt_key": prompt_key}))
        else:
            normalized_criteria.append(criterion)

    if any(c is not orig for c, orig in zip(normalized_criteria, rubric.criteria)):
        return rubric.model_copy(update={"criteria": normalized_criteria})
    return rubric


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


__all__ = [
    "audit_coverage",
    "audit_projection_completeness",
    "audit_scale_consistency",
    "bind",
    "normalize",
]
