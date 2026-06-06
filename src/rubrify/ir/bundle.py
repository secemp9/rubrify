"""RubricBundle — the immutable, locked, executable form of a rubric.

Once locked, a bundle can be audited and reproduced exactly.

LLMs propose and execute rubrics. LLMs do not own rubrics.
The bundle is the source of truth.
"""

from __future__ import annotations

import re
from pydantic import ConfigDict, Field

from rubrify.ir.types import Rubric, SchemaModel
from rubrify.ir.constraints import AuthorityBlock, ConstraintBinding, OutputConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy


class RubricBundle(SchemaModel):
    """Immutable locked rubric bundle. The unit of execution.

    Contains everything needed to run a judgment:
      - The frozen rubric
      - All constraint bindings (criterion <-> surface <-> output)
      - Authority blocks for instruction/data separation
      - Surface policy for rendering
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    rubric: Rubric
    compiled_patterns: dict[str, re.Pattern[str]] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "Runtime cache of compiled regex patterns, keyed by pattern/disqualifier ID. "
            "This is NOT part of the bundle's identity or serialized form (exclude=True). "
            "Populated once by lock_bundle() at construction time and never modified after. "
            "The dict is technically mutable despite the frozen model, but treated as "
            "write-once: any post-construction mutation is a bug."
        ),
    )
    bindings: list[ConstraintBinding] = []
    authority_blocks: list[AuthorityBlock] = []
    surface_policy: SurfacePolicy = SurfacePolicy()
    output_constraints: list[OutputConstraint] = []
    locked: bool = False


def lock_bundle(
    rubric: Rubric,
    bindings: list[ConstraintBinding],
    policy: SurfacePolicy,
    output_constraints: list[OutputConstraint] | None = None,
    authority_blocks: list[AuthorityBlock] | None = None,
) -> RubricBundle:
    """Compile a Rubric into an immutable RubricBundle."""
    output_constraints = output_constraints or []
    authority_blocks = authority_blocks or []

    # Compile PatternEntry patterns — fail loudly on invalid regex
    compiled: dict[str, re.Pattern[str]] = {}
    for p in rubric.patterns:
        flags = re.IGNORECASE if "i" in p.flags else 0
        try:
            compiled[p.id] = re.compile(p.pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex in PatternEntry '{p.id}': {e}") from e
    for dq in rubric.disqualifiers:
        if dq.pattern:
            try:
                compiled[f"dq_{dq.id}"] = re.compile(dq.pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"Invalid regex in Disqualifier '{dq.id}': {e}") from e

    return RubricBundle(
        rubric=rubric,
        bindings=bindings,
        authority_blocks=authority_blocks,
        surface_policy=policy,
        output_constraints=output_constraints,
        locked=True,
        compiled_patterns=compiled,
    )


__all__ = [
    "RubricBundle",
    "lock_bundle",
]
