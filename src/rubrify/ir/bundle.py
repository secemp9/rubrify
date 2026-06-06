"""RubricBundle — the immutable, locked, executable form of a rubric.

Once locked, a bundle can be audited and reproduced exactly.

LLMs propose and execute rubrics. LLMs do not own rubrics.
The bundle is the source of truth.
"""

from __future__ import annotations

import re
from pydantic import ConfigDict, Field

from rubrify.ir.types import Rubric, SchemaModel
from rubrify.ir.constraints import AuthorityBlock, ConstraintBinding, RitualConstraint
from rubrify.ir.roles import GenreModule, RoleSpec, SurfacePolicy


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
    compiled_patterns: dict[str, re.Pattern[str]] = Field(default_factory=dict, exclude=True)
    bindings: list[ConstraintBinding] = []
    authority_blocks: list[AuthorityBlock] = []
    surface_policy: SurfacePolicy = SurfacePolicy()
    genre_modules: list[GenreModule] = []
    rituals: list[RitualConstraint] = []
    locked: bool = False


def lock_bundle(
    rubric: Rubric,
    bindings: list[ConstraintBinding],
    policy: SurfacePolicy,
    rituals: list[RitualConstraint] | None = None,
    authority_blocks: list[AuthorityBlock] | None = None,
    genre_modules: list[GenreModule] | None = None,
) -> RubricBundle:
    """Compile a Rubric into an immutable RubricBundle."""
    rituals = rituals or []
    authority_blocks = authority_blocks or []
    genre_modules = genre_modules or []

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
        genre_modules=genre_modules,
        rituals=rituals,
        locked=True,
        compiled_patterns=compiled,
    )


__all__ = [
    "RubricBundle",
    "lock_bundle",
]
