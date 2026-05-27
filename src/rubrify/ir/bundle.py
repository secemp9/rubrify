"""RubricBundle — the immutable, locked, executable form of a rubric.

Like a compiled binary: once locked, a bundle is content-addressed and can
be verified, audited, and reproduced exactly. The hash changes if ANY
criterion, binding, or policy changes.

LLMs propose and execute rubrics. LLMs do not own rubrics.
The bundle is the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import ConfigDict, Field

from rubrify.ir.types import Rubric, SchemaModel
from rubrify.ir.constraints import AuthorityBlock, ConstraintBinding, RitualConstraint
from rubrify.ir.roles import GenreModule, RoleSpec, SurfacePolicy


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON for content-addressing."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RubricBundle(SchemaModel):
    """Immutable locked rubric bundle. The unit of execution.

    Contains everything needed to run a judgment:
      - The frozen rubric
      - All constraint bindings (criterion <-> surface <-> output)
      - Authority blocks for instruction/data separation
      - Surface policy for rendering
      - Content hash for reproducibility
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    rubric: Rubric
    compiled_patterns: dict[str, re.Pattern[str]] = Field(default_factory=dict, exclude=True)
    bindings: list[ConstraintBinding] = []
    authority_blocks: list[AuthorityBlock] = []
    surface_policy: SurfacePolicy = SurfacePolicy()
    genre_modules: list[GenreModule] = []
    rituals: list[RitualConstraint] = []
    canonical: str = ""
    hash: str = ""
    locked: bool = False

    def verify(self) -> bool:
        """Verify the bundle hash matches its content."""
        if not self.locked:
            return False
        canonical = _canonical_json({
            "rubric": self.rubric.model_dump(mode="json"),
            "bindings": [b.model_dump(mode="json") for b in self.bindings],
            "policy": self.surface_policy.model_dump(mode="json"),
            "rituals": [r.model_dump(mode="json") for r in self.rituals],
        })
        expected = _compute_hash(canonical)
        return self.hash == expected


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

    canonical = _canonical_json({
        "rubric": rubric.model_dump(mode="json"),
        "bindings": [b.model_dump(mode="json") for b in bindings],
        "policy": policy.model_dump(mode="json"),
        "rituals": [r.model_dump(mode="json") for r in rituals],
    })
    h = _compute_hash(canonical)

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
        canonical=canonical,
        hash=h,
        locked=True,
        compiled_patterns=compiled,
    )


def _compute_hash(canonical: str) -> str:
    return "rubric_" + hashlib.blake2s(canonical.encode(), digest_size=16).hexdigest()


__all__ = [
    "RubricBundle",
    "lock_bundle",
]
