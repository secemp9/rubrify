"""RubricCompiler: orchestrates the compilation pipeline.

    Rubric (mutable)
      → normalize (IDs, weights, prompt_keys)
      → bind (generate ConstraintBindings)
      → lock (produce immutable RubricBundle with hash)
      → audit (verify coverage, scale consistency)
"""

from __future__ import annotations

from dataclasses import dataclass

from rubrify.ir.types import Rubric
from rubrify.ir.constraints import AuthorityBlock, RitualConstraint
from rubrify.ir.roles import SurfacePolicy
from rubrify.ir.bundle import RubricBundle, lock_bundle
from rubrify.compiler.passes import (
    audit_coverage,
    audit_projection_completeness,
    audit_scale_consistency,
    bind,
    normalize,
)


@dataclass(slots=True)
class CompilationResult:
    bundle: RubricBundle
    issues: list[str]

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def compile_rubric(
    rubric: Rubric,
    *,
    policy: SurfacePolicy | None = None,
    rituals: list[RitualConstraint] | None = None,
) -> CompilationResult:
    """Full compiler pipeline: Rubric → RubricBundle.

    Returns the locked bundle and any audit issues found.
    This is a synchronous, pure operation — no LLM calls.
    """
    policy = policy or SurfacePolicy()
    rituals = rituals or []

    # Pass 1: Normalize
    normalized = normalize(rubric)

    # Pass 2: Bind
    bindings = bind(normalized, policy)

    # Pass 2.5: Generate AuthorityBlocks
    authority_blocks = [
        AuthorityBlock(
            id="rubric_spec",
            authority="instruction",
            kind="rubric_spec",
            model_should_follow=True,
        ),
        AuthorityBlock(
            id="response_under_test",
            authority="data",
            kind="response_under_test",
            model_should_follow=False,
        ),
        AuthorityBlock(
            id="judge_instructions",
            authority="instruction",
            kind="judge_instructions",
            model_should_follow=True,
        ),
        AuthorityBlock(
            id="context_document",
            authority="data",
            kind="context_document",
            model_should_follow=False,
        ),
    ]

    # Pass 3: Lock
    bundle = lock_bundle(normalized, bindings, policy, rituals=rituals, authority_blocks=authority_blocks)

    # Audit passes
    issues: list[str] = []
    issues.extend(audit_coverage(normalized, bindings))
    issues.extend(audit_projection_completeness(normalized, bindings, policy))
    issues.extend(audit_scale_consistency(normalized))

    return CompilationResult(bundle=bundle, issues=issues)


__all__ = [
    "CompilationResult",
    "compile_rubric",
]
