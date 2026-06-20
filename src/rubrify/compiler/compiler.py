"""RubricCompiler: orchestrates the compilation pipeline.

    Rubric (mutable)
      → bind (generate ConstraintBindings)
      → authority_blocks (define instruction/data separation)
      → lock (produce immutable RubricBundle)
      → audit (verify coverage, projection completeness, scale consistency,
               output constraints)
"""

from __future__ import annotations

from dataclasses import dataclass

from rubrify.ir.types import Rubric
from rubrify.ir.constraints import AuthorityBlock, OutputConstraint
from rubrify.ir.roles import SurfacePolicy
from rubrify.ir.bundle import RubricBundle, lock_bundle
from rubrify.compiler.passes import (
    audit_coverage,
    audit_hypothesis_neutrality,
    audit_output_constraints,
    audit_projection_completeness,
    audit_scale_consistency,
    audit_scope_completeness,
    bind,
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
    output_constraints: list[OutputConstraint] | None = None,
) -> CompilationResult:
    """Full compiler pipeline: Rubric → RubricBundle.

    Returns the locked bundle and any audit issues found.
    This is a synchronous, pure operation — no LLM calls.
    """
    policy = policy or SurfacePolicy()
    output_constraints = output_constraints or []

    # Pass 1: Bind
    bindings = bind(rubric, policy)

    # Pass 1.5: Generate AuthorityBlocks
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

    # Pass 2: Lock
    bundle = lock_bundle(rubric, bindings, policy, output_constraints=output_constraints, authority_blocks=authority_blocks)

    # Audit passes
    issues: list[str] = []
    issues.extend(audit_coverage(rubric, bindings))
    issues.extend(audit_projection_completeness(rubric, bindings, policy))
    issues.extend(audit_scale_consistency(rubric))
    issues.extend(audit_output_constraints(output_constraints, rubric, execution_strategy=policy.execution_strategy))
    issues.extend(audit_scope_completeness(rubric))
    issues.extend(audit_hypothesis_neutrality(rubric))

    return CompilationResult(bundle=bundle, issues=issues)


__all__ = [
    "CompilationResult",
    "compile_rubric",
]
