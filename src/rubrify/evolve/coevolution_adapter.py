"""CoEvolutionAdapter: GEPAAdapter for Mode 3 co-evolutionary rubric optimization.

Evaluates the target rubric against human annotations (same as Mode 1), but
builds reflective records that cover both target components AND meta-components
(gate rubric, reflection templates, acceptance parameters).

The evaluation function scores ONLY the target rubric -- meta-components
influence the search process, not the evaluation score.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from harn_ai.types import Model as HarnifyModel

from rubrify.engine.judgment import Judgment
from rubrify.ir.constraints import RitualConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy
from rubrify.ir.types import Rubric

from rubrify.evolve.adapter import RubricEvolverAdapter
from rubrify.evolve.coevolution_candidate import (
    PREFIX_ACCEPTANCE,
    PREFIX_GATE,
    PREFIX_REFLECTION,
    PREFIX_TARGET,
    CoEvolutionComponents,
    candidate_to_coevolution,
)
from rubrify.evolve.types import AnnotatedExample, JudgmentTrajectory


# ── Trajectory type ───────────────────────────────────────────────


@dataclass(slots=True)
class CoEvolutionTrajectory:
    """Per-example trajectory that wraps the inner JudgmentTrajectory with meta-component state."""
    inner: JudgmentTrajectory
    active_gate_rubric: Rubric
    active_acceptance_params: dict[str, float]
    active_reflection_templates: dict[str, str]


# ── Adapter ───────────────────────────────────────────────────────


class CoEvolutionAdapter(
    GEPAAdapter[AnnotatedExample, CoEvolutionTrajectory, Judgment]
):
    """GEPA adapter that co-evolves target rubric + meta-components.

    Evaluation is delegated to the inner RubricEvolverAdapter (Mode 1).
    Reflective dataset construction adds meta-component diagnostic records.
    """

    def __init__(
        self,
        base_target_rubric: Rubric,
        base_gate_rubric: Rubric,
        base_reflection_templates: dict[str, str],
        base_acceptance_params: dict[str, float],
        judge_model: HarnifyModel,
        *,
        base_target_role: RoleSpec | None = None,
        base_policy: SurfacePolicy | None = None,
        base_rituals: list[RitualConstraint] | None = None,
        judge_api_key: str | None = None,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 2048,
        consistency_runs: int = 1,
        agreement_weight: float = 0.6,
        consistency_weight: float = 0.2,
        discrimination_weight: float = 0.2,
    ):
        self._base_target_rubric = base_target_rubric
        self._base_target_role = base_target_role
        self._base_gate_rubric = base_gate_rubric
        self._base_reflection_templates = base_reflection_templates
        self._base_acceptance_params = base_acceptance_params

        # Create a single RubricEvolverAdapter for the inner evaluation
        self._inner_adapter = RubricEvolverAdapter(
            base_rubric=base_target_rubric,
            judge_model=judge_model,
            base_role=base_target_role,
            base_policy=base_policy,
            base_rituals=base_rituals,
            judge_api_key=judge_api_key,
            judge_temperature=judge_temperature,
            judge_max_tokens=judge_max_tokens,
            consistency_runs=consistency_runs,
            agreement_weight=agreement_weight,
            consistency_weight=consistency_weight,
            discrimination_weight=discrimination_weight,
        )

    # ── evaluate ──────────────────────────────────────────────────

    def evaluate(
        self,
        batch: list[AnnotatedExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[CoEvolutionTrajectory, Judgment]:
        """Evaluate the target rubric from the co-evolution candidate.

        Meta-components (gate, reflection, acceptance) do NOT affect the
        evaluation score -- they only affect the search process.
        """
        # 1. Reconstruct all components (needed for trajectory metadata)
        components = candidate_to_coevolution(
            candidate,
            self._base_target_rubric,
            self._base_target_role,
            self._base_gate_rubric,
            self._base_reflection_templates,
            self._base_acceptance_params,
        )

        # 2. Strip the target.* prefix to give the inner adapter a standard candidate
        target_sub = {
            k[len(PREFIX_TARGET):]: v
            for k, v in candidate.items()
            if k.startswith(PREFIX_TARGET)
        }

        inner_result = self._inner_adapter.evaluate(batch, target_sub, capture_traces)

        # 3. If capture_traces, wrap inner trajectories with meta-component info
        co_trajectories: list[CoEvolutionTrajectory] | None = None
        if capture_traces and inner_result.trajectories:
            co_trajectories = [
                CoEvolutionTrajectory(
                    inner=traj,
                    active_gate_rubric=components.gate_rubric,
                    active_acceptance_params=components.acceptance_params,
                    active_reflection_templates=components.reflection_templates,
                )
                for traj in inner_result.trajectories
            ]

        return EvaluationBatch(
            outputs=inner_result.outputs,
            scores=inner_result.scores,
            trajectories=co_trajectories,
            objective_scores=inner_result.objective_scores,
        )

    # ── make_reflective_dataset ───────────────────────────────────

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset covering both target and meta components.

        - target.* components: delegate to the inner RubricEvolverAdapter
        - gate.* components: diagnostic records about gate influence
        - reflection.template.* components: diagnostic records about template impact
        - acceptance.* components: diagnostic records about acceptance behavior
        """
        assert eval_batch.trajectories is not None

        dataset: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            if component.startswith(PREFIX_TARGET):
                # Delegate to inner adapter with stripped prefix
                inner_key = component[len(PREFIX_TARGET):]
                target_sub = {
                    k[len(PREFIX_TARGET):]: v
                    for k, v in candidate.items()
                    if k.startswith(PREFIX_TARGET)
                }
                # Rebuild inner trajectories from co-evolution trajectories
                inner_trajectories = [t.inner for t in eval_batch.trajectories]
                inner_eval = EvaluationBatch(
                    outputs=eval_batch.outputs,
                    scores=eval_batch.scores,
                    trajectories=inner_trajectories,
                    objective_scores=eval_batch.objective_scores,
                )
                inner_dataset = self._inner_adapter.make_reflective_dataset(
                    target_sub, inner_eval, [inner_key],
                )
                dataset[component] = list(inner_dataset.get(inner_key, []))

            elif component.startswith(PREFIX_GATE):
                dataset[component] = self._build_gate_reflective(
                    component, candidate, eval_batch,
                )

            elif component.startswith(PREFIX_REFLECTION):
                dataset[component] = self._build_reflection_reflective(
                    component, candidate, eval_batch,
                )

            elif component.startswith(PREFIX_ACCEPTANCE):
                dataset[component] = self._build_acceptance_reflective(
                    component, candidate, eval_batch,
                )

        return dataset

    # ── Meta-component reflective record builders ─────────────────

    def _compute_overall_agreement(
        self,
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
    ) -> float:
        """Compute the overall agreement score from objective scores."""
        if eval_batch.objective_scores:
            agreements = [
                obj.get("agreement", 0.0) for obj in eval_batch.objective_scores
            ]
            return sum(agreements) / len(agreements) if agreements else 0.0
        if eval_batch.scores:
            return sum(eval_batch.scores) / len(eval_batch.scores)
        return 0.0

    def _get_persistent_disagreement_criteria(
        self,
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
    ) -> list[str]:
        """Find criteria with consistently high error across examples."""
        assert eval_batch.trajectories is not None
        error_sums: dict[str, float] = {}
        error_counts: dict[str, int] = {}
        for traj in eval_batch.trajectories:
            for crit_id, error in traj.inner.per_criterion_errors.items():
                error_sums[crit_id] = error_sums.get(crit_id, 0.0) + error
                error_counts[crit_id] = error_counts.get(crit_id, 0) + 1
        return [
            cid for cid, total in error_sums.items()
            if error_counts.get(cid, 1) > 0
            and (total / error_counts[cid]) > 0.3
        ]

    def _build_gate_reflective(
        self,
        component: str,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
    ) -> list[dict[str, Any]]:
        """Build reflective records for gate rubric components."""
        overall_agreement = self._compute_overall_agreement(eval_batch)
        persistent_disagreements = self._get_persistent_disagreement_criteria(eval_batch)

        # Summarize gate criteria from candidate
        gate_criteria_parts = []
        for k, v in sorted(candidate.items()):
            if k.startswith(PREFIX_GATE) and k.endswith(".description"):
                crit_id = k[len(PREFIX_GATE):].replace(".description", "")
                gate_criteria_parts.append(f"{crit_id}: {v[:80]}")
        gate_criteria_summary = "; ".join(gate_criteria_parts) if gate_criteria_parts else "(none)"

        records: list[dict[str, Any]] = []
        assert eval_batch.trajectories is not None
        for traj in eval_batch.trajectories:
            per_crit_errors = {
                k: round(v, 2) for k, v in traj.inner.per_criterion_errors.items()
            }
            record: dict[str, Any] = {
                "Inputs": {
                    "component_being_evolved": component,
                    "current_component_value": candidate.get(component, ""),
                    "target_rubric_performance": {
                        "overall_agreement": round(overall_agreement, 3),
                        "per_criterion_errors": per_crit_errors,
                    },
                },
                "Generated Outputs": {
                    "gate_rubric_goal": candidate.get(f"{PREFIX_GATE}rubric.goal", ""),
                    "gate_criteria_summary": gate_criteria_summary,
                },
                "Feedback": (
                    "The proposal quality gate controls which rubric mutations reach evaluation. "
                    f"Current target rubric agreement: {overall_agreement:.3f}. "
                    f"Criteria with persistent disagreement: {persistent_disagreements}. "
                    "If the gate is too strict, good exploratory mutations for these criteria "
                    "may be rejected before evaluation. If too loose, noise from bad mutations "
                    "wastes evaluation budget. Consider: does this gate criterion accurately "
                    "distinguish high-quality proposals from low-quality ones?"
                ),
            }
            records.append(record)
        return records

    def _build_reflection_reflective(
        self,
        component: str,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
    ) -> list[dict[str, Any]]:
        """Build reflective records for reflection template components."""
        overall_agreement = self._compute_overall_agreement(eval_batch)
        persistent_disagreements = self._get_persistent_disagreement_criteria(eval_batch)

        # Determine which target components this template type applies to
        # The component name is like "reflection.template.criterion_description"
        template_type = component[len(PREFIX_REFLECTION):]
        applicable_components = []
        for k in candidate:
            if k.startswith(PREFIX_TARGET):
                target_key = k[len(PREFIX_TARGET):]
                # Check if this target key maps to the same template type
                from rubrify.evolve.coevolution_candidate import _classify_component
                if _classify_component(target_key) == template_type:
                    applicable_components.append(target_key)

        # Pick a sample component value to show in the record
        sample_value = ""
        if applicable_components:
            sample_key = f"{PREFIX_TARGET}{applicable_components[0]}"
            sample_value = candidate.get(sample_key, "")

        records: list[dict[str, Any]] = []
        assert eval_batch.trajectories is not None
        for traj in eval_batch.trajectories:
            per_crit_errors = {
                k: round(v, 2) for k, v in traj.inner.per_criterion_errors.items()
            }
            record: dict[str, Any] = {
                "Inputs": {
                    "component_being_evolved": component,
                    "current_template": candidate.get(component, ""),
                    "sample_component_value": sample_value[:300],
                },
                "Generated Outputs": {
                    "applicable_target_components": applicable_components,
                    "current_target_performance": {
                        "overall_agreement": round(overall_agreement, 3),
                        "per_criterion_errors": per_crit_errors,
                    },
                },
                "Feedback": (
                    "This template guides the reflection LM when proposing improvements to "
                    f"{template_type.replace('_', ' ')} components. "
                    f"Currently, criteria {persistent_disagreements} have high error rates. "
                    "The template should help the reflection LM identify WHY these criteria "
                    "produce disagreements and propose targeted fixes. "
                    "Does the template ask for specific enough analysis? "
                    "Does it guide toward concrete, observable language?"
                ),
            }
            records.append(record)
        return records

    def _build_acceptance_reflective(
        self,
        component: str,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[CoEvolutionTrajectory, Judgment],
    ) -> list[dict[str, Any]]:
        """Build reflective records for acceptance parameter components."""
        overall_agreement = self._compute_overall_agreement(eval_batch)

        # Compute agreement variance across examples
        if eval_batch.objective_scores:
            agreements = [
                obj.get("agreement", 0.0) for obj in eval_batch.objective_scores
            ]
        elif eval_batch.scores:
            agreements = list(eval_batch.scores)
        else:
            agreements = []

        if len(agreements) > 1:
            mean_agr = sum(agreements) / len(agreements)
            variance = sum((a - mean_agr) ** 2 for a in agreements) / len(agreements)
        else:
            variance = 0.0

        # Extract the parameter name from the component key
        param_name = component[len(PREFIX_ACCEPTANCE):]
        current_value = candidate.get(component, "0.0")

        records: list[dict[str, Any]] = []
        assert eval_batch.trajectories is not None
        for traj in eval_batch.trajectories:
            record: dict[str, Any] = {
                "Inputs": {
                    "component_being_evolved": component,
                    "current_value": current_value,
                },
                "Generated Outputs": {
                    "current_agreement_score": round(overall_agreement, 3),
                    "agreement_variance_across_examples": round(variance, 4),
                },
                "Feedback": (
                    f"This tolerance ({param_name}) controls how much "
                    f"{'agreement' if 'agreement' in param_name else param_name.replace('_tolerance', '')} "
                    f"can drop when accepting a mutation that improves another dimension. "
                    f"Current agreement: {overall_agreement:.3f}. "
                    f"If tolerance is too tight (0.0), promising mutations that trade tiny "
                    f"drops for large gains in other dimensions are rejected. "
                    f"If too loose, the rubric drifts away from human alignment. "
                    f"Current variance suggests tolerance of ~{max(0.01, variance ** 0.5 * 0.5):.3f} "
                    f"would allow exploration without significant regression."
                ),
            }
            records.append(record)
        return records


__all__ = [
    "CoEvolutionAdapter",
    "CoEvolutionTrajectory",
]
