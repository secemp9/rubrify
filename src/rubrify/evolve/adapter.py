"""RubricEvolverAdapter: GEPAAdapter implementation for rubric evolution.

The "program" under optimization is a rubric -- decomposed into its textual
components (goal, criterion descriptions, anchor descriptions, weights, role
persona, instructions). GEPA evolves these text parameters; the adapter
evaluates each candidate rubric by:
  1. Reconstructing a Rubric from the candidate dict
  2. Compiling it into a RubricBundle
  3. Running rubrify's Judge on each annotated example
  4. Computing agreement with human annotations as the score
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from harnify_ai.types import Model as HarnifyModel

from rubrify.compiler.compiler import CompilationResult, compile_rubric
from rubrify.engine.judge import Judge, JudgeConfig
from rubrify.engine.judgment import Judgment
from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import RitualConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy
from rubrify.ir.types import Rubric

from rubrify.evolve.candidate import candidate_to_rubric, rubric_to_candidate
from rubrify.evolve.meta_metric import (
    _get_scale_range,
    _to_numeric,
    compute_discrimination,
    compute_consistency,
)
from rubrify.evolve.types import AnnotatedExample, JudgmentTrajectory


class RubricEvolverAdapter(
    GEPAAdapter[AnnotatedExample, JudgmentTrajectory, Judgment]
):
    """GEPA adapter that evolves rubric text components.

    All LLM calls go through harnify_ai (via rubrify's Judge).
    """

    def __init__(
        self,
        base_rubric: Rubric,
        judge_model: HarnifyModel,
        *,
        base_role: RoleSpec | None = None,
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
        self._base_rubric = base_rubric
        self._base_role = base_role
        self._base_policy = base_policy or SurfacePolicy()
        self._base_rituals = base_rituals or []
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_temperature = judge_temperature
        self._judge_max_tokens = judge_max_tokens
        self._consistency_runs = consistency_runs
        self._agreement_weight = agreement_weight
        self._consistency_weight = consistency_weight
        self._discrimination_weight = discrimination_weight

    def _build_judge(self) -> Judge:
        """Create a fresh Judge instance using harnify_ai's Model."""
        return Judge(JudgeConfig(
            model=self._judge_model,
            api_key=self._judge_api_key,
            temperature=self._judge_temperature,
            max_tokens=self._judge_max_tokens,
        ))

    def _compile_candidate(
        self, candidate: dict[str, str]
    ) -> tuple[Rubric, RoleSpec | None, CompilationResult]:
        """Reconstruct and compile a rubric from a GEPA candidate."""
        rubric, role = candidate_to_rubric(
            candidate, self._base_rubric, self._base_role
        )
        policy = self._base_policy
        if role is not None:
            policy = policy.model_copy(update={"role": role})
        result = compile_rubric(rubric, policy=policy, rituals=self._base_rituals)
        return rubric, role, result

    def _run_judgments(
        self,
        bundle: RubricBundle,
        examples: list[AnnotatedExample],
    ) -> list[Judgment]:
        """Run the judge on all examples. Bridges async rubrify to sync GEPA."""
        judge = self._build_judge()

        async def _evaluate_all() -> list[Judgment]:
            judgments = []
            for example in examples:
                judgment = await judge.evaluate(
                    bundle,
                    example.response_text,
                    context_text=example.context_text,
                    genre=example.genre,
                )
                judgments.append(judgment)
            return judgments

        # Bridge to sync: use existing event loop or create one
        try:
            asyncio.get_running_loop()
            # Already in an async context -- run in a thread to avoid nesting
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _evaluate_all())
                return future.result()
        except RuntimeError:
            return asyncio.run(_evaluate_all())

    def evaluate(
        self,
        batch: list[AnnotatedExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[JudgmentTrajectory, Judgment]:
        """Evaluate a candidate rubric against annotated examples.

        For each example:
          1. Compile candidate -> RubricBundle
          2. Run judge on response_text
          3. Compare judgment scores against human_scores
          4. Score = agreement (higher is better)
        """
        # Step 1: Compile the candidate rubric
        rubric, role, compilation = self._compile_candidate(candidate)
        bundle = compilation.bundle

        if not bundle.locked:
            # Compilation failed -- return zero scores with diagnostic info
            dummy_judgment = Judgment(rubric_hash="", violations=["compilation_failed"])
            return EvaluationBatch(
                outputs=[dummy_judgment] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=(
                    [
                        JudgmentTrajectory(
                            example=ex,
                            judgment=dummy_judgment,
                            per_criterion_errors={},
                            compilation_issues=compilation.issues,
                        )
                        for ex in batch
                    ]
                    if capture_traces else None
                ),
                objective_scores=[
                    {"agreement": 0.0, "discrimination": 0.0, "consistency": 0.0}
                    for _ in batch
                ],
            )

        # Step 2: Run judgments
        judgments = self._run_judgments(bundle, batch)

        # Step 3: Compute per-example agreement scores
        criteria = rubric.criteria
        outputs: list[Judgment] = []
        scores: list[float] = []
        trajectories: list[JudgmentTrajectory] | None = [] if capture_traces else None
        objective_scores_list: list[dict[str, float]] = []

        # Compute batch-level discrimination up front so it feeds into
        # per-example composite scores.
        disc = compute_discrimination(judgments, criteria)

        # Default consistency for single-run; overwritten below when
        # consistency_runs > 1.
        consistency = 1.0

        # Optional: multi-run consistency measurement
        if self._consistency_runs > 1:
            all_runs = [judgments]
            for _ in range(self._consistency_runs - 1):
                all_runs.append(self._run_judgments(bundle, batch))
            consistency, _ = compute_consistency(all_runs, criteria)

        for example, judgment in zip(batch, judgments):
            outputs.append(judgment)

            # Per-example agreement: average across criteria
            per_crit_errors: dict[str, float] = {}
            error_sum = 0.0
            n_criteria = 0
            for criterion in criteria:
                if criterion.id not in example.human_scores:
                    continue
                cj = next(
                    (cj for cj in judgment.criterion_judgments
                     if cj.criterion_id == criterion.id),
                    None,
                )
                if cj is None or cj.value is None:
                    per_crit_errors[criterion.id] = 1.0
                    error_sum += 1.0
                else:
                    human_num = _to_numeric(example.human_scores[criterion.id], criterion.scale)
                    judge_num = _to_numeric(cj.value, criterion.scale)
                    scale_range = _get_scale_range(criterion.scale)
                    error = abs(human_num - judge_num) / scale_range if scale_range > 0 else 0.0
                    per_crit_errors[criterion.id] = error
                    error_sum += error
                n_criteria += 1

            agreement = 1.0 - (error_sum / n_criteria) if n_criteria > 0 else 0.0

            # Composite score: blend per-example agreement with batch-level
            # discrimination and consistency so GEPA's acceptance criterion
            # (which operates on sum(scores)) accounts for all three axes.
            composite = (
                agreement * self._agreement_weight
                + disc * self._discrimination_weight
                + consistency * self._consistency_weight
            )
            scores.append(composite)

            # Objective scores for multi-objective Pareto front
            objective_scores_list.append({
                "agreement": agreement,
                "discrimination": disc,
                "consistency": consistency,
                "composite": composite,
            })

            if trajectories is not None:
                trajectories.append(JudgmentTrajectory(
                    example=example,
                    judgment=judgment,
                    per_criterion_errors=per_crit_errors,
                    compilation_issues=compilation.issues,
                ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_list,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[JudgmentTrajectory, Judgment],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build the reflective dataset that GEPA's reflection LM sees.

        For each component being updated, constructs records showing:
        - What the current component text says
        - How the rubric performed on specific examples
        - Where disagreements with human annotations occurred
        - Specific diagnostic information relevant to that component type
        """
        assert eval_batch.trajectories is not None

        dataset: dict[str, list[dict[str, Any]]] = {}

        for component_name in components_to_update:
            records: list[dict[str, Any]] = []

            for trajectory in eval_batch.trajectories:
                record = self._build_reflective_record(
                    component_name, candidate, trajectory
                )
                records.append(record)

            dataset[component_name] = records

        return dataset

    def _build_reflective_record(
        self,
        component_name: str,
        candidate: dict[str, str],
        trajectory: JudgmentTrajectory,
    ) -> dict[str, Any]:
        """Build a single reflective record for one component + one example.

        The record structure follows GEPA's recommended schema:
        {Inputs, Generated Outputs, Feedback}.
        """
        example = trajectory.example
        judgment = trajectory.judgment
        errors = trajectory.per_criterion_errors

        # Common fields
        record: dict[str, Any] = {
            "Inputs": {
                "component_being_evolved": component_name,
                "current_component_value": candidate.get(component_name, ""),
                "response_under_test": example.response_text[:500],
                "human_annotations": {
                    cid: str(val)
                    for cid, val in example.human_scores.items()
                },
            },
            "Generated Outputs": {},
            "Feedback": "",
        }

        # Component-type-specific diagnostics
        if component_name.startswith("criterion.") and ".description" in component_name:
            crit_id = component_name.split(".")[1]
            cj = next(
                (cj for cj in judgment.criterion_judgments if cj.criterion_id == crit_id),
                None,
            )
            if cj is not None:
                gen_out: dict[str, Any] = {
                    "judge_score": str(cj.value),
                    "judge_unit_score": f"{cj.unit_score:.3f}",
                    "judge_rationale": cj.rationale,
                    "judge_evidence": [eq.text for eq in cj.evidence],
                }
                if cj.warnings:
                    gen_out["judge_warnings"] = cj.warnings
                if cj.confidence is not None:
                    gen_out["judge_confidence"] = f"{cj.confidence:.2f}"
                record["Generated Outputs"] = gen_out
                human_val = example.human_scores.get(crit_id, "N/A")
                error = errors.get(crit_id, 0.0)
                evidence_str = "; ".join(eq.text for eq in cj.evidence) if cj.evidence else "(no evidence cited)"
                warnings_str = "; ".join(cj.warnings) if cj.warnings else ""
                confidence_str = f" Confidence: {cj.confidence:.2f}." if cj.confidence is not None else ""
                if error > 0.3:
                    record["Feedback"] = (
                        f"DISAGREEMENT on criterion {crit_id}: "
                        f"Judge scored {cj.value} but human annotated {human_val} "
                        f"(error={error:.2f}).{confidence_str} "
                        f"Judge rationale: '{cj.rationale}'. "
                        f"Evidence cited: [{evidence_str}]. "
                        + (f"Warnings: [{warnings_str}]. " if warnings_str else "")
                        + f"The criterion description may be ambiguous or misaligned "
                        f"with human intent. Consider: Is the description specific "
                        f"enough? Does it clearly distinguish between score levels? "
                        f"Does the cited evidence suggest the judge misinterpreted "
                        f"the criterion?"
                    )
                else:
                    record["Feedback"] = (
                        f"AGREEMENT on criterion {crit_id}: "
                        f"Judge scored {cj.value}, human annotated {human_val}.{confidence_str} "
                        f"Evidence: [{evidence_str}]. "
                        f"The description appears well-calibrated for this case."
                    )

        elif component_name.startswith("criterion.") and ".anchors" in component_name:
            crit_id = component_name.split(".")[1]
            cj = next(
                (cj for cj in judgment.criterion_judgments if cj.criterion_id == crit_id),
                None,
            )
            if cj is not None:
                human_val = example.human_scores.get(crit_id, "N/A")
                error = errors.get(crit_id, 0.0)
                gen_out_anchors: dict[str, Any] = {
                    "judge_assigned_score": str(cj.value),
                    "judge_rationale": cj.rationale,
                    "judge_evidence": [eq.text for eq in cj.evidence],
                    "human_score": str(human_val),
                    "current_anchors": candidate.get(f"criterion.{crit_id}.anchors", ""),
                }
                if cj.warnings:
                    gen_out_anchors["judge_warnings"] = cj.warnings
                if cj.confidence is not None:
                    gen_out_anchors["judge_confidence"] = f"{cj.confidence:.2f}"
                record["Generated Outputs"] = gen_out_anchors
                evidence_str = "; ".join(eq.text for eq in cj.evidence) if cj.evidence else "(no evidence cited)"
                warnings_str = "; ".join(cj.warnings) if cj.warnings else ""
                confidence_str = f" Confidence: {cj.confidence:.2f}." if cj.confidence is not None else ""
                if error > 0.3:
                    record["Feedback"] = (
                        f"ANCHOR MISALIGNMENT on {crit_id}: Judge chose {cj.value} "
                        f"but human chose {human_val} (error={error:.2f}).{confidence_str} "
                        f"Judge rationale: '{cj.rationale}'. "
                        f"Evidence cited: [{evidence_str}]. "
                        + (f"Warnings: [{warnings_str}]. " if warnings_str else "")
                        + f"The anchor descriptions at these levels may not clearly "
                        f"distinguish between them. Review the current anchors and "
                        f"consider whether the descriptions at score levels "
                        f"{cj.value} and {human_val} need sharper differentiation."
                    )
                else:
                    record["Feedback"] = (
                        f"Anchors for {crit_id} led to correct scoring "
                        f"(judge={cj.value}, human={human_val}).{confidence_str} "
                        f"Evidence: [{evidence_str}]."
                    )

        elif component_name.startswith("criterion.") and ".weight" in component_name:
            crit_id = component_name.split(".")[1]
            # Full per-criterion detail so the reflection LM sees the complete picture
            all_cj_detail = {}
            for cj_item in judgment.criterion_judgments:
                cj_entry: dict[str, Any] = {
                    "unit_score": f"{cj_item.unit_score:.3f}",
                    "value": str(cj_item.value),
                    "rationale": cj_item.rationale,
                    "evidence": [eq.text for eq in cj_item.evidence],
                }
                if cj_item.confidence is not None:
                    cj_entry["confidence"] = f"{cj_item.confidence:.2f}"
                all_cj_detail[cj_item.criterion_id] = cj_entry

            gen_out_weight: dict[str, Any] = {
                "all_criterion_judgments": all_cj_detail,
                "overall_judgment_score": f"{judgment.aggregation.normalized_score:.1f}",
                "aggregation_method": judgment.aggregation.method,
            }
            if judgment.pattern_hits:
                gen_out_weight["pattern_hits"] = judgment.pattern_hits
            record["Generated Outputs"] = gen_out_weight

            overall_error = sum(errors.values()) / len(errors) if errors else 0.0
            high_err = {k: round(v, 2) for k, v in errors.items() if v > 0.3}
            low_err = {k: round(v, 2) for k, v in errors.items() if v <= 0.3}
            pattern_str = (
                f" Pattern hits: {json.dumps(judgment.pattern_hits)}."
                if judgment.pattern_hits else ""
            )
            record["Feedback"] = (
                f"Overall agreement for this example: {1.0 - overall_error:.2f}. "
                f"Per-criterion errors: {json.dumps({k: round(v, 2) for k, v in errors.items()})}. "
                f"Unreliable criteria (error>0.3): {json.dumps(high_err)}. "
                f"Reliable criteria (error<=0.3): {json.dumps(low_err)}.{pattern_str} "
                f"Consider: down-weight criteria with consistently high error, "
                f"up-weight criteria that are reliable AND important. "
                f"Objective mechanical pattern evidence can inform weighting decisions."
            )

        elif component_name == "rubric.goal":
            per_crit_summary = {
                cj_item.criterion_id: {
                    "judge": str(cj_item.value),
                    "human": str(example.human_scores.get(cj_item.criterion_id, "N/A")),
                    "error": round(errors.get(cj_item.criterion_id, 0.0), 2),
                }
                for cj_item in judgment.criterion_judgments
            }
            gen_out_goal: dict[str, Any] = {
                "overall_score": f"{judgment.aggregation.normalized_score:.1f}",
                "decision": judgment.decision or "",
                "per_criterion_summary": per_crit_summary,
            }
            if judgment.pattern_hits:
                gen_out_goal["pattern_hits"] = judgment.pattern_hits
            if judgment.ritual_warnings:
                gen_out_goal["ritual_warnings"] = judgment.ritual_warnings
            record["Generated Outputs"] = gen_out_goal

            disagreements = [cid for cid, err in errors.items() if err > 0.3]
            pattern_str = (
                f" Pattern hits: {json.dumps(judgment.pattern_hits)}."
                if judgment.pattern_hits else ""
            )
            ritual_str = (
                f" Ritual warnings: {judgment.ritual_warnings}."
                if judgment.ritual_warnings else ""
            )
            record["Feedback"] = (
                f"The rubric's goal guides the judge's overall interpretation. "
                f"Human label: {example.human_label or 'N/A'}. "
                f"Judge decision: {judgment.decision}. "
                f"Per-criterion disagreements: {disagreements}. "
                f"Per-criterion scores: {json.dumps(per_crit_summary)}."
                f"{pattern_str}{ritual_str} "
                f"Consider whether the goal statement adequately conveys "
                f"the evaluation intent and scope."
            )

        elif component_name == "rubric.instructions":
            per_crit_summary_instr = {
                cj_item.criterion_id: {
                    "judge": str(cj_item.value),
                    "human": str(example.human_scores.get(cj_item.criterion_id, "N/A")),
                    "error": round(errors.get(cj_item.criterion_id, 0.0), 2),
                }
                for cj_item in judgment.criterion_judgments
            }
            gen_out_instr: dict[str, Any] = {
                "decision": judgment.decision or "",
                "violations": judgment.violations,
                "ritual_warnings": judgment.ritual_warnings,
                "per_criterion_summary": per_crit_summary_instr,
            }
            if judgment.pattern_hits:
                gen_out_instr["pattern_hits"] = judgment.pattern_hits
            record["Generated Outputs"] = gen_out_instr

            pattern_str = (
                f" Pattern hits: {json.dumps(judgment.pattern_hits)}."
                if judgment.pattern_hits else ""
            )
            record["Feedback"] = (
                f"Instructions guide the judge's evaluation procedure. "
                f"Violations detected: {judgment.violations}. "
                f"Ritual warnings: {judgment.ritual_warnings}.{pattern_str} "
                f"Compilation issues: {trajectory.compilation_issues}. "
                f"Per-criterion results: {json.dumps(per_crit_summary_instr)}. "
                f"Consider whether instructions need to be more explicit about "
                f"evaluation steps, evidence requirements, or scoring constraints."
            )

        elif component_name.startswith("role."):
            gen_out_role: dict[str, Any] = {
                "decision": judgment.decision or "",
                "overall_score": f"{judgment.aggregation.normalized_score:.1f}",
                "aggregation_method": judgment.aggregation.method,
                "violations": judgment.violations,
            }
            if judgment.pattern_hits:
                gen_out_role["pattern_hits"] = judgment.pattern_hits
            record["Generated Outputs"] = gen_out_role

            high_error_criteria = [cid for cid, err in errors.items() if err > 0.3]
            pattern_str = (
                f" Pattern hits: {json.dumps(judgment.pattern_hits)}."
                if judgment.pattern_hits else ""
            )
            record["Feedback"] = (
                f"The role specification constrains the judge's behavioral space. "
                f"Judge decision: {judgment.decision}. "
                f"Overall score: {judgment.aggregation.normalized_score:.1f}. "
                f"Criteria with human disagreement: {high_error_criteria}.{pattern_str} "
                f"Violations: {judgment.violations}. "
                f"Consider whether the role persona/obligations/constraints need "
                f"adjustment to better guide consistent evaluation."
            )

        elif component_name == "rubric.definitions":
            # Identify which definitions are relevant to disagreements
            disagreement_crit_ids = {cid for cid, err in errors.items() if err > 0.3}
            # Surface all criterion judgments where disagreements occurred
            disagreement_details = {}
            for cj_item in judgment.criterion_judgments:
                if cj_item.criterion_id in disagreement_crit_ids:
                    disagreement_details[cj_item.criterion_id] = {
                        "judge_score": str(cj_item.value),
                        "human_score": str(example.human_scores.get(cj_item.criterion_id, "N/A")),
                        "rationale": cj_item.rationale,
                        "evidence": [eq.text for eq in cj_item.evidence],
                    }
            # Include the current definitions from the candidate
            current_defs = candidate.get("rubric.definitions", "")
            record["Generated Outputs"] = {
                "current_definitions": current_defs,
                "disagreement_criteria": disagreement_details,
                "overall_score": f"{judgment.aggregation.normalized_score:.1f}",
            }
            record["Feedback"] = (
                f"Definitions provide the judge with precise vocabulary. "
                f"Criteria with disagreements: {list(disagreement_crit_ids)}. "
                f"Review the rationales for these disagreements to determine if "
                f"ambiguous or missing definitions caused misinterpretation. "
                f"Disagreement details: {json.dumps(disagreement_details)}. "
                f"Consider adding, clarifying, or removing definitions to align "
                f"the judge's understanding with human intent."
            )

        elif component_name == "rubric.advice_rules":
            # Show which patterns fired and what advice would have been given
            current_rules = candidate.get("rubric.advice_rules", "")
            gen_out_advice: dict[str, Any] = {
                "current_advice_rules": current_rules,
                "overall_score": f"{judgment.aggregation.normalized_score:.1f}",
                "decision": judgment.decision or "",
            }
            if judgment.pattern_hits:
                gen_out_advice["pattern_hits"] = judgment.pattern_hits
            record["Generated Outputs"] = gen_out_advice

            if judgment.pattern_hits:
                fired_patterns = list(judgment.pattern_hits.keys())
                pattern_detail = json.dumps(judgment.pattern_hits)
                record["Feedback"] = (
                    f"Advice rules map pattern triggers to fix imperatives. "
                    f"Patterns that fired: {fired_patterns}. "
                    f"Pattern hit details: {pattern_detail}. "
                    f"Per-criterion errors: {json.dumps({k: round(v, 2) for k, v in errors.items()})}. "
                    f"Consider: Are fired patterns leading to correct advice? "
                    f"Are there unfired patterns that should trigger? "
                    f"Do the fix imperatives help the judge evaluate accurately?"
                )
            else:
                record["Feedback"] = (
                    f"Advice rules map pattern triggers to fix imperatives. "
                    f"No patterns fired for this example. "
                    f"Per-criterion errors: {json.dumps({k: round(v, 2) for k, v in errors.items()})}. "
                    f"Consider whether new pattern triggers are needed to catch "
                    f"issues in this type of response."
                )

        elif component_name == "rubric.calibration_examples":
            # Show which calibration examples are misaligned with human annotations
            current_cal = candidate.get("rubric.calibration_examples", "")
            per_crit_summary_cal = {
                cj_item.criterion_id: {
                    "judge": str(cj_item.value),
                    "human": str(example.human_scores.get(cj_item.criterion_id, "N/A")),
                    "error": round(errors.get(cj_item.criterion_id, 0.0), 2),
                }
                for cj_item in judgment.criterion_judgments
            }
            overall_error = sum(errors.values()) / len(errors) if errors else 0.0
            record["Generated Outputs"] = {
                "current_calibration_examples": current_cal,
                "per_criterion_results": per_crit_summary_cal,
                "overall_agreement": f"{1.0 - overall_error:.2f}",
                "human_label": example.human_label or "N/A",
                "judge_decision": judgment.decision or "",
            }
            misaligned = [
                cid for cid, detail in per_crit_summary_cal.items()
                if detail["error"] > 0.3
            ]
            record["Feedback"] = (
                f"Calibration examples are few-shot demonstrations embedded in "
                f"the judge prompt. Overall agreement: {1.0 - overall_error:.2f}. "
                f"Misaligned criteria: {misaligned}. "
                f"Human label: {example.human_label or 'N/A'}, "
                f"Judge decision: {judgment.decision}. "
                f"Results: {json.dumps(per_crit_summary_cal)}. "
                f"Consider: Do the calibration examples reflect the same scoring "
                f"standards as the human annotations? Are there edge cases that "
                f"need a new calibration example? Should any existing examples "
                f"be revised to match observed human scoring patterns?"
            )

        else:
            # Generic fallback
            record["Generated Outputs"] = {
                "judgment_summary": (
                    f"Score: {judgment.aggregation.normalized_score:.1f}, "
                    f"Decision: {judgment.decision}"
                ),
            }
            record["Feedback"] = (
                f"Per-criterion errors: {json.dumps({k: round(v, 2) for k, v in errors.items()})}"
            )

        return record


__all__ = [
    "RubricEvolverAdapter",
]
