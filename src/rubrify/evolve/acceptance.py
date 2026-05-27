"""Multi-dimensional acceptance criterion for rubric evolution.

Instead of StrictImprovementAcceptance (sum(new) > sum(old)), this uses
the objective_scores (agreement, discrimination, consistency) to make
nuanced acceptance decisions. Zero LLM cost — pure logic over existing scores.
"""

from __future__ import annotations
from dataclasses import dataclass

# Import GEPA's protocol
from gepa.strategies.acceptance import AcceptanceCriterion


@dataclass
class RubricAwareAcceptance(AcceptanceCriterion):
    """Accept mutations based on multi-dimensional rubric quality.

    Accepts if:
    1. Composite score improved (sum(new) > sum(old)), OR
    2. Agreement improved AND discrimination did not degrade beyond tolerance, OR
    3. Any single objective improved AND no objective degraded beyond tolerance
       (Pareto improvement)
    """
    agreement_tolerance: float = 0.0      # agreement must not drop below this delta
    discrimination_tolerance: float = 0.05  # discrimination can drop by up to this
    consistency_tolerance: float = 0.05     # consistency can drop by up to this
    require_composite_improvement: bool = False  # if True, composite must also improve

    def should_accept(self, proposal, state) -> bool:
        """Decide whether to accept a proposed mutation."""
        old_scores = proposal.subsample_scores_before or []
        new_scores = proposal.subsample_scores_after or []

        if not old_scores or not new_scores:
            return False

        # Basic: composite improvement
        old_sum = sum(old_scores)
        new_sum = sum(new_scores)
        composite_improved = new_sum > old_sum

        if self.require_composite_improvement and not composite_improved:
            return False

        # Multi-objective check using objective_scores if available
        old_objectives = getattr(proposal, 'eval_before', None)
        new_objectives = getattr(proposal, 'eval_after', None)

        if old_objectives and new_objectives:
            old_obj = old_objectives.objective_scores
            new_obj = new_objectives.objective_scores

            if old_obj and new_obj:
                # Average objectives across examples
                old_avg = self._avg_objectives(old_obj)
                new_avg = self._avg_objectives(new_obj)

                # Check per-dimension deltas
                agreement_delta = new_avg.get("agreement", 0) - old_avg.get("agreement", 0)
                discrimination_delta = new_avg.get("discrimination", 0) - old_avg.get("discrimination", 0)
                consistency_delta = new_avg.get("consistency", 0) - old_avg.get("consistency", 0)

                # Reject if any dimension degrades beyond tolerance
                if agreement_delta < -self.agreement_tolerance:
                    return False
                if discrimination_delta < -self.discrimination_tolerance:
                    return False
                if consistency_delta < -self.consistency_tolerance:
                    return False

                # Accept if at least one dimension improved
                any_improved = (
                    agreement_delta > 0 or
                    discrimination_delta > 0 or
                    consistency_delta > 0
                )
                if any_improved:
                    return True

        # Fallback: accept on composite improvement
        return composite_improved

    @staticmethod
    def _avg_objectives(obj_list: list[dict[str, float]]) -> dict[str, float]:
        if not obj_list:
            return {}
        keys = obj_list[0].keys()
        return {k: sum(d.get(k, 0) for d in obj_list) / len(obj_list) for k in keys}
