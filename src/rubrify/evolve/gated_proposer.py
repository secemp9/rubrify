"""Gated proposer: wraps GEPA's reflection flow with proposal quality filtering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from rubrify.evolve.proposal_gate import ProposalQualityGate


class GatedProposalFn:
    """A ProposalFn that runs the default reflection, then gates proposals.

    GEPA's ReflectiveMutationProposer accepts custom_candidate_proposer
    as a callable with signature:
        (candidate, reflective_dataset, components_to_update) -> dict[str, str]

    This class implements that callable, internally running the standard
    InstructionProposalSignature for each component, then validating
    the result through the ProposalQualityGate.
    """

    def __init__(
        self,
        reflection_lm: Any,
        reflection_template: str | dict[str, str],
        gate: ProposalQualityGate,
    ):
        self._lm = reflection_lm
        self._template = reflection_template
        self._gate = gate

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new texts with quality gating."""
        results: dict[str, str] = {}

        for component in components_to_update:
            # Get the right template for this component
            if isinstance(self._template, dict):
                template = self._template.get(
                    component,
                    list(self._template.values())[0] if self._template else "",
                )
            else:
                template = self._template

            current_text = candidate.get(component, "")
            side_info = reflective_dataset.get(component, [])

            # Run the default GEPA reflection to get proposed text
            proposed = self._run_reflection(template, current_text, side_info)

            if not proposed or proposed == current_text:
                results[component] = current_text
                continue

            # Gate the proposal
            passed, feedback, score = self._gate.evaluate_proposal(
                component, current_text, proposed
            )

            if passed:
                results[component] = proposed
            else:
                # Try re-proposing with gate feedback
                for _retry in range(self._gate._max_retries):
                    enhanced_side_info = list(side_info) + [
                        {
                            "Inputs": {
                                "gate_feedback": feedback,
                                "rejected_proposal": proposed,
                            },
                            "Generated Outputs": {"gate_score": f"{score:.2f}"},
                            "Feedback": (
                                f"Previous proposal was rejected (score {score:.2f}): "
                                f"{feedback}. Propose a better version."
                            ),
                        }
                    ]
                    proposed = self._run_reflection(
                        template, current_text, enhanced_side_info
                    )

                    if proposed and proposed != current_text:
                        passed, feedback, score = self._gate.evaluate_proposal(
                            component, current_text, proposed
                        )
                        if passed:
                            results[component] = proposed
                            break

                if component not in results:
                    # All retries exhausted — keep original
                    results[component] = current_text

        return results

    def _run_reflection(
        self,
        template: str,
        current_text: str,
        side_info: Sequence[Mapping[str, Any]],
    ) -> str:
        """Run InstructionProposalSignature to generate a proposal."""
        # Format the side_info as text
        side_info_text = json.dumps(list(side_info), indent=2, default=str)

        # Build the prompt using the template
        prompt = template.replace("<curr_param>", current_text).replace(
            "<side_info>", side_info_text
        )

        # Call the reflection LM
        response = self._lm(prompt)

        # Extract text from ``` blocks if present
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                # Take content of first code block
                block = parts[1]
                # Strip language identifier if present
                if block.startswith(("json", "python", "text")):
                    block = block.split("\n", 1)[1] if "\n" in block else block
                return block.strip()

        return response.strip()


__all__ = [
    "GatedProposalFn",
]
