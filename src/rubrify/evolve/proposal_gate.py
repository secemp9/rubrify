"""Proposal quality gate: pre-filters proposed rubric text before expensive evaluation.

Uses rubrify's own Judge to evaluate proposed text against a lightweight
quality rubric. Costs 1 LLM call per proposal (vs N_examples * N_criteria
for a full eval round).
"""

from __future__ import annotations

from harn_ai.types import Model

from rubrify.evolve.async_bridge import run_async

from rubrify.compiler.compiler import compile_rubric
from rubrify.engine.judge import Judge, JudgeConfig
from rubrify.ir.types import (
    Criterion,
    Disqualifier,
    NumericScale,
    Rubric,
    RubricMeta,
    ScaleAnchor,
)


# ── Quality rubric factory ────────────────────────────────────────


def make_proposal_quality_rubric() -> Rubric:
    """Create the proposal quality assessment rubric."""
    return Rubric(
        meta=RubricMeta(
            name="proposal_quality",
            version="1.0",
            author="rubrify.evolve",
            description="Evaluates the quality of proposed rubric component text.",
        ),
        goal="Assess whether a proposed rubric component revision is structurally valid, semantically specific, and clearly improved over the original.",
        criteria=[
            Criterion(
                id="PQ1",
                title="Structural Validity",
                description=(
                    "Does the proposed text parse correctly? For anchors, is it "
                    "valid JSON with value/label/description fields? For weights, "
                    "is it a positive number? For descriptions, is it non-empty?"
                ),
                scale=NumericScale(
                    minimum=0,
                    maximum=2,
                    step=1,
                    anchors=[
                        ScaleAnchor(
                            value=0,
                            label="invalid",
                            description="Invalid structure — malformed JSON, empty text, or unparseable",
                        ),
                        ScaleAnchor(
                            value=1,
                            label="partial",
                            description="Parseable but missing expected fields or has structural oddities",
                        ),
                        ScaleAnchor(
                            value=2,
                            label="clean",
                            description="Clean structure, all expected fields present",
                        ),
                    ],
                ),
                weight=1.0,
            ),
            Criterion(
                id="PQ2",
                title="Semantic Specificity",
                description=(
                    "Does the text contain concrete, observable indicators rather "
                    "than vague qualitative language?"
                ),
                scale=NumericScale(
                    minimum=0,
                    maximum=2,
                    step=1,
                    anchors=[
                        ScaleAnchor(
                            value=0,
                            label="vague",
                            description=(
                                "Mostly vague — uses words like 'various', 'generally', "
                                "'might', 'could' without concrete criteria"
                            ),
                        ),
                        ScaleAnchor(
                            value=1,
                            label="mixed",
                            description=(
                                "Mix of specific and vague — some concrete indicators "
                                "but also relies on subjective terms"
                            ),
                        ),
                        ScaleAnchor(
                            value=2,
                            label="specific",
                            description=(
                                "Specific and observable — cites concrete behaviors, "
                                "features, or patterns to look for"
                            ),
                        ),
                    ],
                ),
                weight=1.0,
            ),
            Criterion(
                id="PQ3",
                title="Improvement Clarity",
                description=(
                    "Is the proposed text clearly different from and potentially "
                    "better than the original?"
                ),
                scale=NumericScale(
                    minimum=0,
                    maximum=2,
                    step=1,
                    anchors=[
                        ScaleAnchor(
                            value=0,
                            label="identical",
                            description="Identical or trivially rephrased — no substantive change",
                        ),
                        ScaleAnchor(
                            value=1,
                            label="unclear",
                            description=(
                                "Different but unclear whether it improves — changes "
                                "phrasing without addressing a specific problem"
                            ),
                        ),
                        ScaleAnchor(
                            value=2,
                            label="targeted",
                            description=(
                                "Targeted change — addresses a specific weakness "
                                "identified in the feedback"
                            ),
                        ),
                    ],
                ),
                weight=1.0,
            ),
        ],
        disqualifiers=[
            Disqualifier(
                id="DQ1",
                description="Identical to original — no change proposed",
            ),
        ],
    )


# ── Quality gate ──────────────────────────────────────────────────


class ProposalQualityGate:
    """Pre-filters proposed rubric component text before expensive evaluation.

    Uses rubrify's Judge to evaluate proposed text against a quality rubric.
    Costs 1 LLM call per proposal (vs N_examples * N_criteria for a full eval).
    """

    def __init__(
        self,
        gate_model: Model,
        *,
        gate_api_key: str | None = None,
        threshold: float = 0.5,  # minimum normalized score to pass
        max_retries: int = 1,    # re-propose with feedback if rejected
    ):
        self._rubric = make_proposal_quality_rubric()
        self._compiled = compile_rubric(self._rubric)
        self._model = gate_model
        self._api_key = gate_api_key
        self._threshold = threshold
        self._max_retries = max_retries

    def evaluate_proposal(
        self,
        component_name: str,
        original_text: str,
        proposed_text: str,
    ) -> tuple[bool, str, float]:
        """Evaluate a proposed text. Returns (passed, feedback, score).

        Uses async bridge since GEPA calls this synchronously.
        """
        # Build the response text for the quality rubric judge:
        # The "response" being judged is the proposal itself
        response = (
            f"Component: {component_name}\n\n"
            f"Original:\n{original_text}\n\n"
            f"Proposed:\n{proposed_text}"
        )

        judge = Judge(JudgeConfig(model=self._model, api_key=self._api_key, use_tool=True))

        # Async bridge
        async def _run():
            return await judge.evaluate(self._compiled.bundle, response)

        judgment = run_async(_run())

        score = judgment.aggregation.normalized_score / 100.0
        passed = score >= self._threshold

        # Build feedback from the judgment for potential re-proposal
        feedback_parts = []
        for cj in judgment.criterion_judgments:
            if cj.unit_score < 0.5:
                feedback_parts.append(f"{cj.criterion_id}: {cj.rationale}")
        feedback = "; ".join(feedback_parts) if feedback_parts else "Proposal passed quality check."

        return passed, feedback, score


__all__ = [
    "ProposalQualityGate",
    "make_proposal_quality_rubric",
]
