"""Applied-text feedback loop.

Phase 4 deliverable. A rubric's advice is supposed to improve the text it
judges; this module closes that loop. ``improve_text`` evaluates a text,
extracts advice strings, asks the model to rewrite the text against those
strings, and re-evaluates. It returns an :class:`ImproveReport` with the
before/after scores and the exact advice that was applied.

This is strictly opt-in and NEVER invoked from :meth:`Rubric.evaluate`
(philosophy anchor 10 / guard rail 8). Callers reach for it explicitly
when they want the coaching loop. When advice is empty the loop short
circuits without a second model call — guard rail 6 forbids silent
fallbacks, so the skipped-improvement path is fully observable via the
returned report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rubrify.client import ChatClient
from rubrify.result import EvaluationResult
from rubrify.rubric import Rubric


@dataclass(frozen=True, slots=True)
class ImproveReport:
    """Outcome of a single :func:`improve_text` call.

    ``applied_advice`` is the exact tuple of advice strings handed to the
    improvement model, preserved so callers can audit what the loop acted
    on. When the loop short-circuits (no advice extracted), ``improved_text``
    equals ``original_text`` and the before/after scores are identical.
    """

    before_score: int | None
    after_score: int | None
    applied_advice: tuple[str, ...]
    original_text: str
    improved_text: str
    before_result: EvaluationResult
    after_result: EvaluationResult

    @property
    def improved(self) -> bool:
        """``True`` iff both scores are known and ``after_score > before_score``."""
        return (
            self.before_score is not None
            and self.after_score is not None
            and self.after_score > self.before_score
        )


def default_advice_extractor(result: EvaluationResult) -> list[str]:
    """Extract actionable advice strings from an :class:`EvaluationResult`.

    Reads ``result.advice`` first (the canonical list), then walks
    ``result.actions`` for the well-known keys ``coaching``, ``edits``, and
    ``next_steps`` (strings are appended as-is; lists have their entries
    coerced to strings). Unknown keys are ignored so the extractor stays
    conservative.
    """
    advice: list[str] = []
    if result.advice:
        advice.extend(result.advice)
    if result.actions:
        for key in ("coaching", "edits", "next_steps"):
            val = result.actions.get(key)
            if isinstance(val, list):
                advice.extend(str(item) for item in val)
            elif isinstance(val, str):
                advice.append(val)
    return advice


_DEFAULT_IMPROVEMENT_TEMPLATE = (
    "Rewrite the following text to address this feedback:\n\n"
    "FEEDBACK:\n{advice_block}\n\n"
    "ORIGINAL TEXT:\n{text}\n\n"
    "Output only the improved text. No preamble."
)

_IMPROVEMENT_SYSTEM_MSG = "You improve text according to feedback. Return only the improved text."


def improve_text(
    rubric: Rubric,
    text: str,
    *,
    client: ChatClient,
    model: str,
    advice_extractor: Callable[[EvaluationResult], list[str]] = default_advice_extractor,
    improvement_prompt_template: str | None = None,
) -> ImproveReport:
    """Opt-in applied-text feedback loop.

    1. Evaluate ``text`` with ``rubric`` to obtain the before-result.
    2. Extract advice via ``advice_extractor`` (defaults to
       :func:`default_advice_extractor`).
    3. If no advice was extracted, return an :class:`ImproveReport` with
       the original text and identical before/after scores — no second
       model call happens. This short circuit is explicit so downstream
       code can distinguish "had nothing to say" from "tried and failed".
    4. Otherwise ask the model to rewrite the text against the advice.
    5. Re-evaluate the improved text and return the full
       :class:`ImproveReport`.

    ``improvement_prompt_template`` may override the default template; it
    must contain ``{advice_block}`` and ``{text}`` format fields.
    """
    before_result = rubric.evaluate(text, client=client, model=model)
    applied_advice = tuple(advice_extractor(before_result))

    if not applied_advice:
        return ImproveReport(
            before_score=before_result.score,
            after_score=before_result.score,
            applied_advice=(),
            original_text=text,
            improved_text=text,
            before_result=before_result,
            after_result=before_result,
        )

    template = improvement_prompt_template or _DEFAULT_IMPROVEMENT_TEMPLATE
    advice_block = "\n".join(f"- {line}" for line in applied_advice)
    improvement_prompt = template.format(advice_block=advice_block, text=text)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _IMPROVEMENT_SYSTEM_MSG},
        {"role": "user", "content": improvement_prompt},
    ]
    improved_text = str(client.chat(messages=messages, model=model, temperature=0.0)).strip()

    after_result = rubric.evaluate(improved_text, client=client, model=model)

    return ImproveReport(
        before_score=before_result.score,
        after_score=after_result.score,
        applied_advice=applied_advice,
        original_text=text,
        improved_text=improved_text,
        before_result=before_result,
        after_result=after_result,
    )
