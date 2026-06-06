"""Judge: stateful public API for rubric-based evaluation.

The Judge accepts a harn_ai Model directly — any model harn can
talk to, rubrify can judge with. No wrapper, no abstraction layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harn_ai.env_api_keys import get_env_api_key
from harn_ai.types import Model

from rubrify.ir.bundle import RubricBundle
from rubrify.engine.judgment import CriterionJudgment, Judgment, JudgeUsage
from rubrify.engine.judge_loop import run_judge_loop


@dataclass(slots=True)
class JudgeConfig:
    """Configuration for a Judge instance.

    model: A harn_ai Model — same object you'd pass to complete_simple().
           Works with any provider harn supports (OpenAI, Anthropic,
           DeepSeek, local proxies, etc.)
    api_key: Optional. If None, auto-discovered from environment via
             harn's get_env_api_key(model.provider).
    """
    model: Model
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    parallel: bool = False
    use_tool: bool = True


class Judge:
    """Stateful judge that evaluates responses against rubric bundles."""

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config
        self._total_usage = JudgeUsage()
        self._evaluation_count = 0
        if self._config.api_key is None:
            self._config.api_key = get_env_api_key(config.model.provider)

    @property
    def model(self) -> Model:
        return self._config.model

    @property
    def total_usage(self) -> JudgeUsage:
        return self._total_usage

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    async def evaluate(
        self,
        bundle: RubricBundle,
        response_text: str,
        *,
        context_text: str | None = None,
        genre: str | None = None,
        on_criterion_start: Callable[[str], None] | None = None,
        on_criterion_done: Callable[[str, CriterionJudgment], None] | None = None,
    ) -> Judgment:
        """Evaluate a response against a locked rubric bundle."""
        judgment = await run_judge_loop(
            bundle=bundle,
            response_text=response_text,
            model=self._config.model,
            context_text=context_text,
            active_genre=genre,
            api_key=self._config.api_key,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            parallel=self._config.parallel,
            use_tool=self._config.use_tool,
            on_criterion_start=on_criterion_start,
            on_criterion_done=on_criterion_done,
        )

        self._total_usage += judgment.usage
        self._evaluation_count += 1
        return judgment


__all__ = [
    "Judge",
    "JudgeConfig",
]
