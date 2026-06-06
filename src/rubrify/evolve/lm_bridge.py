"""LM bridge: wrap a harn_ai Model as GEPA's LanguageModel protocol.

This ensures ALL LLM calls -- both judging and reflection -- go through
harn_ai. No litellm, no direct API calls.

GEPA's LanguageModel protocol is: (str | list[dict]) -> str
harn_ai's complete_simple is: async (Model, Context, opts) -> AssistantMessage
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from harn_ai.stream import complete_simple
from harn_ai.types import Context, Model, SimpleStreamOptions, UserMessage

from rubrify.evolve.async_bridge import run_async


def make_harn_lm(
    model: Model,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Callable[[str | list[dict[str, Any]]], str]:
    """Wrap a harn_ai Model as a GEPA LanguageModel for the reflection step.

    The returned callable conforms to GEPA's LanguageModel protocol:
        (str | list[dict[str, Any]]) -> str

    When prompt is a string, creates a single UserMessage.
    When prompt is a list[dict], concatenates content into a single UserMessage.
    """

    def lm_fn(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [UserMessage(
                role="user",
                content=prompt,
                timestamp=int(time.time() * 1000),
            )]
        else:
            # Handle chat-messages format from GEPA: concatenate all content
            # into a single UserMessage since harn_ai's simple API expects
            # a flat message list.
            parts = []
            for msg in prompt:
                parts.append(msg.get("content", ""))
            messages = [UserMessage(
                role="user",
                content="\n\n".join(parts),
                timestamp=int(time.time() * 1000),
            )]

        context = Context(messages=messages)
        opts = SimpleStreamOptions(
            apiKey=api_key,
            temperature=temperature,
            maxTokens=max_tokens,
        )

        async def _call() -> str:
            result = await complete_simple(model, context, opts)
            text_parts = []
            for block in result.content:
                if block.type == "text":
                    text_parts.append(block.text)
            return "\n".join(text_parts)

        return run_async(_call())

    return lm_fn


__all__ = [
    "make_harn_lm",
]
