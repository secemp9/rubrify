from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import httpx


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for any LLM client. Implement this to use your own client."""

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send chat messages and return the completion text."""
        ...


class Client:
    """Built-in httpx-based client for OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.Client(timeout=120.0)

    @classmethod
    def from_env(cls) -> Client:
        return cls(
            base_url=os.environ.get("RUBRIFY_BASE_URL", ""),
            api_key=os.environ.get("RUBRIFY_API_KEY", ""),
        )

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        result: str = data["choices"][0]["message"]["content"]
        return result

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class OpenRouterClient:
    """Client for OpenRouter API via httpx.

    Uses the OpenRouter OpenAI-compatible endpoint with provider-specific
    headers (``X-Title``, ``HTTP-Referer``).

    Model names use OpenRouter format: ``anthropic/claude-sonnet-4-6``,
    ``openai/gpt-4o``, ``google/gemini-2.5-pro``, etc.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api"

    def __init__(
        self,
        api_key: str = "",
        *,
        app_name: str = "rubrify",
        site_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.app_name = app_name
        self.site_url = site_url
        self._http = httpx.Client(timeout=120.0)

    @classmethod
    def from_env(cls) -> OpenRouterClient:
        return cls(api_key=os.environ.get("OPENROUTER_API_KEY", ""))

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        url = f"{self.OPENROUTER_BASE_URL}/v1/chat/completions"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        result: str = data["choices"][0]["message"]["content"]
        return result

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class OpenAIClient:
    """Client wrapping the official ``openai`` Python SDK.

    Requires: ``pip install rubrify[openai]``

    Model names: ``gpt-5``, ``gpt-4o``, ``gpt-4.1-mini``, etc.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIClient. "
                "Install it with: pip install rubrify[openai]"
            ) from e

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**kwargs)

    @classmethod
    def from_env(cls) -> OpenAIClient:
        return cls(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return content or ""

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AnthropicClient:
    """Client wrapping the official ``anthropic`` Python SDK.

    Requires: ``pip install rubrify[anthropic]``

    Model names: ``claude-sonnet-4-6``, ``claude-opus-4-5``, etc.

    The ``ChatClient`` protocol passes the system prompt as a message with
    ``role='system'``.  This client extracts it and forwards it via the
    Anthropic ``system`` parameter, which is the SDK's expected shape.
    """

    def __init__(
        self,
        api_key: str = "",
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install rubrify[anthropic]"
            ) from e

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = _anthropic.Anthropic(**kwargs)

    @classmethod
    def from_env(cls) -> AnthropicClient:
        return cls(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        system_text = ""
        user_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
        }
        if system_text:
            kwargs["system"] = system_text
        if temperature > 0:
            kwargs["temperature"] = temperature

        resp = self._client.messages.create(**kwargs)
        text_parts = [block.text for block in resp.content if block.type == "text"]
        return "".join(text_parts)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AnthropicClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
