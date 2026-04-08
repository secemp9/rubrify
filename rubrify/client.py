from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

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
