"""Tests for Client and ChatClient Protocol."""

import os
from unittest.mock import patch

import pytest

from rubrify.client import AnthropicClient, ChatClient, Client, OpenAIClient, OpenRouterClient


class TestClientConstruction:
    def test_defaults(self) -> None:
        c = Client()
        assert c.base_url == ""
        assert c.api_key == ""

    def test_with_args(self) -> None:
        c = Client(base_url="https://api.example.com/", api_key="sk-test")
        assert c.base_url == "https://api.example.com"  # Trailing slash stripped
        assert c.api_key == "sk-test"

    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"RUBRIFY_BASE_URL": "https://api.test.com", "RUBRIFY_API_KEY": "sk-env"},
        ):
            c = Client.from_env()
            assert c.base_url == "https://api.test.com"
            assert c.api_key == "sk-env"

    def test_from_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            c = Client.from_env()
            assert c.base_url == ""
            assert c.api_key == ""

    def test_context_manager(self) -> None:
        with Client() as c:
            assert isinstance(c, Client)


class TestChatClientProtocol:
    def test_client_is_instance(self) -> None:
        c = Client()
        assert isinstance(c, ChatClient)

    def test_custom_client_conforms(self) -> None:
        class MyClient:
            def chat(
                self,
                *,
                messages: list[dict[str, str]],
                model: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> str:
                return "response"

        mc = MyClient()
        assert isinstance(mc, ChatClient)

    def test_non_conforming_class(self) -> None:
        class BadClient:
            def do_stuff(self) -> str:
                return "nope"

        bc = BadClient()
        assert not isinstance(bc, ChatClient)


class TestOpenRouterClient:
    def test_is_chatclient(self) -> None:
        c = OpenRouterClient(api_key="test")
        assert isinstance(c, ChatClient)

    def test_default_base_url(self) -> None:
        assert OpenRouterClient.OPENROUTER_BASE_URL == "https://openrouter.ai/api"

    def test_from_env(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}):
            c = OpenRouterClient.from_env()
            assert c.api_key == "sk-or-test"

    def test_context_manager(self) -> None:
        with OpenRouterClient(api_key="test") as c:
            assert isinstance(c, OpenRouterClient)

    def test_custom_headers(self) -> None:
        c = OpenRouterClient(api_key="test", app_name="myapp", site_url="https://example.com")
        assert c.app_name == "myapp"
        assert c.site_url == "https://example.com"


class TestOpenAIClient:
    def test_is_chatclient(self) -> None:
        c = OpenAIClient(api_key="sk-test")
        assert isinstance(c, ChatClient)

    def test_from_env(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            c = OpenAIClient.from_env()
            assert isinstance(c, ChatClient)

    def test_context_manager(self) -> None:
        with OpenAIClient(api_key="sk-test") as c:
            assert isinstance(c, OpenAIClient)


class TestAnthropicClient:
    def test_is_chatclient(self) -> None:
        c = AnthropicClient(api_key="sk-ant-test")
        assert isinstance(c, ChatClient)

    def test_from_env(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            c = AnthropicClient.from_env()
            assert isinstance(c, ChatClient)

    def test_context_manager(self) -> None:
        with AnthropicClient(api_key="sk-ant-test") as c:
            assert isinstance(c, AnthropicClient)


class TestOpenRouterLive:
    @pytest.mark.integration
    def test_openrouter_chat(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")
        c = OpenRouterClient(api_key=api_key)
        result = c.chat(
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            model="anthropic/claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=50,
        )
        assert len(result) > 0, "OpenRouter returned empty response"
        print(f"OpenRouter response: {result!r}")

    @pytest.mark.integration
    def test_openrouter_rubric_evaluate(self) -> None:
        import rubrify

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")
        c = OpenRouterClient(api_key=api_key)
        rubric = rubrify.load("tests/fixtures/on_writing_well_v3.xml")
        result = rubric.evaluate(
            "The sunset was very beautiful and quite breathtaking.",
            client=c,
            model="anthropic/claude-sonnet-4-6",
        )
        assert result.score is not None, "Score should not be None"
        assert result.raw, "Raw should be populated"
        print(f"OpenRouter rubric eval: score={result.score}, label={result.label}")
