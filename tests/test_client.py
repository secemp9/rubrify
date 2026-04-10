"""Tests for Client and ChatClient Protocol."""

import os
from unittest.mock import patch

import pytest

from rubrify.client import (
    AnthropicClient,
    ChatClient,
    Client,
    OpenAIClient,
    OpenRouterClient,
    _resolve_model_for_openrouter,
    _strip_provider_prefix,
)


class TestClientConstruction:
    def test_defaults(self) -> None:
        c = Client()
        assert c.base_url == ""
        assert c.api_key == ""
        assert c.provider == "generic"

    def test_with_base_url(self) -> None:
        c = Client(base_url="https://api.example.com/", api_key="sk-test")
        assert c.base_url == "https://api.example.com"  # Trailing slash stripped
        assert c.api_key == "sk-test"
        assert c.provider == "generic"

    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"RUBRIFY_BASE_URL": "https://api.test.com", "RUBRIFY_API_KEY": "sk-env"},
            clear=True,
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


class TestClientAutoDetection:
    """Client auto-detects provider from API key prefix."""

    def test_generic_with_base_url(self) -> None:
        c = Client(base_url="http://localhost:8317", api_key="anything")
        assert c.provider == "generic"

    def test_openrouter_from_key(self) -> None:
        c = Client(api_key="sk-or-v1-abc123")
        assert c.provider == "openrouter"

    def test_anthropic_from_key(self) -> None:
        c = Client(api_key="sk-ant-api03-abc123")
        assert c.provider == "anthropic"

    def test_openai_from_key(self) -> None:
        c = Client(api_key="sk-proj-abc123")
        assert c.provider == "openai"

    def test_explicit_provider_overrides_detection(self) -> None:
        c = Client(api_key="sk-or-v1-abc", provider="openai")
        assert c.provider == "openai"

    def test_empty_key_is_generic(self) -> None:
        c = Client()
        assert c.provider == "generic"

    def test_unknown_key_is_generic(self) -> None:
        c = Client(api_key="some-random-key")
        assert c.provider == "generic"

    def test_base_url_forces_generic_even_with_or_key(self) -> None:
        c = Client(base_url="http://localhost:8317", api_key="sk-or-v1-abc")
        assert c.provider == "generic"

    def test_is_chatclient_regardless_of_provider(self) -> None:
        for key in ["sk-or-v1-abc", "sk-ant-api03-abc", "sk-proj-abc", "generic-key"]:
            c = Client(api_key=key)
            assert isinstance(c, ChatClient), f"Client(api_key={key!r}) is not ChatClient"

    def test_from_env_openrouter(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-test"}, clear=True):
            c = Client.from_env()
            assert c.provider == "openrouter"

    def test_from_env_anthropic(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            c = Client.from_env()
            assert c.provider == "anthropic"

    def test_from_env_openai(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            c = Client.from_env()
            assert c.provider == "openai"

    def test_from_env_generic_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"RUBRIFY_BASE_URL": "http://localhost", "RUBRIFY_API_KEY": "key"},
            clear=True,
        ):
            c = Client.from_env()
            assert c.provider == "generic"

    def test_from_env_priority_order(self) -> None:
        """OpenRouter takes priority over Anthropic over OpenAI."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-v1-test",
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "OPENAI_API_KEY": "sk-test",
            },
            clear=True,
        ):
            c = Client.from_env()
            assert c.provider == "openrouter"

    def test_context_manager_with_delegate(self) -> None:
        with Client(api_key="sk-or-v1-abc") as c:
            assert c.provider == "openrouter"


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


class TestModelNameResolution:
    """Model names resolve correctly across providers."""

    # --- OpenRouter resolution ---
    def test_bare_claude_gets_anthropic_prefix(self) -> None:
        assert _resolve_model_for_openrouter("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"

    def test_bare_gpt_gets_openai_prefix(self) -> None:
        assert _resolve_model_for_openrouter("gpt-4o") == "openai/gpt-4o"

    def test_bare_gemini_gets_google_prefix(self) -> None:
        assert _resolve_model_for_openrouter("gemini-2.5-pro") == "google/gemini-2.5-pro"

    def test_bare_llama_gets_meta_prefix(self) -> None:
        assert _resolve_model_for_openrouter("llama-3-70b") == "meta-llama/llama-3-70b"

    def test_bare_mistral_gets_mistralai_prefix(self) -> None:
        assert _resolve_model_for_openrouter("mistral-large") == "mistralai/mistral-large"

    def test_bare_deepseek_gets_deepseek_prefix(self) -> None:
        assert _resolve_model_for_openrouter("deepseek-r1") == "deepseek/deepseek-r1"

    def test_already_prefixed_passthrough(self) -> None:
        assert (
            _resolve_model_for_openrouter("anthropic/claude-sonnet-4-6")
            == "anthropic/claude-sonnet-4-6"
        )

    def test_unknown_bare_passthrough(self) -> None:
        assert _resolve_model_for_openrouter("some-custom-model") == "some-custom-model"

    # --- Prefix stripping ---
    def test_strip_anthropic_prefix(self) -> None:
        assert _strip_provider_prefix("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_strip_openai_prefix(self) -> None:
        assert _strip_provider_prefix("openai/gpt-4o") == "gpt-4o"

    def test_bare_name_unchanged(self) -> None:
        assert _strip_provider_prefix("claude-sonnet-4-6") == "claude-sonnet-4-6"

    # --- Unified Client resolves correctly ---
    def test_unified_client_openrouter_resolves_bare_claude(self) -> None:
        c = Client(api_key="sk-or-v1-test")
        assert c.provider == "openrouter"
        assert _resolve_model_for_openrouter("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"

    def test_unified_client_anthropic_strips_prefix(self) -> None:
        c = Client(api_key="sk-ant-test")
        assert c.provider == "anthropic"
        assert _strip_provider_prefix("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_unified_client_generic_passthrough(self) -> None:
        c = Client(base_url="http://localhost", api_key="test")
        assert c.provider == "generic"
