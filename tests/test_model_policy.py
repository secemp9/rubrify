"""Tests for Phase 5: model policy tiers and warn_unsupported helper."""

from __future__ import annotations

import warnings

import pytest

from rubrify.model_policy import (
    DISCOURAGED,
    EXPERIMENTAL,
    RECOMMENDED,
    SUPPORTED,
    ModelTier,
    check_model,
    normalize_model_name,
    warn_unsupported,
)


class TestModelTierEnum:
    def test_model_tier_enum(self) -> None:
        """ModelTier has the five expected members."""
        assert ModelTier.RECOMMENDED.value == "recommended"
        assert ModelTier.SUPPORTED.value == "supported"
        assert ModelTier.EXPERIMENTAL.value == "experimental"
        assert ModelTier.DISCOURAGED.value == "discouraged"
        assert ModelTier.UNKNOWN.value == "unknown"
        # Exactly five tiers — guard rail 8 keeps this list stable.
        assert set(ModelTier) == {
            ModelTier.RECOMMENDED,
            ModelTier.SUPPORTED,
            ModelTier.EXPERIMENTAL,
            ModelTier.DISCOURAGED,
            ModelTier.UNKNOWN,
        }

    def test_tier_constants_are_tuples(self) -> None:
        """Guard rail 4: tiers are plain fnmatch-pattern tuples, no registry."""
        assert isinstance(RECOMMENDED, tuple)
        assert isinstance(SUPPORTED, tuple)
        assert isinstance(EXPERIMENTAL, tuple)
        assert isinstance(DISCOURAGED, tuple)


class TestCheckModelRecommended:
    def test_check_model_recommended_claude(self) -> None:
        tier, message = check_model("claude-sonnet-4-6")
        assert tier is ModelTier.RECOMMENDED
        assert "recommended" in message

    def test_check_model_recommended_gpt(self) -> None:
        tier, _ = check_model("gpt-5")
        assert tier is ModelTier.RECOMMENDED

    @pytest.mark.parametrize(
        "name",
        ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"],
    )
    def test_check_model_recommended_gpt_variants(self, name: str) -> None:
        tier, _ = check_model(name)
        assert tier is ModelTier.RECOMMENDED

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
        ],
    )
    def test_check_model_recommended_claude_variants(self, name: str) -> None:
        tier, _ = check_model(name)
        assert tier is ModelTier.RECOMMENDED

    @pytest.mark.parametrize(
        "name",
        ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
    )
    def test_check_model_recommended_gpt4_newer(self, name: str) -> None:
        tier, _ = check_model(name)
        assert tier is ModelTier.RECOMMENDED


class TestCheckModelSupported:
    def test_check_model_supported_gpt4(self) -> None:
        tier, message = check_model("gpt-4")
        assert tier is ModelTier.SUPPORTED
        assert "supported" in message

    def test_check_model_supported_gpt4_turbo(self) -> None:
        tier, _ = check_model("gpt-4-turbo")
        assert tier is ModelTier.SUPPORTED

    def test_check_model_supported_claude_3(self) -> None:
        tier, _ = check_model("claude-3-haiku-20240307")
        assert tier is ModelTier.SUPPORTED


class TestCheckModelExperimental:
    def test_check_model_experimental_mistral(self) -> None:
        tier, message = check_model("mistral-large-2411")
        assert tier is ModelTier.EXPERIMENTAL
        assert "experimental" in message

    def test_check_model_experimental_llama(self) -> None:
        tier, _ = check_model("llama-3.1-70b-instruct")
        assert tier is ModelTier.EXPERIMENTAL

    @pytest.mark.parametrize(
        "name",
        [
            "gpt-3.5-turbo",
            "mistral-small",
            "llama-3",
            "qwen-2.5-72b",
            "gemini-1.5-pro",
            "deepseek-v2",
        ],
    )
    def test_check_model_experimental_families(self, name: str) -> None:
        tier, _ = check_model(name)
        assert tier is ModelTier.EXPERIMENTAL


class TestCheckModelDiscouraged:
    def test_check_model_discouraged_text_davinci(self) -> None:
        tier, message = check_model("text-davinci-003")
        assert tier is ModelTier.DISCOURAGED
        assert "discouraged" in message

    @pytest.mark.parametrize(
        "name",
        [
            "davinci-002",
            "curie",
            "babbage-002",
            "ada",
            "text-curie-001",
        ],
    )
    def test_check_model_discouraged_legacy(self, name: str) -> None:
        tier, _ = check_model(name)
        assert tier is ModelTier.DISCOURAGED


class TestCheckModelUnknown:
    def test_check_model_unknown_returns_guidance(self) -> None:
        tier, message = check_model("totally-made-up-model-xyz")
        assert tier is ModelTier.UNKNOWN
        assert "not in the policy" in message

    def test_check_model_empty_string_unknown(self) -> None:
        tier, _ = check_model("")
        assert tier is ModelTier.UNKNOWN


class TestCheckModelContract:
    def test_check_model_returns_tuple(self) -> None:
        """check_model always returns a (ModelTier, str) tuple."""
        for name in [
            "gpt-5",
            "gpt-4",
            "mistral-7b",
            "text-davinci-003",
            "no-such-model",
        ]:
            result = check_model(name)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], ModelTier)
            assert isinstance(result[1], str)
            assert result[1] != ""

    def test_check_model_message_includes_name(self) -> None:
        """The message always quotes the model name so guard rail 6 holds."""
        for name in [
            "gpt-5",
            "gpt-4",
            "mistral-7b",
            "text-davinci-003",
            "no-such-model",
        ]:
            _, message = check_model(name)
            assert repr(name) in message


class TestWarnUnsupported:
    def test_warn_unsupported_recommended_no_warning(self) -> None:
        """No warning is emitted for a model in the recommended tier."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any warning becomes an error.
            warn_unsupported("claude-sonnet-4-6")
            warn_unsupported("gpt-5")

    def test_warn_unsupported_supported_no_warning(self) -> None:
        """No warning is emitted for a model in the supported tier."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_unsupported("gpt-4")

    def test_warn_unsupported_experimental_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="experimental"):
            warn_unsupported("mistral-large")

    def test_warn_unsupported_discouraged_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="discouraged"):
            warn_unsupported("text-davinci-003")

    def test_warn_unsupported_unknown_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="not in the policy"):
            warn_unsupported("some-random-model")

    def test_warn_unsupported_warning_contains_policy_prefix(self) -> None:
        """The emitted warning is explicitly namespaced to rubrify model policy."""
        with pytest.warns(UserWarning, match="rubrify model policy:"):
            warn_unsupported("mistral-large")


class TestNormalizeModelName:
    def test_strip_openai_prefix(self) -> None:
        assert normalize_model_name("openai/gpt-5") == "gpt-5"

    def test_strip_anthropic_prefix(self) -> None:
        assert normalize_model_name("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_strip_google_prefix(self) -> None:
        assert normalize_model_name("google/gemini-2.5-pro") == "gemini-2.5-pro"

    def test_strip_meta_prefix(self) -> None:
        assert normalize_model_name("meta-llama/llama-3-70b") == "llama-3-70b"

    def test_strip_deepseek_prefix(self) -> None:
        assert normalize_model_name("deepseek/deepseek-r1") == "deepseek-r1"

    def test_no_prefix(self) -> None:
        assert normalize_model_name("gpt-5") == "gpt-5"

    def test_unknown_prefix_unchanged(self) -> None:
        assert normalize_model_name("someprovider/somemodel") == "someprovider/somemodel"


class TestCheckModelWithProviderPrefix:
    def test_openrouter_openai_gpt5(self) -> None:
        tier, msg = check_model("openai/gpt-5")
        assert tier == ModelTier.RECOMMENDED

    def test_openrouter_anthropic_claude(self) -> None:
        tier, msg = check_model("anthropic/claude-sonnet-4-6")
        assert tier == ModelTier.RECOMMENDED

    def test_openrouter_google_gemini(self) -> None:
        tier, msg = check_model("google/gemini-2.5-pro")
        assert tier == ModelTier.EXPERIMENTAL

    def test_openrouter_meta_llama(self) -> None:
        tier, msg = check_model("meta-llama/llama-3-70b")
        assert tier == ModelTier.EXPERIMENTAL

    def test_bare_name_still_works(self) -> None:
        tier, msg = check_model("claude-sonnet-4-6")
        assert tier == ModelTier.RECOMMENDED
