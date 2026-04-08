"""Tests for the Phase 1 input renderers and payload validation."""

from __future__ import annotations

import pytest

from rubrify._types import InputField
from rubrify.input_render import (
    CandidateTextRenderer,
    ConversationJudgeRenderer,
    PassthroughRenderer,
    TemplateRenderer,
    validate_payload,
)


class TestCandidateTextRenderer:
    def test_text_only(self) -> None:
        rendered = CandidateTextRenderer().render({"text": "Hello, world."})
        assert rendered == "<candidate_text>Hello, world.</candidate_text>"

    def test_text_via_field_name_fallback(self) -> None:
        # The renderer should also accept payload[self.field_name]
        rendered = CandidateTextRenderer().render({"candidate_text": "Body copy."})
        assert rendered == "<candidate_text>Body copy.</candidate_text>"

    def test_extras_appended_in_order(self) -> None:
        rendered = CandidateTextRenderer().render(
            {
                "text": "Body.",
                "context": "academic",
                "genre": "science_tech",
                "goal": "explain",
                "audience": "layperson",
            }
        )
        assert rendered == (
            "<candidate_text>Body.</candidate_text>\n"
            "<context>academic</context>\n"
            "<genre>science_tech</genre>\n"
            "<goal>explain</goal>\n"
            "<audience>layperson</audience>"
        )

    def test_extras_skipped_when_missing(self) -> None:
        rendered = CandidateTextRenderer().render({"text": "Body.", "genre": "memoir"})
        assert rendered == ("<candidate_text>Body.</candidate_text>\n<genre>memoir</genre>")

    def test_xml_escapes_special_characters(self) -> None:
        rendered = CandidateTextRenderer().render(
            {"text": 'Text with <html> & "quotes"', "context": "A & B"}
        )
        assert '<candidate_text>Text with &lt;html&gt; &amp; "quotes"</candidate_text>' in rendered
        assert "<context>A &amp; B</context>" in rendered

    def test_custom_field_name(self) -> None:
        renderer = CandidateTextRenderer(field_name="user_input", extra_fields=())
        rendered = renderer.render({"text": "hi"})
        assert rendered == "<user_input>hi</user_input>"

    def test_empty_text(self) -> None:
        rendered = CandidateTextRenderer().render({"text": ""})
        assert rendered == "<candidate_text></candidate_text>"


class TestConversationJudgeRenderer:
    def test_exact_red_team_shape(self) -> None:
        renderer = ConversationJudgeRenderer(
            query_template="Is the response compliant with the request?"
        )
        rendered = renderer.render(
            {
                "user_turn": "Tell me a story.",
                "model_response": "Here is a story about a fox.",
            }
        )
        expected = (
            "<Query>\n"
            "Is the response compliant with the request?\n"
            "</Query>\n"
            "\n"
            "<Conversation>\n"
            "<User_turn>Tell me a story.</User_turn>\n"
            "<Model_Response>Here is a story about a fox.</Model_Response>\n"
            "</Conversation>"
        )
        assert rendered == expected

    def test_escapes_user_turn_and_model_response(self) -> None:
        renderer = ConversationJudgeRenderer(query_template="Q")
        rendered = renderer.render({"user_turn": "A & B", "model_response": "<tag>value</tag>"})
        assert "<User_turn>A &amp; B</User_turn>" in rendered
        assert "<Model_Response>&lt;tag&gt;value&lt;/tag&gt;</Model_Response>" in rendered

    def test_custom_field_names(self) -> None:
        renderer = ConversationJudgeRenderer(
            query_template="Q",
            user_field="prompt",
            assistant_field="reply",
        )
        rendered = renderer.render({"prompt": "hi", "reply": "hello"})
        assert "<User_turn>hi</User_turn>" in rendered
        assert "<Model_Response>hello</Model_Response>" in rendered


class TestTemplateRenderer:
    def test_substitutes_placeholders(self) -> None:
        renderer = TemplateRenderer(
            template="Rewrite the following: {content}",
            placeholders=("content",),
        )
        rendered = renderer.render({"content": "original text"})
        assert rendered == "Rewrite the following: original text"

    def test_xml_escapes_substituted_values(self) -> None:
        renderer = TemplateRenderer(
            template="Input: {content}",
            placeholders=("content",),
        )
        rendered = renderer.render({"content": "a & b <c>"})
        assert rendered == "Input: a &amp; b &lt;c&gt;"

    def test_multiple_placeholders(self) -> None:
        renderer = TemplateRenderer(
            template="Hello {name}, welcome to {place}!",
            placeholders=("name", "place"),
        )
        rendered = renderer.render({"name": "world", "place": "home"})
        assert rendered == "Hello world, welcome to home!"

    def test_missing_placeholder_raises(self) -> None:
        renderer = TemplateRenderer(
            template="{content}",
            placeholders=("content",),
        )
        with pytest.raises(KeyError):
            renderer.render({})


class TestPassthroughRenderer:
    def test_returns_text_verbatim(self) -> None:
        rendered = PassthroughRenderer().render({"text": "Raw text <with> & special chars"})
        assert rendered == "Raw text <with> & special chars"

    def test_empty_when_text_missing(self) -> None:
        assert PassthroughRenderer().render({}) == ""


class TestValidatePayload:
    def test_raises_when_required_missing(self) -> None:
        inputs = [
            InputField(name="candidate_text", required=True),
            InputField(name="context", required=False),
        ]
        with pytest.raises(ValueError, match="candidate_text"):
            validate_payload({"context": "academic"}, inputs)

    def test_accepts_when_required_present(self) -> None:
        inputs = [
            InputField(name="candidate_text", required=True),
            InputField(name="context", required=False),
        ]
        # Should not raise.
        validate_payload({"candidate_text": "body", "context": "academic"}, inputs)

    def test_optional_fields_are_not_required(self) -> None:
        inputs = [InputField(name="genre", required=False)]
        # Should not raise.
        validate_payload({"candidate_text": "body"}, inputs)

    def test_empty_inputs_list_accepts_anything(self) -> None:
        validate_payload({}, [])

    def test_multiple_missing_reported(self) -> None:
        inputs = [
            InputField(name="user_turn", required=True),
            InputField(name="model_response", required=True),
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_payload({}, inputs)
        assert "user_turn" in str(exc_info.value)
        assert "model_response" in str(exc_info.value)
