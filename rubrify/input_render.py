"""Input renderers — prompt-program input morphologies.

Phase 1 deliverable. Makes the user-message shape pluggable so rubrics can
represent the full set of reference input morphologies (legacy single-text,
red-team conversation judging, template-based transformation) without hardcoding
any particular layout inside ``Rubric.evaluate``.

No renderer branches on metadata. Each renderer is a small dataclass with a
single ``render`` method and explicit fields. All XML escaping is done via
``xml.sax.saxutils.escape``. No silent fallbacks: missing required input fields
raise ``ValueError`` via ``validate_payload``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from xml.sax.saxutils import escape as xml_escape

from rubrify._types import InputField


@runtime_checkable
class InputRenderer(Protocol):
    """Protocol for rendering an input payload into the user-message string."""

    def render(self, payload: dict[str, Any]) -> str:
        """Render ``payload`` into the user message sent to the model."""
        ...


@dataclass(slots=True)
class CandidateTextRenderer:
    """Reproduces the legacy ``Rubric.evaluate`` user-message shape.

    Wraps the main text in ``<{field_name}>...</{field_name}>`` and appends any
    of the declared ``extra_fields`` that are present in the payload as their
    own XML elements, joined with newlines.
    """

    field_name: str = "candidate_text"
    extra_fields: tuple[str, ...] = ("context", "genre", "goal", "audience")

    def render(self, payload: dict[str, Any]) -> str:
        main_text = payload.get("text")
        if main_text is None:
            main_text = payload.get(self.field_name, "")
        escaped_main = xml_escape(str(main_text))
        parts: list[str] = [f"<{self.field_name}>{escaped_main}</{self.field_name}>"]
        for key in self.extra_fields:
            if key in payload:
                escaped = xml_escape(str(payload[key]))
                parts.append(f"<{key}>{escaped}</{key}>")
        return "\n".join(parts)


@dataclass(slots=True)
class ConversationJudgeRenderer:
    """Emits the exact red-team conversation-judging prompt shape.

    Matches the user-message structure from ``red_team_rubric.py``:

        <Query>
        {query_template}
        </Query>

        <Conversation>
        <User_turn>...</User_turn>
        <Model_Response>...</Model_Response>
        </Conversation>
    """

    query_template: str
    user_field: str = "user_turn"
    assistant_field: str = "model_response"

    def render(self, payload: dict[str, Any]) -> str:
        user_turn = xml_escape(str(payload.get(self.user_field, "")))
        model_response = xml_escape(str(payload.get(self.assistant_field, "")))
        return (
            "<Query>\n"
            f"{self.query_template}\n"
            "</Query>\n"
            "\n"
            "<Conversation>\n"
            f"<User_turn>{user_turn}</User_turn>\n"
            f"<Model_Response>{model_response}</Model_Response>\n"
            "</Conversation>"
        )


@dataclass(slots=True)
class TemplateRenderer:
    """Substitutes ``{placeholder}`` tokens in a free-form template.

    Each name in ``placeholders`` is looked up in the payload and its XML-escaped
    value is substituted for the literal ``{name}`` token in ``template``.
    """

    template: str
    placeholders: tuple[str, ...] = ("content",)

    def render(self, payload: dict[str, Any]) -> str:
        result = self.template
        for name in self.placeholders:
            token = "{" + name + "}"
            value = xml_escape(str(payload[name]))
            result = result.replace(token, value)
        return result


@dataclass(slots=True)
class PassthroughRenderer:
    """Returns ``payload['text']`` verbatim — no wrapping, no escaping."""

    def render(self, payload: dict[str, Any]) -> str:
        return str(payload.get("text", ""))


def validate_payload(payload: dict[str, Any], inputs: list[InputField]) -> None:
    """Raise ``ValueError`` if any required ``InputField`` is missing from ``payload``.

    Non-required fields are not checked. Empty-string values are considered
    present; the caller decides what constitutes a meaningful value.
    """
    missing: list[str] = []
    for input_field in inputs:
        if input_field.required and input_field.name not in payload:
            missing.append(input_field.name)
    if missing:
        raise ValueError("payload is missing required input field(s): " + ", ".join(missing))
