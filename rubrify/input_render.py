"""Input renderers — prompt-program input morphologies.

Phase 1 deliverable. Makes the user-message shape pluggable so rubrics can
represent the full set of reference input morphologies (legacy single-text,
red-team conversation judging, template-based transformation) without hardcoding
any particular layout inside ``Rubric.evaluate``.

No renderer branches on metadata. Each renderer is a small dataclass with a
single ``render`` method and explicit fields. XML construction uses
``xml.etree.ElementTree`` where possible; ``TemplateRenderer`` uses
``xml.sax.saxutils.escape`` for value substitution into free-form templates.
No silent fallbacks: missing required input fields raise ``ValueError`` via
``validate_payload``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
        elem = ET.Element(self.field_name)
        elem.text = str(main_text)
        parts: list[str] = [ET.tostring(elem, encoding="unicode", short_empty_elements=False)]
        for key in self.extra_fields:
            if key in payload:
                child = ET.Element(key)
                child.text = str(payload[key])
                parts.append(ET.tostring(child, encoding="unicode", short_empty_elements=False))
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
        query = ET.Element("Query")
        query.text = "\n" + self.query_template + "\n"

        conversation = ET.Element("Conversation")
        conversation.text = "\n"

        user_turn = ET.SubElement(conversation, "User_turn")
        user_turn.text = str(payload.get(self.user_field, ""))
        user_turn.tail = "\n"

        model_response = ET.SubElement(conversation, "Model_Response")
        model_response.text = str(payload.get(self.assistant_field, ""))
        model_response.tail = "\n"

        query_str = ET.tostring(query, encoding="unicode", short_empty_elements=False)
        conv_str = ET.tostring(conversation, encoding="unicode", short_empty_elements=False)
        return query_str + "\n\n" + conv_str


@dataclass(slots=True)
class TemplateRenderer:
    """Substitutes ``{placeholder}`` tokens in a free-form template.

    Each name in ``placeholders`` is looked up in the payload and its XML-escaped
    value is substituted for the literal ``{name}`` token in ``template``.
    """

    template: str
    placeholders: tuple[str, ...] = ("content",)

    def render(self, payload: dict[str, Any]) -> str:
        from xml.sax.saxutils import escape as xml_escape

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
