"""Tests for the Phase 1 opt-in EvaluationTrace observability surface."""

from __future__ import annotations

import json
from typing import Any

import rubrify
from rubrify._types import Criterion, OutputSchema
from rubrify.result import EvaluationTrace


class MockClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.last_messages = messages
        return self._response


def _minimal_json_rubric() -> rubrify.Rubric:
    r = rubrify.Rubric(name="T", mission="Test.")
    r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
    r.output_schema = OutputSchema(constraints={"must_be_json": True})
    return r


class TestRubricEvaluateTrace:
    def test_observe_true_attaches_trace(self) -> None:
        rubric = _minimal_json_rubric()
        client = MockClient(json.dumps({"score": 50}))
        result = rubric.evaluate(
            "some text",
            client=client,
            model="test-model",
            observe=True,
        )
        assert result.trace is not None
        assert isinstance(result.trace, EvaluationTrace)

    def test_trace_contains_expected_fields(self) -> None:
        rubric = _minimal_json_rubric()
        client = MockClient(json.dumps({"score": 50}))
        result = rubric.evaluate(
            "sample body",
            client=client,
            model="test-model",
            observe=True,
        )
        trace = result.trace
        assert trace is not None
        assert "LLM_JUDGE_SPEC" in trace.system_prompt
        assert "<candidate_text>sample body</candidate_text>" in trace.user_message
        assert trace.model == "test-model"
        assert trace.parser == "json"
        assert trace.elapsed_seconds >= 0.0
        assert trace.repair_notes == ()

    def test_default_observe_returns_none_trace(self) -> None:
        rubric = _minimal_json_rubric()
        client = MockClient(json.dumps({"score": 50}))
        result = rubric.evaluate("x", client=client, model="m")
        assert result.trace is None

    def test_trace_parser_is_xml_for_compliance_schema(self) -> None:
        rubric = rubrify.Rubric(name="C", mission="compliance")
        rubric.output_schema = OutputSchema(constraints={"must_use_xml_tags": True})
        client = MockClient("<Judgement>Yes</Judgement>")
        result = rubric.evaluate("x", client=client, model="m", observe=True)
        assert result.trace is not None
        assert result.trace.parser == "xml"

    def test_trace_parser_is_raw_when_no_schema(self) -> None:
        rubric = rubrify.Rubric(name="N", mission="none")
        client = MockClient("anything")
        result = rubric.evaluate("x", client=client, model="m", observe=True)
        assert result.trace is not None
        assert result.trace.parser == "raw"


class TestRubricApplyTrace:
    def test_apply_with_observe_returns_tuple(self) -> None:
        cr = rubrify.Rubric(name="G", instructions="Generate.", output_format="<foo>")
        client = MockClient("raw output")
        result = cr.apply("input text", client=client, model="test-model", observe=True)
        assert isinstance(result, tuple)
        output, trace = result
        assert output == "raw output"
        assert isinstance(trace, EvaluationTrace)
        assert trace.model == "test-model"
        assert trace.user_message == "input text"
        assert trace.parser == "raw"

    def test_apply_default_returns_output_only(self) -> None:
        cr = rubrify.Rubric(name="G", instructions="Generate.", output_format="<foo>")
        client = MockClient("raw output")
        result: Any = cr.apply("input text", client=client, model="m")
        assert result == "raw output"
        assert not isinstance(result, tuple)

    def test_apply_with_observe_and_json_parser(self) -> None:
        cr = rubrify.Rubric(name="G", instructions="Generate.", output_format="json")
        client = MockClient('{"key": "value"}')
        result = cr.apply(
            "do the thing",
            client=client,
            model="m",
            parse_as="json",
            observe=True,
        )
        assert isinstance(result, tuple)
        output, trace = result
        assert output == {"key": "value"}
        assert trace.parser == "json"
