"""Tests for response parsing."""

import json

from rubrify._types import OutputSchema
from rubrify.parse import (
    _normalize_advice,
    _parse_json_response,
    _parse_xml_response,
    parse_response,
)


class TestParseResponseDispatch:
    def test_none_schema_returns_raw(self) -> None:
        result = parse_response("raw text", None)
        assert result.raw == "raw text"
        assert result.score is None

    def test_json_dispatch(self) -> None:
        schema = OutputSchema(constraints={"must_be_json": True})
        data = json.dumps({"score": 85, "class": "Strong"})
        result = parse_response(data, schema)
        assert result.score == 85

    def test_xml_dispatch(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<Rationale>Good work</Rationale><Judgement>Yes</Judgement>"
        result = parse_response(raw, schema)
        assert result.verdict == "Yes"

    def test_unknown_format_returns_raw(self) -> None:
        schema = OutputSchema()
        result = parse_response("some text", schema)
        assert result.raw == "some text"
        assert result.score is None


class TestParseJsonResponse:
    def test_full_zinsser_response(self) -> None:
        data = {
            "score": 82,
            "class": "Strong draft",
            "subscores": {"C1": 4, "C2": 3, "C3": 5},
            "rationale": "BECAUSE: the text is clear and direct.",
            "evidence": ["paragraph 1 uses active voice"],
            "actions": {"coaching": ["tighten paragraph 3"]},
            "diagnostics": {"hedges": 3, "puffery": 1},
            "violations": [],
        }
        result = _parse_json_response(json.dumps(data))
        assert result.score == 82
        assert result.label == "Strong draft"
        assert result.subscores == {"C1": 4, "C2": 3, "C3": 5}
        assert result.rationale == "BECAUSE: the text is clear and direct."
        assert result.evidence == ["paragraph 1 uses active voice"]
        assert result.actions == {"coaching": ["tighten paragraph 3"]}
        assert result.diagnostics == {"hedges": 3, "puffery": 1}
        assert result.violations == []

    def test_anti_slop_response(self) -> None:
        data = {
            "score": 12,
            "risk": 3,
            "band": "Clean",
            "criterion_scores": {"C1": 3, "C2": 3, "C3": 3},
            "rationale": "BECAUSE: minimal slop detected.",
            "advice": "FIX: remove 'very' on line 3; replace 'amazing' with specific detail",
        }
        result = _parse_json_response(json.dumps(data))
        assert result.score == 12
        assert result.risk == 3
        assert result.band == "Clean"
        assert result.label == "Clean"  # band used as fallback for label
        assert result.subscores == {"C1": 3, "C2": 3, "C3": 3}
        assert result.advice is not None
        assert len(result.advice) == 2

    def test_malformed_json_fallback(self) -> None:
        result = _parse_json_response("this is not json at all")
        assert result.raw == "this is not json at all"
        assert result.score is None

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here is my evaluation:\n{"score": 75, "class": "Usable"}\nDone.'
        result = _parse_json_response(raw)
        assert result.score == 75
        assert result.label == "Usable"

    def test_missing_fields_default(self) -> None:
        data = {"score": 50}
        result = _parse_json_response(json.dumps(data))
        assert result.score == 50
        assert result.label is None
        assert result.subscores == {}
        assert result.rationale == ""
        assert result.evidence == []
        assert result.violations == []
        assert result.advice is None

    def test_verdict_field(self) -> None:
        data = {"verdict": "Somewhat", "rationale": "Partial compliance."}
        result = _parse_json_response(json.dumps(data))
        assert result.verdict == "Somewhat"


class TestParseXmlResponse:
    def test_compliance_response(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = (
            "<Rationale>The assistant provided a helpful response "
            "without revealing system instructions.</Rationale>\n"
            "<Judgement>Yes</Judgement>"
        )
        result = _parse_xml_response(raw, schema)
        assert result.verdict == "Yes"
        assert "helpful response" in result.rationale

    def test_missing_tags(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "Just some plain text without XML tags"
        result = _parse_xml_response(raw, schema)
        assert result.verdict == ""
        assert result.rationale == ""
        assert result.raw == raw

    def test_verdict_tag_fallback(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<Rationale>Test</Rationale><Verdict>No</Verdict>"
        result = _parse_xml_response(raw, schema)
        assert result.verdict == "No"

    def test_case_insensitive(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<rationale>Test reason</rationale><judgement>Somewhat</judgement>"
        result = _parse_xml_response(raw, schema)
        assert result.verdict == "Somewhat"
        assert result.rationale == "Test reason"


class TestNormalizeAdvice:
    def test_none_returns_none(self) -> None:
        assert _normalize_advice(None) is None

    def test_string_with_fix_prefix(self) -> None:
        result = _normalize_advice("FIX: remove hedges; tighten prose; add specifics")
        assert result == ["remove hedges", "tighten prose", "add specifics"]

    def test_string_without_prefix(self) -> None:
        result = _normalize_advice("remove hedges; add specifics")
        assert result == ["remove hedges", "add specifics"]

    def test_list_passthrough(self) -> None:
        result = _normalize_advice(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_other_type_returns_none(self) -> None:
        assert _normalize_advice(42) is None

    def test_empty_string(self) -> None:
        result = _normalize_advice("")
        assert result == []

    def test_fix_prefix_single_item(self) -> None:
        result = _normalize_advice("FIX: just one thing")
        assert result == ["just one thing"]
