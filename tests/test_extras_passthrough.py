"""Tests for the Phase 1 extras passthrough on JSON and XML parsing."""

from __future__ import annotations

import json

from rubrify._types import OutputSchema
from rubrify.parse import _parse_json_response, _parse_xml_response


class TestJsonExtrasPassthrough:
    def test_unknown_top_level_key_lands_in_extras(self) -> None:
        data = {
            "score": 75,
            "rationale": "BECAUSE: good.",
            "meta": {"genre": "general", "word_count": 150},
        }
        result = _parse_json_response(json.dumps(data))
        assert result.score == 75
        assert "meta" in result.extras
        assert result.extras["meta"] == {"genre": "general", "word_count": 150}

    def test_nested_meta_preserved_with_nesting_intact(self) -> None:
        data = {
            "score": 80,
            "meta": {
                "genre": "science_tech",
                "detection": {"hedges": 3, "puffery": 0},
                "revisions": [{"line": 1, "fix": "tighten"}],
            },
        }
        result = _parse_json_response(json.dumps(data))
        assert result.extras["meta"]["detection"]["hedges"] == 3
        assert result.extras["meta"]["revisions"][0]["fix"] == "tighten"

    def test_multiple_unknown_keys_all_preserved(self) -> None:
        data = {
            "score": 60,
            "novel_field_a": "alpha",
            "novel_field_b": [1, 2, 3],
            "novel_field_c": {"nested": True},
        }
        result = _parse_json_response(json.dumps(data))
        assert result.extras["novel_field_a"] == "alpha"
        assert result.extras["novel_field_b"] == [1, 2, 3]
        assert result.extras["novel_field_c"] == {"nested": True}

    def test_known_keys_are_not_in_extras(self) -> None:
        data = {
            "score": 50,
            "class": "Usable",
            "subscores": {"C1": 3},
            "criterion_scores": {"C1": 3},
            "rationale": "r",
            "evidence": [],
            "actions": None,
            "diagnostics": None,
            "violations": [],
            "advice": None,
            "risk": 0,
            "band": "Clean",
            "verdict": "Yes",
        }
        result = _parse_json_response(json.dumps(data))
        assert result.extras == {}

    def test_extras_are_json_serializable(self) -> None:
        data = {
            "score": 10,
            "meta": {"nested": {"deeply": ["a", "b"]}},
            "other": 42,
        }
        result = _parse_json_response(json.dumps(data))
        # Must not raise.
        serialized = json.dumps(result.extras)
        roundtrip = json.loads(serialized)
        assert roundtrip["meta"]["nested"]["deeply"] == ["a", "b"]
        assert roundtrip["other"] == 42


class TestXmlExtrasPassthrough:
    def test_unknown_tag_appears_in_extras(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<Rationale>Good</Rationale><Judgement>Yes</Judgement><Meta>extra data</Meta>"
        result = _parse_xml_response(raw, schema)
        assert result.verdict == "Yes"
        assert result.rationale == "Good"
        assert result.extras["Meta"] == "extra data"

    def test_nested_unknown_tag_preserved_as_dict(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = (
            "<Rationale>ok</Rationale>"
            "<Judgement>Yes</Judgement>"
            "<Details>"
            "<Genre>science_tech</Genre>"
            "<Score>82</Score>"
            "</Details>"
        )
        result = _parse_xml_response(raw, schema)
        assert isinstance(result.extras["Details"], dict)
        assert result.extras["Details"]["Genre"] == "science_tech"
        assert result.extras["Details"]["Score"] == "82"

    def test_extras_json_serializable_for_xml(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<Judgement>Yes</Judgement><Flags><HasHype>true</HasHype></Flags>"
        result = _parse_xml_response(raw, schema)
        # Must not raise.
        json.dumps(result.extras)

    def test_known_tags_are_not_in_extras(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "<Rationale>r</Rationale><Judgement>Yes</Judgement>"
        result = _parse_xml_response(raw, schema)
        assert result.extras == {}

    def test_unparseable_xml_returns_empty_extras(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        raw = "totally not xml"
        result = _parse_xml_response(raw, schema)
        assert result.extras == {}
