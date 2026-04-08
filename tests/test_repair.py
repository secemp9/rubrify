"""Tests for the Phase 1 local-only repair layer."""

from __future__ import annotations

import json

from rubrify._types import OutputSchema
from rubrify.repair import (
    RepairResult,
    attempt_schema_repair,
    extract_json_candidate,
    extract_xml_candidate,
)


class TestExtractJsonCandidate:
    def test_clean_json_parses_directly(self) -> None:
        raw = json.dumps({"score": 82, "class": "Good"})
        result = extract_json_candidate(raw)
        assert isinstance(result, RepairResult)
        assert result.repaired is False
        assert result.notes == ()
        assert result.text == raw

    def test_extracts_from_json_code_fence(self) -> None:
        raw = (
            "Here is my evaluation:\n"
            "```json\n"
            '{"score": 70, "class": "Usable"}\n'
            "```\n"
            "Done."
        )
        result = extract_json_candidate(raw)
        assert result.repaired is True
        assert "json" in result.notes[0]
        data = json.loads(result.text)
        assert data["score"] == 70

    def test_extracts_from_unlabeled_code_fence(self) -> None:
        raw = "```\n" '{"score": 55}\n' "```"
        result = extract_json_candidate(raw)
        assert result.repaired is True
        assert result.notes  # populated
        data = json.loads(result.text)
        assert data["score"] == 55

    def test_finds_json_via_brace_matching_in_prose(self) -> None:
        raw = (
            "Before the JSON there is chatter. "
            '{"score": 42, "rationale": "BECAUSE: ok"} '
            "And after the JSON there is more chatter."
        )
        result = extract_json_candidate(raw)
        assert result.repaired is True
        assert "brace" in result.notes[0]
        data = json.loads(result.text)
        assert data["score"] == 42

    def test_brace_matching_handles_nested_structures(self) -> None:
        raw = 'prose {"outer": {"inner": [1, 2, 3]}, "score": 10} more'
        result = extract_json_candidate(raw)
        assert result.repaired is True
        data = json.loads(result.text)
        assert data["outer"]["inner"] == [1, 2, 3]
        assert data["score"] == 10

    def test_brace_matching_ignores_braces_in_strings(self) -> None:
        raw = 'prose {"message": "a { pretend { brace", "score": 1} end'
        result = extract_json_candidate(raw)
        assert result.repaired is True
        data = json.loads(result.text)
        assert data["score"] == 1

    def test_malformed_input_returns_unrepaired_with_notes(self) -> None:
        result = extract_json_candidate("no json here, just words and {{{ garbage")
        assert result.repaired is False
        assert result.notes == ("no valid JSON object found",)
        assert result.text == "no json here, just words and {{{ garbage"

    def test_non_object_json_is_not_accepted(self) -> None:
        # Arrays and scalars are valid JSON but we only want objects.
        result = extract_json_candidate("[1, 2, 3]")
        assert result.repaired is False
        assert result.notes == ("no valid JSON object found",)


class TestExtractXmlCandidate:
    def test_clean_xml_fragment_parses_directly(self) -> None:
        raw = "<Rationale>ok</Rationale><Judgement>Yes</Judgement>"
        result = extract_xml_candidate(raw)
        assert result.repaired is False
        assert result.notes == ()
        assert result.text == raw

    def test_extracts_from_xml_code_fence(self) -> None:
        # Outer text contains a bare ampersand to force strategy 1 (direct
        # ElementTree parse) to fail so strategy 2 (code-fence extraction)
        # actually runs.
        raw = (
            "Output & notes:\n"
            "```xml\n"
            "<Rationale>Sufficient.</Rationale><Judgement>Yes</Judgement>\n"
            "```"
        )
        result = extract_xml_candidate(raw)
        assert result.repaired is True
        assert "xml" in result.notes[0]
        assert "<Judgement>Yes</Judgement>" in result.text

    def test_extracts_required_tags_via_regex(self) -> None:
        # Bare ampersand poisons strategy 1 (direct ET parse) and strategy 2
        # (no code fences), forcing the required-tag regex scrape path.
        raw = (
            "Analysis with & a raw ampersand. "
            "<Judgement>Yes</Judgement> and for the reason: "
            "<Rationale>Because it is.</Rationale>"
        )
        result = extract_xml_candidate(raw, required_tags=("Rationale", "Judgement"))
        assert result.repaired is True
        assert "regex-scraped" in result.notes[0]
        assert "<Rationale>Because it is.</Rationale>" in result.text
        assert "<Judgement>Yes</Judgement>" in result.text

    def test_unrecoverable_input_returns_unrepaired(self) -> None:
        result = extract_xml_candidate("no tags and < bad syntax", required_tags=())
        assert result.repaired is False
        assert result.notes == ("no valid XML structure found",)

    def test_unrecoverable_without_required_tags(self) -> None:
        result = extract_xml_candidate("totally unbalanced <foo>")
        assert result.repaired is False
        assert result.notes == ("no valid XML structure found",)


class TestAttemptSchemaRepair:
    def test_none_schema_returns_unrepaired_with_notes(self) -> None:
        result = attempt_schema_repair("anything", None)
        assert result.repaired is False
        assert result.notes == ("no schema provided",)

    def test_dispatches_to_json_for_must_be_json(self) -> None:
        schema = OutputSchema(constraints={"must_be_json": True})
        raw = 'some prose {"score": 99} trailing'
        result = attempt_schema_repair(raw, schema)
        assert result.repaired is True
        assert json.loads(result.text)["score"] == 99

    def test_dispatches_to_xml_for_must_use_xml_tags(self) -> None:
        schema = OutputSchema(constraints={"must_use_xml_tags": True})
        # Bare ampersand poisons strategy 1 (direct ET parse) and forces the
        # required-tag regex scrape path to fire.
        raw = "Chat chat & chat. <Judgement>No</Judgement> " "more chat <Rationale>why</Rationale>."
        result = attempt_schema_repair(raw, schema)
        assert result.repaired is True
        assert "<Judgement>No</Judgement>" in result.text
        assert "<Rationale>why</Rationale>" in result.text

    def test_unrecognized_schema_returns_unrepaired(self) -> None:
        schema = OutputSchema(constraints={})
        result = attempt_schema_repair("raw data", schema)
        assert result.repaired is False
        assert result.notes == ("schema has no recognized format",)

    def test_clean_json_with_schema_returns_unrepaired(self) -> None:
        schema = OutputSchema(constraints={"must_be_json": True})
        raw = json.dumps({"score": 10})
        result = attempt_schema_repair(raw, schema)
        assert result.repaired is False
        assert result.notes == ()
