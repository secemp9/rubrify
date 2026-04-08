"""Tests for XML serialization/deserialization round-trips."""

from pathlib import Path

import pytest

import rubrify

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadV3:
    def test_load_basic_properties(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert r.name == "ZinsserJudge-XXL"
        assert r.version == "3.0"
        assert "nonfiction" in r.mission.lower()

    def test_criteria_count(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        # 12 core C + 10 genre G + 3 attitude A = 25 total
        assert len(r.criteria) == 25

    def test_disqualifiers(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert len(r.disqualifiers) == 5

    def test_pattern_library(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert r.pattern_library is not None
        assert len(r.pattern_library.entries) == 11

    def test_scoring_labels(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert r.scoring is not None
        assert len(r.scoring.labels) == 6

    def test_output_format_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert r.output_format == "json"

    def test_inputs(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert len(r.inputs) == 5
        assert r.inputs[0].name == "candidate_text"
        assert r.inputs[0].required is True

    def test_genre_criterion(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert "G_BUS" in r.criteria
        assert r.criteria["G_BUS"].genre == "business,email"

    def test_attitude_criterion(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        assert "A_VOX" in r.criteria
        assert r.criteria["A_VOX"].weight == 2


class TestLoadAntiSlop:
    def test_load_basic(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert r.name == "AntiLLMY"
        assert r.version == "1.0"

    def test_criteria(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert len(r.criteria) == 5
        for cid in ["C1", "C2", "C3", "C4", "C5"]:
            assert cid in r.criteria

    def test_uses_patterns(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        c1 = r.criteria["C1"]
        assert c1.uses_patterns is not None
        assert "puffery_words" in c1.uses_patterns

    def test_regex_library(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert r.pattern_library is not None
        assert r.pattern_library.flags == "i"
        assert len(r.pattern_library.entries) == 27

    def test_disqualifiers(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert len(r.disqualifiers) == 3

    def test_advice_rules(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert len(r.advice_rules) == 8

    def test_validation_musts(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert len(r.validation_musts) == 3

    def test_inverted_scoring(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        assert r.scoring is not None
        assert r.scoring.inverted is True


class TestLoadComplianceJudge:
    def test_load_basic(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert r.name == "ComplianceJudge"
        assert r.version == "2.0"

    def test_criteria(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert len(r.criteria) == 3

    def test_disqualifiers(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert len(r.disqualifiers) == 2

    def test_decision_logic(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert len(r.decision_logic) == 4

    def test_mapping_examples(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert len(r.mapping_examples) == 5

    def test_xml_output_format(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert r.output_format == "xml"

    def test_definitions(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert len(r.definitions) >= 5
        assert "COMPLY" in r.definitions

    def test_what_to_judge(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert r.what_to_judge != ""

    def test_pattern_library_groups(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert r.pattern_library is not None
        assert len(r.pattern_library.entries) > 0

    def test_scoring_guidance(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        assert r.scoring_guidance != ""


class TestLoadV1:
    def test_load_basic(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v1.xml"))
        assert r.name == "ZinsserJudge"
        assert r.version == "1.0"

    def test_criteria_count(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v1.xml"))
        # 12 core C + 7 genre G = 19
        assert len(r.criteria) == 19

    def test_disqualifiers(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v1.xml"))
        assert len(r.disqualifiers) == 4

    def test_no_pattern_library(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v1.xml"))
        assert r.pattern_library is None


class TestRoundTrip:
    """Test that load -> to_xml -> loads produces structurally equivalent rubrics."""

    @pytest.mark.parametrize(
        "filename",
        [
            "on_writing_well_v1.xml",
            "on_writing_well_v3.xml",
            "anti_slop_rubric.xml",
            "compliance_judge.xml",
        ],
    )
    def test_round_trip_structure(self, filename: str) -> None:
        path = str(FIXTURES / filename)
        original = rubrify.load(path)
        xml_out = original.to_xml()

        # Verify the XML is parseable
        reloaded = rubrify.loads(xml_out)

        # Verify key structural properties
        assert reloaded.name == original.name
        assert reloaded.version == original.version
        assert len(reloaded.criteria) == len(original.criteria)
        assert len(reloaded.disqualifiers) == len(original.disqualifiers)

        # Verify criteria IDs match
        assert set(reloaded.criteria.keys()) == set(original.criteria.keys())

        # Verify criterion anchors preserved
        for cid in original.criteria:
            orig_c = original.criteria[cid]
            reload_c = reloaded.criteria[cid]
            assert set(orig_c.anchors.keys()) == set(reload_c.anchors.keys())

    @pytest.mark.parametrize(
        "filename",
        [
            "on_writing_well_v3.xml",
            "anti_slop_rubric.xml",
        ],
    )
    def test_round_trip_pattern_library(self, filename: str) -> None:
        path = str(FIXTURES / filename)
        original = rubrify.load(path)
        reloaded = rubrify.loads(original.to_xml())

        assert original.pattern_library is not None
        assert reloaded.pattern_library is not None
        assert set(reloaded.pattern_library.entries.keys()) == set(
            original.pattern_library.entries.keys()
        )

    def test_round_trip_scoring(self) -> None:
        original = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        reloaded = rubrify.loads(original.to_xml())

        assert original.scoring is not None
        assert reloaded.scoring is not None
        assert len(reloaded.scoring.labels) == len(original.scoring.labels)

    def test_round_trip_decision_logic(self) -> None:
        original = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        reloaded = rubrify.loads(original.to_xml())

        assert len(reloaded.decision_logic) == len(original.decision_logic)
        assert len(reloaded.mapping_examples) == len(original.mapping_examples)

    def test_round_trip_advice_rules(self) -> None:
        original = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        reloaded = rubrify.loads(original.to_xml())

        assert len(reloaded.advice_rules) == len(original.advice_rules)
        assert len(reloaded.validation_musts) == len(original.validation_musts)


class TestSpecialCharacters:
    def test_regex_patterns_survive_round_trip(self) -> None:
        """Regex patterns with special chars should survive round-trip."""
        original = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        reloaded = rubrify.loads(original.to_xml())

        assert reloaded.pattern_library is not None
        assert original.pattern_library is not None

        # Check that pattern content survived
        for pat_id in original.pattern_library.entries:
            assert pat_id in reloaded.pattern_library.entries


class TestEmptyRubric:
    def test_empty_rubric_serialization(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test mission.")
        xml = r.to_xml()
        assert "<LLM_JUDGE_SPEC" in xml
        assert 'name="Test"' in xml

        reloaded = rubrify.loads(xml)
        assert reloaded.name == "Test"
        assert reloaded.mission == "Test mission."
