"""Tests for property predicates and validation."""

from pathlib import Path

import rubrify
from rubrify._properties import (
    p_advice,
    p_aligned,
    p_anchored,
    p_criteria,
    p_decision,
    p_dq,
    p_economy,
    p_examples,
    p_inverted,
    p_mechanical,
    p_mission,
    p_patterns,
    p_schema,
    p_steering,
    p_validation,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestPredicatesOnV3:
    """Test predicates against the well-formed ZinsserJudge v3 fixture."""

    def setup_method(self) -> None:
        self.r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))

    def test_p_mission(self) -> None:
        assert p_mission(self.r) is True

    def test_p_criteria(self) -> None:
        assert p_criteria(self.r) == 25  # all criteria have anchors

    def test_p_anchored(self) -> None:
        assert p_anchored(self.r) == 1.0  # all have >= 2 anchors

    def test_p_mechanical(self) -> None:
        assert p_mechanical(self.r) > 0

    def test_p_dq(self) -> None:
        assert p_dq(self.r) == 5

    def test_p_schema(self) -> None:
        assert p_schema(self.r) is True

    def test_p_aligned(self) -> None:
        # v3 JSON template uses "subscores" not individual criterion IDs, so alignment is 0
        assert p_aligned(self.r) == 0.0

    def test_p_steering(self) -> None:
        assert p_steering(self.r) >= 1

    def test_p_patterns(self) -> None:
        assert p_patterns(self.r) == 11

    def test_p_examples(self) -> None:
        assert p_examples(self.r) == 0  # v3 has no mapping examples

    def test_p_economy(self) -> None:
        # 25 criteria is outside [3,7] range
        assert p_economy(self.r) is False

    def test_p_inverted(self) -> None:
        assert p_inverted(self.r) is False

    def test_p_decision(self) -> None:
        assert p_decision(self.r) is False

    def test_p_advice(self) -> None:
        assert p_advice(self.r) == 0

    def test_p_validation(self) -> None:
        assert p_validation(self.r) == 0


class TestPredicatesOnAntiSlop:
    def setup_method(self) -> None:
        self.r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))

    def test_inverted(self) -> None:
        assert p_inverted(self.r) is True

    def test_patterns(self) -> None:
        assert p_patterns(self.r) == 27

    def test_advice(self) -> None:
        assert p_advice(self.r) == 8

    def test_economy(self) -> None:
        assert p_economy(self.r) is True  # 5 criteria


class TestPredicatesOnCompliance:
    def setup_method(self) -> None:
        self.r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))

    def test_decision(self) -> None:
        assert p_decision(self.r) is True

    def test_examples(self) -> None:
        assert p_examples(self.r) == 5

    def test_economy(self) -> None:
        assert p_economy(self.r) is True  # 3 criteria


class TestValidation:
    def test_v3_is_valid(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        result = validate(r)
        assert result.is_valid is True

    def test_anti_slop_is_valid(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        result = validate(r)
        assert result.is_valid is True

    def test_compliance_is_valid(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        result = validate(r)
        assert result.is_valid is True

    def test_empty_rubric_fails_n1_n2_n3(self) -> None:
        r = rubrify.Rubric()
        result = validate(r)
        assert result.is_valid is False
        assert len(result.errors) == 3
        error_names = {e.name for e in result.errors}
        assert "N1_mission" in error_names
        assert "N2_structure" in error_names
        assert "N3_output" in error_names

    def test_n1_failure_empty_mission(self) -> None:
        r = rubrify.Rubric(mission="")
        r.criteria["C1"] = rubrify.Criterion(id="C1", name="Test", anchors={0: "a", 1: "b"})
        r.output_schema = rubrify.OutputSchema(template="{}")
        result = validate(r)
        assert result.is_valid is False
        error_names = {e.name for e in result.errors}
        assert "N1_mission" in error_names

    def test_n2_failure_no_structure(self) -> None:
        r = rubrify.Rubric(mission="Test mission")
        r.output_schema = rubrify.OutputSchema(template="{}")
        result = validate(r)
        assert result.is_valid is False
        error_names = {e.name for e in result.errors}
        assert "N2_structure" in error_names

    def test_n3_failure_no_output(self) -> None:
        r = rubrify.Rubric(mission="Test mission")
        r.criteria["C1"] = rubrify.Criterion(id="C1", name="Test")
        result = validate(r)
        assert result.is_valid is False
        error_names = {e.name for e in result.errors}
        assert "N3_output" in error_names

    def test_s1_warning_single_anchor(self) -> None:
        r = rubrify.Rubric(mission="Test mission")
        r.criteria["C1"] = rubrify.Criterion(id="C1", name="Test", anchors={0: "only one"})
        r.output_schema = rubrify.OutputSchema(template="{}")
        result = validate(r)
        assert result.is_valid is True
        warning_names = {w.name for w in result.warnings}
        assert "S1_anchored" in warning_names

    def test_well_formed_vs_valid(self) -> None:
        """is_well_formed requires all S checks to pass too."""
        r = rubrify.Rubric(mission="Test mission")
        r.criteria["C1"] = rubrify.Criterion(id="C1", name="Test", anchors={0: "a", 1: "b"})
        r.output_schema = rubrify.OutputSchema(template="{}")
        result = validate(r)
        assert result.is_valid is True
        # Will fail some S checks (no DQs, no steering constraints, etc.)
        assert result.is_well_formed is False
