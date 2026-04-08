"""Tests for the calibration -> mutations bridge."""

from __future__ import annotations

import rubrify
from rubrify._mutations import (
    AddMappingExample,
    AddPattern,
    AddSteeringConstraint,
)
from rubrify.calibration import (
    CalibrationCase,
    CalibrationReport,
    CalibrationResult,
    calibration_to_mutations,
)
from rubrify.result import EvaluationResult


def _make_rubric() -> rubrify.Rubric:
    r = rubrify.Rubric(name="Sample", mission="Sample mission.")
    r.add_criterion(rubrify.Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
    return r


def _passing_result(case_id: str, case: CalibrationCase | None) -> CalibrationResult:
    return CalibrationResult(
        case_id=case_id,
        passed=True,
        actual=EvaluationResult(score=90),
        expected_summary="ok",
        actual_summary="ok",
        case=case,
    )


def _failing_result(
    case_id: str,
    case: CalibrationCase,
    actual: EvaluationResult,
) -> CalibrationResult:
    return CalibrationResult(
        case_id=case_id,
        passed=False,
        actual=actual,
        expected_summary="want",
        actual_summary="got",
        case=case,
    )


def _report(results: list[CalibrationResult], name: str = "t") -> CalibrationReport:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    return CalibrationReport(suite_name=name, passed=passed, failed=failed, results=tuple(results))


class TestCalibrationToMutations:
    def test_empty_report_returns_empty_list(self) -> None:
        rubric = _make_rubric()
        report = _report([])
        assert calibration_to_mutations(rubric, report) == []

    def test_all_passed_returns_empty_list(self) -> None:
        rubric = _make_rubric()
        case = CalibrationCase(id="c1", payload={"text": "x"}, expected_verdict="Yes")
        report = _report([_passing_result("c1", case)])
        assert calibration_to_mutations(rubric, report) == []

    def test_verdict_failure_suggests_mapping_example(self) -> None:
        rubric = _make_rubric()
        case = CalibrationCase(id="case1", payload={"text": "x"}, expected_verdict="Yes")
        failing = _failing_result("case1", case, EvaluationResult(verdict="No"))
        report = _report([failing])

        mutations = calibration_to_mutations(rubric, report)
        assert len(mutations) == 1
        m = mutations[0]
        assert isinstance(m, AddMappingExample)
        assert m.example.id == "E_case1"
        assert m.example.verdict == "Yes"
        assert "(fill in)" in str(m.example.assistant)
        assert "(placeholder)" in str(m.example.user)

    def test_band_failure_adds_pattern_when_library_missing(self) -> None:
        rubric = _make_rubric()
        assert rubric.pattern_library is None
        case = CalibrationCase(id="b1", payload={"text": "x"}, expected_band="Clean")
        failing = _failing_result("b1", case, EvaluationResult(band="Severe"))
        report = _report([failing])

        mutations = calibration_to_mutations(rubric, report)
        assert any(isinstance(m, AddPattern) for m in mutations)
        patterns = [m for m in mutations if isinstance(m, AddPattern)]
        assert patterns[0].pattern_id == "pattern_b1"
        assert "(pattern placeholder)" in patterns[0].pattern

    def test_band_failure_skipped_when_pattern_library_exists(self) -> None:
        rubric = _make_rubric()
        rubric.pattern_library = rubrify.PatternLibrary(entries={"p1": "existing"})
        case = CalibrationCase(id="b1", payload={"text": "x"}, expected_band="Clean")
        failing = _failing_result("b1", case, EvaluationResult(band="Severe"))
        report = _report([failing])

        mutations = calibration_to_mutations(rubric, report)
        assert not any(isinstance(m, AddPattern) for m in mutations)

    def test_schema_violation_suggests_schema_reminder(self) -> None:
        rubric = _make_rubric()
        case = CalibrationCase(id="s1", payload={"text": "x"}, expected_verdict="Yes")
        failing = _failing_result(
            "s1",
            case,
            EvaluationResult(verdict="No", repaired=True, repair_notes=("fixed",)),
        )
        report = _report([failing])

        mutations = calibration_to_mutations(rubric, report)
        schema_fixes = [
            m
            for m in mutations
            if isinstance(m, AddSteeringConstraint) and m.key == "schema_reminder"
        ]
        assert len(schema_fixes) == 1

    def test_score_range_failure_adds_scoring_guidance_reminder(self) -> None:
        rubric = _make_rubric()
        assert rubric.scoring_guidance == ""
        case = CalibrationCase(
            id="sr1",
            payload={"text": "x"},
            expected_score_min=80,
            expected_score_max=100,
        )
        failing = _failing_result("sr1", case, EvaluationResult(score=40))
        report = _report([failing])

        mutations = calibration_to_mutations(rubric, report)
        reminders = [
            m
            for m in mutations
            if isinstance(m, AddSteeringConstraint) and m.key == "scoring_guidance_reminder"
        ]
        assert len(reminders) == 1

    def test_no_duplicate_mutations_for_repeated_failure_type(self) -> None:
        rubric = _make_rubric()
        # Two failing cases both with schema violations → only one schema
        # reminder; each also has a distinct verdict scaffold.
        case_a = CalibrationCase(id="a", payload={"text": "x"}, expected_verdict="Yes")
        case_b = CalibrationCase(id="b", payload={"text": "y"}, expected_verdict="No")
        results = [
            _failing_result(
                "a",
                case_a,
                EvaluationResult(verdict="No", repaired=True),
            ),
            _failing_result(
                "b",
                case_b,
                EvaluationResult(verdict="Yes", repaired=True),
            ),
        ]
        report = _report(results)
        mutations = calibration_to_mutations(rubric, report)

        schema_fixes = [
            m
            for m in mutations
            if isinstance(m, AddSteeringConstraint) and m.key == "schema_reminder"
        ]
        assert len(schema_fixes) == 1  # deduped

        mapping_examples = [m for m in mutations if isinstance(m, AddMappingExample)]
        assert len(mapping_examples) == 2  # distinct ids → both kept
        ids = {m.example.id for m in mapping_examples}
        assert ids == {"E_a", "E_b"}

    def test_no_content_invention(self) -> None:
        """Every scaffold must carry placeholder markers, not invented content."""
        rubric = _make_rubric()
        case_verdict = CalibrationCase(id="v1", payload={"text": "x"}, expected_verdict="Yes")
        case_band = CalibrationCase(id="b1", payload={"text": "x"}, expected_band="Clean")
        case_score = CalibrationCase(
            id="s1",
            payload={"text": "x"},
            expected_score_min=80,
            expected_score_max=100,
        )
        results = [
            _failing_result("v1", case_verdict, EvaluationResult(verdict="No", repaired=True)),
            _failing_result("b1", case_band, EvaluationResult(band="Severe")),
            _failing_result("s1", case_score, EvaluationResult(score=10)),
        ]
        report = _report(results)
        mutations = calibration_to_mutations(rubric, report)
        assert mutations  # we should have at least one structural fix

        for m in mutations:
            if isinstance(m, AddMappingExample):
                assert "(fill in)" in str(m.example.assistant)
                assert "(placeholder)" in str(m.example.user)
            elif isinstance(m, AddPattern):
                assert "(pattern placeholder)" in m.pattern
            elif isinstance(m, AddSteeringConstraint):
                assert "(placeholder)" in m.value
            else:
                raise AssertionError(
                    f"Unexpected mutation type produced by calibration_to_mutations: "
                    f"{type(m).__name__}"
                )
