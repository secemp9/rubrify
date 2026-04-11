"""Phase 3 tests: calibration as unit testing.

Covers the frozen dataclass contracts, the expectation checkers (verdict /
score-range / band / label / valid), the suite runner for both ``Rubric`` and
``ConstraintRubric``, ``assert_calibration`` / ``summarize_report`` behavior,
the structure of the reference calibration suites, and the opt-in live
integration runs for ``COMPLIANCE_JUDGE_SUITE``, ``ANTI_SLOP_DISCRIMINANT_SUITE``,
and META_EVALUATOR self-calibration.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest

import rubrify
from rubrify._calibration_suites import (
    ANTI_SLOP_DISCRIMINANT_SUITE,
    COMPLETENESS_FORCING_SUITE,
    COMPLIANCE_JUDGE_SUITE,
    QUERY,
)
from rubrify.calibration import (
    CalibrationCase,
    CalibrationReport,
    CalibrationResult,
    assert_calibration,
    run_calibration_suite,
    run_meta_evaluator_self_calibration,
    summarize_report,
)
from rubrify.result import ConstraintResult, EvaluationResult
from rubrify.rubric import Rubric

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fakes ────────────────────────────────────────────────────────────────


class _FakeRubric(Rubric):
    """A ``Rubric`` subclass whose ``evaluate`` returns a preset result.

    Used to exercise ``run_calibration_suite`` end-to-end without a live
    model or an HTTP client. The ``isinstance(rubric, Rubric)`` dispatch
    branch in the runner still fires correctly because this is a real
    subclass.
    """

    def __init__(self, canned: EvaluationResult) -> None:
        super().__init__(name="fake-rubric")
        self._canned = canned

    def evaluate(  # type: ignore[override]
        self,
        text: str,
        *,
        client: Any,
        model: str,
        repair: bool = False,
        observe: bool = False,
        **kwargs: Any,
    ) -> EvaluationResult:
        return self._canned


class _FakeConstraintRubric(Rubric):
    """A ``Rubric`` subclass with validators whose ``apply_and_validate`` is preset."""

    def __init__(self, canned: ConstraintResult) -> None:
        super().__init__(
            name="fake-constraint", instructions="test", validators=[lambda _: (True, None)]
        )
        self._canned = canned

    def apply_and_validate(  # type: ignore[override]
        self,
        text: str,
        *,
        client: Any,
        model: str,
        repair: bool = False,
        observe: bool = False,
        **kwargs: Any,
    ) -> ConstraintResult:
        return self._canned


def _mk_case(case_id: str = "c1", **expected: Any) -> CalibrationCase:
    return CalibrationCase(id=case_id, payload={"text": "sample"}, **expected)


# ── Dataclass frozenness ─────────────────────────────────────────────────


class TestFrozenness:
    def test_calibration_case_frozen(self) -> None:
        case = _mk_case(expected_verdict="Yes")
        with pytest.raises(dataclasses.FrozenInstanceError):
            case.id = "other"  # type: ignore[misc]

    def test_calibration_result_frozen(self) -> None:
        result = CalibrationResult(
            case_id="c1",
            passed=True,
            actual=None,
            expected_summary="verdict='Yes'",
            actual_summary="verdict='Yes'",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False  # type: ignore[misc]

    def test_calibration_report_frozen(self) -> None:
        report = CalibrationReport(suite_name="s", passed=0, failed=0, results=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.passed = 5  # type: ignore[misc]

    def test_calibration_report_totals(self) -> None:
        results = (
            CalibrationResult(
                case_id="a",
                passed=True,
                actual=None,
                expected_summary="",
                actual_summary="",
            ),
            CalibrationResult(
                case_id="b",
                passed=False,
                actual=None,
                expected_summary="",
                actual_summary="",
            ),
        )
        report = CalibrationReport(suite_name="s", passed=1, failed=1, results=results)
        assert report.total == 2
        assert report.all_passed is False

        clean = CalibrationReport(suite_name="s", passed=3, failed=0, results=())
        assert clean.total == 3
        assert clean.all_passed is True


class TestCaseValidation:
    def test_case_with_no_expectations_raises(self) -> None:
        with pytest.raises(ValueError, match="no expected fields"):
            CalibrationCase(id="empty", payload={"text": "x"})


# ── Expectation matching via the runner ─────────────────────────────────


class TestVerdictMatching:
    def test_verdict_matching_pass(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="Yes"))
        case = _mk_case(expected_verdict="Yes")
        report = run_calibration_suite(rubric, [case], client=None, model="m", suite_name="v")
        assert report.all_passed
        assert report.passed == 1
        assert report.failed == 0

    def test_verdict_matching_fail(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="No"))
        case = _mk_case(expected_verdict="Yes")
        report = run_calibration_suite(rubric, [case], client=None, model="m", suite_name="v")
        assert not report.all_passed
        assert report.failed == 1
        assert "verdict='Yes'" in report.results[0].expected_summary
        assert "verdict='No'" in report.results[0].actual_summary


class TestScoreRangeMatching:
    def test_score_range_in_range(self) -> None:
        rubric = _FakeRubric(EvaluationResult(score=10))
        case = _mk_case(expected_score_min=5, expected_score_max=15)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed

    def test_score_range_below(self) -> None:
        rubric = _FakeRubric(EvaluationResult(score=2))
        case = _mk_case(expected_score_min=5, expected_score_max=15)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed
        assert "score in [5, 15]" in report.results[0].expected_summary
        assert "score=2" in report.results[0].actual_summary

    def test_score_range_above(self) -> None:
        rubric = _FakeRubric(EvaluationResult(score=99))
        case = _mk_case(expected_score_min=5, expected_score_max=15)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed

    def test_score_range_with_only_min(self) -> None:
        rubric = _FakeRubric(EvaluationResult(score=100))
        case = _mk_case(expected_score_min=5)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed
        assert "-inf" not in report.results[0].expected_summary
        assert "+inf" in report.results[0].expected_summary

    def test_score_range_none_actual(self) -> None:
        rubric = _FakeRubric(EvaluationResult(score=None))
        case = _mk_case(expected_score_min=0, expected_score_max=100)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed


class TestBandMatching:
    def test_band_matches(self) -> None:
        rubric = _FakeRubric(EvaluationResult(band="Clean"))
        case = _mk_case(expected_band="Clean")
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed

    def test_band_mismatches(self) -> None:
        rubric = _FakeRubric(EvaluationResult(band="Severe"))
        case = _mk_case(expected_band="Clean")
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed


class TestLabelMatching:
    def test_label_matches(self) -> None:
        rubric = _FakeRubric(EvaluationResult(label="Publish-ready"))
        case = _mk_case(expected_label="Publish-ready")
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed

    def test_label_mismatches(self) -> None:
        rubric = _FakeRubric(EvaluationResult(label="Draft"))
        case = _mk_case(expected_label="Publish-ready")
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed


class TestValidMatchingConstraint:
    def test_valid_true(self) -> None:
        rubric = _FakeConstraintRubric(ConstraintResult(output="ok", valid=True))
        case = _mk_case(expected_valid=True)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed

    def test_valid_false(self) -> None:
        rubric = _FakeConstraintRubric(
            ConstraintResult(output="bad", valid=False, violations=("missing tag",))
        )
        case = _mk_case(expected_valid=True)
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed


class TestMultipleExpectations:
    def test_all_must_pass(self) -> None:
        rubric = _FakeRubric(
            EvaluationResult(verdict="Yes", score=50, band="Clean", label="Publish-ready")
        )
        case = _mk_case(
            expected_verdict="Yes",
            expected_score_min=40,
            expected_score_max=60,
            expected_band="Clean",
            expected_label="Publish-ready",
        )
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert report.all_passed

    def test_one_failure_fails_case(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="Yes", score=50, band="Severe"))
        case = _mk_case(
            expected_verdict="Yes",
            expected_score_min=40,
            expected_score_max=60,
            expected_band="Clean",
        )
        report = run_calibration_suite(rubric, [case], client=None, model="m")
        assert not report.all_passed
        # Should still record the expected/actual summary of every field
        assert "band='Clean'" in report.results[0].expected_summary
        assert "band='Severe'" in report.results[0].actual_summary


class TestAssertCalibration:
    def test_passes_silently(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="Yes"))
        report = run_calibration_suite(
            rubric, [_mk_case(expected_verdict="Yes")], client=None, model="m"
        )
        assert_calibration(report)

    def test_raises_on_failure_with_full_list(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="No"))
        cases = [
            _mk_case(case_id="c1", expected_verdict="Yes"),
            _mk_case(case_id="c2", expected_verdict="Yes"),
        ]
        report = run_calibration_suite(rubric, cases, client=None, model="m", suite_name="mysuite")
        with pytest.raises(AssertionError) as exc_info:
            assert_calibration(report)
        msg = str(exc_info.value)
        assert "mysuite" in msg
        assert "c1" in msg
        assert "c2" in msg
        assert "2/2" in msg


class TestSummarizeReport:
    def test_summary_is_multiline_and_nonempty(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="No"))
        report = run_calibration_suite(
            rubric,
            [_mk_case(case_id="c1", expected_verdict="Yes")],
            client=None,
            model="m",
            suite_name="summary-test",
        )
        summary = summarize_report(report)
        assert summary
        assert "\n" in summary
        assert "summary-test" in summary
        assert "c1" in summary
        assert "passed: 0" in summary
        assert "failed: 1" in summary

    def test_summary_on_passing_report(self) -> None:
        rubric = _FakeRubric(EvaluationResult(verdict="Yes"))
        report = run_calibration_suite(
            rubric,
            [_mk_case(expected_verdict="Yes")],
            client=None,
            model="m",
            suite_name="clean",
        )
        summary = summarize_report(report)
        assert "passed: 1" in summary
        assert "failed: 0" in summary
        assert "failing cases" not in summary


# ── Suite structure tests ───────────────────────────────────────────────


class TestComplianceSuiteStructure:
    def test_has_4_cases(self) -> None:
        assert len(COMPLIANCE_JUDGE_SUITE) == 4
        case_ids = {c.id for c in COMPLIANCE_JUDGE_SUITE}
        assert case_ids == {
            "case1_meta_yes",
            "case2_clean_yes",
            "case3_refuse_deflect_no",
            "case4_refuse_no",
        }

    def test_all_have_expected_verdict(self) -> None:
        for c in COMPLIANCE_JUDGE_SUITE:
            assert c.expected_verdict is not None
            assert c.expected_verdict in {"Yes", "Somewhat", "No"}

    def test_all_carry_conversation_payload(self) -> None:
        for c in COMPLIANCE_JUDGE_SUITE:
            assert "user_turn" in c.payload
            assert "model_response" in c.payload


class TestAntiSlopSuiteStructure:
    def test_discriminant_non_overlapping(self) -> None:
        assert len(ANTI_SLOP_DISCRIMINANT_SUITE) == 2
        by_id = {c.id: c for c in ANTI_SLOP_DISCRIMINANT_SUITE}
        clean = by_id["clean_text"]
        sloppy = by_id["sloppy_text"]
        # Non-overlapping score ranges prove the clean > sloppy invariant.
        assert clean.expected_score_min is not None
        assert sloppy.expected_score_max is not None
        assert clean.expected_score_min > sloppy.expected_score_max
        # Both must carry a band expectation.
        assert clean.expected_band == "Minimal"
        assert sloppy.expected_band == "Severe"


class TestCompletenessForcingSuiteStructure:
    def test_has_2_cases(self) -> None:
        assert len(COMPLETENESS_FORCING_SUITE) == 2
        case_ids = {c.id for c in COMPLETENESS_FORCING_SUITE}
        assert case_ids == {"fibonacci_forcing", "hello_world_forcing"}

    def test_all_have_expected_valid(self) -> None:
        for c in COMPLETENESS_FORCING_SUITE:
            assert c.expected_valid is True

    def test_all_carry_text_payload(self) -> None:
        for c in COMPLETENESS_FORCING_SUITE:
            assert "text" in c.payload
            assert isinstance(c.payload["text"], str)
            assert len(c.payload["text"]) > 0


class TestPublicExports:
    def test_phase3_symbols_exported(self) -> None:
        for name in (
            "CalibrationCase",
            "CalibrationResult",
            "CalibrationReport",
            "run_calibration_suite",
            "assert_calibration",
            "summarize_report",
            "run_meta_evaluator_self_calibration",
            "COMPLIANCE_JUDGE_SUITE",
            "ANTI_SLOP_DISCRIMINANT_SUITE",
            "COMPLETENESS_FORCING_SUITE",
        ):
            assert hasattr(rubrify, name), f"rubrify missing export: {name}"


# ── Live integration tests ──────────────────────────────────────────────


def _live_env() -> tuple[str, str, str] | None:
    base_url = os.environ.get("RUBRIFY_BASE_URL", "")
    api_key = os.environ.get("RUBRIFY_API_KEY", "")
    model = os.environ.get("RUBRIFY_MODEL", "")
    if not (base_url and api_key and model):
        return None
    return base_url, api_key, model


@pytest.mark.integration
def test_compliance_judge_live_suite() -> None:
    """Run the 4-case ComplianceJudge suite live and assert every verdict matches."""
    env = _live_env()
    if env is None:
        pytest.skip("live API env vars not set")
    base_url, api_key, model = env

    client = rubrify.Client(base_url=base_url, api_key=api_key)
    try:
        rubric = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        rubric.input_renderer = rubrify.ConversationJudgeRenderer(query_template=QUERY)
        report = run_calibration_suite(
            rubric,
            COMPLIANCE_JUDGE_SUITE,
            client=client,
            model=model,
            suite_name="ComplianceJudge",
        )
    finally:
        client.close()

    assert_calibration(report)


@pytest.mark.integration
def test_anti_slop_live_discriminant() -> None:
    """Run the anti-slop clean-vs-sloppy discriminant suite live."""
    env = _live_env()
    if env is None:
        pytest.skip("live API env vars not set")
    base_url, api_key, model = env

    client = rubrify.Client(base_url=base_url, api_key=api_key)
    try:
        rubric = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        report = run_calibration_suite(
            rubric,
            ANTI_SLOP_DISCRIMINANT_SUITE,
            client=client,
            model=model,
            suite_name="AntiSlopDiscriminant",
        )
    finally:
        client.close()

    assert_calibration(report)


@pytest.mark.integration
def test_meta_evaluator_self_calibration_live() -> None:
    """Run META_EVALUATOR against every reference rubric and assert ordering."""
    env = _live_env()
    if env is None:
        pytest.skip("live API env vars not set")
    base_url, api_key, model = env

    client = rubrify.Client(base_url=base_url, api_key=api_key)
    try:
        report = run_meta_evaluator_self_calibration(client=client, model=model)
    finally:
        client.close()

    assert_calibration(report)
