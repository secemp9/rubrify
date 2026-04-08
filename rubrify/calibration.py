"""Calibration as unit testing for behavioral claims.

Phase 3 deliverable. A rubric that makes a behavioral claim (a compliance
verdict, a discriminant band, a forced output shape) must ship calibration
cases that assert that claim against a live model. This module provides the
runner, the case / result / report types, and the META_EVALUATOR self-
calibration helper that dogfoods the meta-rubric against the reference set.

Per guard rail 6 of ``PHILOSOPHY.md`` there are no silent fallbacks: a
``CalibrationCase`` with no expected fields raises at construction, and
``assert_calibration`` surfaces every failing case with its expected vs
actual summary. Per guard rail 10, META_EVALUATOR is not exempt — its own
self-calibration runs through the same runner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rubrify.result import ConstraintResult, EvaluationResult
from rubrify.rubric import ConstraintRubric, Rubric

if TYPE_CHECKING:
    from rubrify._mutations import RubricMutation


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """A single calibration case declaring expected behavior for one input.

    ``payload`` is the renderer input. Rubrics that use a custom
    ``InputRenderer`` receive the whole payload as kwargs; legacy rubrics
    receive ``payload["text"]`` as the positional ``text`` argument to
    ``Rubric.evaluate``. At least one ``expected_*`` field must be set; a
    case with no expectations raises ``ValueError`` at construction.
    """

    id: str
    payload: dict[str, Any]
    expected_verdict: str | None = None
    expected_score_min: int | None = None
    expected_score_max: int | None = None
    expected_band: str | None = None
    expected_label: str | None = None
    expected_valid: bool | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self._has_any_expectation():
            raise ValueError(
                f"CalibrationCase {self.id!r} has no expected fields; at least one of "
                "expected_verdict, expected_score_min, expected_score_max, expected_band, "
                "expected_label, or expected_valid must be set."
            )

    def _has_any_expectation(self) -> bool:
        return (
            self.expected_verdict is not None
            or self.expected_score_min is not None
            or self.expected_score_max is not None
            or self.expected_band is not None
            or self.expected_label is not None
            or self.expected_valid is not None
        )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of a single ``CalibrationCase`` evaluation.

    ``actual`` carries the raw ``EvaluationResult`` / ``ConstraintResult`` the
    rubric produced, or ``None`` for synthesized assertions (e.g., the
    META_EVALUATOR ordering invariants which compare multiple runs).
    ``expected_summary`` and ``actual_summary`` are short human-readable
    strings used by ``assert_calibration`` and ``summarize_report``.
    ``case`` is the originating :class:`CalibrationCase` when the result came
    from the regular suite runner; it is ``None`` for synthesized assertions
    and lets downstream tools (e.g. ``calibration_to_mutations``) recover
    the expected fields without re-parsing ``expected_summary``.
    """

    case_id: str
    passed: bool
    actual: EvaluationResult | ConstraintResult | None
    expected_summary: str
    actual_summary: str
    notes: tuple[str, ...] = ()
    case: CalibrationCase | None = None


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Aggregate result of a calibration suite run.

    ``passed`` and ``failed`` are stored directly so the report is
    self-contained; ``results`` preserves per-case detail for diagnostics.
    """

    suite_name: str
    passed: int
    failed: int
    results: tuple[CalibrationResult, ...]

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# --- Case checking ----------------------------------------------------------


def _check_case(
    case: CalibrationCase, actual: EvaluationResult | ConstraintResult
) -> tuple[bool, str, str]:
    """Compare a case's expectations against an actual result.

    Returns ``(passed, expected_summary, actual_summary)``. All set
    ``expected_*`` fields must match for ``passed`` to be true. Expectations
    that do not apply to the actual result type (e.g. ``expected_verdict``
    against a :class:`ConstraintResult`) are treated as mismatches.
    """
    if isinstance(actual, EvaluationResult):
        actual_verdict: str | None = actual.verdict
        actual_score: int | None = actual.score
        actual_band: str | None = actual.band
        actual_label: str | None = actual.label
        actual_valid: bool | None = None
    else:
        actual_verdict = None
        actual_score = None
        actual_band = None
        actual_label = None
        actual_valid = actual.valid

    expected_parts: list[str] = []
    actual_parts: list[str] = []
    passed = True

    if case.expected_verdict is not None:
        expected_parts.append(f"verdict={case.expected_verdict!r}")
        actual_parts.append(f"verdict={actual_verdict!r}")
        if actual_verdict != case.expected_verdict:
            passed = False

    if case.expected_score_min is not None or case.expected_score_max is not None:
        lo_repr = str(case.expected_score_min) if case.expected_score_min is not None else "-inf"
        hi_repr = str(case.expected_score_max) if case.expected_score_max is not None else "+inf"
        expected_parts.append(f"score in [{lo_repr}, {hi_repr}]")
        actual_parts.append(f"score={actual_score!r}")
        if actual_score is None:
            passed = False
        else:
            if case.expected_score_min is not None and actual_score < case.expected_score_min:
                passed = False
            if case.expected_score_max is not None and actual_score > case.expected_score_max:
                passed = False

    if case.expected_band is not None:
        expected_parts.append(f"band={case.expected_band!r}")
        actual_parts.append(f"band={actual_band!r}")
        if actual_band != case.expected_band:
            passed = False

    if case.expected_label is not None:
        expected_parts.append(f"label={case.expected_label!r}")
        actual_parts.append(f"label={actual_label!r}")
        if actual_label != case.expected_label:
            passed = False

    if case.expected_valid is not None:
        expected_parts.append(f"valid={case.expected_valid}")
        actual_parts.append(f"valid={actual_valid}")
        if actual_valid != case.expected_valid:
            passed = False

    expected_summary = ", ".join(expected_parts) if expected_parts else "(no expectations)"
    actual_summary = ", ".join(actual_parts) if actual_parts else "(no matching fields)"
    return passed, expected_summary, actual_summary


# --- Suite runner -----------------------------------------------------------


def run_calibration_suite(
    rubric: Rubric | ConstraintRubric,
    cases: Iterable[CalibrationCase],
    *,
    client: Any,
    model: str,
    suite_name: str = "",
    **kwargs: Any,
) -> CalibrationReport:
    """Run a calibration suite against a live rubric.

    Dispatches on the concrete rubric type: a :class:`Rubric` is invoked via
    :meth:`Rubric.evaluate`, a :class:`ConstraintRubric` via
    :meth:`ConstraintRubric.apply_and_validate`. For each case the runner
    extracts ``text = payload.get("text", "")`` as the positional argument and
    forwards the remainder of the payload as keyword arguments so that
    renderer-based rubrics (e.g. ``ConversationJudgeRenderer``) receive the
    structured payload they expect. Additional ``**kwargs`` (e.g.
    ``temperature``) are forwarded to every call.
    """
    results: list[CalibrationResult] = []
    passed_count = 0
    failed_count = 0

    for case in cases:
        text = str(case.payload.get("text", ""))
        extra_payload = {k: v for k, v in case.payload.items() if k != "text"}
        merged_kwargs: dict[str, Any] = {**extra_payload, **kwargs}

        actual: EvaluationResult | ConstraintResult
        if isinstance(rubric, Rubric):
            actual = rubric.evaluate(text, client=client, model=model, **merged_kwargs)
        elif isinstance(rubric, ConstraintRubric):
            actual = rubric.apply_and_validate(text, client=client, model=model, **merged_kwargs)
        else:
            raise TypeError(
                f"run_calibration_suite expects a Rubric or ConstraintRubric, "
                f"got {type(rubric).__name__}"
            )

        passed, expected_summary, actual_summary = _check_case(case, actual)
        notes = (case.notes,) if case.notes else ()
        results.append(
            CalibrationResult(
                case_id=case.id,
                passed=passed,
                actual=actual,
                expected_summary=expected_summary,
                actual_summary=actual_summary,
                notes=notes,
                case=case,
            )
        )
        if passed:
            passed_count += 1
        else:
            failed_count += 1

    return CalibrationReport(
        suite_name=suite_name,
        passed=passed_count,
        failed=failed_count,
        results=tuple(results),
    )


def assert_calibration(report: CalibrationReport) -> None:
    """Raise ``AssertionError`` if any case in ``report`` failed.

    The error message lists every failing case with its expected and actual
    summaries so test output shows exactly which invariants the rubric
    violated. Succeeds silently if ``report.all_passed``.
    """
    if report.all_passed:
        return

    failing = [r for r in report.results if not r.passed]
    lines = [
        f"calibration suite {report.suite_name!r} failed: "
        f"{report.failed}/{report.total} cases failed",
    ]
    for r in failing:
        lines.append(f"  - {r.case_id}: expected {r.expected_summary}; actual {r.actual_summary}")
        for note in r.notes:
            if note:
                lines.append(f"      note: {note}")
    raise AssertionError("\n".join(lines))


def summarize_report(report: CalibrationReport) -> str:
    """Return a human-readable multi-line summary of ``report``."""
    lines = [
        f"Calibration suite: {report.suite_name or '<unnamed>'}",
        f"  total: {report.total}",
        f"  passed: {report.passed}",
        f"  failed: {report.failed}",
    ]
    failing = [r for r in report.results if not r.passed]
    if failing:
        lines.append("  failing cases:")
        for r in failing:
            lines.append(
                f"    - {r.case_id}: expected {r.expected_summary}; actual {r.actual_summary}"
            )
            for note in r.notes:
                if note:
                    lines.append(f"        note: {note}")
    return "\n".join(lines)


# --- META_EVALUATOR self-calibration ---------------------------------------


_META_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

_META_FIXTURE_FILES: dict[str, str] = {
    "v1": "on_writing_well_v1.xml",
    "v2": "on_writing_well_v2.xml",
    "v3": "on_writing_well_v3.xml",
    "anti_slop": "anti_slop_rubric.xml",
    "compliance_judge": "compliance_judge.xml",
    "slurs": "slurs.xml",
}

# Tolerance for v3>=v2 and v2>=v1 ordering invariants. Empirically, META_EVALUATOR
# has ~10 point run-to-run variance on the same rubric because the judge model
# noise compounds across 5 criteria. 15 absorbs that noise without letting a real
# quality regression slip through. See research/ for the full investigation.
_META_ORDERING_TOLERANCE = 15
_META_VALIDITY_THRESHOLD = 60


def run_meta_evaluator_self_calibration(client: Any, model: str) -> CalibrationReport:
    """Run META_EVALUATOR against every reference rubric and assert ordering.

    Loads each fixture XML, passes its ``to_xml()`` form through
    ``META_EVALUATOR.evaluate``, and then builds a tuple of synthesized
    :class:`CalibrationResult` entries encoding the ordering invariants from
    the Phase 3 plan:

    * ``v3 >= v2 - tolerance`` and ``v2 >= v1 - tolerance`` (Zinsser lineage).
    * ``anti_slop meta score >= 60`` and ``compliance_judge >= 60`` (valid).
    * ``slurs meta score < 60`` (expected failure per guard rail 9).

    The tolerance lets models return slightly lower scores for a later
    version without breaking the test; the absolute thresholds encode the
    validity / expected-failure claim.
    """
    from rubrify._meta_rubric import META_EVALUATOR
    from rubrify.xml_io import rubric_from_xml

    scores: dict[str, int] = {}
    raw_results: dict[str, EvaluationResult] = {}
    for key, filename in _META_FIXTURE_FILES.items():
        path = _META_FIXTURES_DIR / filename
        rubric = rubric_from_xml(path.read_text(encoding="utf-8"))
        meta_result = META_EVALUATOR.evaluate(rubric.to_xml(), client=client, model=model)
        scores[key] = int(meta_result.score) if meta_result.score is not None else 0
        raw_results[key] = meta_result

    results: list[CalibrationResult] = []

    # Ordering: v3 >= v2 - tolerance
    v3_ge_v2 = scores["v3"] >= scores["v2"] - _META_ORDERING_TOLERANCE
    results.append(
        CalibrationResult(
            case_id="v3_ge_v2",
            passed=v3_ge_v2,
            actual=raw_results["v3"],
            expected_summary=f"meta(v3) >= meta(v2) - {_META_ORDERING_TOLERANCE}",
            actual_summary=f"meta(v3)={scores['v3']}, meta(v2)={scores['v2']}",
            notes=("Zinsser lineage monotonicity",),
        )
    )

    # Ordering: v2 >= v1 - tolerance
    v2_ge_v1 = scores["v2"] >= scores["v1"] - _META_ORDERING_TOLERANCE
    results.append(
        CalibrationResult(
            case_id="v2_ge_v1",
            passed=v2_ge_v1,
            actual=raw_results["v2"],
            expected_summary=f"meta(v2) >= meta(v1) - {_META_ORDERING_TOLERANCE}",
            actual_summary=f"meta(v2)={scores['v2']}, meta(v1)={scores['v1']}",
            notes=("Zinsser lineage monotonicity",),
        )
    )

    # Validity: anti_slop >= threshold
    anti_slop_valid = scores["anti_slop"] >= _META_VALIDITY_THRESHOLD
    results.append(
        CalibrationResult(
            case_id="anti_slop_valid",
            passed=anti_slop_valid,
            actual=raw_results["anti_slop"],
            expected_summary=f"meta(anti_slop) >= {_META_VALIDITY_THRESHOLD}",
            actual_summary=f"meta(anti_slop)={scores['anti_slop']}",
            notes=("well-formed detection rubric",),
        )
    )

    # Validity: compliance_judge >= threshold
    compliance_valid = scores["compliance_judge"] >= _META_VALIDITY_THRESHOLD
    results.append(
        CalibrationResult(
            case_id="compliance_judge_valid",
            passed=compliance_valid,
            actual=raw_results["compliance_judge"],
            expected_summary=f"meta(compliance_judge) >= {_META_VALIDITY_THRESHOLD}",
            actual_summary=f"meta(compliance_judge)={scores['compliance_judge']}",
            notes=("well-formed compliance rubric",),
        )
    )

    # Expected failure: slurs < threshold
    slurs_low = scores["slurs"] < _META_VALIDITY_THRESHOLD
    results.append(
        CalibrationResult(
            case_id="slurs_low",
            passed=slurs_low,
            actual=raw_results["slurs"],
            expected_summary=(f"meta(slurs) < {_META_VALIDITY_THRESHOLD} (failure-mode artifact)"),
            actual_summary=f"meta(slurs)={scores['slurs']}",
            notes=("guard rail 9: failure is data, not bug",),
        )
    )

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    return CalibrationReport(
        suite_name="META_EVALUATOR self-calibration",
        passed=passed_count,
        failed=failed_count,
        results=tuple(results),
    )


# --- Calibration → mutations bridge ----------------------------------------


def calibration_to_mutations(
    rubric: Rubric,
    report: CalibrationReport,
) -> list[RubricMutation]:
    """Map calibration failures to conservative structural mutations.

    Phase 4 deliverable. Deterministic, LLM-free. Given a rubric and a
    calibration report, propose a small set of *scaffold* mutations that a
    human or model can fill with content later. No content is invented and
    no calls are made to any model.

    Rules (all conservative, all deduplicated):

    * **Failing ``expected_verdict``** → suggest an
      :class:`rubrify._mutations.AddMappingExample` whose scaffold uses a
      placeholder ``user`` summary, an ``"(fill in)"`` assistant, and the
      expected verdict verbatim.
    * **Failing ``expected_band``** (detection rubric) with
      ``rubric.pattern_library is None`` → suggest an
      :class:`rubrify._mutations.AddPattern` skeleton named
      ``pattern_<case.id>`` with a placeholder regex.
    * **Failing ``expected_score_min/max``** with empty
      ``rubric.scoring_guidance`` → suggest an
      :class:`rubrify._mutations.AddSteeringConstraint` on a
      ``scoring_guidance_reminder`` key.
    * **Schema violation** (actual result ``repaired=True``) → suggest an
      :class:`rubrify._mutations.AddSteeringConstraint` on a
      ``schema_reminder`` key.

    Returns an empty list when no structural fixes apply. Guard rail 6: the
    caller can still inspect the report to see *which* cases failed. Guard
    rail 8: no hypothetical future rules.
    """
    from rubrify._mutations import (
        AddMappingExample,
        AddPattern,
        AddSteeringConstraint,
    )
    from rubrify._types import MappingExample

    mutations: list[RubricMutation] = []
    seen_mapping_ids: set[str] = set()
    seen_pattern_ids: set[str] = set()
    seen_steering_keys: set[str] = set()

    for result in report.results:
        if result.passed or result.case is None:
            continue
        case = result.case

        schema_repaired = False
        if isinstance(result.actual, EvaluationResult | ConstraintResult):
            schema_repaired = result.actual.repaired

        if schema_repaired and "schema_reminder" not in seen_steering_keys:
            mutations.append(
                AddSteeringConstraint(
                    key="schema_reminder",
                    value=(
                        "(placeholder) remind the model to follow the declared "
                        "output schema exactly; human fills in specifics."
                    ),
                )
            )
            seen_steering_keys.add("schema_reminder")

        if case.expected_verdict is not None:
            mapping_id = f"E_{case.id}"
            if mapping_id not in seen_mapping_ids:
                mutations.append(
                    AddMappingExample(
                        example=MappingExample(
                            id=mapping_id,
                            user=f"(placeholder) payload summary for case {case.id}",
                            assistant="(fill in)",
                            verdict=case.expected_verdict,
                        )
                    )
                )
                seen_mapping_ids.add(mapping_id)

        if case.expected_band is not None and rubric.pattern_library is None:
            pattern_id = f"pattern_{case.id}"
            if pattern_id not in seen_pattern_ids:
                mutations.append(
                    AddPattern(
                        pattern_id=pattern_id,
                        pattern=r"(pattern placeholder)",
                    )
                )
                seen_pattern_ids.add(pattern_id)

        has_score_expectation = (
            case.expected_score_min is not None or case.expected_score_max is not None
        )
        if has_score_expectation and not rubric.scoring_guidance:
            key = "scoring_guidance_reminder"
            if key not in seen_steering_keys:
                mutations.append(
                    AddSteeringConstraint(
                        key=key,
                        value=(
                            "(placeholder) remind the model of the scoring "
                            "guidance band boundaries; human fills in specifics."
                        ),
                    )
                )
                seen_steering_keys.add(key)

    return mutations
