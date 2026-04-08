"""Tests for Phase 4 thresholded ``refine()`` loop."""

from __future__ import annotations

import dataclasses
import json
import sys

import pytest

import rubrify
import rubrify.generate  # noqa: F401  # ensure module is loaded into sys.modules
from rubrify._mutations import AddSteeringConstraint
from rubrify.provenance import RefinementReport

# rubrify/__init__.py does `from rubrify.generate import generate, refine`, so the
# attribute `rubrify.generate` resolves to the function. Grab the module via
# ``sys.modules`` so monkeypatch can reach ``_suggest_mutations`` on it.
generate_mod = sys.modules["rubrify.generate"]


def _meta_response(score: int, weak: bool = False) -> str:
    if weak:
        sub = {"C1": 2, "C2": 2, "C3": 2, "C4": 2, "C5": 2}
    else:
        sub = {"C1": 5, "C2": 4, "C3": 4, "C4": 4, "C5": 5}
    return json.dumps(
        {
            "score": score,
            "class": "x",
            "subscores": sub,
            "rationale": "BECAUSE: r.",
        }
    )


class MockClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.call_count += 1
        return self._response


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        if self.call_count >= len(self._responses):
            resp = self._responses[-1]
        else:
            resp = self._responses[self.call_count]
        self.call_count += 1
        return resp


def _build_rubric() -> rubrify.Rubric:
    r = rubrify.Rubric(name="Sample", mission="Sample rubric.")
    r.add_criterion(rubrify.Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
    r.output_schema = rubrify.OutputSchema(
        format="json",
        template='{"score":0}',
        constraints={"must_be_json": True},
    )
    r.scoring = rubrify.Scoring(formula="Sum C1.")
    return r


class TestRefinementReport:
    def test_refine_report_frozen(self) -> None:
        report = RefinementReport(
            iterations=0,
            start_score=50,
            end_score=50,
            stopped_reason="target_met",
            steps=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.iterations = 1  # type: ignore[misc]

    def test_improved_property_true_when_end_greater(self) -> None:
        r = RefinementReport(
            iterations=1,
            start_score=50,
            end_score=80,
            stopped_reason="target_met",
            steps=(),
        )
        assert r.improved is True

    def test_improved_property_false_when_equal(self) -> None:
        r = RefinementReport(
            iterations=0,
            start_score=80,
            end_score=80,
            stopped_reason="target_met",
            steps=(),
        )
        assert r.improved is False

    def test_improved_property_false_when_end_missing(self) -> None:
        r = RefinementReport(
            iterations=0,
            start_score=50,
            end_score=None,
            stopped_reason="max_iters",
            steps=(),
        )
        assert r.improved is False


class TestRefineStopping:
    def test_refine_target_met_on_first_attempt(self) -> None:
        rubric = _build_rubric()
        client = MockClient(_meta_response(95))
        result = rubrify.refine(
            rubric,
            client=client,
            model="m",
            target_score=80,
            max_iters=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        _, report = result
        assert report.stopped_reason == "target_met"
        assert report.iterations == 0
        assert report.start_score == 95
        assert report.end_score == 95
        # Only the initial meta-eval should have been called.
        assert client.call_count == 1

    def test_refine_no_mutations(self) -> None:
        rubric = _build_rubric()
        # High score with non-weak subscores; _suggest_mutations returns [].
        client = MockClient(_meta_response(90))
        result = rubrify.refine(
            rubric,
            client=client,
            model="m",
            max_iters=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        _, report = result
        assert report.stopped_reason == "no_mutations"
        assert report.iterations == 1

    def test_refine_reaches_max_iters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rubric = _build_rubric()
        # Always return the same (below-target) score so neither target_met
        # nor score_regressed ever fires.
        client = MockClient(_meta_response(40, weak=True))

        call_counter = {"n": 0}

        def always_mutate(
            rubric: rubrify.Rubric,
            weak_properties: list[str],
            meta_result: rubrify.EvaluationResult,
        ) -> list[object]:
            call_counter["n"] += 1
            return [
                AddSteeringConstraint(
                    key=f"iter_{call_counter['n']}",
                    value="(placeholder)",
                )
            ]

        monkeypatch.setattr(generate_mod, "_suggest_mutations", always_mutate)

        result = rubrify.refine(
            rubric,
            client=client,
            model="m",
            target_score=90,
            max_iters=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        _, report = result
        assert report.stopped_reason == "max_iters"
        assert report.iterations == 3

    def test_refine_return_report_tuple_shape(self) -> None:
        rubric = _build_rubric()
        client = MockClient(_meta_response(90))
        result = rubrify.refine(rubric, client=client, model="m", return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        out_rubric, report = result
        assert isinstance(out_rubric, rubrify.Rubric)
        assert isinstance(report, RefinementReport)

    def test_refine_return_rubric_by_default(self) -> None:
        rubric = _build_rubric()
        client = MockClient(_meta_response(90))
        result = rubrify.refine(rubric, client=client, model="m")
        assert isinstance(result, rubrify.Rubric)

    def test_refine_score_regressed_reverts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rubric = _build_rubric()
        # Initial meta-eval -> 70, iteration meta-eval -> 40 (regression).
        client = SequenceClient([_meta_response(70, weak=True), _meta_response(40, weak=True)])

        def always_mutate(
            rubric: rubrify.Rubric,
            weak_properties: list[str],
            meta_result: rubrify.EvaluationResult,
        ) -> list[object]:
            return [
                AddSteeringConstraint(
                    key="regression_test",
                    value="(placeholder)",
                )
            ]

        monkeypatch.setattr(generate_mod, "_suggest_mutations", always_mutate)

        original_version = rubric.version
        result = rubrify.refine(
            rubric,
            client=client,
            model="m",
            target_score=100,
            max_iters=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        returned, report = result
        assert report.stopped_reason == "score_regressed"
        assert report.start_score == 70
        assert report.end_score == 70  # reverted, not the regressed 40
        # The returned rubric must be the pre-mutation one.
        assert returned.version == original_version
