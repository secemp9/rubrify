"""Tests for the applied-text feedback loop (``improve_text``)."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

import rubrify
from rubrify.improve import (
    ImproveReport,
    default_advice_extractor,
    improve_text,
)
from rubrify.result import EvaluationResult


def _make_rubric() -> rubrify.Rubric:
    r = rubrify.Rubric(name="Sample", mission="Sample mission.")
    r.add_criterion(rubrify.Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
    r.output_schema = rubrify.OutputSchema(
        format="json",
        template='{"score":0,"advice":[]}',
        constraints={"must_be_json": True},
    )
    r.scoring = rubrify.Scoring(formula="Sum C1.")
    return r


def _eval_response(
    score: int,
    advice: list[str] | None = None,
) -> str:
    payload: dict[str, object] = {
        "score": score,
        "class": "x",
        "subscores": {"C1": 3},
        "rationale": "BECAUSE: test.",
    }
    if advice is not None:
        payload["advice"] = advice
    return json.dumps(payload)


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


class TestImproveReport:
    def test_improve_report_frozen(self) -> None:
        r = ImproveReport(
            before_score=50,
            after_score=80,
            applied_advice=("fix1",),
            original_text="orig",
            improved_text="better",
            before_result=EvaluationResult(score=50),
            after_result=EvaluationResult(score=80),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.before_score = 10  # type: ignore[misc]

    def test_improved_property_true(self) -> None:
        r = ImproveReport(
            before_score=50,
            after_score=80,
            applied_advice=(),
            original_text="",
            improved_text="",
            before_result=EvaluationResult(score=50),
            after_result=EvaluationResult(score=80),
        )
        assert r.improved is True

    def test_improved_property_false_when_equal(self) -> None:
        r = ImproveReport(
            before_score=50,
            after_score=50,
            applied_advice=(),
            original_text="",
            improved_text="",
            before_result=EvaluationResult(score=50),
            after_result=EvaluationResult(score=50),
        )
        assert r.improved is False

    def test_improved_property_false_when_missing(self) -> None:
        r = ImproveReport(
            before_score=None,
            after_score=80,
            applied_advice=(),
            original_text="",
            improved_text="",
            before_result=EvaluationResult(),
            after_result=EvaluationResult(score=80),
        )
        assert r.improved is False


class TestDefaultAdviceExtractor:
    def test_extracts_from_advice_list(self) -> None:
        result = EvaluationResult(advice=["Cut hedges.", "Use verbs."])
        assert default_advice_extractor(result) == ["Cut hedges.", "Use verbs."]

    def test_extracts_from_actions_coaching(self) -> None:
        result = EvaluationResult(actions={"coaching": ["Tighten lead.", "Trim clauses."]})
        assert default_advice_extractor(result) == [
            "Tighten lead.",
            "Trim clauses.",
        ]

    def test_extracts_from_actions_edits_and_next_steps(self) -> None:
        result = EvaluationResult(
            actions={
                "edits": ["Delete paragraph 2."],
                "next_steps": "Run the rubric again.",
            }
        )
        advice = default_advice_extractor(result)
        assert "Delete paragraph 2." in advice
        assert "Run the rubric again." in advice

    def test_combined_sources(self) -> None:
        result = EvaluationResult(
            advice=["A"],
            actions={"coaching": ["B"], "edits": "C"},
        )
        advice = default_advice_extractor(result)
        assert advice == ["A", "B", "C"]

    def test_empty_returns_empty_list(self) -> None:
        assert default_advice_extractor(EvaluationResult()) == []


class TestImproveText:
    def test_no_advice_short_circuits(self) -> None:
        rubric = _make_rubric()
        # Single eval response, no advice → only one chat call.
        client = SequenceClient([_eval_response(50, advice=[])])
        report = improve_text(rubric, "original text", client=client, model="m")
        assert report.before_score == 50
        assert report.after_score == 50
        assert report.original_text == "original text"
        assert report.improved_text == "original text"
        assert report.applied_advice == ()
        assert client.call_count == 1

    def test_with_advice_triggers_three_chat_calls(self) -> None:
        rubric = _make_rubric()
        # 1st: evaluate (advice present), 2nd: improvement text, 3rd: re-evaluate.
        client = SequenceClient(
            [
                _eval_response(40, advice=["Cut the hedges.", "Use strong verbs."]),
                "Rewritten text with stronger verbs.",
                _eval_response(80, advice=[]),
            ]
        )
        report = improve_text(rubric, "weak original text", client=client, model="m")
        assert client.call_count == 3
        assert report.before_score == 40
        assert report.after_score == 80
        assert report.original_text == "weak original text"
        assert report.improved_text == "Rewritten text with stronger verbs."
        assert report.applied_advice == ("Cut the hedges.", "Use strong verbs.")
        assert report.improved is True

    def test_report_structure(self) -> None:
        rubric = _make_rubric()
        client = SequenceClient(
            [
                _eval_response(40, advice=["Fix this."]),
                "Better text.",
                _eval_response(70, advice=[]),
            ]
        )
        report = improve_text(rubric, "text", client=client, model="m")
        assert isinstance(report, ImproveReport)
        assert isinstance(report.before_result, EvaluationResult)
        assert isinstance(report.after_result, EvaluationResult)
        assert report.applied_advice == ("Fix this.",)
        assert report.original_text == "text"
        assert report.improved_text == "Better text."

    def test_custom_advice_extractor(self) -> None:
        rubric = _make_rubric()
        client = SequenceClient(
            [
                _eval_response(50, advice=["ignored"]),
                "improved",
                _eval_response(60, advice=[]),
            ]
        )

        def always_extract(result: EvaluationResult) -> list[str]:
            return ["custom advice"]

        report = improve_text(
            rubric,
            "t",
            client=client,
            model="m",
            advice_extractor=always_extract,
        )
        assert report.applied_advice == ("custom advice",)


@pytest.mark.integration
class TestImproveTextLive:
    def test_improve_text_live(self) -> None:
        """Live run against the Zinsser v3 rubric with a deliberately weak text."""
        base_url = os.environ.get("RUBRIFY_BASE_URL")
        api_key = os.environ.get("RUBRIFY_API_KEY")
        model = os.environ.get("RUBRIFY_MODEL")
        if not (base_url and api_key and model):
            pytest.skip("RUBRIFY_BASE_URL, RUBRIFY_API_KEY, RUBRIFY_MODEL required")

        fixture = os.path.dirname(__file__) + "/fixtures/on_writing_well_v3.xml"
        rubric = rubrify.load(fixture)
        client = rubrify.Client(base_url=base_url, api_key=api_key)

        weak_text = (
            "In the realm of software, one might perhaps arguably consider "
            "that leveraging synergistic solutions could potentially enable "
            "stakeholders to actualize transformative outcomes."
        )

        report = improve_text(rubric, weak_text, client=client, model=model)
        if report.before_score is None or report.after_score is None:
            pytest.skip("rubric did not produce integer scores")
        assert report.after_score > report.before_score
