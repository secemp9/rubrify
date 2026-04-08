"""Tests for Phase 4 thresholded ``generate()`` loop."""

from __future__ import annotations

import json

import rubrify
from rubrify.provenance import RefinementReport

VALID_RUBRIC_XML = """\
<LLM_JUDGE_SPEC version="1.0" name="TestRubric">
  <mission>Evaluate test quality.</mission>
  <rubric>
    <criterion id="C1" name="Clarity" weight="40">
      <anchor_0>Unclear.</anchor_0>
      <anchor_5>Crystal clear.</anchor_5>
    </criterion>
    <criterion id="C2" name="Completeness" weight="30">
      <anchor_0>Missing everything.</anchor_0>
      <anchor_5>Fully complete.</anchor_5>
    </criterion>
    <criterion id="C3" name="Accuracy" weight="30">
      <anchor_0>All wrong.</anchor_0>
      <anchor_5>Perfectly accurate.</anchor_5>
    </criterion>
    <disqualifiers>
      <dq id="DQ1">Empty submission.</dq>
    </disqualifiers>
  </rubric>
  <output_schema>
    <json_template>{"score":0,"class":"","subscores":{"C1":0,"C2":0,"C3":0},"rationale":""}</json_template>
    <constraints>
      <must_be_json>true</must_be_json>
      <no_prose_outside_json>true</no_prose_outside_json>
    </constraints>
  </output_schema>
  <scoring>
    <formula>Sum weighted C1-C3. Normalize to 100.</formula>
    <labels>
      <label min="80" max="100">Excellent</label>
      <label min="50" max="79">Good</label>
      <label min="0" max="49">Needs Work</label>
    </labels>
  </scoring>
</LLM_JUDGE_SPEC>"""


def _meta_response(score: int) -> str:
    return json.dumps(
        {
            "score": score,
            "class": "x",
            "subscores": {"C1": 4, "C2": 4, "C3": 4, "C4": 4, "C5": 4},
            "rationale": "BECAUSE: test rationale.",
        }
    )


class SequenceClient:
    """Client that returns a sequence of responses, one per ``chat`` call."""

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


class SingleResponseClient:
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


class TestGenerateLoop:
    def test_min_meta_score_met_first_attempt(self) -> None:
        client = SequenceClient([VALID_RUBRIC_XML, _meta_response(90)])
        result = rubrify.generate(
            "source",
            client=client,
            model="m",
            min_meta_score=80,
            max_attempts=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        rubric, report = result
        assert isinstance(rubric, rubrify.Rubric)
        assert isinstance(report, RefinementReport)
        assert report.stopped_reason == "target_met"
        assert report.iterations == 1
        # Only one generate + one meta-eval should have been made.
        assert client.call_count == 2

    def test_min_meta_score_retries(self) -> None:
        client = SequenceClient(
            [
                VALID_RUBRIC_XML,
                _meta_response(50),
                VALID_RUBRIC_XML,
                _meta_response(85),
            ]
        )
        result = rubrify.generate(
            "source",
            client=client,
            model="m",
            min_meta_score=80,
            max_attempts=3,
            return_report=True,
        )
        assert isinstance(result, tuple)
        _, report = result
        assert report.stopped_reason == "target_met"
        assert report.iterations == 2
        assert report.end_score == 85

    def test_max_attempts_exhausted_below_threshold(self) -> None:
        client = SequenceClient(
            [
                VALID_RUBRIC_XML,
                _meta_response(50),
                VALID_RUBRIC_XML,
                _meta_response(55),
            ]
        )
        result = rubrify.generate(
            "source",
            client=client,
            model="m",
            min_meta_score=80,
            max_attempts=2,
            return_report=True,
        )
        assert isinstance(result, tuple)
        rubric, report = result
        assert isinstance(rubric, rubrify.Rubric)
        assert report.stopped_reason == "max_iters"
        assert report.iterations == 2
        assert report.end_score == 55

    def test_return_report_tuple(self) -> None:
        client = SequenceClient([VALID_RUBRIC_XML])
        result = rubrify.generate("source", client=client, model="m", return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        out_rubric, report = result
        assert isinstance(out_rubric, rubrify.Rubric)
        assert isinstance(report, RefinementReport)

    def test_populates_provenance(self) -> None:
        client = SequenceClient([VALID_RUBRIC_XML])
        result = rubrify.generate(
            "Build a scoring rubric for short essays",
            client=client,
            model="m",
        )
        assert isinstance(result, rubrify.Rubric)
        assert result.provenance is not None
        kinds = [s.kind for s in result.provenance.refinement_steps]
        assert "generate" in kinds
        assert result.provenance.source_kind == "concept"
        assert result.provenance.source_summary.startswith("Build a scoring")

    def test_backwards_compat_returns_rubric(self) -> None:
        """Default args match pre-Phase-4 behavior: a single generate call."""
        client = SingleResponseClient(VALID_RUBRIC_XML)
        result = rubrify.generate("source", client=client, model="m")
        assert isinstance(result, rubrify.Rubric)
        # Single chat call, no meta-eval with defaults.
        assert client.call_count == 1
