"""Tests for Phase E: batch eval, to_json, parse_as, __version__."""

import json
from pathlib import Path

import rubrify
from rubrify._types import Criterion, OutputSchema, Scoring

FIXTURES = Path(__file__).parent / "fixtures"


class MockClient:
    """Mock client that returns a predetermined response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count: int = 0

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


class TestEvaluateBatch:
    def test_batch_returns_correct_count(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test.")
        r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = OutputSchema(
            constraints={"must_be_json": True},
        )

        response = json.dumps({"score": 75, "class": "Good"})
        client = MockClient(response)
        results = r.evaluate_batch(
            ["text1", "text2", "text3"],
            client=client,
            model="m",
        )

        assert len(results) == 3
        assert all(res.score == 75 for res in results)
        assert client.call_count == 3

    def test_batch_empty_list(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test.")
        r.output_schema = OutputSchema(constraints={"must_be_json": True})

        client = MockClient("{}")
        results = r.evaluate_batch([], client=client, model="m")

        assert results == []
        assert client.call_count == 0

    def test_batch_single_item(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test.")
        r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = OutputSchema(constraints={"must_be_json": True})

        response = json.dumps({"score": 50})
        client = MockClient(response)
        results = r.evaluate_batch(["only one"], client=client, model="m")

        assert len(results) == 1
        assert results[0].score == 50


class TestToJson:
    def test_basic_rubric_to_json(self) -> None:
        r = rubrify.Rubric(name="Test", version="1.0", mission="Test mission.")
        r.add_criterion(
            Criterion(id="C1", name="Clarity", weight=50, anchors={0: "bad", 5: "good"})
        )
        r.add_disqualifier(rubrify.Disqualifier(id="DQ1", description="Empty text."))
        r.output_schema = OutputSchema(
            format="json",
            template='{"score":0}',
            constraints={"must_be_json": True},
        )
        r.scoring = Scoring(
            formula="Sum C1.",
            labels={(0, 49): "Low", (50, 100): "High"},
        )

        json_str = r.to_json()
        data = json.loads(json_str)

        assert data["name"] == "Test"
        assert data["version"] == "1.0"
        assert data["mission"] == "Test mission."
        assert "C1" in data["criteria"]
        assert data["criteria"]["C1"]["name"] == "Clarity"
        assert data["criteria"]["C1"]["weight"] == 50
        assert data["criteria"]["C1"]["anchors"]["0"] == "bad"
        assert data["criteria"]["C1"]["anchors"]["5"] == "good"
        assert len(data["disqualifiers"]) == 1
        assert data["output_schema"]["format"] == "json"
        assert data["scoring"]["formula"] == "Sum C1."

    def test_v3_fixture_to_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        json_str = r.to_json()
        data = json.loads(json_str)

        assert data["name"] == "ZinsserJudge-XXL"
        assert len(data["criteria"]) == 25
        assert "pattern_library" in data
        assert data["scoring"]["inverted"] is False

    def test_anti_slop_to_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        json_str = r.to_json()
        data = json.loads(json_str)

        assert data["name"] == "AntiLLMY"
        assert data["scoring"]["inverted"] is True
        assert "pattern_library" in data
        assert data["pattern_library"]["flags"] == "i"

    def test_compliance_to_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        json_str = r.to_json()
        data = json.loads(json_str)

        assert data["name"] == "ComplianceJudge"
        assert "decision_logic" in data
        assert "mapping_examples" in data
        assert "definitions" in data

    def test_empty_rubric_to_json(self) -> None:
        r = rubrify.Rubric(name="Empty", mission="Minimal.")
        json_str = r.to_json()
        data = json.loads(json_str)

        assert data["name"] == "Empty"
        assert data["mission"] == "Minimal."
        assert "criteria" not in data
        assert "disqualifiers" not in data

    def test_to_json_is_valid_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        json_str = r.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestParseAs:
    def test_apply_parse_as_json(self) -> None:
        cr = rubrify.Rubric(
            name="TestGen",
            instructions="Generate.",
        )
        response = json.dumps({"key": "value", "count": 42})
        client = MockClient(response)
        result = cr.apply("input", client=client, model="m", parse_as="json")

        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["count"] == 42

    def test_apply_without_parse_as_returns_string(self) -> None:
        cr = rubrify.Rubric(
            name="TestGen",
            instructions="Generate.",
        )
        client = MockClient("raw text response")
        result = cr.apply("input", client=client, model="m")

        assert isinstance(result, str)
        assert result == "raw text response"

    def test_apply_parse_as_none_returns_string(self) -> None:
        cr = rubrify.Rubric(
            name="TestGen",
            instructions="Generate.",
        )
        client = MockClient("raw text")
        result = cr.apply("input", client=client, model="m", parse_as=None)

        assert isinstance(result, str)


class TestVersion:
    def test_version_exists(self) -> None:
        assert hasattr(rubrify, "__version__")
        assert rubrify.__version__ == "0.1.0"

    def test_version_is_string(self) -> None:
        assert isinstance(rubrify.__version__, str)
