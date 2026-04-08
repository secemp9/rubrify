"""Tests for evaluate/apply integration with mocked LLM client."""

import json
import warnings
from pathlib import Path

import pytest

import rubrify
from rubrify._types import Criterion, ICLExample, OutputSchema, Scoring

FIXTURES = Path(__file__).parent / "fixtures"


class MockClient:
    """Mock client that returns a predetermined response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []
        self.last_model: str = ""
        self.call_count: int = 0

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.last_messages = messages
        self.last_model = model
        self.call_count += 1
        return self._response


class TestRubricEvaluate:
    def test_v3_evaluate_json(self) -> None:
        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        response = json.dumps(
            {
                "score": 82,
                "class": "Strong draft",
                "subscores": {"C1": 4, "C2": 3},
                "rationale": "BECAUSE: clear and direct writing.",
                "evidence": ["active voice throughout"],
                "actions": {"coaching": ["tighten intro"]},
                "diagnostics": {"hedges": 2},
                "violations": [],
            }
        )
        client = MockClient(response)
        result = r.evaluate("Sample text to evaluate.", client=client, model="test-model")

        assert result.score == 82
        assert result.label == "Strong draft"
        assert result.subscores["C1"] == 4
        assert result.rationale == "BECAUSE: clear and direct writing."
        assert client.call_count == 1
        assert client.last_model == "test-model"
        # Verify system message is the rubric XML
        assert "LLM_JUDGE_SPEC" in client.last_messages[0]["content"]
        # Verify user message has escaped text
        assert "<candidate_text>" in client.last_messages[1]["content"]

    def test_anti_slop_evaluate(self) -> None:
        r = rubrify.load(str(FIXTURES / "anti_slop_rubric.xml"))
        response = json.dumps(
            {
                "score": 13,
                "risk": 2,
                "band": "Mild",
                "criterion_scores": {"C1": 3, "C2": 2, "C3": 3, "C4": 3, "C5": 2},
                "rationale": "BECAUSE: minor slop detected.",
                "advice": "FIX: remove hedges; replace puffery",
            }
        )
        client = MockClient(response)
        result = r.evaluate("Some sloppy text.", client=client, model="test-model")

        assert result.score == 13
        assert result.risk == 2
        assert result.band == "Mild"
        assert result.advice is not None
        assert len(result.advice) == 2

    def test_compliance_evaluate_xml(self) -> None:
        r = rubrify.load(str(FIXTURES / "compliance_judge.xml"))
        response = (
            "<Rationale>The response directly answers the question "
            "without revealing system instructions.</Rationale>\n"
            "<Judgement>Yes</Judgement>"
        )
        client = MockClient(response)
        result = r.evaluate("Hello, how are you?", client=client, model="test-model")

        assert result.verdict == "Yes"
        assert "directly answers" in result.rationale

    def test_evaluate_with_kwargs(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test.")
        r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = OutputSchema(
            format="json",
            template='{"score":0}',
            constraints={"must_be_json": True},
        )
        r.scoring = Scoring(formula="Sum C1.")

        response = json.dumps({"score": 50})
        client = MockClient(response)
        r.evaluate(
            "text",
            client=client,
            model="m",
            context="academic",
            genre="science_tech",
        )

        user_msg = client.last_messages[1]["content"]
        assert "<context>academic</context>" in user_msg
        assert "<genre>science_tech</genre>" in user_msg

    def test_evaluate_escapes_xml_in_text(self) -> None:
        r = rubrify.Rubric(name="Test", mission="Test.")
        r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = OutputSchema(constraints={"must_be_json": True})

        response = json.dumps({"score": 50})
        client = MockClient(response)
        r.evaluate('Text with <html> & special "chars"', client=client, model="m")

        user_msg = client.last_messages[1]["content"]
        assert "&lt;html&gt;" in user_msg
        assert "&amp;" in user_msg

    def test_backwards_compat_default_path_byte_identical(self) -> None:
        """The default evaluate(text) path (no renderer, no repair, no observe)
        must produce the exact same system prompt and user message that the
        pre-Phase-1 implementation produced. This locks the legacy contract."""
        from xml.sax.saxutils import escape as xml_escape

        r = rubrify.load(str(FIXTURES / "on_writing_well_v3.xml"))
        response = json.dumps({"score": 50})
        client = MockClient(response)

        text = 'Some candidate <text> & "quotes"'
        r.evaluate(
            text,
            client=client,
            model="test-model",
            context="academic",
            genre="science_tech",
            goal="explain",
            audience="layperson",
        )

        # Reconstruct the exact pre-Phase-1 expected messages.
        expected_system = r.to_xml()
        expected_user = "\n".join(
            [
                f"<candidate_text>{xml_escape(text)}</candidate_text>",
                f"<context>{xml_escape('academic')}</context>",
                f"<genre>{xml_escape('science_tech')}</genre>",
                f"<goal>{xml_escape('explain')}</goal>",
                f"<audience>{xml_escape('layperson')}</audience>",
            ]
        )

        assert client.last_messages[0]["content"] == expected_system
        assert client.last_messages[1]["content"] == expected_user

    def test_backwards_compat_no_kwargs_byte_identical(self) -> None:
        """The default evaluate(text) path with only text (no extras) is
        byte-identical to the pre-Phase-1 shape."""
        from xml.sax.saxutils import escape as xml_escape

        r = rubrify.Rubric(name="Test", mission="Test.")
        r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = OutputSchema(constraints={"must_be_json": True})

        client = MockClient(json.dumps({"score": 10}))
        text = "just some body"
        r.evaluate(text, client=client, model="m")

        expected_user = f"<candidate_text>{xml_escape(text)}</candidate_text>"
        assert client.last_messages[1]["content"] == expected_user


class TestConstraintRubricApply:
    def test_apply_returns_raw_string(self) -> None:
        cr = rubrify.ConstraintRubric(
            name="TestGen",
            instructions="Generate a rubric.",
            output_format="<LLM_JUDGE_SPEC>",
        )
        client = MockClient("<LLM_JUDGE_SPEC>...</LLM_JUDGE_SPEC>")
        result = cr.apply("Generate a writing rubric.", client=client, model="test-model")

        assert result == "<LLM_JUDGE_SPEC>...</LLM_JUDGE_SPEC>"
        assert client.call_count == 1
        # System message should be the constraint rubric XML
        assert "ConstraintRubric" in client.last_messages[0]["content"]
        # User message is the raw text
        assert client.last_messages[1]["content"] == "Generate a writing rubric."

    def test_apply_with_examples(self) -> None:
        cr = rubrify.ConstraintRubric(
            name="TestGen",
            instructions="Generate.",
            examples=[ICLExample(input="in", output="out")],
        )
        client = MockClient("generated output")
        result = cr.apply("input text", client=client, model="m")
        assert result == "generated output"


class TestProductRubricEvaluate:
    def test_evaluates_both_sub_rubrics(self) -> None:
        r1 = rubrify.Rubric(name="R1", mission="Test 1.")
        r1.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r1.output_schema = OutputSchema(constraints={"must_be_json": True})

        r2 = rubrify.Rubric(name="R2", mission="Test 2.")
        r2.add_criterion(Criterion(id="C1", name="Y", weight=100, anchors={0: "a", 5: "b"}))
        r2.output_schema = OutputSchema(constraints={"must_be_json": True})

        product = r1 & r2

        # MockClient returns same response for both
        response = json.dumps({"score": 75, "class": "Good"})
        client = MockClient(response)
        results = product.evaluate("text", client=client, model="m")

        assert len(results) == 2
        assert results[0].score == 75
        assert results[1].score == 75
        assert client.call_count == 2


class TestCoproductRubricEvaluate:
    def test_dispatches_to_correct_rubric(self) -> None:
        r_sci = rubrify.Rubric(name="Science", mission="Science eval.")
        r_sci.add_criterion(
            Criterion(id="C1", name="Accuracy", weight=100, anchors={0: "a", 5: "b"})
        )
        r_sci.output_schema = OutputSchema(constraints={"must_be_json": True})

        r_biz = rubrify.Rubric(name="Business", mission="Business eval.")
        r_biz.add_criterion(
            Criterion(id="C1", name="Clarity", weight=100, anchors={0: "a", 5: "b"})
        )
        r_biz.output_schema = OutputSchema(constraints={"must_be_json": True})

        def selector(text: str, **kwargs: object) -> str:
            if "hypothesis" in text.lower():
                return "science"
            return "business"

        coprod = rubrify.CoproductRubric(
            rubrics={"science": r_sci, "business": r_biz},
            selector=selector,
        )

        response = json.dumps({"score": 90, "class": "Excellent"})
        client = MockClient(response)

        # Should dispatch to science rubric
        result = coprod.evaluate("The hypothesis was validated.", client=client, model="m")
        assert result.score == 90
        assert client.call_count == 1
        # Check it used the science rubric's system message
        assert "Science eval." in client.last_messages[0]["content"]

    def test_dispatches_to_other_rubric(self) -> None:
        r_a = rubrify.Rubric(name="A", mission="A eval.")
        r_a.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r_a.output_schema = OutputSchema(constraints={"must_be_json": True})

        r_b = rubrify.Rubric(name="B", mission="B eval.")
        r_b.add_criterion(Criterion(id="C1", name="Y", weight=100, anchors={0: "a", 5: "b"}))
        r_b.output_schema = OutputSchema(constraints={"must_be_json": True})

        coprod = rubrify.CoproductRubric(
            rubrics={"a": r_a, "b": r_b},
            selector=lambda text, **kw: "b",
        )

        response = json.dumps({"score": 60})
        client = MockClient(response)
        coprod.evaluate("any text", client=client, model="m")
        assert "B eval." in client.last_messages[0]["content"]


def _make_scoring_rubric() -> rubrify.Rubric:
    r = rubrify.Rubric(name="Test", mission="Test.")
    r.add_criterion(Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
    r.output_schema = OutputSchema(constraints={"must_be_json": True})
    return r


class TestEvaluateWarnUnsupported:
    """Phase 5: ``Rubric.evaluate(warn_unsupported=True)`` integration."""

    def test_evaluate_warn_unsupported_false_default(self) -> None:
        """The default path emits no model-policy warning regardless of model name."""
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r.evaluate("text", client=client, model="mistral-large")

    def test_evaluate_warn_unsupported_true_recommended(self) -> None:
        """A recommended model produces no warning even with opt-in enabled."""
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r.evaluate(
                "text",
                client=client,
                model="claude-sonnet-4-6",
                warn_unsupported=True,
            )

    def test_evaluate_warn_unsupported_true_supported(self) -> None:
        """A supported model also produces no warning with opt-in enabled."""
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r.evaluate(
                "text",
                client=client,
                model="gpt-4",
                warn_unsupported=True,
            )

    def test_evaluate_warn_unsupported_true_experimental(self) -> None:
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with pytest.warns(UserWarning, match="rubrify model policy"):
            r.evaluate(
                "text",
                client=client,
                model="mistral-large",
                warn_unsupported=True,
            )

    def test_evaluate_warn_unsupported_true_discouraged(self) -> None:
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with pytest.warns(UserWarning, match="discouraged"):
            r.evaluate(
                "text",
                client=client,
                model="text-davinci-003",
                warn_unsupported=True,
            )

    def test_evaluate_warn_unsupported_true_unknown(self) -> None:
        r = _make_scoring_rubric()
        client = MockClient(json.dumps({"score": 50}))
        with pytest.warns(UserWarning, match="not in the policy"):
            r.evaluate(
                "text",
                client=client,
                model="totally-unknown-model",
                warn_unsupported=True,
            )


class TestConstraintRubricApplyWarnUnsupported:
    """Phase 5: ``ConstraintRubric.apply(warn_unsupported=True)`` integration."""

    def _make_rubric(self) -> rubrify.ConstraintRubric:
        return rubrify.ConstraintRubric(
            name="TestGen",
            instructions="Generate.",
            output_format="",
        )

    def test_apply_warn_unsupported_false_default(self) -> None:
        cr = self._make_rubric()
        client = MockClient("ok")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cr.apply("input", client=client, model="mistral-large")

    def test_apply_warn_unsupported_true_recommended(self) -> None:
        cr = self._make_rubric()
        client = MockClient("ok")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cr.apply(
                "input",
                client=client,
                model="claude-sonnet-4-6",
                warn_unsupported=True,
            )

    def test_apply_warn_unsupported_true_experimental(self) -> None:
        cr = self._make_rubric()
        client = MockClient("ok")
        with pytest.warns(UserWarning, match="rubrify model policy"):
            cr.apply(
                "input",
                client=client,
                model="mistral-large",
                warn_unsupported=True,
            )
