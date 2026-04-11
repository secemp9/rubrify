"""Tests for Phase D: generation, meta-rubric system, and instruction primitives."""

import json

import pytest

import rubrify
from rubrify._meta_rubric import (
    COMPLIANCE_GENERATOR,
    DETECTION_GENERATOR,
    INSTRUCTION_PRIMITIVES,
    META_EVALUATOR,
    SCORING_GENERATOR,
    compose_instructions,
)
from rubrify._types import Instruction
from rubrify.generate import _extract_xml, generate, refine

# A valid rubric XML that the mock client can return
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

# Invalid rubric XML missing mission (fails N1)
INVALID_RUBRIC_XML = """\
<LLM_JUDGE_SPEC version="1.0" name="BadRubric">
  <rubric>
    <criterion id="C1" name="X" weight="100">
      <anchor_0>Bad.</anchor_0>
      <anchor_5>Good.</anchor_5>
    </criterion>
  </rubric>
  <output_schema>
    <json_template>{"score":0}</json_template>
    <constraints><must_be_json>true</must_be_json></constraints>
  </output_schema>
</LLM_JUDGE_SPEC>"""


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


class TestExtractXml:
    """Tests for _extract_xml: markdown fence stripping and raw XML extraction."""

    def test_raw_xml_no_fences(self) -> None:
        raw = VALID_RUBRIC_XML
        assert _extract_xml(raw) == raw

    def test_xml_in_xml_fence(self) -> None:
        wrapped = f"```xml\n{VALID_RUBRIC_XML}\n```"
        assert _extract_xml(wrapped) == VALID_RUBRIC_XML

    def test_xml_in_plain_fence(self) -> None:
        wrapped = f"```\n{VALID_RUBRIC_XML}\n```"
        assert _extract_xml(wrapped) == VALID_RUBRIC_XML

    def test_prose_before_and_after_fence(self) -> None:
        text = (
            "Here is the rubric I generated for you:\n\n"
            f"```xml\n{VALID_RUBRIC_XML}\n```\n\n"
            "Let me know if you need any changes!"
        )
        assert _extract_xml(text) == VALID_RUBRIC_XML

    def test_multiple_fences_picks_llm_judge_spec(self) -> None:
        text = (
            "```python\nprint('hello')\n```\n\n"
            f"```xml\n{VALID_RUBRIC_XML}\n```\n\n"
            "```json\n{{}}\n```"
        )
        assert _extract_xml(text) == VALID_RUBRIC_XML

    def test_no_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="does not contain valid"):
            _extract_xml("Just some random text with no XML at all.")


class TestMetaEvaluatorStructure:
    def test_has_five_criteria(self) -> None:
        assert len(META_EVALUATOR.criteria) == 5
        assert set(META_EVALUATOR.criteria.keys()) == {"C1", "C2", "C3", "C4", "C5"}

    def test_has_four_disqualifiers(self) -> None:
        assert len(META_EVALUATOR.disqualifiers) == 4
        dq_ids = [dq.id for dq in META_EVALUATOR.disqualifiers]
        assert dq_ids == ["DQ1", "DQ2", "DQ3", "DQ4"]

    def test_has_output_schema(self) -> None:
        assert META_EVALUATOR.output_schema is not None
        assert META_EVALUATOR.output_schema.constraints.get("must_be_json") is True

    def test_has_scoring(self) -> None:
        assert META_EVALUATOR.scoring is not None
        assert len(META_EVALUATOR.scoring.labels) == 6

    def test_criteria_weights_sum_to_100(self) -> None:
        total = sum(c.weight for c in META_EVALUATOR.criteria.values())
        assert total == 100

    def test_all_criteria_have_anchors(self) -> None:
        for c in META_EVALUATOR.criteria.values():
            assert len(c.anchors) >= 2, f"{c.id} has fewer than 2 anchors"

    def test_validates_as_valid(self) -> None:
        result = rubrify.validate(META_EVALUATOR)
        assert result.is_valid


class TestGeneratorStructure:
    def test_scoring_generator_is_rubric(self) -> None:
        assert isinstance(SCORING_GENERATOR, rubrify.Rubric)
        assert SCORING_GENERATOR.name == "ScoringRubricGenerator"

    def test_scoring_generator_has_instructions(self) -> None:
        assert "mission" in SCORING_GENERATOR.instructions.lower()
        assert "criteria" in SCORING_GENERATOR.instructions.lower()
        assert "disqualifier" in SCORING_GENERATOR.instructions.lower()

    def test_scoring_generator_has_example(self) -> None:
        assert len(SCORING_GENERATOR.examples) == 1
        assert "ZinsserJudge" in SCORING_GENERATOR.examples[0].output

    def test_detection_generator_has_regex_instructions(self) -> None:
        assert isinstance(DETECTION_GENERATOR, rubrify.Rubric)
        assert "regex_library" in DETECTION_GENERATOR.instructions.lower()
        assert "inverted" in DETECTION_GENERATOR.instructions.lower()

    def test_detection_generator_has_example(self) -> None:
        assert len(DETECTION_GENERATOR.examples) == 1
        assert "AntiLLMY" in DETECTION_GENERATOR.examples[0].output

    def test_compliance_generator_has_decision_logic_instructions(self) -> None:
        assert isinstance(COMPLIANCE_GENERATOR, rubrify.Rubric)
        assert "decision_logic" in COMPLIANCE_GENERATOR.instructions.lower()
        assert "xml" in COMPLIANCE_GENERATOR.instructions.lower()

    def test_compliance_generator_has_example(self) -> None:
        assert len(COMPLIANCE_GENERATOR.examples) == 1
        assert "ComplianceJudge" in COMPLIANCE_GENERATOR.examples[0].output


class TestInstructionPrimitives:
    def test_all_15_primitives_exist(self) -> None:
        expected_keys = {
            "mission",
            "criteria_structure",
            "disqualifiers",
            "output_schema",
            "scoring",
            "mechanical",
            "steering",
            "mirror",
            "regex_library",
            "inverted_scoring",
            "advice_rules",
            "decision_logic",
            "xml_output",
            "mapping_examples",
            "definitions",
        }
        assert set(INSTRUCTION_PRIMITIVES.keys()) == expected_keys

    def test_all_primitives_are_instruction_type(self) -> None:
        for key, inst in INSTRUCTION_PRIMITIVES.items():
            assert isinstance(inst, Instruction), f"{key} is not an Instruction"
            assert inst.text, f"{key} has empty text"

    def test_compose_instructions_produces_numbered_list(self) -> None:
        instructions = [
            INSTRUCTION_PRIMITIVES["mission"],
            INSTRUCTION_PRIMITIVES["criteria_structure"],
        ]
        result = compose_instructions(instructions)
        assert result.startswith("Generate a valid <LLM_JUDGE_SPEC>")
        assert "1. " in result
        assert "2. " in result

    def test_compose_instructions_empty_list(self) -> None:
        result = compose_instructions([])
        assert "Generate a valid" in result


class TestGenerate:
    def test_generate_with_mock_returns_rubric(self) -> None:
        client = MockClient(VALID_RUBRIC_XML)
        result = generate(
            "Create a writing quality rubric",
            client=client,
            model="test-model",
            rubric_type="scoring",
        )
        assert isinstance(result, rubrify.Rubric)
        assert result.name == "TestRubric"
        assert len(result.criteria) == 3
        assert client.call_count == 1

    def test_generate_with_name_override(self) -> None:
        client = MockClient(VALID_RUBRIC_XML)
        result = generate(
            "source",
            client=client,
            model="m",
            name="CustomName",
        )
        assert isinstance(result, rubrify.Rubric)
        assert result.name == "CustomName"

    def test_generate_with_evaluate_returns_tuple(self) -> None:
        meta_response = json.dumps(
            {
                "score": 85,
                "class": "Strong rubric",
                "subscores": {"C1": 4, "C2": 4, "C3": 3, "C4": 4, "C5": 5},
                "rationale": "BECAUSE: well structured rubric.",
            }
        )

        call_count = 0
        responses = [VALID_RUBRIC_XML, meta_response]

        class MultiClient:
            def chat(
                self,
                *,
                messages: list[dict[str, str]],
                model: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> str:
                nonlocal call_count
                resp = responses[call_count]
                call_count += 1
                return resp

        result = generate(
            "source",
            client=MultiClient(),
            model="m",
            evaluate=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        rubric, eval_result = result
        assert isinstance(rubric, rubrify.Rubric)
        assert isinstance(eval_result, rubrify.EvaluationResult)
        assert eval_result.score == 85

    def test_generate_invalid_rubric_type_raises(self) -> None:
        client = MockClient("")
        with pytest.raises(ValueError, match="Unknown rubric_type"):
            generate("source", client=client, model="m", rubric_type="unknown")

    def test_generate_invalid_xml_raises(self) -> None:
        client = MockClient(INVALID_RUBRIC_XML)
        with pytest.raises(ValueError, match="failed validation"):
            generate("source", client=client, model="m")

    def test_generate_detection_type(self) -> None:
        client = MockClient(VALID_RUBRIC_XML)
        result = generate(
            "source",
            client=client,
            model="m",
            rubric_type="detection",
        )
        assert isinstance(result, rubrify.Rubric)

    def test_generate_compliance_type(self) -> None:
        client = MockClient(VALID_RUBRIC_XML)
        result = generate(
            "source",
            client=client,
            model="m",
            rubric_type="compliance",
        )
        assert isinstance(result, rubrify.Rubric)


class TestRefine:
    def test_refine_returns_rubric(self) -> None:
        rubric = rubrify.Rubric(name="Test", mission="Test rubric.")
        rubric.add_criterion(
            rubrify.Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"})
        )
        rubric.output_schema = rubrify.OutputSchema(
            format="json",
            template='{"score":0}',
            constraints={"must_be_json": True},
        )
        rubric.scoring = rubrify.Scoring(formula="Sum C1.")

        meta_response = json.dumps(
            {
                "score": 90,
                "class": "Exemplary",
                "subscores": {"C1": 5, "C2": 4, "C3": 4, "C4": 4, "C5": 5},
                "rationale": "BECAUSE: solid rubric design.",
            }
        )
        client = MockClient(meta_response)
        result = refine(rubric, client=client, model="m")
        # With MVP _suggest_mutations returning empty, result is the original rubric
        assert isinstance(result, rubrify.Rubric)
        assert result.name == "Test"


class TestExports:
    def test_generate_exported(self) -> None:
        assert hasattr(rubrify, "generate")
        assert callable(rubrify.generate)

    def test_refine_exported(self) -> None:
        assert hasattr(rubrify, "refine")
        assert callable(rubrify.refine)

    def test_meta_evaluator_exported(self) -> None:
        assert hasattr(rubrify, "META_EVALUATOR")
        assert isinstance(rubrify.META_EVALUATOR, rubrify.Rubric)

    def test_generators_exported(self) -> None:
        assert hasattr(rubrify, "SCORING_GENERATOR")
        assert hasattr(rubrify, "DETECTION_GENERATOR")
        assert hasattr(rubrify, "COMPLIANCE_GENERATOR")
        assert isinstance(rubrify.SCORING_GENERATOR, rubrify.Rubric)
        assert isinstance(rubrify.DETECTION_GENERATOR, rubrify.Rubric)
        assert isinstance(rubrify.COMPLIANCE_GENERATOR, rubrify.Rubric)
