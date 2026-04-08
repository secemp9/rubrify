"""Tests for Phase 4 provenance: RefinementStep, RubricProvenance, and the
``Rubric.export_provenance`` sidecar."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import rubrify
from rubrify.provenance import RefinementStep, RubricProvenance

# --- Helpers ---------------------------------------------------------------


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


class MockClient:
    """Minimal mock client returning a sequence of canned responses."""

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
            return self._responses[-1]
        resp = self._responses[self.call_count]
        self.call_count += 1
        return resp


def _meta_response(score: int, subscores: dict[str, int] | None = None) -> str:
    sub = subscores or {"C1": 5, "C2": 4, "C3": 4, "C4": 4, "C5": 5}
    return json.dumps(
        {
            "score": score,
            "class": "Strong rubric" if score >= 80 else "Needs work",
            "subscores": sub,
            "rationale": "BECAUSE: reasons.",
        }
    )


# --- RefinementStep --------------------------------------------------------


class TestRefinementStep:
    def test_refinement_step_frozen(self) -> None:
        step = RefinementStep(
            kind="generate",
            reason="initial",
            before_version="",
            after_version="1.0",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.kind = "evolve"  # type: ignore[misc]

    def test_refinement_step_defaults(self) -> None:
        step = RefinementStep(
            kind="refine_iter",
            reason="mutations applied",
            before_version="1.0",
            after_version="2.0",
        )
        assert step.mutation_names == ()
        assert step.meta_score is None


# --- RubricProvenance ------------------------------------------------------


class TestRubricProvenance:
    def test_rubric_provenance_default_is_none(self) -> None:
        rubric = rubrify.Rubric(name="X", mission="m")
        assert rubric.provenance is None

    def test_provenance_add_step_appends(self) -> None:
        prov = RubricProvenance()
        assert prov.refinement_steps == []
        step = RefinementStep(kind="generate", reason="x", before_version="", after_version="1.0")
        prov.add_step(step)
        assert prov.refinement_steps == [step]
        prov.add_step(step)
        assert len(prov.refinement_steps) == 2

    def test_provenance_to_dict_from_dict_roundtrip(self) -> None:
        prov = RubricProvenance(
            source_kind="concept",
            source_summary="a rubric for writing",
            generated_by_model="test-model",
            evaluated_by_model="test-model",
            parent_name="Parent",
            parent_version="1.0",
            calibration_suites=["suite_a", "suite_b"],
            tags=["tag1"],
        )
        prov.add_step(
            RefinementStep(
                kind="generate",
                reason="initial",
                before_version="",
                after_version="1.0",
                mutation_names=("AddSteeringConstraint",),
                meta_score=75,
            )
        )
        prov.add_step(
            RefinementStep(
                kind="refine_iter",
                reason="applied fix",
                before_version="1.0",
                after_version="2.0",
                mutation_names=(),
                meta_score=82,
            )
        )

        data = prov.to_dict()
        restored = RubricProvenance.from_dict(data)

        assert restored.source_kind == prov.source_kind
        assert restored.source_summary == prov.source_summary
        assert restored.generated_by_model == prov.generated_by_model
        assert restored.evaluated_by_model == prov.evaluated_by_model
        assert restored.parent_name == prov.parent_name
        assert restored.parent_version == prov.parent_version
        assert restored.calibration_suites == prov.calibration_suites
        assert restored.tags == prov.tags
        assert len(restored.refinement_steps) == 2
        assert restored.refinement_steps[0] == prov.refinement_steps[0]
        assert restored.refinement_steps[1] == prov.refinement_steps[1]

    def test_provenance_json_serializable(self) -> None:
        prov = RubricProvenance(
            source_kind="concept",
            source_summary="short",
            generated_by_model="m",
        )
        prov.add_step(
            RefinementStep(
                kind="generate",
                reason="r",
                before_version="",
                after_version="1.0",
                meta_score=80,
            )
        )
        # json.dumps must not raise
        text = json.dumps(prov.to_dict())
        assert "generate" in text
        assert "concept" in text


# --- Rubric integration ----------------------------------------------------


class TestRubricProvenanceIntegration:
    def _make_rubric(self) -> rubrify.Rubric:
        r = rubrify.Rubric(name="Sample", mission="Sample mission.")
        r.add_criterion(rubrify.Criterion(id="C1", name="X", weight=100, anchors={0: "a", 5: "b"}))
        r.output_schema = rubrify.OutputSchema(
            format="json",
            template='{"score":0}',
            constraints={"must_be_json": True},
        )
        r.scoring = rubrify.Scoring(formula="Sum C1.")
        return r

    def test_generate_populates_provenance(self) -> None:
        client = MockClient([VALID_RUBRIC_XML])
        result = rubrify.generate(
            "Create a writing quality rubric",
            client=client,
            model="test-model",
            rubric_type="scoring",
        )
        assert isinstance(result, rubrify.Rubric)
        assert result.provenance is not None
        assert result.provenance.source_kind == "concept"
        assert result.provenance.source_summary.startswith("Create a writing quality")
        assert result.provenance.generated_by_model == "test-model"
        assert len(result.provenance.refinement_steps) >= 1
        assert result.provenance.refinement_steps[0].kind == "generate"

    def test_refine_appends_refinement_step(self) -> None:
        rubric = self._make_rubric()
        meta_resp = _meta_response(90)
        client = MockClient([meta_resp])
        result = rubrify.refine(rubric, client=client, model="m")
        assert isinstance(result, rubrify.Rubric)
        assert result.provenance is not None
        # At least one refine_iter step should have been recorded.
        kinds = [s.kind for s in result.provenance.refinement_steps]
        assert "refine_iter" in kinds

    def test_export_provenance_raises_on_none(self, tmp_path: Path) -> None:
        rubric = self._make_rubric()
        assert rubric.provenance is None
        with pytest.raises(ValueError, match="no provenance"):
            rubric.export_provenance(str(tmp_path / "prov.json"))

    def test_export_provenance_writes_json(self, tmp_path: Path) -> None:
        rubric = self._make_rubric()
        rubric.provenance = RubricProvenance(
            source_kind="concept",
            source_summary="test",
            generated_by_model="m",
        )
        rubric.provenance.add_step(
            RefinementStep(
                kind="generate",
                reason="initial",
                before_version="",
                after_version="1.0",
                meta_score=75,
            )
        )
        out = tmp_path / "sample_provenance.json"
        rubric.export_provenance(str(out))
        assert out.exists()
        data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        assert data["source_kind"] == "concept"
        assert data["generated_by_model"] == "m"
        assert len(data["refinement_steps"]) == 1
        assert data["refinement_steps"][0]["kind"] == "generate"

    def test_rubric_copy_preserves_provenance(self) -> None:
        rubric = self._make_rubric()
        rubric.provenance = RubricProvenance(
            source_kind="concept",
            source_summary="test",
            generated_by_model="m",
        )
        rubric.provenance.add_step(
            RefinementStep(
                kind="generate",
                reason="r",
                before_version="",
                after_version="1.0",
                meta_score=80,
            )
        )
        clone = rubric.copy()
        assert clone.provenance is not None
        assert clone.provenance is not rubric.provenance  # deep-copied
        assert clone.provenance.source_kind == "concept"
        assert clone.provenance.refinement_steps == rubric.provenance.refinement_steps
        # Mutating the clone must not touch the original
        clone.provenance.add_step(
            RefinementStep(
                kind="refine_iter",
                reason="x",
                before_version="1.0",
                after_version="2.0",
            )
        )
        assert len(rubric.provenance.refinement_steps) == 1
        assert len(clone.provenance.refinement_steps) == 2

    def test_to_xml_does_not_include_provenance(self) -> None:
        """Guard rail 5 enforcement: provenance must never leak into XML."""
        rubric = self._make_rubric()
        rubric.provenance = RubricProvenance(
            source_kind="concept",
            source_summary="SHOULD_NOT_APPEAR_IN_XML_MARKER",
            generated_by_model="secret-model-name",
        )
        rubric.provenance.add_step(
            RefinementStep(
                kind="generate",
                reason="FORBIDDEN_REASON_MARKER",
                before_version="",
                after_version="1.0",
            )
        )
        xml = rubric.to_xml()
        assert "SHOULD_NOT_APPEAR_IN_XML_MARKER" not in xml
        assert "secret-model-name" not in xml
        assert "FORBIDDEN_REASON_MARKER" not in xml
        assert "provenance" not in xml.lower()
