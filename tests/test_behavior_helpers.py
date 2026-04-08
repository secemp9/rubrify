"""Tests for Phase 5: behavior-oriented generation helpers.

These tests split cleanly along the LLM-vs-construction axis described in
``rubrify/generate.py``:

* ``generate_evaluator`` / ``generate_detector`` / ``generate_classifier``
  route through :func:`rubrify.generate.generate` and are verified by
  monkey-patching that function.
* ``generate_constraint`` / ``generate_transformer`` /
  ``generate_from_examples`` are pure Python construction helpers that
  never touch the LLM. They are verified by instantiation and attribute
  assertions only.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

import rubrify
from rubrify._types import ICLExample
from rubrify.generate import (
    generate_classifier,
    generate_constraint,
    generate_detector,
    generate_evaluator,
    generate_from_examples,
    generate_transformer,
)
from rubrify.input_render import TemplateRenderer
from rubrify.rubric import ConstraintRubric, Rubric

# ``rubrify.__init__`` re-binds ``rubrify.generate`` to the :func:`generate`
# function, so we reach the underlying module via ``importlib`` to patch
# the ``generate`` symbol that the Phase 5 helpers actually call.
_GENERATE_MODULE = importlib.import_module("rubrify.generate")


class _RecordedCall:
    """Captures the args passed to a monkey-patched ``generate``."""

    def __init__(self) -> None:
        self.source: str | None = None
        self.kwargs: dict[str, Any] = {}
        self.called: bool = False


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _RecordedCall:
    """Replace ``rubrify.generate.generate`` with a recording stub."""
    record = _RecordedCall()

    def fake_generate(source: str, **kwargs: Any) -> Rubric:
        record.source = source
        record.kwargs = dict(kwargs)
        record.called = True
        return Rubric(name="stub", version="1.0", mission="stub mission")

    monkeypatch.setattr(_GENERATE_MODULE, "generate", fake_generate)
    return record


class TestLLMGenerationWrappers:
    def test_generate_evaluator_routes_to_scoring(self, recorded: _RecordedCall) -> None:
        rubric = generate_evaluator("source text", client=object(), model="gpt-5")
        assert recorded.called
        assert recorded.source == "source text"
        assert recorded.kwargs["rubric_type"] == "scoring"
        assert recorded.kwargs["model"] == "gpt-5"
        assert isinstance(rubric, Rubric)

    def test_generate_detector_routes_to_detection(self, recorded: _RecordedCall) -> None:
        generate_detector("source text", client=object(), model="gpt-5")
        assert recorded.kwargs["rubric_type"] == "detection"

    def test_generate_classifier_routes_to_compliance(self, recorded: _RecordedCall) -> None:
        generate_classifier("source text", client=object(), model="gpt-5")
        assert recorded.kwargs["rubric_type"] == "compliance"

    def test_generate_evaluator_forwards_kwargs(self, recorded: _RecordedCall) -> None:
        """Extra kwargs (e.g. min_meta_score) are forwarded unchanged."""
        generate_evaluator(
            "source text",
            client=object(),
            model="gpt-5",
            min_meta_score=80,
            max_attempts=3,
            name="MyRubric",
        )
        assert recorded.kwargs["min_meta_score"] == 80
        assert recorded.kwargs["max_attempts"] == 3
        assert recorded.kwargs["name"] == "MyRubric"

    def test_wrappers_exposed_on_package_root(self) -> None:
        """The three LLM wrappers are also importable from the package root."""
        assert rubrify.generate_evaluator is generate_evaluator
        assert rubrify.generate_detector is generate_detector
        assert rubrify.generate_classifier is generate_classifier


class TestGenerateConstraint:
    def test_generate_constraint_returns_constraint_rubric(self) -> None:
        rubric = generate_constraint("Force the output to match schema X.")
        assert isinstance(rubric, ConstraintRubric)
        assert rubric.instructions == "Force the output to match schema X."
        assert rubric.behaviors == frozenset({"force"})

    def test_generate_constraint_preserves_examples(self) -> None:
        examples = [
            ICLExample(input="in1", output="out1"),
            ICLExample(input="in2", output="out2"),
        ]
        rubric = generate_constraint(
            "Force the output.",
            examples=examples,
        )
        assert rubric.examples == examples
        assert len(rubric.examples) == 2

    def test_generate_constraint_custom_behaviors(self) -> None:
        custom = frozenset({"force", "extract"})
        rubric = generate_constraint(
            "Force and extract.",
            behaviors=custom,
        )
        assert rubric.behaviors == custom

    def test_generate_constraint_preserves_name_and_output_format(self) -> None:
        rubric = generate_constraint(
            "Force.",
            name="ForceRubric",
            output_format="<result>...</result>",
        )
        assert rubric.name == "ForceRubric"
        assert rubric.output_format == "<result>...</result>"

    def test_generate_constraint_default_examples_empty(self) -> None:
        rubric = generate_constraint("Force.")
        assert rubric.examples == []


class TestGenerateTransformer:
    def test_generate_transformer_with_template(self) -> None:
        rubric = generate_transformer(
            "Transform input.",
            template="<data>{content}</data>",
            placeholders=("content",),
        )
        assert isinstance(rubric, ConstraintRubric)
        assert rubric.input_renderer is not None
        assert isinstance(rubric.input_renderer, TemplateRenderer)
        assert rubric.input_renderer.template == "<data>{content}</data>"
        assert rubric.input_renderer.placeholders == ("content",)

    def test_generate_transformer_without_template(self) -> None:
        rubric = generate_transformer("Transform input.")
        assert isinstance(rubric, ConstraintRubric)
        assert rubric.input_renderer is None

    def test_generate_transformer_behaviors_is_transform(self) -> None:
        with_template = generate_transformer("t", template="<x>{content}</x>")
        without_template = generate_transformer("t")
        assert with_template.behaviors == frozenset({"transform"})
        assert without_template.behaviors == frozenset({"transform"})

    def test_generate_transformer_custom_placeholders(self) -> None:
        rubric = generate_transformer(
            "Rewrite the story.",
            template="<prompt>{title}: {body}</prompt>",
            placeholders=("title", "body"),
        )
        assert rubric.input_renderer is not None
        assert isinstance(rubric.input_renderer, TemplateRenderer)
        assert rubric.input_renderer.placeholders == ("title", "body")

    def test_generate_transformer_preserves_examples(self) -> None:
        examples = [ICLExample(input="in", output="out")]
        rubric = generate_transformer(
            "Transform.",
            examples=examples,
        )
        assert rubric.examples == examples


class TestGenerateFromExamples:
    def test_generate_from_examples_sets_instructions(self) -> None:
        examples = [ICLExample(input="in", output="out")]
        rubric = generate_from_examples(
            "Extract structured data from the input.",
            examples,
        )
        assert isinstance(rubric, ConstraintRubric)
        assert rubric.instructions == "Extract structured data from the input."

    def test_generate_from_examples_preserves_example_list(self) -> None:
        examples = [
            ICLExample(input="raw text 1", output="extracted 1"),
            ICLExample(input="raw text 2", output="extracted 2"),
            ICLExample(input="raw text 3", output="extracted 3"),
        ]
        rubric = generate_from_examples("Extract", examples)
        assert rubric.examples == examples
        assert len(rubric.examples) == 3

    def test_generate_from_examples_default_behaviors_extract(self) -> None:
        rubric = generate_from_examples(
            "Extract",
            [ICLExample(input="i", output="o")],
        )
        assert rubric.behaviors == frozenset({"extract"})

    def test_generate_from_examples_custom_behaviors(self) -> None:
        rubric = generate_from_examples(
            "Classify",
            [ICLExample(input="i", output="o")],
            behaviors=frozenset({"extract", "transform"}),
        )
        assert rubric.behaviors == frozenset({"extract", "transform"})

    def test_generate_from_examples_preserves_name(self) -> None:
        rubric = generate_from_examples(
            "Extract",
            [ICLExample(input="i", output="o")],
            name="Extractor",
        )
        assert rubric.name == "Extractor"


class TestConstructionHelpersDoNotCallLLM:
    """Guard rail check: the construction helpers must never touch a client."""

    def test_generate_constraint_accepts_no_client(self) -> None:
        # Call with no client/model at all; must succeed.
        rubric = generate_constraint("Do the thing.")
        assert isinstance(rubric, ConstraintRubric)

    def test_generate_transformer_accepts_no_client(self) -> None:
        rubric = generate_transformer("Transform.")
        assert isinstance(rubric, ConstraintRubric)

    def test_generate_from_examples_accepts_no_client(self) -> None:
        rubric = generate_from_examples(
            "Extract.",
            [ICLExample(input="i", output="o")],
        )
        assert isinstance(rubric, ConstraintRubric)
