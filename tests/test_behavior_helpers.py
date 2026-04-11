"""Tests for Phase 5: behavior-oriented generation helpers.

These tests cover the LLM-calling wrappers (generate_evaluator,
generate_detector, generate_classifier) which route through
:func:`rubrify.generate.generate`.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

import rubrify
from rubrify.generate import (
    generate_classifier,
    generate_detector,
    generate_evaluator,
)
from rubrify.rubric import Rubric

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
