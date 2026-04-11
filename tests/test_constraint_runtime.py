"""Phase 2 tests: constraint runtime as first-class behavior layer.

Covers the metadata-only ``behaviors`` frozenset, the validators pipeline,
``apply_and_validate`` / ``apply_with_repair`` return shapes, and the cautious
XML round-trip extension for ``<behaviors>``. Includes an opt-in live
integration test for completeness-style forcing against the real API.
"""

from __future__ import annotations

import dataclasses
import os
import re

import pytest

import rubrify
from rubrify._examples import COMPLETENESS_EXAMPLE
from rubrify.repair import RepairResult
from rubrify.result import ConstraintResult, EvaluationTrace
from rubrify.xml_io import rubric_from_xml


def _strip_behaviors_tag(xml: str) -> str:
    """Remove the metadata-only ``<behaviors>...</behaviors>`` element from XML."""
    return re.sub(r"\s*<behaviors>[^<]*</behaviors>\s*", "", xml)


class MockClient:
    """Mock client that returns a predetermined response and records calls."""

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


# ── behaviors: metadata only ──────────────────────────────────────────────


class TestBehaviorsMetadata:
    def test_behaviors_frozenset_default(self) -> None:
        cr = rubrify.Rubric(name="X", instructions="do a thing")
        assert isinstance(cr.behaviors, frozenset)
        assert cr.behaviors == frozenset()

    def test_behaviors_immutable(self) -> None:
        cr = rubrify.Rubric(
            name="X",
            instructions="do a thing",
            behaviors=frozenset({"force"}),
        )
        assert not hasattr(cr.behaviors, "add")
        assert not hasattr(cr.behaviors, "discard")
        new = cr.behaviors | {"transform"}
        assert cr.behaviors == frozenset({"force"})
        assert new == frozenset({"force", "transform"})

    def test_behaviors_compose(self) -> None:
        cr = rubrify.Rubric(
            name="Composite",
            instructions="do many things",
            behaviors=frozenset({"force", "transform", "extract"}),
        )
        assert cr.behaviors == frozenset({"force", "transform", "extract"})
        assert cr.behaviors <= rubrify.CONSTRAINT_BEHAVIORS

    def test_behaviors_not_dispatched(self) -> None:
        """Two rubrics that differ only in ``behaviors`` must execute identically."""
        cr_a = rubrify.Rubric(
            name="Same",
            instructions="do the thing",
            output_format="raw",
            behaviors=frozenset({"judge"}),
        )
        cr_b = rubrify.Rubric(
            name="Same",
            instructions="do the thing",
            output_format="raw",
            behaviors=frozenset({"force", "transform", "extract", "calibrate"}),
        )
        client_a = MockClient("output A")
        client_b = MockClient("output B")

        out_a = cr_a.apply("hello", client=client_a, model="m")
        out_b = cr_b.apply("hello", client=client_b, model="m")

        assert client_a.call_count == client_b.call_count == 1
        assert out_a == "output A"
        assert out_b == "output B"
        assert (
            client_a.last_messages[1]["content"] == client_b.last_messages[1]["content"] == "hello"
        )
        sys_a = _strip_behaviors_tag(client_a.last_messages[0]["content"])
        sys_b = _strip_behaviors_tag(client_b.last_messages[0]["content"])
        assert sys_a == sys_b

    def test_behaviors_taxonomy_exhaustive(self) -> None:
        expected = frozenset(
            {"judge", "score", "detect", "force", "transform", "extract", "calibrate"}
        )
        assert expected == rubrify.CONSTRAINT_BEHAVIORS


# ── validate_output ───────────────────────────────────────────────────────


class TestValidateOutput:
    def test_validate_output_empty_validators(self) -> None:
        cr = rubrify.Rubric(name="X", instructions="x")
        valid, violations = cr.validate_output("anything at all")
        assert valid is True
        assert violations == []

    def test_validate_output_passing_validator(self) -> None:
        def always_pass(_: str) -> tuple[bool, str | None]:
            return (True, None)

        cr = rubrify.Rubric(name="X", instructions="x", validators=[always_pass])
        valid, violations = cr.validate_output("hello")
        assert valid is True
        assert violations == []

    def test_validate_output_failing_validator(self) -> None:
        def always_fail(_: str) -> tuple[bool, str | None]:
            return (False, "nope")

        cr = rubrify.Rubric(name="X", instructions="x", validators=[always_fail])
        valid, violations = cr.validate_output("hello")
        assert valid is False
        assert violations == ["nope"]

    def test_validate_output_multiple_validators(self) -> None:
        def v_pass(_: str) -> tuple[bool, str | None]:
            return (True, None)

        def v_fail_1(_: str) -> tuple[bool, str | None]:
            return (False, "missing tag A")

        def v_fail_2(_: str) -> tuple[bool, str | None]:
            return (False, "missing tag B")

        cr = rubrify.Rubric(
            name="X",
            instructions="x",
            validators=[v_pass, v_fail_1, v_fail_2],
        )
        valid, violations = cr.validate_output("hello")
        assert valid is False
        assert violations == ["missing tag A", "missing tag B"]


# ── apply_and_validate ────────────────────────────────────────────────────


def _has_response_wrapper(output: str) -> tuple[bool, str | None]:
    if "<response>" in output and "</response>" in output:
        return (True, None)
    return (False, "missing <response> wrapper")


def _has_code_block_tag(output: str) -> tuple[bool, str | None]:
    if "<full_entire_complete_updated_code_in_a_code_block_here>" in output:
        return (True, None)
    return (False, "missing code block tag")


class TestApplyAndValidate:
    def test_apply_and_validate_passing(self) -> None:
        good = (
            "<response>\n"
            "<full_entire_complete_updated_code_in_a_code_block_here>"
            "print('ok')"
            "</full_entire_complete_updated_code_in_a_code_block_here>\n"
            "</response>"
        )
        cr = rubrify.Rubric(
            name="Forcer",
            instructions="force structure",
            validators=[_has_response_wrapper, _has_code_block_tag],
            behaviors=frozenset({"force"}),
        )
        client = MockClient(good)
        result = cr.apply_and_validate("request", client=client, model="m")

        assert isinstance(result, ConstraintResult)
        assert result.valid is True
        assert result.violations == ()
        assert result.output == good
        assert result.repaired is False
        assert result.repair_notes == ()
        assert result.trace is None

    def test_apply_and_validate_failing(self) -> None:
        bad = "Sure thing! Here is the code: print('hi')"
        cr = rubrify.Rubric(
            name="Forcer",
            instructions="force structure",
            validators=[_has_response_wrapper, _has_code_block_tag],
        )
        client = MockClient(bad)
        result = cr.apply_and_validate("request", client=client, model="m")

        assert isinstance(result, ConstraintResult)
        assert result.valid is False
        assert "missing <response> wrapper" in result.violations
        assert "missing code block tag" in result.violations
        assert result.output == bad

    def test_apply_and_validate_with_observe(self) -> None:
        cr = rubrify.Rubric(
            name="X",
            instructions="do",
            validators=[lambda s: (True, None)],
        )
        client = MockClient("output")
        result = cr.apply_and_validate("input", client=client, model="m", observe=True)
        assert isinstance(result, ConstraintResult)
        assert result.trace is not None
        assert isinstance(result.trace, EvaluationTrace)
        assert result.trace.model == "m"
        assert result.trace.parser == "raw"
        assert result.trace.user_message == "input"

    def test_apply_and_validate_rejects_parse_as(self) -> None:
        cr = rubrify.Rubric(name="X", instructions="do")
        client = MockClient("output")
        with pytest.raises(TypeError, match="parse_as"):
            cr.apply_and_validate("input", client=client, model="m", parse_as="json")


# ── apply_with_repair ─────────────────────────────────────────────────────


class TestApplyWithRepair:
    def test_apply_with_repair_no_repair_fn(self) -> None:
        cr = rubrify.Rubric(
            name="X",
            instructions="do",
            validators=[_has_response_wrapper],
        )
        client = MockClient("no wrapper here")
        result = cr.apply_with_repair("input", client=client, model="m")
        assert isinstance(result, ConstraintResult)
        assert result.valid is False
        assert result.repaired is False
        assert result.repair_notes == ()
        assert result.output == "no wrapper here"

    def test_apply_with_repair_skips_when_valid(self) -> None:
        cr = rubrify.Rubric(
            name="X",
            instructions="do",
            validators=[_has_response_wrapper],
        )
        client = MockClient("<response>ok</response>")

        def should_not_be_called(_: str) -> RepairResult:
            raise AssertionError("repair_fn must not be called when valid")

        result = cr.apply_with_repair(
            "input", client=client, model="m", repair_fn=should_not_be_called
        )
        assert result.valid is True
        assert result.repaired is False

    def test_apply_with_repair_with_repair_fn(self) -> None:
        cr = rubrify.Rubric(
            name="X",
            instructions="do",
            validators=[_has_response_wrapper],
        )
        client = MockClient("bare text")

        def wrap_in_response(raw: str) -> RepairResult:
            return RepairResult(
                text=f"<response>{raw}</response>",
                repaired=True,
                notes=("wrapped bare text in <response>",),
            )

        result = cr.apply_with_repair("input", client=client, model="m", repair_fn=wrap_in_response)
        assert isinstance(result, ConstraintResult)
        assert result.valid is True
        assert result.repaired is True
        assert result.repair_notes == ("wrapped bare text in <response>",)
        assert result.output == "<response>bare text</response>"
        assert result.violations == ()


# ── XML round-trip for <behaviors> ────────────────────────────────────────


class TestXMLBehaviorsRoundTrip:
    def test_xml_roundtrip_with_behaviors(self) -> None:
        cr = rubrify.Rubric(
            name="RoundTrip",
            instructions="force the shape",
            output_format="<foo/>",
            behaviors=frozenset({"force", "transform"}),
        )
        xml = cr.to_xml()
        assert "<behaviors>" in xml
        assert "force transform" in xml

        restored = rubric_from_xml(xml)
        assert restored.name == "RoundTrip"
        assert restored.instructions == "force the shape"
        assert restored.output_format_str == "<foo/>"
        assert restored.behaviors == frozenset({"force", "transform"})

    def test_xml_roundtrip_without_behaviors(self) -> None:
        cr = rubrify.Rubric(
            name="Plain",
            instructions="just do the thing",
        )
        xml = cr.to_xml()
        assert "<behaviors>" not in xml

        restored = rubric_from_xml(xml)
        assert restored.name == "Plain"
        assert restored.instructions == "just do the thing"
        assert restored.behaviors == frozenset()


# ── ConstraintResult immutability ─────────────────────────────────────────


class TestConstraintResult:
    def test_constraint_result_frozen(self) -> None:
        result = ConstraintResult(output="x", valid=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.output = "y"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.valid = False  # type: ignore[misc]

    def test_constraint_result_defaults(self) -> None:
        result = ConstraintResult(output="hello", valid=True)
        assert result.violations == ()
        assert result.repaired is False
        assert result.repair_notes == ()
        assert result.trace is None


# ── Example constants are importable ──────────────────────────────────────


class TestExampleConstants:
    def test_completeness_example_shape(self) -> None:
        assert isinstance(rubrify.COMPLETENESS_EXAMPLE, rubrify.Rubric)
        assert "force" in rubrify.COMPLETENESS_EXAMPLE.behaviors
        assert "transform" in rubrify.COMPLETENESS_EXAMPLE.behaviors

    def test_extraction_example_shape(self) -> None:
        assert isinstance(rubrify.EXTRACTION_EXAMPLE, rubrify.Rubric)
        assert rubrify.EXTRACTION_EXAMPLE.behaviors == frozenset({"extract"})

    def test_transform_example_has_template_renderer(self) -> None:
        assert isinstance(rubrify.TRANSFORM_EXAMPLE, rubrify.Rubric)
        assert rubrify.TRANSFORM_EXAMPLE.behaviors == frozenset({"transform"})
        assert rubrify.TRANSFORM_EXAMPLE.input_renderer is not None


# ── Live integration test ────────────────────────────────────────────────


@pytest.mark.integration
def test_completeness_forcing_live() -> None:
    """Build a completeness-style rubric in Python and force the structure live."""
    base_url = os.environ.get("RUBRIFY_BASE_URL", "")
    api_key = os.environ.get("RUBRIFY_API_KEY", "")
    model = os.environ.get("RUBRIFY_MODEL", "")
    if not (base_url and api_key and model):
        pytest.skip("live API env vars not set")

    client = rubrify.Client(base_url=base_url, api_key=api_key)

    def has_response_wrapper(output: str) -> tuple[bool, str | None]:
        if "<response>" in output and "</response>" in output:
            return (True, None)
        return (False, "missing <response> wrapper")

    def has_code_block_tag(output: str) -> tuple[bool, str | None]:
        if "<full_entire_complete_updated_code_in_a_code_block_here>" in output:
            return (True, None)
        return (False, "missing full_entire_complete_updated_code_in_a_code_block_here tag")

    rubric = rubrify.Rubric(
        name=COMPLETENESS_EXAMPLE.name,
        instructions=COMPLETENESS_EXAMPLE.instructions,
        output_format=COMPLETENESS_EXAMPLE.output_format_str,
        examples=list(COMPLETENESS_EXAMPLE.examples),
        behaviors=COMPLETENESS_EXAMPLE.behaviors,
        validators=[has_response_wrapper, has_code_block_tag],
    )

    try:
        result = rubric.apply_and_validate(
            "Write a Python function that returns the sum of two numbers.",
            client=client,
            model=model,
        )
    finally:
        client.close()

    assert isinstance(result, ConstraintResult)
    assert result.valid, f"live output violated structural template: {result.violations}"
