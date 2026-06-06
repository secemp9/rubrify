"""Comprehensive test suite for rubrify — IR types, compiler, codecs, and
end-to-end integration via harn's faux provider.

Every test is deterministic: no real LLM calls, no network, no randomness.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from rubrify.ir.types import (
    BinaryScale,
    Criterion,
    CriterionGroup,
    Disqualifier,
    NumericScale,
    OrdinalScale,
    PatternEntry,
    Rubric,
    RubricMeta,
    ScaleAnchor,
)
from rubrify.ir.roles import SurfacePolicy
from rubrify.ir.bundle import RubricBundle
from rubrify.compiler.compiler import CompilationResult, compile_rubric
from rubrify.codecs.xml_codec import render_rubric_xml
from rubrify.codecs.json_codec import (
    ParseError,
    build_judgment_model,
    build_judgment_tool,
    parse_judgment_json,
    validate_judgment_output,
)
from rubrify.engine.judge import Judge, JudgeConfig
from rubrify.engine.judgment import CriterionJudgment
from rubrify.ir.constraints import (
    CharLimitConstraint,
    ItemCountConstraint,
    PrefixSuffixConstraint,
    TokenConstraint,
    WordCountConstraint,
)
from rubrify.compiler.passes import audit_output_constraints


# ── Helpers ───────────────────────────────────────────────────────


def _simple_criterion(cid: str, title: str | None = None, **overrides) -> Criterion:
    defaults = dict(
        id=cid,
        title=title or f"Criterion {cid}",
        description=f"Test criterion {cid}",
        scale=NumericScale(minimum=0, maximum=5),
        weight=1.0,
    )
    defaults.update(overrides)
    return Criterion(**defaults)


def _make_simple_rubric(
    *,
    criteria: list[Criterion] | None = None,
    name: str = "TestRubric",
    goal: str = "Test evaluation.",
    **kwargs,
) -> Rubric:
    return Rubric(
        meta=RubricMeta(name=name, version="1.0"),
        goal=goal,
        criteria=criteria or [_simple_criterion("C1"), _simple_criterion("C2")],
        **kwargs,
    )


def _compile(rubric: Rubric | None = None, **kwargs) -> CompilationResult:
    return compile_rubric(rubric or _make_simple_rubric(), **kwargs)


# ═════════════════════════════════════════════════════════════════
# 1. IR Type Validation
# ═════════════════════════════════════════════════════════════════


class TestIRValidation:
    """Pydantic model validators reject structurally invalid rubric definitions."""

    def test_numeric_scale_rejects_max_lte_min(self):
        with pytest.raises(ValidationError, match="maximum.*must be greater than minimum"):
            NumericScale(minimum=5, maximum=3)

    def test_numeric_scale_rejects_max_eq_min(self):
        with pytest.raises(ValidationError, match="maximum.*must be greater than minimum"):
            NumericScale(minimum=5, maximum=5)

    def test_numeric_scale_rejects_nonpositive_step(self):
        with pytest.raises(ValidationError, match="step must be positive"):
            NumericScale(minimum=0, maximum=10, step=0)

    def test_ordinal_scale_rejects_empty_anchors(self):
        with pytest.raises(ValidationError, match="at least one anchor"):
            OrdinalScale(anchors=[])

    def test_criterion_rejects_negative_weight(self):
        with pytest.raises(ValidationError):
            Criterion(
                id="C1", title="t", description="d",
                scale=NumericScale(minimum=0, maximum=5), weight=-1,
            )

    def test_rubric_rejects_duplicate_criterion_ids(self):
        with pytest.raises(ValidationError, match="Duplicate criterion IDs"):
            Rubric(
                meta=RubricMeta(name="t", version="1.0"),
                goal="g",
                criteria=[
                    _simple_criterion("C1"),
                    _simple_criterion("C1", title="duplicate"),
                ],
            )

    def test_rubric_rejects_invalid_group_refs(self):
        with pytest.raises(ValidationError, match="unknown criteria"):
            Rubric(
                meta=RubricMeta(name="t", version="1.0"),
                goal="g",
                criteria=[_simple_criterion("C1")],
                groups=[CriterionGroup(id="G1", title="group", children=["C99"])],
            )

    def test_rubric_rejects_invalid_disqualifier_refs(self):
        with pytest.raises(ValidationError, match="unknown criteria"):
            Rubric(
                meta=RubricMeta(name="t", version="1.0"),
                goal="g",
                criteria=[_simple_criterion("C1")],
                disqualifiers=[Disqualifier(id="DQ1", description="bad", criterion_ids=["C99"])],
            )

    def test_surface_policy_rejects_invalid_codec(self):
        with pytest.raises(ValidationError):
            SurfacePolicy(input_codec="yaml")

    def test_schema_model_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            NumericScale(minimum=0, maximum=5, nonexistent_field="boom")


# ═════════════════════════════════════════════════════════════════
# 1b. Execution Strategy & Constraint Scope Validation
# ═════════════════════════════════════════════════════════════════


class TestExecutionStrategy:
    """Validate execution_strategy on SurfacePolicy and scope on OutputConstraints."""

    def test_surface_policy_accepts_valid_strategies(self):
        for strategy in ("holistic", "grouped", "per_criterion"):
            sp = SurfacePolicy(execution_strategy=strategy)
            assert sp.execution_strategy == strategy

    def test_surface_policy_rejects_invalid_strategy(self):
        with pytest.raises(ValidationError):
            SurfacePolicy(execution_strategy="round_robin")

    def test_constraint_scope_defaults_to_call(self):
        """All constraint variants default their scope field to 'call'."""
        ps = PrefixSuffixConstraint(
            id="ps", description="d", target_field="rationale", prefix="X"
        )
        wc = WordCountConstraint(
            id="wc", description="d", target_field="rationale", count=5, mode="exactly"
        )
        cl = CharLimitConstraint(
            id="cl", description="d", target_field="rationale", limit=100, mode="max"
        )
        ic = ItemCountConstraint(
            id="ic", description="d", target_field="evidence", count=2, mode="min"
        )
        tk = TokenConstraint(
            id="tk", description="d", target_field="rationale", token="X"
        )
        for c in (ps, wc, cl, ic, tk):
            assert c.scope == "call", f"{type(c).__name__} did not default scope to 'call'"

    def test_constraint_accepts_valid_scopes(self):
        for scope in ("call", "criterion", "judgment"):
            c = TokenConstraint(
                id="tk", description="d", target_field="rationale",
                token="X", scope=scope,
            )
            assert c.scope == scope

    def test_constraint_rejects_invalid_scope(self):
        with pytest.raises(ValidationError):
            TokenConstraint(
                id="tk", description="d", target_field="rationale",
                token="X", scope="global",
            )


# ═════════════════════════════════════════════════════════════════
# 2. Scale Normalization
# ═════════════════════════════════════════════════════════════════


class TestScaleNormalization:
    """Scale.to_unit() maps raw scores into [0, 1] with correct clamping."""

    def test_numeric_to_unit_bounds(self):
        scale = NumericScale(minimum=0, maximum=10)
        assert scale.to_unit(0) == 0.0
        assert scale.to_unit(10) == 1.0
        assert scale.to_unit(5) == 0.5

    def test_numeric_to_unit_clamps_above(self):
        scale = NumericScale(minimum=0, maximum=10)
        assert scale.to_unit(15) == 1.0

    def test_numeric_to_unit_clamps_below(self):
        scale = NumericScale(minimum=0, maximum=10)
        assert scale.to_unit(-5) == 0.0

    def test_ordinal_to_unit_by_value(self):
        scale = OrdinalScale(anchors=[
            ScaleAnchor(value=0, label="bad"),
            ScaleAnchor(value=5, label="good"),
        ])
        assert scale.to_unit(0) == 0.0
        assert scale.to_unit(5) == 1.0

    def test_ordinal_to_unit_by_label(self):
        scale = OrdinalScale(anchors=[
            ScaleAnchor(value=0, label="bad"),
            ScaleAnchor(value=5, label="good"),
        ])
        assert scale.to_unit("bad") == 0.0
        assert scale.to_unit("good") == 1.0

    def test_ordinal_to_unit_clamps(self):
        scale = OrdinalScale(anchors=[
            ScaleAnchor(value=0, label="bad"),
            ScaleAnchor(value=5, label="good"),
        ])
        assert scale.to_unit(10) == 1.0
        assert scale.to_unit(-3) == 0.0

    def test_ordinal_unknown_label_raises(self):
        scale = OrdinalScale(anchors=[
            ScaleAnchor(value=0, label="bad"),
            ScaleAnchor(value=5, label="good"),
        ])
        with pytest.raises(ValueError, match="Unknown ordinal label"):
            scale.to_unit("nonexistent")

    def test_binary_to_unit(self):
        scale = BinaryScale()
        assert scale.to_unit(True) == 1.0
        assert scale.to_unit(False) == 0.0


# ═════════════════════════════════════════════════════════════════
# 3. Compiler Pipeline
# ═════════════════════════════════════════════════════════════════


class TestCompiler:
    """compile_rubric() normalizes, binds, locks, and audits."""

    def test_compile_produces_locked_bundle(self):
        result = _compile()
        assert result.ok
        assert result.bundle.locked is True

    def test_bundle_is_frozen(self):
        result = _compile()
        with pytest.raises(ValidationError):
            result.bundle.locked = False

    def test_compilation_preserves_criteria(self):
        rubric = _make_simple_rubric()
        result = compile_rubric(rubric)
        assert len(result.bundle.rubric.criteria) == len(rubric.criteria)
        for orig, compiled in zip(rubric.criteria, result.bundle.rubric.criteria):
            assert compiled.id == orig.id

    def test_bindings_generated_for_each_criterion(self):
        rubric = _make_simple_rubric()
        result = compile_rubric(rubric)
        assert len(result.bundle.bindings) == len(rubric.criteria)

    def test_each_binding_has_xml_and_json_projections(self):
        result = _compile()
        for binding in result.bundle.bindings:
            codecs = {p.codec for p in binding.projections}
            assert "xml" in codecs
            assert "json" in codecs

    def test_invalid_pattern_rejected_at_compile(self):
        rubric = _make_simple_rubric(
            patterns=[PatternEntry(id="bad", pattern="[broken")],
        )
        with pytest.raises(ValueError, match="Invalid regex"):
            compile_rubric(rubric)

    def test_valid_pattern_compiled(self):
        rubric = _make_simple_rubric(
            patterns=[PatternEntry(id="p1", pattern=r"\bfoo\b")],
        )
        result = compile_rubric(rubric)
        assert "p1" in result.bundle.compiled_patterns

    def test_compilation_issues_empty_for_clean_rubric(self):
        result = _compile()
        assert result.issues == []
        assert result.ok

    def test_audit_warns_per_criterion_hard_call_scope(self):
        """Compiling with per_criterion strategy + hard call-scoped constraint produces a warning."""
        rubric = _make_simple_rubric()
        policy = SurfacePolicy(execution_strategy="per_criterion")
        constraints = [
            TokenConstraint(
                id="tk_hard",
                description="must contain VERDICT",
                target_field="rationale",
                token="VERDICT",
                enforcement="hard",
                scope="call",
            ),
        ]
        result = compile_rubric(rubric, policy=policy, output_constraints=constraints)
        # Should have an issue about scope='call' + enforcement='hard' in per_criterion mode
        assert any(
            "tk_hard" in issue and "per_criterion" in issue
            for issue in result.issues
        )


# ═════════════════════════════════════════════════════════════════
# 4. XML Codec
# ═════════════════════════════════════════════════════════════════


class TestXmlCodec:
    """render_rubric_xml() produces well-formed, binding-driven XML."""

    def test_xml_round_trips(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        assert root.tag == "LLM_JUDGE_SPEC"

    def test_root_has_version_and_name(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        assert root.get("version") == "1.0"
        assert root.get("name") == "TestRubric"

    def test_mission_text_matches_goal(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        assert root.find("mission").text == "Test evaluation."

    def test_special_chars_escaped(self):
        rubric = Rubric(
            meta=RubricMeta(name='Test & "Escape"', version="1.0"),
            goal="Check A > B & C < D",
            criteria=[_simple_criterion("C1", title='Score & "Quality"')],
        )
        result = compile_rubric(rubric)
        xml = render_rubric_xml(result.bundle)
        # ET.fromstring would raise on malformed XML with unescaped entities
        root = ET.fromstring(xml)
        assert "&" in root.find("mission").text
        assert '"' in root.get("name")

    def test_binding_drives_criterion_attributes(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        crit_el = root.find(".//criterion")
        binding = result.bundle.bindings[0]
        xml_proj = next(p for p in binding.projections if p.codec == "xml")
        assert crit_el.get("id") == xml_proj.attributes["id"]
        assert crit_el.get("name") == xml_proj.attributes["name"]

    def test_criterion_count_matches_rubric(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        crit_elements = root.findall(".//criterion")
        assert len(crit_elements) == 2

    def test_output_schema_present(self):
        result = _compile()
        xml = render_rubric_xml(result.bundle)
        root = ET.fromstring(xml)
        assert root.find("output_schema") is not None
        assert root.find("output_schema/json_template") is not None


# ═════════════════════════════════════════════════════════════════
# 5. JSON Codec
# ═════════════════════════════════════════════════════════════════


class TestJsonCodec:
    """JSON parsing, dynamic model construction, and validation."""

    def test_parse_valid_json(self):
        result = parse_judgment_json('{"score": 5, "rationale": "good"}')
        assert result["score"] == 5
        assert result["rationale"] == "good"

    def test_parse_empty_raises(self):
        with pytest.raises(ParseError):
            parse_judgment_json("")

    def test_parse_whitespace_only_raises(self):
        with pytest.raises(ParseError):
            parse_judgment_json("   \n  ")

    def test_parse_non_json_raises(self):
        with pytest.raises(ParseError):
            parse_judgment_json("This is plain text, not JSON.")

    def test_build_judgment_model_cached(self):
        bundle = _compile().bundle
        m1 = build_judgment_model(bundle)
        m2 = build_judgment_model(bundle)
        assert m1 is m2

    def test_build_judgment_model_has_expected_fields(self):
        bundle = _compile().bundle
        Model = build_judgment_model(bundle)
        fields = set(Model.model_fields.keys())
        assert "score" in fields
        assert "rationale" in fields
        assert "evidence" in fields
        assert "criterion_scores" in fields

    def test_validate_valid_output(self):
        bundle = _compile().bundle
        parsed = {
            "score": 4,
            "rationale": "Good quality.",
            "evidence": ["quote1"],
            "violations": [],
            "criterion_scores": {"C1": 4, "C2": 3},
        }
        validated, warnings = validate_judgment_output(parsed, bundle)
        assert validated is not None
        assert warnings == []

    def test_validate_coerces_string_to_number(self):
        bundle = _compile().bundle
        parsed = {
            "score": "3",
            "rationale": "ok",
            "criterion_scores": {"C1": "2", "C2": "4"},
        }
        validated, warnings = validate_judgment_output(parsed, bundle)
        assert validated is not None
        assert validated.score == 3

    def test_validate_rejects_missing_required(self):
        bundle = _compile().bundle
        parsed = {"rationale": "ok"}  # missing score, criterion_scores
        validated, warnings = validate_judgment_output(parsed, bundle)
        assert validated is None
        assert len(warnings) > 0

    def test_build_judgment_tool(self):
        bundle = _compile().bundle
        tool = build_judgment_tool(bundle)
        assert tool.name == "submit_judgment"
        assert "submit" in tool.description.lower() or "judgment" in tool.description.lower()


# ═════════════════════════════════════════════════════════════════
# 6. Output Constraints
# ═════════════════════════════════════════════════════════════════


class TestOutputConstraints:
    """OutputConstraint variants: check() logic, validation, and audit pass."""

    # ── PrefixSuffixConstraint ────────────────────────────────────

    def test_prefix_suffix_check_pass(self):
        c = PrefixSuffixConstraint(
            id="ps1", description="starts with BECAUSE:", target_field="rationale",
            prefix="BECAUSE:", suffix=".",
        )
        assert c.check("BECAUSE: the answer is correct.") is None

    def test_prefix_suffix_check_fail(self):
        c = PrefixSuffixConstraint(
            id="ps2", description="must start with BECAUSE:", target_field="rationale",
            prefix="BECAUSE:",
        )
        result = c.check("The answer is correct.")
        assert result is not None
        assert "BECAUSE:" in result

    # ── WordCountConstraint ───────────────────────────────────────

    def test_word_count_check_exactly(self):
        c = WordCountConstraint(
            id="wc1", description="exactly five words", target_field="rationale",
            count=5, mode="exactly",
        )
        assert c.check("one two three four five") is None
        result = c.check("one two three")
        assert result is not None
        assert "exactly 5" in result

    def test_word_count_check_min_max(self):
        c_min = WordCountConstraint(
            id="wc_min", description="at least 3 words", target_field="rationale",
            count=3, mode="min",
        )
        assert c_min.check("one two three four") is None
        result_min = c_min.check("one two")
        assert result_min is not None
        assert "at least 3" in result_min

        c_max = WordCountConstraint(
            id="wc_max", description="at most 3 words", target_field="rationale",
            count=3, mode="max",
        )
        assert c_max.check("one two") is None
        result_max = c_max.check("one two three four")
        assert result_max is not None
        assert "at most 3" in result_max

    # ── CharLimitConstraint ───────────────────────────────────────

    def test_char_limit_check(self):
        c = CharLimitConstraint(
            id="cl1", description="at most 10 chars", target_field="rationale",
            limit=10, mode="max",
        )
        assert c.check("short") is None
        result = c.check("this is way too long")
        assert result is not None
        assert "at most 10" in result

    # ── ItemCountConstraint ───────────────────────────────────────

    def test_item_count_check(self):
        c = ItemCountConstraint(
            id="ic1", description="exactly 3 items", target_field="evidence",
            count=3, mode="exactly", delimiter=",",
        )
        assert c.check("a, b, c") is None
        result = c.check("a, b")
        assert result is not None
        assert "exactly 3" in result

    # ── TokenConstraint ───────────────────────────────────────────

    def test_token_check_contains(self):
        c = TokenConstraint(
            id="tk1", description="must contain VERDICT", target_field="rationale",
            token="VERDICT", must_contain=True,
        )
        assert c.check("The VERDICT is guilty.") is None
        result = c.check("The answer is guilty.")
        assert result is not None
        assert "VERDICT" in result

    def test_token_check_must_not_contain(self):
        c = TokenConstraint(
            id="tk2", description="must not contain TODO", target_field="rationale",
            token="TODO", must_contain=False,
        )
        assert c.check("The answer is complete.") is None
        result = c.check("This is TODO later.")
        assert result is not None
        assert "TODO" in result

    # ── Validation errors ─────────────────────────────────────────

    def test_prefix_suffix_requires_at_least_one(self):
        with pytest.raises(ValidationError, match="At least one"):
            PrefixSuffixConstraint(
                id="ps_bad", description="no prefix or suffix", target_field="rationale",
            )

    def test_word_count_rejects_nonpositive(self):
        with pytest.raises(ValidationError, match="count must be positive"):
            WordCountConstraint(
                id="wc_bad", description="zero count", target_field="rationale",
                count=0, mode="exactly",
            )

    # ── audit_output_constraints ──────────────────────────────────

    def test_audit_catches_duplicate_ids(self):
        constraints = [
            TokenConstraint(id="dup", description="a", target_field="rationale", token="X"),
            TokenConstraint(id="dup", description="b", target_field="rationale", token="Y"),
        ]
        rubric = _make_simple_rubric()
        issues = audit_output_constraints(constraints, rubric)
        assert any("Duplicate" in i and "dup" in i for i in issues)

    def test_audit_catches_unknown_target_field(self):
        constraints = [
            TokenConstraint(id="unk", description="a", target_field="nonexistent_field", token="X"),
        ]
        rubric = _make_simple_rubric()
        issues = audit_output_constraints(constraints, rubric)
        assert any("nonexistent_field" in i and "not a recognized field" in i for i in issues)


# ═════════════════════════════════════════════════════════════════
# 7. Integration: Full Pipeline with Faux Provider
# ═════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end: compile rubric, evaluate with faux provider, verify judgment."""

    @pytest.fixture
    def faux(self):
        from harn_ai.providers.faux import register_faux_provider
        reg = register_faux_provider()
        yield reg
        reg.unregister()

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tool_call(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_tool_call

        rubric = _make_simple_rubric()
        result = compile_rubric(rubric)
        assert result.ok

        # Queue one faux tool-call response per criterion (C1, C2)
        faux.set_responses([
            faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 4,
                    "rationale": "BECAUSE: Good quality.",
                    "evidence": ["quote1"],
                    "violations": [],
                    "criterion_scores": {"C1": 4, "C2": 3},
                }),
                {"stopReason": "toolUse"},
            ),
            faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 3,
                    "rationale": "BECAUSE: Decent.",
                    "evidence": [],
                    "violations": [],
                    "criterion_scores": {"C1": 4, "C2": 3},
                }),
                {"stopReason": "toolUse"},
            ),
        ])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        judgment = await judge.evaluate(result.bundle, "Test response text.")

        assert len(judgment.criterion_judgments) == 2
        assert judgment.aggregation.normalized_score > 0
        assert judgment.decision is not None

    @pytest.mark.asyncio
    async def test_text_fallback_when_no_tool_call(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_text

        rubric = _make_simple_rubric(criteria=[_simple_criterion("C1")])
        result = compile_rubric(rubric)

        json_body = '{"score": 4, "rationale": "solid", "evidence": [], "violations": [], "criterion_scores": {"C1": 4}}'
        faux.set_responses([
            faux_assistant_message(faux_text(json_body), {"stopReason": "stop"}),
        ])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        judgment = await judge.evaluate(result.bundle, "Test response text.")

        assert len(judgment.criterion_judgments) == 1
        cj = judgment.criterion_judgments[0]
        assert cj.criterion_id == "C1"
        assert cj.unit_score > 0

    @pytest.mark.asyncio
    async def test_judge_tracks_usage(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_tool_call

        rubric = _make_simple_rubric(criteria=[_simple_criterion("C1")])
        result = compile_rubric(rubric)

        faux.set_responses([
            faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 5,
                    "rationale": "Excellent.",
                    "evidence": [],
                    "violations": [],
                    "criterion_scores": {"C1": 5},
                }),
                {"stopReason": "toolUse"},
            ),
        ])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        await judge.evaluate(result.bundle, "Response.")

        assert judge.evaluation_count == 1
        assert judge.total_usage.api_calls >= 1

    @pytest.mark.asyncio
    async def test_disqualifier_zeros_score(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_tool_call

        rubric = _make_simple_rubric(
            criteria=[_simple_criterion("C1")],
            disqualifiers=[Disqualifier(id="DQ1", description="plagiarism", pattern=r"copied verbatim")],
        )
        result = compile_rubric(rubric)

        faux.set_responses([
            faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 5,
                    "rationale": "Excellent work.",
                    "evidence": [],
                    "violations": [],
                    "criterion_scores": {"C1": 5},
                }),
                {"stopReason": "toolUse"},
            ),
        ])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        # The response text triggers the disqualifier pattern
        judgment = await judge.evaluate(result.bundle, "This was copied verbatim from Wikipedia.")

        assert judgment.aggregation.normalized_score == 0.0
        assert "DQ1" in judgment.violations
        assert judgment.decision == "Rejected"

    @pytest.mark.asyncio
    async def test_binary_scale_integration(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_tool_call

        rubric = _make_simple_rubric(
            criteria=[Criterion(
                id="B1", title="Pass/Fail Check", description="Binary test",
                scale=BinaryScale(), weight=1.0,
            )],
        )
        result = compile_rubric(rubric)

        faux.set_responses([
            faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 1,
                    "rationale": "Passes.",
                    "evidence": [],
                    "violations": [],
                    "criterion_scores": {"B1": True},
                }),
                {"stopReason": "toolUse"},
            ),
        ])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        judgment = await judge.evaluate(result.bundle, "Test.")

        cj = judgment.criterion_judgments[0]
        assert cj.criterion_id == "B1"
        assert cj.unit_score == 1.0

    @pytest.mark.asyncio
    async def test_multiple_evaluations_accumulate_usage(self, faux):
        from harn_ai.providers.faux import faux_assistant_message, faux_tool_call

        rubric = _make_simple_rubric(criteria=[_simple_criterion("C1")])
        result = compile_rubric(rubric)

        def _make_response():
            return faux_assistant_message(
                faux_tool_call("submit_judgment", {
                    "score": 3, "rationale": "ok",
                    "evidence": [], "violations": [],
                    "criterion_scores": {"C1": 3},
                }),
                {"stopReason": "toolUse"},
            )

        faux.set_responses([_make_response(), _make_response()])

        model = faux.get_model()
        judge = Judge(JudgeConfig(model=model))
        await judge.evaluate(result.bundle, "First.")
        await judge.evaluate(result.bundle, "Second.")

        assert judge.evaluation_count == 2
        assert judge.total_usage.api_calls >= 2
