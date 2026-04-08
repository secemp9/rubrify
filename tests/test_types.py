"""Tests for kernel primitive dataclasses."""

from rubrify._types import (
    AdviceRule,
    Criterion,
    DecisionRule,
    Disqualifier,
    ICLExample,
    InputField,
    Instruction,
    MappingExample,
    OutputSchema,
    PatternLibrary,
    Scoring,
    ValidationMust,
)


class TestCriterion:
    def test_minimal(self) -> None:
        c = Criterion(id="C1", name="Test")
        assert c.id == "C1"
        assert c.name == "Test"
        assert c.weight == 0
        assert c.anchors == {}
        assert c.mechanical_rules == []
        assert c.uses_patterns is None
        assert c.genre is None
        assert c.notes is None

    def test_full(self) -> None:
        c = Criterion(
            id="C1",
            name="Clarity",
            weight=13,
            anchors={0: "bad", 5: "good"},
            mechanical_rules=["rule1"],
            uses_patterns=["puffery"],
            genre="science_tech",
            notes="some notes",
        )
        assert c.weight == 13
        assert c.anchors[0] == "bad"
        assert c.uses_patterns == ["puffery"]
        assert c.genre == "science_tech"

    def test_scale_property_empty(self) -> None:
        c = Criterion(id="C1", name="Test")
        assert c.scale == (0, 0)

    def test_scale_property(self) -> None:
        c = Criterion(id="C1", name="Test", anchors={0: "a", 3: "b", 5: "c"})
        assert c.scale == (0, 5)

    def test_slots(self) -> None:
        c = Criterion(id="C1", name="Test")
        assert not hasattr(c, "__dict__")


class TestDisqualifier:
    def test_construction(self) -> None:
        dq = Disqualifier(id="DQ1", description="Test failure")
        assert dq.id == "DQ1"
        assert dq.description == "Test failure"

    def test_slots(self) -> None:
        dq = Disqualifier(id="DQ1", description="x")
        assert not hasattr(dq, "__dict__")


class TestOutputSchema:
    def test_defaults(self) -> None:
        os = OutputSchema()
        assert os.format == "json"
        assert os.template == ""
        assert os.constraints == {}

    def test_full(self) -> None:
        os = OutputSchema(
            format="xml",
            template="<test/>",
            constraints={"must_use_xml_tags": True},
        )
        assert os.format == "xml"
        assert os.constraints["must_use_xml_tags"] is True


class TestScoring:
    def test_defaults(self) -> None:
        s = Scoring()
        assert s.formula == ""
        assert s.labels == {}
        assert s.inverted is False

    def test_weighted_sum_helper(self) -> None:
        result = Scoring.weighted_sum(["C1", "C2", "C3"])
        assert "C1, C2, C3" in result
        assert "DQ" in result

    def test_inverted_sum_helper(self) -> None:
        result = Scoring.inverted_sum(["C1", "C2"], 10)
        assert "C1+C2" in result
        assert "risk" in result


class TestPatternLibrary:
    def test_defaults(self) -> None:
        pl = PatternLibrary()
        assert pl.entries == {}
        assert pl.flags == ""
        assert pl._entry_types == {}

    def test_add(self) -> None:
        pl = PatternLibrary()
        pl.add("test_pat", r"\btest\b")
        assert pl.entries["test_pat"] == r"\btest\b"
        assert pl._entry_types["test_pat"] == "regex"

    def test_add_list(self) -> None:
        pl = PatternLibrary()
        pl.add_list("hedges", "very|quite|rather")
        assert pl.entries["hedges"] == "very|quite|rather"
        assert pl._entry_types["hedges"] == "list"


class TestDecisionRule:
    def test_construction(self) -> None:
        r = DecisionRule(id="R1", condition="If any DQ => No")
        assert r.id == "R1"
        assert r.condition == "If any DQ => No"


class TestAdviceRule:
    def test_construction(self) -> None:
        r = AdviceRule(when=["puffery", "weasel"], advice="Fix it")
        assert r.when == ["puffery", "weasel"]
        assert r.advice == "Fix it"


class TestMappingExample:
    def test_minimal(self) -> None:
        e = MappingExample(id="E1")
        assert e.id == "E1"
        assert e.user is None
        assert e.assistant is None
        assert e.verdict == ""

    def test_full(self) -> None:
        e = MappingExample(id="E1", user="test", assistant="resp", verdict="Yes")
        assert e.user == "test"
        assert e.verdict == "Yes"


class TestValidationMust:
    def test_construction(self) -> None:
        v = ValidationMust(description="Output JSON only")
        assert v.description == "Output JSON only"


class TestInputField:
    def test_defaults(self) -> None:
        f = InputField(name="candidate_text")
        assert f.name == "candidate_text"
        assert f.required is False
        assert f.description == ""


class TestInstruction:
    def test_construction(self) -> None:
        i = Instruction(text="Do something")
        assert i.text == "Do something"


class TestICLExample:
    def test_construction(self) -> None:
        e = ICLExample(input="in", output="out")
        assert e.input == "in"
        assert e.output == "out"
