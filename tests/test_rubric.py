"""Tests for Rubric class (unified: scoring + constraint)."""

import rubrify
from rubrify._types import ICLExample


class TestRubricConstruction:
    def test_defaults(self) -> None:
        r = rubrify.Rubric()
        assert r.name == ""
        assert r.version == "1.0"
        assert r.mission == ""
        assert r.criteria == {}
        assert r.disqualifiers == []

    def test_with_args(self) -> None:
        r = rubrify.Rubric(name="Test", version="2.0", mission="Do something")
        assert r.name == "Test"
        assert r.version == "2.0"
        assert r.mission == "Do something"

    def test_repr(self) -> None:
        r = rubrify.Rubric(name="Test", version="1.0")
        assert "Test" in repr(r)
        assert "criteria=0" not in repr(r) or "examples=0" in repr(r)


class TestAddCriterion:
    def test_add_criterion(self) -> None:
        r = rubrify.Rubric()
        c = rubrify.Criterion(id="C1", name="Clarity", weight=10)
        r.add_criterion(c)
        assert "C1" in r.criteria
        assert r.criteria["C1"].name == "Clarity"

    def test_add_disqualifier(self) -> None:
        r = rubrify.Rubric()
        dq = rubrify.Disqualifier(id="DQ1", description="Bad thing")
        r.add_disqualifier(dq)
        assert len(r.disqualifiers) == 1
        assert r.disqualifiers[0].id == "DQ1"


class TestOutputFormat:
    def test_json_format(self) -> None:
        r = rubrify.Rubric()
        r.output_schema = rubrify.OutputSchema(
            constraints={"must_be_json": True},
        )
        assert r.output_format == "json"

    def test_xml_format(self) -> None:
        r = rubrify.Rubric()
        r.output_schema = rubrify.OutputSchema(
            constraints={"must_use_xml_tags": True},
        )
        assert r.output_format == "xml"

    def test_unknown_format(self) -> None:
        r = rubrify.Rubric()
        assert r.output_format == "unknown"

    def test_default_format(self) -> None:
        r = rubrify.Rubric()
        r.output_schema = rubrify.OutputSchema()
        assert r.output_format == "json"


class TestGenreCriteria:
    def test_genre_filter(self) -> None:
        r = rubrify.Rubric()
        r.add_criterion(rubrify.Criterion(id="C1", name="General", weight=10))
        r.add_criterion(
            rubrify.Criterion(id="G_SCI", name="Science", weight=5, genre="science_tech")
        )
        r.add_criterion(
            rubrify.Criterion(id="G_BUS", name="Business", weight=5, genre="business,email")
        )

        sci = r.genre_criteria("science_tech")
        assert len(sci) == 1
        assert sci[0].id == "G_SCI"

        bus = r.genre_criteria("business")
        assert len(bus) == 1
        assert bus[0].id == "G_BUS"

        email = r.genre_criteria("email")
        assert len(email) == 1
        assert email[0].id == "G_BUS"

        gen = r.genre_criteria("general")
        assert len(gen) == 0


class TestCopy:
    def test_copy_independence(self) -> None:
        r = rubrify.Rubric(name="Original", mission="test")
        r.add_criterion(rubrify.Criterion(id="C1", name="Test", weight=10))

        c = r.copy()
        c.name = "Copy"
        c.criteria["C1"].weight = 99

        assert r.name == "Original"
        assert r.criteria["C1"].weight == 10


class TestConstraintStyleRubric:
    def test_construction(self) -> None:
        cr = rubrify.Rubric(
            name="TestGen",
            instructions="Generate rubric",
            output_format="<LLM_JUDGE_SPEC>",
        )
        assert cr.name == "TestGen"
        assert cr.instructions == "Generate rubric"
        assert cr.examples == []

    def test_with_examples(self) -> None:
        cr = rubrify.Rubric(
            name="TestGen",
            examples=[ICLExample(input="in", output="out")],
        )
        assert len(cr.examples) == 1

    def test_repr(self) -> None:
        cr = rubrify.Rubric(name="TestGen")
        assert "TestGen" in repr(cr)
