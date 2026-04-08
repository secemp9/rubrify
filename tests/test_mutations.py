"""Tests for mutation dataclasses and evolve()."""

import pytest

import rubrify
from rubrify._mutations import (
    AddCriterion,
    AddDisqualifier,
    AddPattern,
    AdjustWeight,
    RemoveCriterion,
    _bump_version,
)
from rubrify._types import Criterion, Disqualifier


def _simple_rubric() -> rubrify.Rubric:
    """Create a simple rubric for mutation testing."""
    r = rubrify.Rubric(name="Test", version="1.0", mission="Test mission.")
    r.add_criterion(Criterion(id="C1", name="Clarity", weight=50, anchors={0: "bad", 5: "good"}))
    r.add_criterion(Criterion(id="C2", name="Brevity", weight=50, anchors={0: "bad", 5: "good"}))
    r.add_disqualifier(Disqualifier(id="DQ1", description="Plagiarism"))
    return r


class TestBumpVersion:
    def test_bump_1_0(self) -> None:
        assert _bump_version("1.0") == "2.0"

    def test_bump_3_0(self) -> None:
        assert _bump_version("3.0") == "4.0"

    def test_bump_10_0(self) -> None:
        assert _bump_version("10.0") == "11.0"


class TestAddCriterion:
    def test_adds_criterion(self) -> None:
        r = _simple_rubric()
        new_crit = Criterion(id="C3", name="Style", weight=10)
        m = AddCriterion(criterion=new_crit)
        m.apply(r)
        assert "C3" in r.criteria
        assert r.criteria["C3"].name == "Style"

    def test_overwrites_existing(self) -> None:
        r = _simple_rubric()
        new_crit = Criterion(id="C1", name="Replaced", weight=99)
        m = AddCriterion(criterion=new_crit)
        m.apply(r)
        assert r.criteria["C1"].name == "Replaced"
        assert r.criteria["C1"].weight == 99


class TestRemoveCriterion:
    def test_removes_criterion(self) -> None:
        r = _simple_rubric()
        m = RemoveCriterion(criterion_id="C1")
        m.apply(r)
        assert "C1" not in r.criteria
        assert "C2" in r.criteria

    def test_missing_raises(self) -> None:
        r = _simple_rubric()
        m = RemoveCriterion(criterion_id="C99")
        with pytest.raises(KeyError, match="C99"):
            m.apply(r)


class TestAdjustWeight:
    def test_adjusts_weight(self) -> None:
        r = _simple_rubric()
        m = AdjustWeight(criterion_id="C1", new_weight=75)
        m.apply(r)
        assert r.criteria["C1"].weight == 75

    def test_missing_raises(self) -> None:
        r = _simple_rubric()
        m = AdjustWeight(criterion_id="C99", new_weight=10)
        with pytest.raises(KeyError, match="C99"):
            m.apply(r)


class TestAddPattern:
    def test_adds_pattern_creates_library(self) -> None:
        r = _simple_rubric()
        assert r.pattern_library is None
        m = AddPattern(pattern_id="hedges", pattern=r"\b(very|quite)\b")
        m.apply(r)
        assert r.pattern_library is not None
        assert "hedges" in r.pattern_library.entries

    def test_adds_pattern_to_existing_library(self) -> None:
        r = _simple_rubric()
        r.pattern_library = rubrify.PatternLibrary()
        r.pattern_library.add("existing", r"\btest\b")
        m = AddPattern(pattern_id="new_pat", pattern=r"\bfoo\b")
        m.apply(r)
        assert "existing" in r.pattern_library.entries
        assert "new_pat" in r.pattern_library.entries


class TestAddDisqualifier:
    def test_appends_dq(self) -> None:
        r = _simple_rubric()
        assert len(r.disqualifiers) == 1
        m = AddDisqualifier(disqualifier=Disqualifier(id="DQ2", description="Hate speech"))
        m.apply(r)
        assert len(r.disqualifiers) == 2
        assert r.disqualifiers[1].id == "DQ2"


class TestEvolve:
    def test_evolve_applies_mutations_and_bumps_version(self) -> None:
        r = _simple_rubric()
        mutations = [
            AddCriterion(criterion=Criterion(id="C3", name="Style", weight=10)),
            AdjustWeight(criterion_id="C1", new_weight=40),
            AddDisqualifier(disqualifier=Disqualifier(id="DQ2", description="Spam")),
        ]
        evolved = r.evolve(mutations)

        # Original unchanged
        assert r.version == "1.0"
        assert len(r.criteria) == 2
        assert r.criteria["C1"].weight == 50

        # Evolved has mutations applied
        assert evolved.version == "2.0"
        assert len(evolved.criteria) == 3
        assert evolved.criteria["C1"].weight == 40
        assert "C3" in evolved.criteria
        assert len(evolved.disqualifiers) == 2

    def test_evolve_remove_and_add(self) -> None:
        r = _simple_rubric()
        mutations = [
            RemoveCriterion(criterion_id="C2"),
            AddCriterion(criterion=Criterion(id="C2_new", name="New Brevity", weight=50)),
        ]
        evolved = r.evolve(mutations)
        assert "C2" not in evolved.criteria
        assert "C2_new" in evolved.criteria
        assert evolved.version == "2.0"

    def test_evolve_with_pattern(self) -> None:
        r = _simple_rubric()
        mutations = [
            AddPattern(pattern_id="hedges", pattern=r"\b(very|quite)\b"),
        ]
        evolved = r.evolve(mutations)
        assert evolved.pattern_library is not None
        assert "hedges" in evolved.pattern_library.entries
        # Original still has no pattern library
        assert r.pattern_library is None

    def test_double_evolve(self) -> None:
        r = _simple_rubric()
        v2 = r.evolve([AddCriterion(criterion=Criterion(id="C3", name="X", weight=5))])
        v3 = v2.evolve([AdjustWeight(criterion_id="C3", new_weight=15)])
        assert r.version == "1.0"
        assert v2.version == "2.0"
        assert v3.version == "3.0"
        assert v3.criteria["C3"].weight == 15
