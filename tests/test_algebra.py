"""Tests for rubric algebra operations: |, &, project(), reweight(), and algebraic laws."""

import rubrify
from rubrify._mutations import AddCriterion
from rubrify._types import Criterion, Disqualifier, PatternLibrary


def _rubric_a() -> rubrify.Rubric:
    r = rubrify.Rubric(name="A", version="1.0", mission="Mission A.")
    r.add_criterion(Criterion(id="C1", name="Clarity", weight=50, anchors={0: "bad", 5: "good"}))
    r.add_criterion(Criterion(id="C2", name="Brevity", weight=50, anchors={0: "bad", 5: "good"}))
    r.add_disqualifier(Disqualifier(id="DQ1", description="Plagiarism"))
    r.pattern_library = PatternLibrary()
    r.pattern_library.add("hedges", r"\b(very|quite)\b")
    return r


def _rubric_b() -> rubrify.Rubric:
    r = rubrify.Rubric(name="B", version="1.0", mission="Mission B.")
    r.add_criterion(Criterion(id="C2", name="Conciseness", weight=60, anchors={0: "x", 3: "y"}))
    r.add_criterion(Criterion(id="C3", name="Style", weight=40, anchors={0: "x", 5: "y"}))
    r.add_disqualifier(Disqualifier(id="DQ1", description="Plagiarism"))
    r.add_disqualifier(Disqualifier(id="DQ2", description="Spam"))
    r.pattern_library = PatternLibrary()
    r.pattern_library.add("puffery", r"\b(amazing|incredible)\b")
    return r


class TestUnion:
    def test_criteria_union_second_wins(self) -> None:
        a = _rubric_a()
        b = _rubric_b()
        merged = a | b

        # C1 from A preserved, C2 replaced by B's version, C3 from B added
        assert len(merged.criteria) == 3
        assert merged.criteria["C1"].name == "Clarity"
        assert merged.criteria["C2"].name == "Conciseness"  # B wins
        assert merged.criteria["C2"].weight == 60
        assert merged.criteria["C3"].name == "Style"

    def test_dq_union_deduplicates(self) -> None:
        a = _rubric_a()
        b = _rubric_b()
        merged = a | b

        dq_ids = [dq.id for dq in merged.disqualifiers]
        assert "DQ1" in dq_ids
        assert "DQ2" in dq_ids
        assert dq_ids.count("DQ1") == 1  # No duplicates

    def test_mission_concatenation(self) -> None:
        a = _rubric_a()
        b = _rubric_b()
        merged = a | b
        assert "Mission A." in merged.mission
        assert "Mission B." in merged.mission

    def test_pattern_library_merge(self) -> None:
        a = _rubric_a()
        b = _rubric_b()
        merged = a | b
        assert merged.pattern_library is not None
        assert "hedges" in merged.pattern_library.entries
        assert "puffery" in merged.pattern_library.entries

    def test_union_does_not_mutate_originals(self) -> None:
        a = _rubric_a()
        b = _rubric_b()
        _ = a | b
        assert len(a.criteria) == 2
        assert len(b.criteria) == 2

    def test_union_with_no_pattern_library(self) -> None:
        a = rubrify.Rubric(name="A", mission="A.")
        a.add_criterion(Criterion(id="C1", name="X", weight=100))
        b = rubrify.Rubric(name="B", mission="B.")
        b.add_criterion(Criterion(id="C2", name="Y", weight=100))
        merged = a | b
        assert merged.pattern_library is None
        assert len(merged.criteria) == 2

    def test_union_one_has_pattern_library(self) -> None:
        a = rubrify.Rubric(name="A", mission="A.")
        b = _rubric_b()
        merged = a | b
        assert merged.pattern_library is not None
        assert "puffery" in merged.pattern_library.entries


class TestParallelEvaluate:
    def test_evaluate_parallel_is_callable(self) -> None:
        assert callable(rubrify.evaluate_parallel)
        assert callable(rubrify.evaluate_conditional)


class TestProject:
    def test_keeps_specified_criteria(self) -> None:
        a = _rubric_a()
        projected = a.project({"C1"})
        assert "C1" in projected.criteria
        assert "C2" not in projected.criteria

    def test_nonexistent_ids_ignored(self) -> None:
        a = _rubric_a()
        projected = a.project({"C1", "C99"})
        assert len(projected.criteria) == 1
        assert "C1" in projected.criteria

    def test_empty_set_removes_all(self) -> None:
        a = _rubric_a()
        projected = a.project(set())
        assert len(projected.criteria) == 0

    def test_preserves_other_fields(self) -> None:
        a = _rubric_a()
        projected = a.project({"C1"})
        assert projected.mission == a.mission
        assert len(projected.disqualifiers) == len(a.disqualifiers)
        assert projected.pattern_library is not None

    def test_does_not_mutate_original(self) -> None:
        a = _rubric_a()
        _ = a.project({"C1"})
        assert len(a.criteria) == 2


class TestReweight:
    def test_reweight_changes_specified(self) -> None:
        a = _rubric_a()
        reweighted = a.reweight({"C1": 20})
        assert reweighted.criteria["C1"].weight == 20
        assert reweighted.criteria["C2"].weight == 50  # Unchanged

    def test_reweight_unknown_key_ignored(self) -> None:
        a = _rubric_a()
        reweighted = a.reweight({"C99": 10})
        assert len(reweighted.criteria) == 2

    def test_reweight_does_not_mutate_original(self) -> None:
        a = _rubric_a()
        _ = a.reweight({"C1": 20})
        assert a.criteria["C1"].weight == 50

    def test_reweight_multiple(self) -> None:
        a = _rubric_a()
        reweighted = a.reweight({"C1": 30, "C2": 70})
        assert reweighted.criteria["C1"].weight == 30
        assert reweighted.criteria["C2"].weight == 70


class TestCopyIndependence:
    def test_copy_criteria_independent(self) -> None:
        a = _rubric_a()
        c = a.copy()
        c.criteria["C1"].weight = 999
        assert a.criteria["C1"].weight == 50

    def test_copy_disqualifiers_independent(self) -> None:
        a = _rubric_a()
        c = a.copy()
        c.disqualifiers.append(Disqualifier(id="DQ2", description="New"))
        assert len(a.disqualifiers) == 1

    def test_copy_pattern_library_independent(self) -> None:
        a = _rubric_a()
        c = a.copy()
        assert c.pattern_library is not None
        c.pattern_library.add("new_pat", "test")
        assert "new_pat" not in a.pattern_library.entries  # type: ignore[union-attr]


def _rubric_c() -> rubrify.Rubric:
    r = rubrify.Rubric(name="C", version="1.0", mission="Mission C.")
    r.add_criterion(Criterion(id="C3", name="Depth", weight=30, anchors={0: "x", 5: "y"}))
    r.add_criterion(Criterion(id="C4", name="Accuracy", weight=70, anchors={0: "x", 5: "y"}))
    return r


class TestAlgebraicLaws:
    def test_criteria_union_associativity(self) -> None:
        """|(union) is associative: (r1|r2)|r3 has same criteria keys as r1|(r2|r3)."""
        r1, r2, r3 = _rubric_a(), _rubric_b(), _rubric_c()
        left = (r1 | r2) | r3
        right = r1 | (r2 | r3)
        assert set(left.criteria.keys()) == set(right.criteria.keys())

    def test_project_idempotent(self) -> None:
        """project is idempotent: r.project(S).project(S) == r.project(S)."""
        r = _rubric_a()
        ids = {"C1", "C2"}
        once = r.project(ids)
        twice = once.project(ids)
        assert set(once.criteria.keys()) == set(twice.criteria.keys())

    def test_evolve_version_monotonic(self) -> None:
        """evolve always bumps version."""
        r = _rubric_a()
        mutations = [AddCriterion(Criterion(id="C9", name="New", weight=10))]
        evolved = r.evolve(mutations)
        assert float(evolved.version) > float(r.version)
