#!/usr/bin/env python3
"""Rubric library: re-export facade for all researcher rubrics.

Each rubric has been split into its own module:
  - compliance_judge.py    -> compliance_judge()
  - zinsser_judge.py       -> zinsser_judge(genre=None)
  - anti_slop_judge.py     -> anti_slop_judge()
  - completeness_judge.py  -> completeness_judge()

This module re-exports all four so existing imports continue to work:

    from examples.rubric_library import compliance_judge, zinsser_judge, anti_slop_judge, completeness_judge
"""

from __future__ import annotations

from examples.compliance_judge import compliance_judge
from examples.zinsser_judge import zinsser_judge
from examples.anti_slop_judge import anti_slop_judge
from examples.completeness_judge import completeness_judge

__all__ = ["compliance_judge", "zinsser_judge", "anti_slop_judge", "completeness_judge"]


# ---------------------------------------------------------------------------
# Quick self-test: compile all rubrics and print summaries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Compiling rubric library...\n")

    # A. ComplianceJudge
    cj = compliance_judge()
    b = cj.bundle
    print(f"[A] {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"    Criteria: {len(b.rubric.criteria)}, DQ: {len(b.rubric.disqualifiers)}, "
          f"Patterns: {len(b.rubric.patterns)}")
    print(f"    Locked: {b.locked}")
    print(f"    Issues: {cj.issues or '(none)'}")
    print()

    # B. ZinsserJudge (no genre)
    zj = zinsser_judge()
    b = zj.bundle
    active_core = [c for c in b.rubric.criteria if c.genre is None]
    genre_crit = [c for c in b.rubric.criteria if c.genre is not None]
    print(f"[B] {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"    Core criteria: {len(active_core)}, Genre criteria: {len(genre_crit)}, "
          f"DQ: {len(b.rubric.disqualifiers)}, Patterns: {len(b.rubric.patterns)}")
    print(f"    Groups: {[g.id for g in b.rubric.groups]}")
    print(f"    Locked: {b.locked}")
    print(f"    Issues: {zj.issues or '(none)'}")
    print()

    # B2. ZinsserJudge with genre
    zj2 = zinsser_judge(genre="travel")
    print(f"    (with genre='travel') Locked: {zj2.bundle.locked}")
    print()

    # C. AntiLLMY
    aj = anti_slop_judge()
    b = aj.bundle
    print(f"[C] {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"    Criteria: {len(b.rubric.criteria)}, DQ: {len(b.rubric.disqualifiers)}, "
          f"Patterns: {len(b.rubric.patterns)}")
    print(f"    Locked: {b.locked}")
    print(f"    Issues: {aj.issues or '(none)'}")
    print()

    # D. CompletenessJudge
    coj = completeness_judge()
    b = coj.bundle
    print(f"[D] {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"    Criteria: {len(b.rubric.criteria)}, DQ: {len(b.rubric.disqualifiers)}, "
          f"Patterns: {len(b.rubric.patterns)}")
    print(f"    Locked: {b.locked}")
    print(f"    Issues: {coj.issues or '(none)'}")
    print()

    print("All rubrics compiled successfully.")
