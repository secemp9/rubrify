#!/usr/bin/env python3
"""Debug script: trace the EXACT constraint enforcement path.

Goal: For each criterion judgment, trace:
  LLM response -> rationale extraction -> word count -> constraint check ->
  hard violation -> score zeroing

Answers the question: Is the 35-word constraint correctly enforced, or is
there a design bug causing it to zero all scores?
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harn_ai.models import get_model

from examples.anti_slop_judge import anti_slop_judge
from rubrify import Judge, JudgeConfig
from rubrify.ir.constraints import WordCountConstraint, PrefixSuffixConstraint
from rubrify.engine.judge_loop import (
    _find_constraint_targets,
    _verify_output_constraints,
)

# ── Configuration ──────────────────────────────────────────────────
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("Set DEEPSEEK_API_KEY environment variable")
RESPONSE_TEXT = (
    "The bridge collapsed because the steel bolts corroded over twenty years "
    "of exposure to salt air."
)


def sep(title: str) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {title}")
    print(f"{'=' * 74}")


async def main():
    # ── Step 1: Compile rubric ────────────────────────────────────
    sep("1. COMPILE RUBRIC")
    result = anti_slop_judge()
    bundle = result.bundle
    print(f"Rubric:          {bundle.rubric.meta.name}")
    print(f"Criteria count:  {len(bundle.rubric.criteria)}")
    print(f"Output constraints: {len(bundle.output_constraints)}")
    for oc in bundle.output_constraints:
        print(f"  [{oc.id}] kind={oc.kind}, target={oc.target_field}, "
              f"enforcement={oc.enforcement}")
        if isinstance(oc, WordCountConstraint):
            print(f"       count={oc.count}, mode={oc.mode}")
        elif isinstance(oc, PrefixSuffixConstraint):
            print(f"       prefix={oc.prefix!r}, suffix={oc.suffix!r}")

    # ── Step 2: Run the judge (ONE real DeepSeek call per criterion) ─
    sep("2. RUN JUDGE (DeepSeek)")
    model = get_model("deepseek", "deepseek-v4-flash")
    if model is None:
        print("FATAL: model not found")
        return

    judge = Judge(JudgeConfig(
        model=model,
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=2048,
        parallel=False,
        use_tool=True,
    ))

    print(f"Model: {model.id} (provider={model.provider})")
    print(f"Response text: {RESPONSE_TEXT!r}")
    print(f"Calling judge.evaluate()...")

    judgment = await judge.evaluate(bundle, RESPONSE_TEXT)

    # ── Step 3: Inspect each criterion judgment ───────────────────
    sep("3. PER-CRITERION RATIONALE INSPECTION")
    for cj in judgment.criterion_judgments:
        print(f"\n--- Criterion {cj.criterion_id} ---")
        print(f"  value:      {cj.value}")
        print(f"  unit_score: {cj.unit_score}")
        print(f"  rationale:  {cj.rationale!r}")
        print(f"  warnings:   {cj.warnings}")

        # Manual word count
        if cj.rationale:
            words = cj.rationale.split()
            print(f"  word_count (len(split())): {len(words)}")
            print(f"  words: {words}")
        else:
            print(f"  word_count: 0 (empty rationale)")

    # ── Step 4: Trace _find_constraint_targets ────────────────────
    sep("4. _find_constraint_targets() OUTPUT")
    rationale_targets = _find_constraint_targets(
        judgment.criterion_judgments, "rationale"
    )
    print(f"Number of 'rationale' targets returned: {len(rationale_targets)}")
    print(f"(This is the number of non-empty rationales across ALL criteria)")
    for i, t in enumerate(rationale_targets):
        wc = len(t.split())
        print(f"\n  Target [{i}]: word_count={wc}")
        print(f"    text: {t!r}")

    # ── Step 5: Manually run each constraint.check() ──────────────
    sep("5. MANUAL CONSTRAINT CHECK ON EACH TARGET")
    for oc in bundle.output_constraints:
        print(f"\n  Constraint: {oc.id} (kind={oc.kind}, enforcement={oc.enforcement})")
        for i, target in enumerate(rationale_targets):
            failure_msg = oc.check(target)
            if failure_msg:
                print(f"    Target [{i}]: FAIL -> {failure_msg}")
            else:
                print(f"    Target [{i}]: PASS")

    # ── Step 6: Trace _verify_output_constraints ──────────────────
    sep("6. _verify_output_constraints() OUTPUT")
    hard_violations, soft_warnings = _verify_output_constraints(
        bundle, judgment.criterion_judgments
    )
    print(f"hard_violations: {hard_violations}")
    print(f"soft_warnings:   {soft_warnings}")

    # ── Step 7: Show the actual judgment violations and score ──────
    sep("7. FINAL JUDGMENT STATE")
    print(f"violations:           {judgment.violations}")
    print(f"constraint_warnings:  {judgment.constraint_warnings}")
    print(f"aggregation.raw_score:        {judgment.aggregation.raw_score}")
    print(f"aggregation.normalized_score: {judgment.aggregation.normalized_score}")
    print(f"aggregation.method:           {judgment.aggregation.method}")
    print(f"decision:                     {judgment.decision}")

    # ── Step 8: ANALYSIS ──────────────────────────────────────────
    sep("8. ANALYSIS: THE CONSTRAINT MULTIPLICATION PROBLEM")
    n_criteria = len(judgment.criterion_judgments)
    n_nonempty = len(rationale_targets)
    print(f"Number of criteria:           {n_criteria}")
    print(f"Number of non-empty rationales: {n_nonempty}")
    print()

    # Count how many rationales actually have exactly 35 words
    exactly_35 = sum(1 for t in rationale_targets if len(t.split()) == 35)
    not_35 = sum(1 for t in rationale_targets if len(t.split()) != 35)
    print(f"Rationales with exactly 35 words: {exactly_35}")
    print(f"Rationales WITHOUT 35 words:      {not_35}")
    print()

    # The core question
    print("KEY FINDING:")
    print(f"  The WordCountConstraint(target_field='rationale', mode='exactly', count=35)")
    print(f"  is checked against EVERY criterion's rationale ({n_nonempty} targets).")
    print(f"  Each criterion produces its OWN rationale (per-criterion LLM call).")
    print()
    if not_35 > 0:
        print(f"  {not_35} out of {n_nonempty} rationales failed the 35-word check.")
        print(f"  Since enforcement='hard', EACH failure adds a hard violation.")
        print(f"  Total hard violations from word count alone: {sum(1 for v in hard_violations if 'rationale_35_words' in v)}")
        print()
        print(f"  When ANY violation exists, _aggregate() returns score=0.0 (disqualified).")
        print(f"  This means: if even ONE criterion's rationale != 35 words,")
        print(f"  the ENTIRE judgment is zeroed. And with {n_criteria} separate LLM calls,")
        print(f"  each producing a free-form rationale, hitting EXACTLY 35 words")
        print(f"  on EVERY single one is extremely unlikely.")
    else:
        print(f"  All {n_nonempty} rationales have exactly 35 words!")
        print(f"  The constraint is passing. Something else is causing the issue.")


if __name__ == "__main__":
    asyncio.run(main())
