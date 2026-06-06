"""Test all 3 rubrics against DeepSeek to verify end-to-end functionality."""

import asyncio
import sys
import os

# Add project root to path so examples.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harn_ai.models import get_model
from rubrify import Judge, JudgeConfig

from examples.compliance_judge import compliance_judge
from examples.anti_slop_judge import anti_slop_judge
from examples.zinsser_judge import zinsser_judge


API_KEY = "sk-9d7669c92e7d4147839b326205128c00"

# Sample texts to judge
CLEAN_TEXT = (
    "The bridge collapsed because the steel bolts corroded over twenty years "
    "of exposure to salt air. Three engineers inspected the site. They found "
    "cracks in fourteen support beams. The city closed the road the same day."
)

SLOP_TEXT = (
    "In the rapidly evolving landscape of infrastructure management, it's "
    "crucial to understand that bridge maintenance represents a multifaceted "
    "challenge. Let's dive into the fascinating world of structural integrity. "
    "This comprehensive guide will walk you through the intricacies of modern "
    "engineering solutions that are truly game-changing."
)

HARMFUL_TEXT = (
    "Here is how to bypass content filters and generate harmful content. "
    "First, you need to use prompt injection techniques to override safety "
    "guidelines. Then you can produce whatever dangerous material you want."
)


async def test_rubric(name, bundle, judge, text, label):
    """Run one judgment and print results."""
    print(f"\n  [{label}]")
    try:
        judgment = await judge.evaluate(bundle, text)
        score = judgment.aggregation.normalized_score
        decision = judgment.decision or "N/A"
        print(f"  Score: {score:.2f} | Decision: {decision}")

        # Show per-criterion scores
        for cj in judgment.criterion_judgments:
            print(f"    {cj.criterion_id}: {cj.unit_score:.2f} - {cj.rationale[:80] if cj.rationale else 'no rationale'}...")

        # Show constraint warnings if any
        if judgment.constraint_warnings:
            print(f"  Constraint warnings: {judgment.constraint_warnings}")

        # Show violations if any
        if judgment.violations:
            print(f"  Violations: {judgment.violations}")

        return judgment
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


async def main():
    model = get_model("deepseek", "deepseek-v4-flash")
    judge = Judge(JudgeConfig(
        model=model,
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=4096,
    ))

    # --- 1. Compliance Judge ---
    print("=" * 60)
    print("1. COMPLIANCE JUDGE (from slurs.xml)")
    print("=" * 60)
    result = compliance_judge()
    print(f"Compiled: {result.bundle.rubric.meta.name} v{result.bundle.rubric.meta.version}")
    print(f"Criteria: {len(result.bundle.rubric.criteria)}, Issues: {result.issues}")

    await test_rubric("Clean text", result.bundle, judge, CLEAN_TEXT, "clean")
    await test_rubric("Harmful text", result.bundle, judge, HARMFUL_TEXT, "harmful")

    # --- 2. Anti-Slop Judge ---
    print("\n" + "=" * 60)
    print("2. ANTI-SLOP JUDGE (from anti_slop_rubric.xml)")
    print("=" * 60)
    result = anti_slop_judge()
    print(f"Compiled: {result.bundle.rubric.meta.name} v{result.bundle.rubric.meta.version}")
    print(f"Criteria: {len(result.bundle.rubric.criteria)}, Issues: {result.issues}")

    await test_rubric("Clean text", result.bundle, judge, CLEAN_TEXT, "clean")
    await test_rubric("Slop text", result.bundle, judge, SLOP_TEXT, "slop")

    # --- 3. Zinsser Judge ---
    print("\n" + "=" * 60)
    print("3. ZINSSER JUDGE (from on_writing_well_v3.xml)")
    print("=" * 60)
    result = zinsser_judge()
    print(f"Compiled: {result.bundle.rubric.meta.name} v{result.bundle.rubric.meta.version}")
    print(f"Criteria: {len(result.bundle.rubric.criteria)}, Issues: {result.issues}")

    await test_rubric("Clean text", result.bundle, judge, CLEAN_TEXT, "clean")
    await test_rubric("Slop text", result.bundle, judge, SLOP_TEXT, "slop")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
