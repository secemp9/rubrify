"""Test all 4 rubrics against DeepSeek to verify end-to-end functionality."""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; API keys must be in environment

import asyncio
import sys
from pathlib import Path

# Add project root to path so examples.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harn_ai.models import get_model
from rubrify import Judge, JudgeConfig

from examples.compliance_judge import compliance_judge
from examples.anti_slop_judge import anti_slop_judge
from examples.zinsser_judge import zinsser_judge
from examples.completeness_judge import completeness_judge


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

COMPLETE_TEXT = (
    "To make a classic French omelette, follow these steps:\n"
    "Step 1: Crack three large eggs into a bowl and whisk vigorously for "
    "30 seconds until the yolks and whites are fully combined.\n"
    "Step 2: Heat a non-stick pan over medium heat and add one tablespoon "
    "of unsalted butter. Swirl until the butter foams and the foam subsides.\n"
    "Step 3: Pour the egg mixture into the pan. Let it set for 10 seconds, "
    "then use a spatula to push the edges toward the center while tilting "
    "the pan to let uncooked egg flow to the edges.\n"
    "Step 4: When the eggs are just barely set on top (still slightly wet), "
    "fold the omelette in thirds by tilting the pan and rolling it onto "
    "a warm plate.\n"
    "Step 5: Garnish with fresh chives and a pinch of flaky salt. Serve "
    "immediately while the interior remains creamy."
)

INCOMPLETE_TEXT = (
    "To make a classic French omelette:\n"
    "Step 1: Crack three eggs into a bowl.\n"
    "Step 2: TODO: implement this feature\n"
    "Step 3: [...]\n"
    "Step 4: The remaining steps follow the same pattern.\n"
    "# ... rest of recipe omitted for brevity"
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

    # --- 4. Completeness Judge ---
    print("\n" + "=" * 60)
    print("4. COMPLETENESS JUDGE (from completeness_rubric.md)")
    print("=" * 60)
    result = completeness_judge()
    print(f"Compiled: {result.bundle.rubric.meta.name} v{result.bundle.rubric.meta.version}")
    print(f"Criteria: {len(result.bundle.rubric.criteria)}, Issues: {result.issues}")

    await test_rubric("Complete text", result.bundle, judge, COMPLETE_TEXT, "complete")
    await test_rubric("Incomplete text", result.bundle, judge, INCOMPLETE_TEXT, "incomplete")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
