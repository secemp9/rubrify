"""Example: rubrify-powered Prime Intellect verifiers environment.

Uses the AntiLLMY rubric as a reward signal for RL training.
The environment evaluates writing quality -- clean prose scores high,
LLM-y slop scores low.

Usage:
    # Evaluate only (no training):
    DEEPSEEK_API_KEY=sk-... python examples/verifiers_env_example.py

    # With verifiers CLI:
    vf eval environments/anti_slop_writing --model deepseek/deepseek-v4-flash
"""

# ---------------------------------------------------------------------------
# Optional: load .env for API keys
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; API keys must be in environment

import asyncio
import sys
from pathlib import Path

# Ensure examples directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harn_ai.models import get_model
from rubrify import Judge, JudgeConfig
from rubrify.bridge.verifiers import make_rubrify_rubric

from examples.anti_slop_judge import anti_slop_judge

# ---------------------------------------------------------------------------
# Hardcoded writing-prompt dataset (mix of clean and slop)
# ---------------------------------------------------------------------------
WRITING_SAMPLES = [
    # --- Clean prose: factual, concrete, no puffery ---
    {
        "prompt": "Write a paragraph about the collapse of the Morandi Bridge.",
        "answer": "clean",
        "text": (
            "The Morandi Bridge in Genoa collapsed on August 14, 2018, killing "
            "43 people. A 200-meter section of the deck fell onto a railway line, "
            "warehouse buildings, and the Polcevera river bed. Investigators found "
            "that the stay cables on pylon 9 had corroded far beyond the threshold "
            "assumed in the 1967 design. Autostrade per l'Italia, the private "
            "concessionaire, had deferred structural reinforcements three times "
            "between 2009 and 2017. The replacement span, designed by Renzo Piano, "
            "opened to traffic on August 3, 2020."
        ),
    },
    # --- Moderate slop: some puffery, formulaic transitions ---
    {
        "prompt": "Write a paragraph about renewable energy trends.",
        "answer": "moderate_slop",
        "text": (
            "In the rapidly evolving landscape of renewable energy, it's important "
            "to note that solar and wind power have emerged as truly game-changing "
            "technologies. Moreover, the stunning growth of battery storage "
            "solutions serves as a testament to human ingenuity. Furthermore, "
            "many believe that the transition to clean energy will create "
            "unprecedented opportunities. In summary, the future of renewable "
            "energy is breathtaking and continues to captivate researchers, "
            "policymakers, and industry analysts alike."
        ),
    },
    # --- Heavy slop: meta-communication, AI tells, chatbot language ---
    {
        "prompt": "Write a paragraph about machine learning applications.",
        "answer": "heavy_slop",
        "text": (
            "Certainly! Let me walk you through the fascinating world of machine "
            "learning. This comprehensive guide will help you understand the "
            "intricacies of AI applications. It's worth noting that these "
            "technologies play a vital role in modern society. On the other hand, "
            "despite its challenges, the field continues to evolve. I hope this "
            "helps! Would you like me to elaborate on any of these points? Here's "
            "a quick summary of the key takeaways that are truly breathtaking."
        ),
    },
    # --- Clean prose: specific, neutral, no filler ---
    {
        "prompt": "Write a paragraph about the 1986 Chernobyl disaster.",
        "answer": "clean",
        "text": (
            "Reactor No. 4 at the Chernobyl Nuclear Power Plant exploded during "
            "a safety test on April 26, 1986, at 1:23 AM local time. The steam "
            "explosion and resulting fire released approximately 400 times the "
            "radioactive fallout of the Hiroshima bomb. Soviet authorities did not "
            "evacuate Pripyat, the nearest city of 49,000 residents, until 36 "
            "hours after the blast. Two plant workers died that night. Another 28 "
            "emergency responders died of acute radiation syndrome within months. "
            "The exclusion zone, a 2,600-square-kilometer area around the plant, "
            "remains largely uninhabited."
        ),
    },
    # --- Watermark / disqualifier trigger: oaicite token ---
    {
        "prompt": "Write a paragraph about the history of chess.",
        "answer": "disqualified",
        "text": (
            "Chess originated in India around the 6th century as a game called "
            "chaturanga. As an AI language model, I should note that the game "
            "spread to Persia, where it became known as shatranj "
            "[contentReference[oaicite:0]]. The rules evolved over centuries, "
            "with the modern queen and bishop gaining their current powers in "
            "15th-century Spain."
        ),
    },
]


# ---------------------------------------------------------------------------
# Environment factory (for `vf eval` integration)
# ---------------------------------------------------------------------------
def load_environment():
    """Build a verifiers SingleTurnEnv using the AntiLLMY rubric.

    Returns a vf.SingleTurnEnv that can be used directly with the
    verifiers training and evaluation pipeline. This function follows
    the same convention as Prime Intellect's built-in environment
    modules (each exposes a top-level load_environment()).
    """
    import verifiers as vf

    # Compile rubric and create judge
    result = anti_slop_judge()
    bundle = result.bundle

    model = get_model("deepseek", "deepseek-v4-flash")
    judge = Judge(JudgeConfig(model=model, temperature=0.0, max_tokens=4096))

    # Build the verifiers Rubric via our bridge
    rubric = make_rubrify_rubric(bundle, judge)

    # Build dataset (in production this would come from HuggingFace or a file)
    def build_dataset():
        from datasets import Dataset
        return Dataset.from_list([
            {"prompt": s["prompt"], "answer": s["answer"]}
            for s in WRITING_SAMPLES
        ])

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        rubric=rubric,
        system_prompt="You are a precise, factual writer. Write clear prose.",
    )


# ---------------------------------------------------------------------------
# Standalone demo: score pre-written samples without a vLLM server
# ---------------------------------------------------------------------------
async def demo():
    """Score pre-written text samples using the rubrify bridge.

    This demonstrates that the bridge correctly converts rubrify judgments
    into verifiers-compatible rewards and per-criterion metrics -- without
    needing a running vLLM server or the full verifiers training pipeline.
    """
    # We import verifiers types directly so we can build State objects
    # by hand. In a real training run, the environment does this for you.
    from verifiers.types import RolloutInput, RolloutTiming, State

    # 1. Compile the rubric
    print("Compiling AntiLLMY rubric...")
    result = anti_slop_judge()
    bundle = result.bundle
    print(
        f"  Name:       {bundle.rubric.meta.name} v{bundle.rubric.meta.version}\n"
        f"  Criteria:   {len(bundle.rubric.criteria)}\n"
        f"  DQs:        {len(bundle.rubric.disqualifiers)}\n"
        f"  Patterns:   {len(bundle.rubric.patterns)}\n"
        f"  Issues:     {result.issues or '(none)'}"
    )

    # 2. Create the judge (auto-discovers DEEPSEEK_API_KEY from env)
    model = get_model("deepseek", "deepseek-v4-flash")
    judge = Judge(JudgeConfig(model=model, temperature=0.0, max_tokens=4096))
    print(f"\nJudge: {model.provider}/{model.id}")

    # 3. Build the verifiers Rubric via our bridge
    rubric = make_rubrify_rubric(bundle, judge)
    reward_names = rubric._get_reward_func_names()
    print(f"Rubric functions: {reward_names}")
    print(f"Rubric weights:   {rubric._get_reward_weights()}")

    # 4. Score each pre-written sample
    print("\n" + "=" * 72)
    print("SCORING PRE-WRITTEN SAMPLES")
    print("=" * 72)

    for i, sample in enumerate(WRITING_SAMPLES):
        prompt_text = sample["prompt"]
        expected = sample["answer"]
        text = sample["text"]

        print(f"\n--- Sample {i + 1}: [{expected}] ---")
        print(f"Prompt: {prompt_text}")
        print(f"Text:   {text[:100]}...")

        # Build a verifiers State just like the environment would.
        # The completion is a list of chat messages; the bridge reads
        # the last message's content as the text to evaluate.
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": prompt_text}],
                answer=expected,
                example_id=i,
            )
        )
        state["completion"] = [
            {"role": "assistant", "content": text},
        ]
        state["trajectory"] = []
        state["timing"] = RolloutTiming()

        # score_rollout calls our bridge's rubrify_reward function,
        # which in turn calls judge.evaluate() under the hood.
        await rubric.score_rollout(state)

        reward = state["reward"]
        metrics = state["metrics"]

        print(f"\n  Reward (weighted):  {reward:.4f}")
        print(f"  Metrics:")
        for name, value in metrics.items():
            print(f"    {name}: {value:.4f}")

        # The bridge stashes the full Judgment on the state for inspection
        judgment = state.get("rubrify_judgment")
        if judgment is not None:
            agg = judgment.aggregation
            print(f"  Normalized score:   {agg.normalized_score:.2f}/100")
            decision = judgment.decision or "N/A"
            print(f"  Decision:           {decision}")
            if judgment.violations:
                print(f"  Violations:         {judgment.violations}")
            for cj in judgment.criterion_judgments:
                label = f"    {cj.criterion_id}"
                score_str = f"{cj.unit_score:.2f}"
                rationale = (cj.rationale or "")[:60]
                print(f"{label}: {score_str}  {rationale}")

    print("\n" + "=" * 72)
    print(f"Judge usage: {judge.total_usage}")
    print(f"Evaluations: {judge.evaluation_count}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(demo())
