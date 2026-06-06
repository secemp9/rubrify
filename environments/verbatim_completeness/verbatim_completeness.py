"""
Verbatim Completeness Environment.

Tests a model's ability to reproduce mixed-content text (lorem ipsum,
normalised data, Python stdlib source) with perfect fidelity.

Five prompt templates are randomly selected per sample to reduce overfitting
to any single phrasing.  Token budgets are computed from the model's context
window so that prompts stay within limits.
"""

import random

import verifiers as vf
from datasets import Dataset

from .data_gen import generate_dataset
from .rewards import REWARD_FUNCS

# ---------------------------------------------------------------------------
# Prompt templates -- {text} is the placeholder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    # 1. Direct
    (
        "Copy the following text exactly as it appears, character for character. "
        "Do not add, remove, or modify anything.\n\n{text}"
    ),
    # 2. Imperative
    (
        "Your task is to reproduce the text below verbatim. "
        "Output only the reproduced text with no commentary.\n\n{text}"
    ),
    # 3. Explicit constraints
    (
        "Output the following text with perfect fidelity.\n"
        "Requirements:\n"
        "- Every character must match exactly\n"
        "- Preserve all whitespace and punctuation\n"
        "- Do not add any preamble or postamble\n\n{text}"
    ),
    # 4. Adversarial clarity
    (
        "I need you to act as a perfect text mirror. "
        "Reflect the text below back to me with zero alterations. "
        "No introductions, no sign-offs, just the exact text.\n\n{text}"
    ),
    # 5. Minimal
    "Copy this 1:1.\n\n{text}",
]

# ---------------------------------------------------------------------------
# Token budget helpers
# ---------------------------------------------------------------------------

PROMPT_OVERHEAD_TOKENS = 150
CHARS_PER_TOKEN = 3.5


def compute_max_chars(
    max_output_tokens: int,
    max_input_tokens: int,
    target_fill_ratio: float,
) -> int:
    """Compute the character budget for generated text samples.

    The budget is the smaller of the output and (overhead-adjusted) input
    windows, scaled by *target_fill_ratio* and clamped to [200, 50_000].
    """
    output_budget = max_output_tokens * CHARS_PER_TOKEN
    input_budget = (max_input_tokens - PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN
    max_chars = target_fill_ratio * min(output_budget, input_budget)
    return int(max(200, min(50000, max_chars)))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_dataset(
    n_samples: int,
    seed: int,
    max_chars: int,
    prompt_templates: list[str],
) -> Dataset:
    """Generate samples and wrap each in a randomly-chosen prompt template."""
    rng = random.Random(seed)
    samples = generate_dataset(n_samples, seed, max_chars)

    records: list[dict] = []
    for sample in samples:
        template = rng.choice(prompt_templates)
        prompt_text = template.format(text=sample["text"])
        records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": sample["text"],
            "info": {
                "source_types": sample["source_types"],
                "char_count": sample["char_count"],
                "prompt_variant": prompt_templates.index(template),
            },
        })

    return Dataset.from_dict({k: [r[k] for r in records] for k in records[0]})


# ---------------------------------------------------------------------------
# Environment entry point
# ---------------------------------------------------------------------------


def load_environment(
    n_samples: int = 500,
    seed: int = 42,
    target_fill_ratio: float = 0.6,
    max_output_tokens: int = 8192,
    max_input_tokens: int = 128000,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Load the verbatim-completeness environment.

    Args:
        n_samples: Number of text samples to generate.
        seed: Base seed for deterministic reproduction.
        target_fill_ratio: Fraction of the token budget to fill with text.
        max_output_tokens: Model output-token limit.
        max_input_tokens: Model input-token limit.
        **kwargs: Forwarded to ``vf.SingleTurnEnv``.

    Returns:
        A configured ``SingleTurnEnv`` ready for evaluation.
    """
    max_chars = compute_max_chars(max_output_tokens, max_input_tokens, target_fill_ratio)

    def dataset_builder() -> Dataset:
        return build_dataset(n_samples, seed, max_chars, PROMPT_TEMPLATES)

    funcs = [fn for fn, _ in REWARD_FUNCS]
    weights = [w for _, w in REWARD_FUNCS]
    rubric = vf.Rubric(funcs=funcs, weights=weights)

    return vf.SingleTurnEnv(
        dataset=dataset_builder,
        rubric=rubric,
        system_prompt="",
        **kwargs,
    )
