"""Exhaustive test suite for verbatim_score.

Tests cover: exact match, truncations, reversed content, preamble/postamble,
paraphrasing, edge cases (empty, whitespace, single char, gibberish),
repetition penalty, single-word deletion, and linearity of truncation scores.
"""

from __future__ import annotations

import pytest

from rubrify.scoring.verbatim import verbatim_score


# ── Reference text used across most tests ────────────────────────

REFERENCE = (
    "The bridge collapsed because the steel bolts corroded over twenty "
    "years of exposure to salt air. Three engineers inspected the site. "
    "They found cracks in fourteen support beams. The city closed the "
    "road the same day."
)

SENTENCES = [
    "The bridge collapsed because the steel bolts corroded over twenty years of exposure to salt air.",
    "Three engineers inspected the site.",
    "They found cracks in fourteen support beams.",
    "The city closed the road the same day.",
]


# ── Helpers ──────────────────────────────────────────────────────

def _truncate_front(text: str, fraction: float) -> str:
    """Return the first `fraction` of `text` (by character count)."""
    n = int(len(text) * fraction)
    return text[:n]


def _truncate_back(text: str, fraction: float) -> str:
    """Return the last `fraction` of `text` (by character count)."""
    n = int(len(text) * fraction)
    return text[-n:]


# ═══════════════════════════════════════════════════════════════════
# 1. Exact match
# ═══════════════════════════════════════════════════════════════════

class TestExactMatch:
    def test_exact_match_returns_one(self):
        assert verbatim_score(REFERENCE, REFERENCE) == 1.0


# ═══════════════════════════════════════════════════════════════════
# 2-4. Front truncation (70%, 50%, 30%)
# ═══════════════════════════════════════════════════════════════════

class TestFrontTruncation:
    def test_first_70_percent(self):
        response = _truncate_front(REFERENCE, 0.70)
        score = verbatim_score(REFERENCE, response)
        assert 0.55 <= score <= 0.85, f"Expected ~0.70, got {score:.4f}"

    def test_first_50_percent(self):
        response = _truncate_front(REFERENCE, 0.50)
        score = verbatim_score(REFERENCE, response)
        assert 0.35 <= score <= 0.65, f"Expected ~0.50, got {score:.4f}"

    def test_first_30_percent(self):
        response = _truncate_front(REFERENCE, 0.30)
        score = verbatim_score(REFERENCE, response)
        assert 0.15 <= score <= 0.45, f"Expected ~0.30, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# 5. Last 70% (should be close to first 70%)
# ═══════════════════════════════════════════════════════════════════

class TestBackTruncation:
    def test_last_70_percent(self):
        response = _truncate_back(REFERENCE, 0.70)
        score = verbatim_score(REFERENCE, response)
        front_score = verbatim_score(REFERENCE, _truncate_front(REFERENCE, 0.70))
        assert 0.55 <= score <= 0.85, f"Expected ~0.70, got {score:.4f}"
        # Front and back 70% should yield similar scores (within 0.15)
        assert abs(score - front_score) < 0.15, (
            f"Front={front_score:.4f}, Back={score:.4f} differ by more than 0.15"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. Reversed sentences -> penalized more than in-order truncation
# ═══════════════════════════════════════════════════════════════════

class TestOrderPenalty:
    def test_reversed_sentences_penalized(self):
        reversed_text = " ".join(reversed(SENTENCES))
        score_reversed = verbatim_score(REFERENCE, reversed_text)
        score_front70 = verbatim_score(REFERENCE, _truncate_front(REFERENCE, 0.70))
        # Reversed should score MUCH lower than front 70% which is in-order
        assert score_reversed < score_front70, (
            f"Reversed ({score_reversed:.4f}) should be < front-70% ({score_front70:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════
# 7-8. Preamble and postamble
# ═══════════════════════════════════════════════════════════════════

class TestPreamblePostamble:
    def test_with_preamble(self):
        response = "Here is the text:\n" + REFERENCE
        score = verbatim_score(REFERENCE, response)
        assert 0.75 <= score <= 0.99, f"Expected ~0.85, got {score:.4f}"

    def test_with_postamble(self):
        response = REFERENCE + "\nI copied it."
        score = verbatim_score(REFERENCE, response)
        assert 0.75 <= score <= 0.99, f"Expected ~0.85, got {score:.4f}"

    def test_preamble_and_postamble_similar(self):
        pre_score = verbatim_score(REFERENCE, "Here is the text:\n" + REFERENCE)
        post_score = verbatim_score(REFERENCE, REFERENCE + "\nI copied it.")
        assert abs(pre_score - post_score) < 0.15, (
            f"Preamble ({pre_score:.4f}) and postamble ({post_score:.4f}) differ too much"
        )


# ═══════════════════════════════════════════════════════════════════
# 9. Paraphrased -> moderate score
# ═══════════════════════════════════════════════════════════════════

class TestParaphrase:
    def test_paraphrased_moderate_score(self):
        paraphrased = (
            "The bridge fell down due to corrosion of steel bolts after two "
            "decades in salty air. A team of three engineers examined the "
            "location and discovered cracks in fourteen beams. The road was "
            "shut down by the city that same day."
        )
        score = verbatim_score(REFERENCE, paraphrased)
        assert 0.25 <= score <= 0.70, f"Expected ~0.30-0.60, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# 10-14. Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_response(self):
        assert verbatim_score(REFERENCE, "") == 0.0

    def test_empty_reference(self):
        assert verbatim_score("", "something") == 0.0

    def test_both_empty(self):
        assert verbatim_score("", "") == 1.0

    def test_random_gibberish(self):
        gibberish = "xkqz plmn wrvy gths jbdf nlkr ztxw"
        score = verbatim_score(REFERENCE, gibberish)
        assert score < 0.10, f"Expected near 0, got {score:.4f}"

    def test_single_character(self):
        score = verbatim_score(REFERENCE, "T")
        assert score < 0.05, f"Expected near 0, got {score:.4f}"

    def test_reference_repeated_3x(self):
        response = REFERENCE * 3
        score = verbatim_score(REFERENCE, response)
        assert score < 1.0, f"Repeated 3x should not be 1.0, got {score:.4f}"
        # The repetition adds a lot of extra content, so it should be penalized
        assert score < 0.60, f"Repeated 3x should be meaningfully penalized, got {score:.4f}"

    def test_only_whitespace(self):
        score = verbatim_score(REFERENCE, "   \n\t  ")
        assert score < 0.05, f"Expected near 0, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# 15. One word missing from middle -> high but not 1.0
# ═══════════════════════════════════════════════════════════════════

class TestMinorEdits:
    def test_one_word_missing_from_middle(self):
        # Remove "steel" from the middle of the reference
        response = REFERENCE.replace("steel ", "")
        score = verbatim_score(REFERENCE, response)
        assert 0.90 <= score < 1.0, f"Expected high but not 1.0, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# 16-21. Linearity check: 10%, 20%, 30%, 50%, 70%, 90% truncation
# ═══════════════════════════════════════════════════════════════════

class TestLinearity:
    """Truncation scores should be roughly monotonic with fraction kept."""

    FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

    def test_truncation_scores_are_monotonic(self):
        scores = []
        for frac in self.FRACTIONS:
            response = _truncate_front(REFERENCE, frac)
            s = verbatim_score(REFERENCE, response)
            scores.append(s)

        # Every successive score should be >= the previous (monotonic)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 0.02, (
                f"Non-monotonic: {self.FRACTIONS[i-1]:.0%} -> {scores[i-1]:.4f}, "
                f"{self.FRACTIONS[i]:.0%} -> {scores[i]:.4f}"
            )

    def test_truncation_score_at_10_percent_is_low(self):
        response = _truncate_front(REFERENCE, 0.10)
        score = verbatim_score(REFERENCE, response)
        assert score < 0.20, f"10% should be low, got {score:.4f}"

    def test_truncation_score_at_90_percent_is_high(self):
        response = _truncate_front(REFERENCE, 0.90)
        score = verbatim_score(REFERENCE, response)
        assert score >= 0.80, f"90% should be high, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Extra edge cases for robustness
# ═══════════════════════════════════════════════════════════════════

class TestAdditionalEdgeCases:
    def test_single_char_reference_exact_match(self):
        assert verbatim_score("a", "a") == 1.0

    def test_single_char_reference_mismatch(self):
        score = verbatim_score("a", "b")
        assert score == 0.0, f"Expected 0.0 for total mismatch, got {score:.4f}"

    def test_score_always_in_unit_interval(self):
        """Score should always be in [0.0, 1.0] for arbitrary inputs."""
        test_pairs = [
            ("hello", "hello"),
            ("hello", "world"),
            ("abc", "abcdef"),
            ("abcdef", "abc"),
            ("", ""),
            ("a", ""),
            ("", "a"),
            (REFERENCE, REFERENCE[:10]),
            (REFERENCE, REFERENCE * 2),
            ("aaaa", "bbbb"),
            ("short", "a much longer string that shares no content"),
        ]
        for ref, resp in test_pairs:
            score = verbatim_score(ref, resp)
            assert 0.0 <= score <= 1.0, (
                f"Score {score:.4f} out of [0,1] for ref={ref!r:.40}, resp={resp!r:.40}"
            )

    def test_swapped_words(self):
        """Swapping two words should produce a score close to but below 1.0."""
        # Swap "bridge" and "road"
        response = REFERENCE.replace("bridge", "XYZZY").replace("road", "bridge").replace("XYZZY", "road")
        score = verbatim_score(REFERENCE, response)
        assert 0.85 <= score < 1.0, f"Expected high but not 1.0, got {score:.4f}"

    def test_entirely_different_text(self):
        other = (
            "Quantum computing leverages superposition and entanglement to "
            "solve problems that classical computers find intractable. "
            "Current implementations use superconducting qubits."
        )
        score = verbatim_score(REFERENCE, other)
        assert score < 0.25, f"Unrelated text should score low, got {score:.4f}"

    def test_case_sensitivity(self):
        """The function is case-sensitive by default."""
        upper = REFERENCE.upper()
        score = verbatim_score(REFERENCE, upper)
        # Very different character-by-character -> should be penalized
        assert score < 0.80, f"Case change should be penalized, got {score:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Visual table (runs when executed directly, not via pytest)
# ═══════════════════════════════════════════════════════════════════

def _print_results_table():
    """Print a formatted table of all test cases and their scores."""
    print("=" * 80)
    print(f"{'#':<4} {'Test Case':<45} {'Score':>8} {'Expected':>12}")
    print("-" * 80)

    cases = [
        ("1",  "Exact match",                    REFERENCE, "1.0"),
        ("2",  "First 70%",                       _truncate_front(REFERENCE, 0.70), "~0.70"),
        ("3",  "First 50%",                       _truncate_front(REFERENCE, 0.50), "~0.50"),
        ("4",  "First 30%",                       _truncate_front(REFERENCE, 0.30), "~0.30"),
        ("5",  "Last 70%",                        _truncate_back(REFERENCE, 0.70), "~0.70"),
        ("6",  "Reversed sentences",              " ".join(reversed(SENTENCES)), "<0.70"),
        ("7",  "With preamble",                   "Here is the text:\n" + REFERENCE, "~0.85"),
        ("8",  "With postamble",                  REFERENCE + "\nI copied it.", "~0.85"),
        ("9",  "Paraphrased",                     (
            "The bridge fell down due to corrosion of steel bolts after two "
            "decades in salty air. A team of three engineers examined the "
            "location and discovered cracks in fourteen beams. The road was "
            "shut down by the city that same day."
        ), "~0.40-0.60"),
        ("10", "Empty response",                  "", "0.0"),
        ("11", "Random gibberish",                "xkqz plmn wrvy gths jbdf nlkr ztxw", "~0.0"),
        ("12", "Single character",                "T", "~0.0"),
        ("13", "Reference repeated 3x",           REFERENCE * 3, "<1.0"),
        ("14", "Only whitespace",                 "   \n\t  ", "~0.0"),
        ("15", "One word missing from middle",    REFERENCE.replace("steel ", ""), ">0.90"),
    ]

    for num, label, response, expected in cases:
        score = verbatim_score(REFERENCE, response)
        print(f"{num:<4} {label:<45} {score:>8.4f} {expected:>12}")

    # Linearity check
    print("-" * 80)
    print("Linearity check (front truncation):")
    print("-" * 80)
    for frac in [0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]:
        response = _truncate_front(REFERENCE, frac)
        score = verbatim_score(REFERENCE, response)
        num = f"L{int(frac*100)}"
        print(f"{num:<4} {f'First {frac:.0%}':<45} {score:>8.4f} {'~' + f'{frac:.2f}':>12}")

    print("=" * 80)


if __name__ == "__main__":
    _print_results_table()
