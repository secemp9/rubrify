"""Verbatim reproduction scoring via Levenshtein + LCS."""

from rapidfuzz.distance import Levenshtein, LCSseq


def verbatim_score(reference: str, response: str) -> float:
    """Score how faithfully response reproduces reference.

    Returns a float in [0.0, 1.0]:
    - 1.0 for exact match
    - Proportional partial credit for in-order partial matches
    - Penalized for out-of-order content (lower than same-content in-order)
    - Proportional penalty for truncation and additions
    - Smooth gradient from 0 to 1

    Algorithm: L^2 / max(C, L, eps) where L = Levenshtein normalized similarity,
    C = LCS normalized similarity. This applies an order penalty when content
    is matched but disordered (C > L), and reverts to raw Levenshtein when
    content is well-ordered (L >= C).
    """
    if not reference and not response:
        return 1.0
    if not reference or not response:
        return 0.0

    lev = Levenshtein.normalized_similarity(reference, response)
    lcs = LCSseq.normalized_similarity(reference, response)

    return lev ** 2 / max(lcs, lev, 1e-9)
