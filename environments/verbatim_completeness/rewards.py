"""Reward functions for the verbatim_completeness environment.

Four deterministic reward signals that together assess how faithfully a model
reproduces a reference text verbatim:

    verbatim_fidelity  (0.50) -- primary Levenshtein/LCS composite score
    chunk_coverage     (0.20) -- fraction of 50-char reference chunks found
    no_additions       (0.15) -- penalty for preambles, postambles, fences
    no_truncation      (0.15) -- penalty for early stopping / truncation
"""

from __future__ import annotations

import re
from typing import Any

from rubrify.scoring.verbatim import verbatim_score

# ---------------------------------------------------------------------------
# Types -- kept loose so the module works with both raw dicts and pydantic
# message objects that the verifiers framework may pass.
# ---------------------------------------------------------------------------
# verifiers.types.Messages  =  list[Message]
# Each Message is either a pydantic model or a plain dict with at minimum
# {"role": str, "content": str | list[...] | None}.
Messages = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def extract_response(completion: Messages) -> str:
    """Extract the text content of the last assistant message.

    Handles three content shapes:
      - ``str``  (common case)
      - ``list`` of content parts (multimodal / tool-use format) -- concatenates
        all ``text`` parts
      - ``None`` / missing -- returns empty string
    """
    if not completion:
        return ""

    # Walk backwards to find the last assistant message.
    for msg in reversed(completion):
        # Support both dict-like and attribute access (pydantic models expose
        # ``get`` via CustomBaseModel).
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role != "assistant":
            continue

        content = (
            msg.get("content", None)
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )

        if content is None:
            return ""
        if isinstance(content, str):
            return content

        # list-of-parts: concatenate text parts
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type", "") == "text":
                        parts.append(part.get("text", ""))
                else:
                    # Pydantic content-part model
                    if getattr(part, "type", "") == "text":
                        parts.append(getattr(part, "text", ""))
            return "".join(parts)

        # Fallback: coerce to string
        return str(content)

    # No assistant message found at all.
    return ""


# ---------------------------------------------------------------------------
# R1  verbatim_fidelity  (weight = 0.50)
# ---------------------------------------------------------------------------

def verbatim_fidelity(completion: Messages, answer: str, **kwargs: Any) -> float:
    """Primary fidelity signal: L^2 / max(C, L, eps).

    Wraps ``rubrify.scoring.verbatim.verbatim_score`` which computes
    Levenshtein-normalised-similarity squared divided by the max of that
    value and the LCS-normalised-similarity.
    """
    response = extract_response(completion)
    return float(verbatim_score(answer, response))


# ---------------------------------------------------------------------------
# R2  chunk_coverage  (weight = 0.20)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 50


def chunk_coverage(completion: Messages, answer: str, **kwargs: Any) -> float:
    """Fraction of 50-character reference chunks found verbatim in the response.

    This catches subtle word substitutions or reorderings that the main
    Levenshtein score might still rate highly.
    """
    response = extract_response(completion)
    if not answer:
        # Nothing to reproduce -- vacuously correct.
        return 1.0
    if not response:
        return 0.0

    chunks: list[str] = [
        answer[i : i + _CHUNK_SIZE]
        for i in range(0, len(answer), _CHUNK_SIZE)
    ]
    if not chunks:
        return 1.0

    found = sum(1 for chunk in chunks if chunk in response)
    return found / len(chunks)


# ---------------------------------------------------------------------------
# R3  no_additions  (weight = 0.15)
# ---------------------------------------------------------------------------

# Patterns the model likes to prepend.
_LEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:sure|okay|of course|certainly|here(?:'s| is| you go))[,!.:]\s*", re.IGNORECASE),
    re.compile(r"^(?:the following is|below is|here is the)\b.*?:\s*", re.IGNORECASE),
]

# Patterns the model likes to append.
_TRAILING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\s*(?:let me know|hope this helps|feel free|is there anything).*$", re.IGNORECASE),
    re.compile(r"\s*(?:---+|===+)\s*$"),
]

# Code-fence wrapping.
_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n", re.MULTILINE)


def no_additions(completion: Messages, answer: str, **kwargs: Any) -> float:
    """Detect extraneous content the model added around the reference.

    Checks four signals and combines them into a single [0, 1] score:
      - length ratio (response much longer than reference)
      - leading preamble patterns
      - trailing postamble patterns
      - code-fence wrapping

    Returns 0.0 when the response is too short to evaluate (<10 chars).
    """
    response = extract_response(completion)
    if len(response) < 10:
        return 0.0
    if not answer:
        return 1.0

    penalties: float = 0.0

    # -- length ratio penalty --
    len_ratio = len(response) / max(len(answer), 1)
    if len_ratio > 1.5:
        # Linearly penalise from 1.5x to 3x (0 -> 0.4 penalty).
        penalties += min((len_ratio - 1.5) / 1.5, 1.0) * 0.4

    # -- leading preamble --
    for pat in _LEADING_PATTERNS:
        if pat.search(response):
            penalties += 0.25
            break

    # -- trailing postamble (regex patterns) --
    for pat in _TRAILING_PATTERNS:
        if pat.search(response):
            penalties += 0.2
            break

    # -- trailing postamble (suffix-diff) --
    # If the response ends with content not at the end of the reference,
    # there is appended text regardless of whether it matches a regex.
    if len(response) > len(answer):
        # Find where the reference ends in the response
        idx = response.rfind(answer[-min(80, len(answer)):])
        if idx >= 0:
            tail_start = idx + min(80, len(answer))
            tail = response[tail_start:].strip()
            if len(tail) > 5:  # non-trivial trailing content
                penalties += min(len(tail) / max(len(answer), 1), 0.3)

    # -- code fence wrapping --
    fence_count = len(_CODE_FENCE_RE.findall(response))
    if fence_count >= 2:
        # Wrapped in code fences.
        penalties += 0.3
    elif fence_count == 1:
        penalties += 0.15

    return max(1.0 - penalties, 0.0)


# ---------------------------------------------------------------------------
# R4  no_truncation  (weight = 0.15)
# ---------------------------------------------------------------------------

_TRUNCATION_SIGNALS = re.compile(
    r"\b(?:truncated|omitted|abbreviated|continued|rest of|remaining|etc\.?)\b",
    re.IGNORECASE,
)
_TRAILING_ELLIPSIS = re.compile(r"\.{3,}\s*$")


def no_truncation(completion: Messages, answer: str, **kwargs: Any) -> float:
    """Detect whether the model stopped reproducing the reference early.

    Three sub-checks:
      1. **Length coverage** -- ratio of response length to reference length.
      2. **Tail matching** -- the last 100 characters of the reference should
         appear somewhere in the last 200 characters of the response.
      3. **Truncation signals** -- words like "truncated", "omitted", trailing
         ellipsis.
    """
    response = extract_response(completion)
    if not answer:
        return 1.0
    if not response:
        return 0.0

    penalties: float = 0.0

    # -- length coverage --
    coverage = len(response) / max(len(answer), 1)
    if coverage < 0.9:
        # Scale penalty: at 0.9 coverage -> 0 penalty, at 0 coverage -> 0.5
        penalties += (1.0 - min(coverage / 0.9, 1.0)) * 0.5

    # -- tail matching --
    tail_len = min(100, len(answer))
    ref_tail = answer[-tail_len:]
    resp_window = response[-200:] if len(response) >= 200 else response
    if ref_tail not in resp_window:
        penalties += 0.35

    # -- truncation signal words / trailing ellipsis --
    if _TRUNCATION_SIGNALS.search(response):
        penalties += 0.15
    if _TRAILING_ELLIPSIS.search(response):
        penalties += 0.15

    return max(1.0 - penalties, 0.0)


# ---------------------------------------------------------------------------
# Registry for easy environment integration
# ---------------------------------------------------------------------------

REWARD_FUNCS: list[tuple[Any, float]] = [
    (verbatim_fidelity, 0.50),
    (chunk_coverage, 0.20),
    (no_additions, 0.15),
    (no_truncation, 0.15),
]
