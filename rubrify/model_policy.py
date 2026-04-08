"""Model support policy.

Declares which model families are recommended, supported, experimental,
or discouraged for rubric evaluation and generation.

Based on hands-on experimentation: GPT-class and Claude-class models
follow XML contracts and steering anchors reliably. Smaller or older
models drift on structured outputs.

This is a declarative policy, not a capability matrix. When a model
needs fine-grained capability descriptions, the appropriate response
is to calibrate it against reference suites, not to invent metadata.
Guard rail 8 (no hypothetical future-proofing) keeps this module to
plain tuples of fnmatch patterns and a single enum.
"""

from __future__ import annotations

import fnmatch
import warnings
from enum import Enum
from typing import Final


class ModelTier(str, Enum):
    RECOMMENDED = "recommended"
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    DISCOURAGED = "discouraged"
    UNKNOWN = "unknown"


# Recommended: fully tested, follows XML and steering anchors reliably.
RECOMMENDED: Final[tuple[str, ...]] = (
    "gpt-5*",
    "gpt-4.1*",
    "gpt-4o*",
    "claude-opus-4*",
    "claude-sonnet-4*",
    "claude-haiku-4*",
    "claude-3-5-sonnet*",
    "claude-3-opus*",
)

# Supported: expected to work but limited production experience.
SUPPORTED: Final[tuple[str, ...]] = (
    "gpt-4*",
    "claude-3-*",
)

# Experimental: may work with hand-holding; calibrate before relying.
EXPERIMENTAL: Final[tuple[str, ...]] = (
    "gpt-3.5*",
    "mistral*",
    "llama*",
    "qwen*",
    "gemini*",
    "deepseek*",
)

# Discouraged: known to drift on structured outputs or fail on anchors.
DISCOURAGED: Final[tuple[str, ...]] = (
    "davinci*",
    "curie*",
    "babbage*",
    "ada*",
    "text-*",
)


def check_model(name: str) -> tuple[ModelTier, str]:
    """Return the tier and a human-readable message for a model name.

    Uses fnmatch patterns so ``claude-sonnet-4-6`` matches
    ``claude-sonnet-4*``. Unknown models return
    ``(ModelTier.UNKNOWN, guidance)``.
    """
    for pattern in RECOMMENDED:
        if fnmatch.fnmatch(name, pattern):
            return (
                ModelTier.RECOMMENDED,
                f"{name!r} is in the recommended tier. "
                "Fully tested for XML and steering anchor compliance.",
            )
    for pattern in SUPPORTED:
        if fnmatch.fnmatch(name, pattern):
            return (
                ModelTier.SUPPORTED,
                f"{name!r} is in the supported tier. "
                "Expected to work; calibrate for production use.",
            )
    for pattern in EXPERIMENTAL:
        if fnmatch.fnmatch(name, pattern):
            return (
                ModelTier.EXPERIMENTAL,
                f"{name!r} is in the experimental tier. "
                "Calibrate against reference suites before relying on it.",
            )
    for pattern in DISCOURAGED:
        if fnmatch.fnmatch(name, pattern):
            return (
                ModelTier.DISCOURAGED,
                f"{name!r} is discouraged. "
                "Known to drift on XML contracts and steering anchors.",
            )
    return (
        ModelTier.UNKNOWN,
        f"{name!r} is not in the policy. " "Treat as experimental and calibrate before use.",
    )


def warn_unsupported(name: str) -> None:
    """Emit a ``UserWarning`` if the model is not recommended or supported.

    Intended to be called from inside the runtime entry points
    (``Rubric.evaluate``, ``ConstraintRubric.apply``, ``generate``) when the
    caller opted in with ``warn_unsupported=True``. ``stacklevel=3`` points
    the warning at the user call site (user -> entry point -> this helper).
    Recommended and supported models never produce a warning.
    """
    tier, message = check_model(name)
    if tier not in (ModelTier.RECOMMENDED, ModelTier.SUPPORTED):
        warnings.warn(
            f"rubrify model policy: {message}",
            category=UserWarning,
            stacklevel=3,
        )
