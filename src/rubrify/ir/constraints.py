"""ConstraintBinding, OutputConstraint variants, SurfacePolicy — the thesis in code.

The researcher's core insight: (roleplaying == jailbreak == context following) == rubrics.

A ConstraintBinding is the explicit triple connecting:
  1. A semantic criterion (what we evaluate)
  2. A surface-layer projection (how the LLM sees it — XML tag, JSON path)
  3. An output field (where the LLM writes its judgment)

OutputConstraint is a discriminated union of typed constraint variants, each
with a concrete `check()` method — no natural-language string parsing.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from harn_ai.types import SchemaModel


# ── Surface projection ────────────────────────────────────────────

class SurfaceProjection(SchemaModel):
    """One surface-layer representation of a constraint."""
    codec: str
    path: str
    attributes: dict[str, str] = {}


class ConstraintBinding(SchemaModel):
    """The triple-layer alignment: criterion <-> surface tag <-> output field."""
    criterion_id: str
    output_field: str
    evidence_source: str = "response"
    authority: Literal["instruction", "data", "meta"] = "instruction"
    projections: list[SurfaceProjection] = []


class AuthorityBlock(SchemaModel):
    """Marks a prompt section as instruction vs. data."""
    id: str
    authority: Literal["instruction", "data", "meta"]
    kind: Literal[
        "system_prompt", "rubric_spec", "judge_instructions",
        "response_under_test", "context_document", "calibration_example",
    ]
    model_should_follow: bool = True


# ── OutputConstraint variants ─────────────────────────────────────

class PrefixSuffixConstraint(SchemaModel):
    """Constraint that checks whether a field starts/ends with specified strings."""
    kind: Literal["prefix_suffix"] = "prefix_suffix"
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    scope: Literal["call", "criterion", "judgment"] = "call"  # when to check: per-call (default), per-criterion, or on final judgment
    prefix: str | None = None
    suffix: str | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> PrefixSuffixConstraint:
        if self.prefix is None and self.suffix is None:
            raise ValueError("At least one of 'prefix' or 'suffix' must be set")
        return self

    def check(self, value: str) -> str | None:
        if self.prefix is not None and not value.startswith(self.prefix):
            return f"Expected value to start with {self.prefix!r}, got {value[:len(self.prefix) + 10]!r}"
        if self.suffix is not None and not value.endswith(self.suffix):
            return f"Expected value to end with {self.suffix!r}, got {value[-len(self.suffix) - 10:]!r}"
        return None


class WordCountConstraint(SchemaModel):
    """Constraint that checks word count of a field."""
    kind: Literal["word_count"] = "word_count"
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    scope: Literal["call", "criterion", "judgment"] = "call"  # when to check: per-call (default), per-criterion, or on final judgment
    count: int
    mode: Literal["exactly", "min", "max"] = "exactly"

    @field_validator("count")
    @classmethod
    def _count_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"count must be positive, got {v}")
        return v

    def check(self, value: str) -> str | None:
        actual = len(value.split())
        if self.mode == "exactly" and actual != self.count:
            return f"Expected exactly {self.count} words, got {actual}"
        if self.mode == "min" and actual < self.count:
            return f"Expected at least {self.count} words, got {actual}"
        if self.mode == "max" and actual > self.count:
            return f"Expected at most {self.count} words, got {actual}"
        return None


class CharLimitConstraint(SchemaModel):
    """Constraint that checks character count of a field."""
    kind: Literal["char_limit"] = "char_limit"
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    scope: Literal["call", "criterion", "judgment"] = "call"  # when to check: per-call (default), per-criterion, or on final judgment
    limit: int
    mode: Literal["exactly", "min", "max"] = "max"

    @field_validator("limit")
    @classmethod
    def _limit_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"limit must be positive, got {v}")
        return v

    def check(self, value: str) -> str | None:
        actual = len(value)
        if self.mode == "exactly" and actual != self.limit:
            return f"Expected exactly {self.limit} characters, got {actual}"
        if self.mode == "min" and actual < self.limit:
            return f"Expected at least {self.limit} characters, got {actual}"
        if self.mode == "max" and actual > self.limit:
            return f"Expected at most {self.limit} characters, got {actual}"
        return None


class ItemCountConstraint(SchemaModel):
    """Constraint that checks the number of delimited items in a field."""
    kind: Literal["item_count"] = "item_count"
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    scope: Literal["call", "criterion", "judgment"] = "call"  # when to check: per-call (default), per-criterion, or on final judgment
    count: int
    mode: Literal["exactly", "min", "max"] = "exactly"
    delimiter: str = ";"

    @field_validator("count")
    @classmethod
    def _count_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"count must be positive, got {v}")
        return v

    def check(self, value: str) -> str | None:
        items = [item.strip() for item in value.split(self.delimiter) if item.strip()]
        actual = len(items)
        if self.mode == "exactly" and actual != self.count:
            return f"Expected exactly {self.count} items (delimiter={self.delimiter!r}), got {actual}"
        if self.mode == "min" and actual < self.count:
            return f"Expected at least {self.count} items (delimiter={self.delimiter!r}), got {actual}"
        if self.mode == "max" and actual > self.count:
            return f"Expected at most {self.count} items (delimiter={self.delimiter!r}), got {actual}"
        return None


class TokenConstraint(SchemaModel):
    """Constraint that checks whether a specific token is present/absent in a field."""
    kind: Literal["token"] = "token"
    id: str
    description: str
    target_field: str
    enforcement: Literal["hard", "soft"] = "hard"
    scope: Literal["call", "criterion", "judgment"] = "call"  # when to check: per-call (default), per-criterion, or on final judgment
    token: str
    must_contain: bool = True

    def check(self, value: str) -> str | None:
        found = self.token in value
        if self.must_contain and not found:
            return f"Expected value to contain {self.token!r}, but it was not found"
        if not self.must_contain and found:
            return f"Expected value to NOT contain {self.token!r}, but it was found"
        return None


# ── Discriminated union ───────────────────────────────────────────

OutputConstraint: TypeAlias = Annotated[
    PrefixSuffixConstraint | WordCountConstraint | CharLimitConstraint | ItemCountConstraint | TokenConstraint,
    Field(discriminator="kind"),
]


__all__ = [
    "AuthorityBlock",
    "CharLimitConstraint",
    "ConstraintBinding",
    "ItemCountConstraint",
    "OutputConstraint",
    "PrefixSuffixConstraint",
    "SurfaceProjection",
    "TokenConstraint",
    "WordCountConstraint",
]
