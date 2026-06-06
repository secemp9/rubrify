"""JSON codec: Pydantic-powered judgment validation and tool construction.

No regex. No cascading fallbacks. The dynamic Pydantic model IS the schema,
the validator, the tool parameter type, and the JSON template source — one
source of truth for all four.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ValidationError, create_model, Field as PydanticField

from harn_ai.types import Tool
from harn_ai.utils.json_parse import parse_json_with_repair


class ParseError(Exception):
    """LLM output could not be parsed as valid JSON."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def parse_judgment_json(raw: str) -> dict[str, Any]:
    """Parse LLM output as JSON using harn's repair-capable parser.

    Raises ParseError on failure. No fallbacks.
    """
    raw = raw.strip()
    if not raw:
        raise ParseError("LLM returned empty output", raw)

    try:
        result = parse_json_with_repair(raw)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise ParseError(f"Failed to parse LLM output as JSON: {e}", raw) from e

    if isinstance(result, dict):
        return result

    raise ParseError(
        f"LLM output parsed as {type(result).__name__}, expected dict",
        raw,
    )


# ── Dynamic model construction (cached per criterion specs) ──────

def _scale_to_field_type(scale: Any) -> tuple:
    """Derive Pydantic field type from scale for dynamic model creation."""
    from rubrify.ir.types import BinaryScale, NumericScale, OrdinalScale, NominalScale
    if isinstance(scale, BinaryScale):
        return (bool | int | float, ...)
    if isinstance(scale, NumericScale):
        return (int | float, ...)
    if isinstance(scale, OrdinalScale):
        return (int | float | str, ...)
    if isinstance(scale, NominalScale):
        return (str, ...)
    return (int | float | str, ...)


@lru_cache(maxsize=32)
def _build_judgment_model_cached(criterion_specs: tuple) -> type:
    """Build and cache a dynamic Pydantic model keyed by criterion specs."""
    criterion_fields = {}
    for cid, scale_kind in criterion_specs:
        # Reconstruct field type from the kind string
        type_map = {
            "binary": (bool | int | float, ...),
            "numeric": (int | float, ...),
            "ordinal": (int | float | str, ...),
            "nominal": (str, ...),
        }
        criterion_fields[cid] = type_map.get(scale_kind, (int | float | str, ...))

    CriterionScores = create_model("CriterionScores", **criterion_fields)

    JudgmentOutput = create_model(
        "JudgmentOutput",
        score=(int | float, ...),
        rationale=(str, ...),
        evidence=(list[str], PydanticField(default_factory=list)),
        violations=(list[str], PydanticField(default_factory=list)),
        criterion_scores=(CriterionScores, ...),
        confidence=(float, PydanticField(default=1.0)),
    )
    return JudgmentOutput


def build_judgment_model(bundle: "RubricBundle") -> type:
    """Build a dynamic Pydantic model for this rubric's expected output.

    Cached per criterion specs — calling this 10 times for a 10-criterion
    rubric creates the model once.
    """
    specs = tuple((c.id, c.scale.kind) for c in bundle.rubric.criteria)
    return _build_judgment_model_cached(specs)


# ── Tool construction for structured output ──────────────────────

def build_judgment_tool(bundle: "RubricBundle") -> Tool:
    """Build a harn Tool that forces structured judgment output.

    When passed to Context(tools=[...]), providers like OpenAI and Anthropic
    force the LLM to produce valid JSON matching the schema via tool-calling.
    This is more reliable than text-prompting for JSON.
    """
    Model = build_judgment_model(bundle)
    return Tool(
        name="submit_judgment",
        description=(
            "Submit your evaluation judgment. You MUST call this tool with your "
            "scores, rationale, and evidence for the criterion being evaluated."
        ),
        parameters=Model,
    )


# ── Schema and template generation ───────────────────────────────

def generate_judgment_schema(bundle: "RubricBundle") -> dict:
    """Generate JSON Schema from the rubric for documentation/prompt embedding."""
    Model = build_judgment_model(bundle)
    return Model.model_json_schema()


def generate_judgment_template(bundle: "RubricBundle") -> str:
    """Generate a JSON template (zero-valued instance) from the model.

    Single source of truth: the template is derived from the same dynamic
    model used for validation and tool construction.
    """
    Model = build_judgment_model(bundle)
    # Build default values per criterion
    defaults: dict[str, Any] = {"score": 0, "rationale": "", "evidence": [], "violations": []}
    criterion_defaults = {}
    for c in bundle.rubric.criteria:
        if c.scale.kind == "binary":
            criterion_defaults[c.id] = False
        else:
            criterion_defaults[c.id] = 0
    defaults["criterion_scores"] = criterion_defaults
    return json.dumps(defaults, separators=(",", ":"))


# ── Validation (returns model instance, not dict) ────────────────

def validate_judgment_output(
    parsed: dict[str, Any],
    bundle: "RubricBundle",
) -> tuple[BaseModel | None, list[str]]:
    """Validate parsed LLM output against the rubric's expected schema.

    Returns (validated_model_instance, warnings). On success, warnings is empty
    and the model instance supports typed attribute access.
    On failure, returns (None, warnings).
    """
    Model = build_judgment_model(bundle)
    try:
        validated = Model.model_validate(parsed)
        return (validated, [])
    except ValidationError as e:
        warnings = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return (None, warnings)


__all__ = [
    "ParseError",
    "build_judgment_model",
    "build_judgment_tool",
    "generate_judgment_schema",
    "generate_judgment_template",
    "parse_judgment_json",
    "validate_judgment_output",
]
