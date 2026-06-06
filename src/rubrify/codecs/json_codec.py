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
from pydantic_core import PydanticUndefined

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


def build_judgment_model(bundle: "RubricBundle", criteria: list | None = None) -> type:
    """Build a dynamic Pydantic model for this rubric's expected output.

    Cached per criterion specs — calling this 10 times for a 10-criterion
    rubric creates the model once.

    If *criteria* is provided, build the model using only those criteria
    (for group/holistic calls where a subset is evaluated in one call).
    If None, uses all criteria from bundle.rubric.criteria (backwards
    compatible).
    """
    source = criteria if criteria is not None else bundle.rubric.criteria
    specs = tuple((c.id, c.scale.kind) for c in source)
    return _build_judgment_model_cached(specs)


# ── Tool construction for structured output ──────────────────────

def build_judgment_tool(bundle: "RubricBundle", criteria: list | None = None) -> Tool:
    """Build a harn Tool that forces structured judgment output.

    When passed to Context(tools=[...]), providers like OpenAI and Anthropic
    force the LLM to produce valid JSON matching the schema via tool-calling.
    This is more reliable than text-prompting for JSON.

    If *criteria* is provided, build the tool for only those criteria
    (subset evaluation in group/holistic execution strategies).
    """
    Model = build_judgment_model(bundle, criteria=criteria)
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


def _zero_value_for_annotation(annotation: Any) -> Any:
    """Return a sensible zero/empty value for a type annotation.

    Handles union types (e.g. ``int | float``) by picking the first concrete
    member, and falls back to ``None`` for anything unrecognised.
    """
    import types as _types
    import typing

    # Unwrap PEP 604 union (int | float | str) — a types.UnionType instance
    if isinstance(annotation, _types.UnionType):
        return _zero_value_for_annotation(annotation.__args__[0])

    # Also handle typing.Union[int, float, str]
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        return _zero_value_for_annotation(annotation.__args__[0])

    # Concrete scalar types
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return ""
    if annotation is list or (origin is list):
        return []
    if annotation is dict or (origin is dict):
        return {}

    # Pydantic sub-model: recurse into its fields
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_to_template(annotation)

    return None


def _model_to_template(model: type[BaseModel]) -> dict[str, Any]:
    """Build a zero-valued template dict from a Pydantic model's fields.

    For each field the logic is:
    1. Use ``field.default`` if one is set (and is not PydanticUndefined).
    2. Call ``field.default_factory()`` if one is set.
    3. Otherwise derive a zero value from the field's type annotation.
    """
    template: dict[str, Any] = {}
    for name, field_info in model.model_fields.items():
        if field_info.default is not PydanticUndefined:
            template[name] = field_info.default
        elif field_info.default_factory is not None:
            template[name] = field_info.default_factory()
        else:
            template[name] = _zero_value_for_annotation(field_info.annotation)
    return template


def generate_judgment_template(bundle: "RubricBundle", criteria: list | None = None) -> str:
    """Generate a JSON template (zero-valued instance) from the model.

    Single source of truth: the template is derived from the same dynamic
    model used for validation and tool construction.

    If *criteria* is provided, generate a template for only those criteria
    (subset evaluation in group/holistic execution strategies).
    """
    Model = build_judgment_model(bundle, criteria=criteria)
    return json.dumps(_model_to_template(Model), separators=(",", ":"))


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
