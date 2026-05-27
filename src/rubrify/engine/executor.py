"""CriterionExecutor: the single LLM call per criterion.

Two execution strategies:
  1. Tool-based: Define judgment output as a harnify Tool. The provider
     forces structured JSON via tool-calling. No text parsing needed.
  2. Text-based: Send text prompt, parse JSON from response text.
     Used when provider does not support tool-calling.

Both strategies share the same extraction logic. The binding's output_field
drives score access via typed Pydantic model attribute access, not dict
navigation with string splitting.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
import defusedxml.ElementTree as SafeET  # noqa: F401

from pydantic import BaseModel

from harnify_ai.stream import complete_simple
from harnify_ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    UserMessage,
)

from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import ConstraintBinding
from rubrify.ir.types import Criterion
from rubrify.codecs.xml_codec import render_rubric_xml, render_criterion_xml
from rubrify.codecs.json_codec import (
    ParseError,
    build_judgment_tool,
    parse_judgment_json,
    validate_judgment_output,
)
from rubrify.engine.judgment import CriterionJudgment, EvidenceQuote, JudgeUsage


async def execute_criterion(
    criterion: Criterion,
    bundle: RubricBundle,
    response_text: str,
    model: Model,
    *,
    binding: ConstraintBinding | None = None,
    context_text: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    usage: JudgeUsage | None = None,
    use_tool: bool = True,
) -> CriterionJudgment:
    """Execute a single criterion judgment via one LLM call.

    When use_tool=True (default), uses harnify's Tool system to force
    structured output via the provider's native tool-calling mechanism.
    Falls back to text-based extraction if the response has no tool call.
    """
    system_prompt = _build_system_prompt(criterion, bundle, context_text)
    user_prompt = _build_user_prompt(criterion, response_text, use_tool)

    tool = build_judgment_tool(bundle) if use_tool else None

    context = Context(
        systemPrompt=system_prompt,
        messages=[
            UserMessage(
                role="user",
                content=user_prompt,
                timestamp=int(time.time() * 1000),
            ),
        ],
        tools=[tool] if tool else None,
    )

    opts = SimpleStreamOptions(
        apiKey=api_key,
        temperature=temperature,
        maxTokens=max_tokens,
    )

    result = await complete_simple(model, context, opts)

    # Track usage
    if usage is not None:
        usage.input_tokens += result.usage.input
        usage.output_tokens += result.usage.output
        usage.total_tokens += result.usage.totalTokens
        usage.api_calls += 1

    # Strategy 1: Extract from tool call (pre-parsed, no repair needed)
    for block in result.content:
        if block.type == "toolCall" and block.name == "submit_judgment":
            validated, warnings = validate_judgment_output(block.arguments, bundle)
            return _extract_from_validated(criterion, validated, block.arguments, binding, warnings)

    # Strategy 2: Extract from text response
    raw_text = ""
    for block in result.content:
        if block.type == "text":
            raw_text += block.text

    try:
        parsed = parse_judgment_json(raw_text)
    except ParseError as e:
        return CriterionJudgment(
            criterion_id=criterion.id,
            value=None,
            unit_score=0.0,
            rationale="",
            confidence=None,
            warnings=[f"JSON parse failed: {e}"],
        )

    validated, warnings = validate_judgment_output(parsed, bundle)
    return _extract_from_validated(criterion, validated, parsed, binding, warnings)


def _extract_from_validated(
    criterion: Criterion,
    validated: BaseModel | None,
    raw_parsed: dict,
    binding: ConstraintBinding | None,
    validation_warnings: list[str],
) -> CriterionJudgment:
    """Extract a CriterionJudgment from a validated Pydantic model instance.

    Uses typed attribute access on the model — no _get_nested, no dict
    navigation with string splitting. The binding tells us which criterion
    field to read; the model type guarantees it exists if validation passed.
    """
    warnings = list(validation_warnings)

    # Score extraction via typed model attribute access
    raw_value = None
    if validated is not None and binding:
        criterion_scores = getattr(validated, "criterion_scores", None)
        if criterion_scores is not None:
            raw_value = getattr(criterion_scores, criterion.id, None)
        if raw_value is None:
            warnings.append(f"Score not found for criterion '{criterion.id}' in validated output")
    elif validated is None:
        # Validation failed — try raw dict as last resort
        scores = raw_parsed.get("criterion_scores", {})
        if isinstance(scores, dict):
            raw_value = scores.get(criterion.id)
        if raw_value is None:
            warnings.append(f"Validation failed and score not found for '{criterion.id}'")
    else:
        warnings.append(f"No binding for criterion {criterion.id}")

    # Normalize to unit [0, 1]
    unit_score = 0.0
    if raw_value is not None:
        try:
            if isinstance(raw_value, bool):
                unit_score = criterion.scale.to_unit(raw_value)
            else:
                unit_score = criterion.scale.to_unit(
                    float(raw_value) if not isinstance(raw_value, str) else raw_value
                )
        except (ValueError, TypeError) as e:
            warnings.append(f"Scale normalization failed for {criterion.id}: {e}")

    # Extract evidence and rationale from validated model or raw dict
    source = validated if validated is not None else raw_parsed
    rationale = _get_field(source, "rationale", "")
    evidence_raw = _get_field(source, "evidence", [])
    confidence = _get_field(source, "confidence", None)

    evidence: list[EvidenceQuote] = []
    for item in evidence_raw:
        if isinstance(item, str):
            evidence.append(EvidenceQuote(source="response", text=item))

    return CriterionJudgment(
        criterion_id=criterion.id,
        value=raw_value,
        unit_score=unit_score,
        evidence=evidence,
        rationale=rationale,
        confidence=confidence,
        warnings=warnings,
    )


def _get_field(source: BaseModel | dict, field: str, default: object = None) -> object:
    """Get a field from either a Pydantic model instance or a dict."""
    if isinstance(source, BaseModel):
        return getattr(source, field, default)
    if isinstance(source, dict):
        return source.get(field, default)
    return default


def _build_system_prompt(
    criterion: Criterion,
    bundle: RubricBundle,
    context_text: str | None,
) -> str:
    """Build the full system prompt with rubric XML."""
    parts: list[str] = []

    policy = bundle.surface_policy
    focus = getattr(policy, "criterion_focus", "full")
    if focus == "focused":
        parts.append(render_criterion_xml(criterion, bundle))
    else:
        parts.append(render_rubric_xml(bundle))

    if context_text:
        data_block = next(
            (b for b in bundle.authority_blocks if b.kind == "context_document"),
            None,
        )
        if data_block is None:
            raise RuntimeError("Bundle missing required authority block 'context_document'")
        context_el = ET.Element("context")
        context_el.set("authority", data_block.authority)
        context_el.set("follow", str(data_block.model_should_follow).lower())
        context_el.text = (
            "\nThe following is the reference context. "
            "Ground your evaluation against this.\n"
            f"{context_text}\n"
        )
        parts.append(ET.tostring(context_el, encoding="unicode"))

    for ritual in bundle.rituals:
        parts.append(f"CONSTRAINT [{ritual.id}]: {ritual.description}")

    return "\n\n".join(parts)


def _build_user_prompt(criterion: Criterion, response_text: str, use_tool: bool) -> str:
    """Build the user-facing prompt for evaluating one criterion."""
    submit_instruction = (
        "Call the submit_judgment tool with your evaluation."
        if use_tool
        else "Return ONLY the JSON specified in the output schema."
    )
    return (
        f"Evaluate the following response for criterion [{criterion.id}] {criterion.title}.\n\n"
        f"Criterion description: {criterion.description}\n\n"
        f"<response_under_test>\n{response_text}\n</response_under_test>\n\n"
        f"Score this response on criterion {criterion.id} using the scale and anchors defined "
        f"in the rubric above. {submit_instruction} "
        f"Focus exclusively on {criterion.id}. Ignore other criteria."
    )


__all__ = [
    "execute_criterion",
]
