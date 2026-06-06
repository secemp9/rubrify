"""Criterion execution: LLM calls for judging rubric criteria.

Three execution modes:
  1. Single criterion (execute_criterion): One LLM call evaluates one
     criterion. Used when execution strategy is "isolated".
  2. Grouped criteria (execute_group): One LLM call evaluates multiple
     criteria and returns per-criterion scores. Used when execution
     strategy is "grouped" or "holistic".
  3. Tool-based vs text-based: Both modes support tool-calling (structured
     output) or text-based (JSON parsing from response text).

Both strategies share the same extraction logic. The binding's output_field
drives score access via typed Pydantic model attribute access, not dict
navigation with string splitting.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

from pydantic import BaseModel

from harn_ai.stream import complete_simple
from harn_ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    UserMessage,
)

from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import ConstraintBinding
from rubrify.ir.types import Criterion
from rubrify.codecs.xml_codec import render_rubric_xml, render_criterion_xml, render_group_xml
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

    When use_tool=True (default), uses harn's Tool system to force
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

    for constraint in bundle.output_constraints:
        parts.append(f"CONSTRAINT [{constraint.id}]: {constraint.description}")

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


async def execute_group(
    criteria: list[Criterion],
    bundle: RubricBundle,
    response_text: str,
    model: Model,
    api_key: str | None = None,
    use_tool: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[list[CriterionJudgment], JudgeUsage]:
    """Execute a grouped judgment: ONE LLM call scores multiple criteria.

    This is the heart of the grouped/holistic execution strategy. Instead of
    "Focus exclusively on C1", the prompt says "Score all criteria" and then
    per-criterion scores are extracted from the response's criterion_scores dict.

    Returns a list of CriterionJudgments (one per criterion in the group) and
    a JudgeUsage summarizing the single API call's token consumption.
    """
    usage = JudgeUsage()

    # A) Build system prompt using render_group_xml for the criteria subset
    system_prompt = _build_group_system_prompt(criteria, bundle)

    # B) Build user prompt that asks for ALL criteria to be scored
    user_prompt = _build_group_user_prompt(criteria, response_text, use_tool)

    # C) Build subset-aware tool
    tool = build_judgment_tool(bundle, criteria=criteria) if use_tool else None

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

    # D) Make ONE call to complete_simple
    result = await complete_simple(model, context, opts)

    # Track usage
    usage.input_tokens += result.usage.input
    usage.output_tokens += result.usage.output
    usage.total_tokens += result.usage.totalTokens
    usage.api_calls += 1

    # E) Parse the response
    parsed: dict = {}

    # Strategy 1: Extract from tool call
    for block in result.content:
        if block.type == "toolCall" and block.name == "submit_judgment":
            parsed = block.arguments if isinstance(block.arguments, dict) else {}
            break

    # Strategy 2: Extract from text response (fallback)
    if not parsed:
        raw_text = ""
        for block in result.content:
            if block.type == "text":
                raw_text += block.text
        try:
            parsed = parse_judgment_json(raw_text)
        except ParseError:
            # Total failure — return empty judgments with warnings
            return (
                [
                    CriterionJudgment(
                        criterion_id=c.id,
                        value=None,
                        unit_score=0.0,
                        rationale="",
                        confidence=None,
                        warnings=["Group execution failed: could not parse LLM response"],
                    )
                    for c in criteria
                ],
                usage,
            )

    # Extract shared fields from response
    rationale = parsed.get("rationale", "")
    evidence_raw = parsed.get("evidence", [])
    violations = parsed.get("violations", [])

    # Build shared evidence list
    evidence: list[EvidenceQuote] = []
    for item in evidence_raw:
        if isinstance(item, str):
            evidence.append(EvidenceQuote(source="response", text=item))

    # Extract criterion_scores dict
    criterion_scores = parsed.get("criterion_scores", {})
    if isinstance(criterion_scores, BaseModel):
        # Pydantic model from tool call — convert to dict via attribute access
        criterion_scores = {
            c.id: getattr(criterion_scores, c.id, None) for c in criteria
        }
    elif not isinstance(criterion_scores, dict):
        criterion_scores = {}

    # F) Construct per-criterion CriterionJudgments
    judgments: list[CriterionJudgment] = []
    for criterion in criteria:
        warnings: list[str] = []

        raw_value = criterion_scores.get(criterion.id)
        if raw_value is None:
            warnings.append(
                f"Score not found for criterion '{criterion.id}' in grouped response"
            )

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

        judgments.append(
            CriterionJudgment(
                criterion_id=criterion.id,
                value=raw_value,
                unit_score=unit_score,
                evidence=evidence,
                rationale=rationale,
                confidence=parsed.get("confidence"),
                warnings=warnings,
            )
        )

    # G) Return list of judgments + usage
    return (judgments, usage)


def _build_group_system_prompt(
    criteria: list[Criterion],
    bundle: RubricBundle,
) -> str:
    """Build the system prompt for a grouped evaluation using render_group_xml."""
    parts: list[str] = []
    parts.append(render_group_xml(criteria, bundle))

    for constraint in bundle.output_constraints:
        parts.append(f"CONSTRAINT [{constraint.id}]: {constraint.description}")

    return "\n\n".join(parts)


def _build_group_user_prompt(
    criteria: list[Criterion],
    response_text: str,
    use_tool: bool,
) -> str:
    """Build the user prompt for evaluating ALL criteria in a group."""
    criteria_summary = ", ".join(
        f"[{c.id}] {c.title}" for c in criteria
    )
    submit_instruction = (
        "Call the submit_judgment tool with your evaluation."
        if use_tool
        else "Return ONLY the JSON specified in the output schema."
    )
    return (
        f"Evaluate the following response on ALL criteria listed in the rubric spec above. "
        f"For each criterion, provide a score. Provide one overall rationale and shared evidence.\n\n"
        f"Criteria to score: {criteria_summary}\n\n"
        f"<response_under_test>\n{response_text}\n</response_under_test>\n\n"
        f"Score this response on every criterion using the scales and anchors defined "
        f"in the rubric above. {submit_instruction}"
    )


__all__ = [
    "execute_criterion",
    "execute_group",
]
