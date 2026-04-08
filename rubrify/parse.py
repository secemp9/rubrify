from __future__ import annotations

import json
import re
from typing import Any

from rubrify._types import OutputSchema
from rubrify.result import EvaluationResult


def parse_response(raw: str, output_schema: OutputSchema | None) -> EvaluationResult:
    """Dispatch to JSON or XML parser based on output_schema."""
    if output_schema is None:
        return EvaluationResult(raw=raw)

    if output_schema.constraints.get("must_be_json"):
        return _parse_json_response(raw)
    elif output_schema.constraints.get("must_use_xml_tags"):
        return _parse_xml_response(raw, output_schema)
    else:
        return EvaluationResult(raw=raw)


def _parse_json_response(raw: str) -> EvaluationResult:
    """Parse JSON response from scoring/detection rubrics."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return EvaluationResult(raw=raw)

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return EvaluationResult(raw=raw)

    return EvaluationResult(
        score=data.get("score"),
        label=data.get("class") or data.get("band"),
        verdict=data.get("verdict"),
        subscores=data.get("subscores") or data.get("criterion_scores") or {},
        rationale=data.get("rationale", ""),
        evidence=data.get("evidence", []),
        actions=data.get("actions"),
        diagnostics=data.get("diagnostics"),
        violations=data.get("violations", []),
        advice=_normalize_advice(data.get("advice")),
        risk=data.get("risk"),
        band=data.get("band"),
        raw=raw,
    )


def _parse_xml_response(raw: str, schema: OutputSchema) -> EvaluationResult:
    """Parse XML-tagged response from compliance rubrics."""

    def extract(tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return EvaluationResult(
        verdict=extract("Judgement") or extract("Verdict"),
        rationale=extract("Rationale"),
        raw=raw,
    )


def _normalize_advice(advice: Any) -> list[str] | None:
    if advice is None:
        return None
    if isinstance(advice, str):
        cleaned = advice.removeprefix("FIX:").strip()
        return [s.strip() for s in cleaned.split(";") if s.strip()]
    if isinstance(advice, list):
        return advice
    return None
