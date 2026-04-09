from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

from rubrify._types import OutputSchema
from rubrify.result import EvaluationResult

# JSON keys that are consumed directly into named EvaluationResult fields.
# Anything not in this set is passed through verbatim in ``result.extras``.
_KNOWN_JSON_KEYS: frozenset[str] = frozenset(
    {
        "score",
        "class",
        "band",
        "verdict",
        "subscores",
        "criterion_scores",
        "rationale",
        "evidence",
        "actions",
        "diagnostics",
        "violations",
        "advice",
        "risk",
    }
)

# XML tags that are consumed directly into named EvaluationResult fields.
# The comparison is case-insensitive to match the legacy regex-based parser.
_KNOWN_XML_TAGS_LOWER: frozenset[str] = frozenset({"rationale", "judgement", "verdict"})


def parse_response(
    raw: str,
    output_schema: OutputSchema | None,
    *,
    repair: bool = False,
) -> EvaluationResult:
    """Dispatch to JSON or XML parser based on ``output_schema``.

    When ``repair=True``, :func:`rubrify.repair.attempt_schema_repair` runs
    before parsing and its notes are attached to the returned result. The
    default path (``repair=False``) is unchanged from the pre-Phase-1
    implementation except that unknown fields are now surfaced via
    ``result.extras``.
    """
    repaired = False
    repair_notes: tuple[str, ...] = ()
    working = raw

    if repair:
        from rubrify.repair import attempt_schema_repair

        repair_result = attempt_schema_repair(raw, output_schema)
        repair_notes = repair_result.notes
        if repair_result.repaired:
            working = repair_result.text
            repaired = True

    if output_schema is None:
        result = EvaluationResult(raw=raw)
    elif output_schema.constraints.get("must_be_json"):
        result = _parse_json_response(working)
    elif output_schema.constraints.get("must_use_xml_tags"):
        result = _parse_xml_response(working, output_schema)
    else:
        result = EvaluationResult(raw=raw)

    if repair:
        result = dataclasses.replace(
            result,
            raw=raw,
            repaired=repaired,
            repair_notes=repair_notes,
        )

    return result


def _parse_json_response(raw: str, *, repair: bool = False) -> EvaluationResult:
    """Parse JSON response from scoring/detection rubrics.

    Two paths controlled by ``repair``:

    * ``repair=False`` (default): try ``json.loads(raw)`` directly. If that
      fails, use ``json.JSONDecoder().raw_decode()`` to find the first valid
      JSON object in the string — this is a standard JSON parsing strategy,
      not repair. If that also fails, return ``EvaluationResult(raw=raw)``
      (``score=None`` + ``raw`` populated indicates parse failure).

    * ``repair=True``: use the repair layer's ``extract_json_candidate``
      which handles code fences, prose wrappers, and balanced-brace
      matching. When salvage occurs, ``repaired=True`` and
      ``repair_notes`` are set on the result.

    Unknown top-level keys are preserved under ``result.extras`` verbatim.
    """
    repaired = False
    repair_notes: tuple[str, ...] = ()

    if repair:
        from rubrify.repair import extract_json_candidate

        extraction = extract_json_candidate(raw)
        repaired = extraction.repaired
        repair_notes = extraction.notes
        text = extraction.text
    else:
        text = raw

    # Try direct parse first.
    data = None
    with contextlib.suppress(json.JSONDecodeError):
        data = json.loads(text)

    # If direct parse failed and we're not in repair mode, try raw_decode
    # to find the first valid JSON object in the string.
    if data is None and not repair:
        try:
            decoder = json.JSONDecoder()
            # Scan forward to the first '{' character.
            idx = text.find("{")
            if idx >= 0:
                data, _ = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, ValueError):
            pass

    if data is None:
        result = EvaluationResult(raw=raw)
        if repair:
            result = dataclasses.replace(
                result,
                repaired=repaired,
                repair_notes=repair_notes,
            )
        return result

    if not isinstance(data, dict):
        result = EvaluationResult(raw=raw)
        if repair:
            result = dataclasses.replace(
                result,
                repaired=repaired,
                repair_notes=repair_notes,
            )
        return result

    extras: dict[str, Any] = {
        key: value for key, value in data.items() if key not in _KNOWN_JSON_KEYS
    }

    result = EvaluationResult(
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
        extras=extras,
    )

    if repair:
        result = dataclasses.replace(
            result,
            repaired=repaired,
            repair_notes=repair_notes,
        )

    return result


def _parse_xml_response(raw: str, schema: OutputSchema) -> EvaluationResult:
    """Parse XML-tagged response from compliance rubrics.

    Known tags (``Rationale``, ``Judgement``, ``Verdict``) feed the named
    result fields. Any other top-level tag is preserved under ``result.extras``
    — as a string for leaf elements, or as a nested dict for elements with
    children.
    """

    def extract(tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    verdict = extract("Judgement") or extract("Verdict")
    rationale = extract("Rationale")
    extras = _extract_xml_extras(raw)

    return EvaluationResult(
        verdict=verdict,
        rationale=rationale,
        raw=raw,
        extras=extras,
    )


def _extract_xml_extras(raw: str) -> dict[str, Any]:
    """Best-effort extraction of unknown top-level XML tags into a dict.

    Wraps ``raw`` in a synthetic root so multi-sibling fragments parse. Known
    tags are skipped. For leaf elements the extra is the stripped text; for
    elements with children the extra is a nested dict produced by
    :func:`_element_to_py`.
    """
    extras: dict[str, Any] = {}
    try:
        root = ET.fromstring(f"<__rubrify_root__>{raw}</__rubrify_root__>")
    except ET.ParseError:
        return extras

    for child in root:
        if child.tag.lower() in _KNOWN_XML_TAGS_LOWER:
            continue
        extras[child.tag] = _element_to_py(child)
    return extras


def _element_to_py(elem: ET.Element) -> Any:
    """Convert an XML element into a JSON-serializable Python value."""
    if len(elem) == 0:
        return (elem.text or "").strip()
    result: dict[str, Any] = {}
    for child in elem:
        result[child.tag] = _element_to_py(child)
    return result


def _normalize_advice(advice: Any) -> list[str] | None:
    if advice is None:
        return None
    if isinstance(advice, str):
        cleaned = advice.removeprefix("FIX:").strip()
        return [s.strip() for s in cleaned.split(";") if s.strip()]
    if isinstance(advice, list):
        return advice
    return None
