"""Repair layer — local-only recovery of malformed model output.

Phase 1 deliverable. When a model wraps JSON in prose, forgets a closing tag,
or emits a code fence around its structured output, the repair helpers in this
module attempt to surface a usable payload **without silent fallbacks**: every
``RepairResult`` carries a notes tuple describing what was (or was not) done.

Strictly local. No model-assisted repair. Guard rail 6 (no silent fallbacks)
is binding: when the helpers cannot extract a usable candidate, they return
``repaired=False`` with explicit notes rather than swallowing the failure.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from rubrify._types import OutputSchema


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Outcome of a repair attempt.

    ``text`` is either the original input (when no repair was applied or
    repair failed) or the extracted candidate. ``repaired`` is ``True`` only
    when the returned ``text`` differs from the input in a repair-meaningful
    way. ``notes`` is always populated when the result is not a clean pass
    (``repaired=False`` with empty notes means: input parsed cleanly on the
    first attempt).
    """

    text: str
    repaired: bool
    notes: tuple[str, ...]


def _iter_code_fences(raw: str) -> list[tuple[str, str]]:
    """Return ``(lang, content)`` pairs for markdown code fences found in ``raw``.

    Uses markdown-it-py so behavior matches the generator's XML extraction path.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    tokens = md.parse(raw)
    fences: list[tuple[str, str]] = []
    for token in tokens:
        if token.type == "fence":
            info = (token.info or "").strip()
            fences.append((info, token.content))
    return fences


def _find_largest_json_object(raw: str) -> str | None:
    """Find the largest substring of ``raw`` that parses as a JSON object.

    Walks every ``{`` and scans forward with string-aware brace counting.
    Returns ``None`` if no balanced object parses successfully.
    """
    best: str | None = None
    for start in (i for i, c in enumerate(raw) if c == "{"):
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(data, dict) and (best is None or len(candidate) > len(best)):
                        best = candidate
                    break
    return best


def extract_json_candidate(raw: str) -> RepairResult:
    """Extract a JSON object from ``raw`` using progressive strategies.

    Strategies, in order:

    1. Parse ``raw`` directly. If it is a valid JSON object, return it
       unrepaired.
    2. Parse markdown code fences (``json`` or unlabeled) found in ``raw``.
    3. Brace-scan for the largest substring that parses as a JSON object.

    If every strategy fails, return the original ``raw`` with
    ``repaired=False`` and a descriptive notes entry.
    """
    # Strategy 1: direct parse.
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return RepairResult(text=raw, repaired=False, notes=())
    except json.JSONDecodeError:
        pass

    # Strategy 2: code fences.
    try:
        fences = _iter_code_fences(raw)
    except Exception:  # pragma: no cover - markdown-it-py internal failure
        fences = []
    for lang, content in fences:
        if lang and lang.lower() not in ("json", "javascript", "js"):
            continue
        stripped = content.strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            label = lang.lower() if lang else "unlabeled"
            return RepairResult(
                text=stripped,
                repaired=True,
                notes=(f"extracted from {label} code fence",),
            )

    # Strategy 3: brace matching.
    candidate = _find_largest_json_object(raw)
    if candidate is not None:
        return RepairResult(
            text=candidate,
            repaired=True,
            notes=("found via brace matching",),
        )

    # Strategy 4: give up explicitly.
    return RepairResult(
        text=raw,
        repaired=False,
        notes=("no valid JSON object found",),
    )


def _parses_as_xml_fragment(raw: str) -> bool:
    """Return ``True`` if ``raw`` parses as XML when wrapped in a synthetic root.

    The wrapping accommodates rubric outputs that emit multiple sibling tags
    (e.g. ``<Rationale>...</Rationale><Judgement>...</Judgement>``) which are
    not a well-formed document on their own but are valid fragments.
    """
    try:
        ET.fromstring(f"<__rubrify_root__>{raw}</__rubrify_root__>")
    except ET.ParseError:
        return False
    return True


def extract_xml_candidate(raw: str, required_tags: tuple[str, ...] = ()) -> RepairResult:
    """Extract an XML fragment from ``raw`` using progressive strategies.

    Strategies, in order:

    1. Parse ``raw`` directly (wrapped in a synthetic root) with ElementTree.
    2. Parse markdown code fences (``xml`` or unlabeled) found in ``raw``.
    3. If ``required_tags`` is provided, regex-extract each named tag pair
       from ``raw`` and splice the results together.

    On failure returns ``repaired=False`` with explicit notes.
    """
    # Strategy 1: direct parse.
    if _parses_as_xml_fragment(raw):
        return RepairResult(text=raw, repaired=False, notes=())

    # Strategy 2: code fences.
    try:
        fences = _iter_code_fences(raw)
    except Exception:  # pragma: no cover - markdown-it-py internal failure
        fences = []
    for lang, content in fences:
        if lang and lang.lower() not in ("xml", "html"):
            continue
        stripped = content.strip()
        if _parses_as_xml_fragment(stripped):
            label = lang.lower() if lang else "unlabeled"
            return RepairResult(
                text=stripped,
                repaired=True,
                notes=(f"extracted from {label} code fence",),
            )

    # Strategy 3: required-tag regex scrape.
    if required_tags:
        recovered: list[str] = []
        found: list[str] = []
        for tag in required_tags:
            pattern = rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>"
            m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
            if m:
                elem = ET.Element(tag)
                elem.text = m.group(1).strip()
                recovered.append(ET.tostring(elem, encoding="unicode", short_empty_elements=False))
                found.append(tag)
        if recovered:
            return RepairResult(
                text="\n".join(recovered),
                repaired=True,
                notes=(f"regex-scraped required tags: {', '.join(found)}",),
            )

    # Strategy 4: give up explicitly.
    return RepairResult(
        text=raw,
        repaired=False,
        notes=("no valid XML structure found",),
    )


def attempt_schema_repair(raw: str, schema: OutputSchema | None) -> RepairResult:
    """Dispatch to the right extractor based on ``schema``.

    * ``schema is None`` → unrepaired with ``("no schema provided",)``.
    * ``must_be_json`` → :func:`extract_json_candidate`.
    * ``must_use_xml_tags`` → :func:`extract_xml_candidate` with the known
      compliance tags as required.
    * Any other schema → unrepaired with ``("schema has no recognized format",)``.
    """
    if schema is None:
        return RepairResult(text=raw, repaired=False, notes=("no schema provided",))

    if schema.constraints.get("must_be_json"):
        return extract_json_candidate(raw)

    if schema.constraints.get("must_use_xml_tags"):
        return extract_xml_candidate(raw, required_tags=("Rationale", "Judgement", "Verdict"))

    return RepairResult(
        text=raw,
        repaired=False,
        notes=("schema has no recognized format",),
    )
