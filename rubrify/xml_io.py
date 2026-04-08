"""XML serialization and deserialization for rubric specs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
from xml.dom.minidom import parseString

from rubrify._types import (
    AdviceRule,
    Criterion,
    DecisionRule,
    Disqualifier,
    InputField,
    MappingExample,
    OutputSchema,
    PatternLibrary,
    Scoring,
    ValidationMust,
)

if TYPE_CHECKING:
    from rubrify.rubric import ConstraintRubric, Rubric


def _text(elem: ET.Element | None) -> str:
    """Get stripped text content of an element, or empty string."""
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _text_raw(elem: ET.Element | None) -> str:
    """Get text content preserving internal whitespace, but strip leading/trailing."""
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _bool_attr(elem: ET.Element, attr: str, default: bool = False) -> bool:
    val = elem.get(attr, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def rubric_from_xml(xml_string: str) -> Rubric:
    """Parse an XML string into a Rubric object."""
    from rubrify.rubric import Rubric

    root = ET.fromstring(xml_string)

    r = Rubric(
        name=root.get("name", ""),
        version=root.get("version", "1.0"),
        mission=_text(root.find("mission")),
    )

    # Step 4: Parse <inputs>/<field>
    inputs_elem = root.find("inputs")
    if inputs_elem is not None:
        for field_elem in inputs_elem.findall("field"):
            r.inputs.append(
                InputField(
                    name=field_elem.get("name", ""),
                    required=_bool_attr(field_elem, "required"),
                    description=_text(field_elem),
                )
            )

    # Step 5: Parse <what_to_judge>
    wtj = root.find("what_to_judge")
    if wtj is not None:
        r.what_to_judge = _text_raw(wtj)

    # Step 6: Parse <definitions>/<def>
    defs_elem = root.find("definitions")
    if defs_elem is not None:
        for def_elem in defs_elem.findall("def"):
            def_id = def_elem.get("id", "")
            r.definitions[def_id] = _text(def_elem)

    # Step 7: Parse <rubric>/<criterion> and step 8: <disqualifiers>
    rubric_elem = root.find("rubric")
    if rubric_elem is not None:
        for crit_elem in rubric_elem.findall("criterion"):
            crit = _parse_criterion(crit_elem)
            r.add_criterion(crit)

        # Step 8: Parse <disqualifiers>/<dq>
        dq_container = rubric_elem.find("disqualifiers")
        if dq_container is not None:
            for dq_elem in dq_container.findall("dq"):
                r.add_disqualifier(
                    Disqualifier(
                        id=dq_elem.get("id", ""),
                        description=_text(dq_elem),
                    )
                )

    # Step 9: Parse <pattern_library> (ZinsserJudge variant with <list>/<regex>)
    # Step 10: Parse <regex_library> (AntiLLMY variant with <pattern>)
    # Also handle ComplianceJudge variant with <refusal_regexes>/<r>, etc.
    pl = _parse_pattern_library(root)
    if pl is not None:
        r.pattern_library = pl

    # Step 11: Parse <decision_logic>/<rule>
    dl_elem = root.find("decision_logic")
    if dl_elem is not None:
        for rule_elem in dl_elem.findall("rule"):
            r.add_decision_rule(
                DecisionRule(
                    id=rule_elem.get("id", ""),
                    condition=_text(rule_elem),
                )
            )

    # Step 12: Parse <mapping_examples>/<example>
    me_elem = root.find("mapping_examples")
    if me_elem is not None:
        for ex_elem in me_elem.findall("example"):
            r.add_mapping_example(
                MappingExample(
                    id=ex_elem.get("id", ""),
                    user=_text(ex_elem.find("user")),
                    assistant=_text(ex_elem.find("assistant")),
                    verdict=_text(ex_elem.find("verdict")),
                )
            )

    # Step 13: Parse <output_schema>
    os_elem = root.find("output_schema")
    if os_elem is not None:
        r.output_schema = _parse_output_schema(os_elem)

    # Step 14: Parse <scoring>
    scoring_elem = root.find("scoring")
    if scoring_elem is not None:
        r.scoring = _parse_scoring(scoring_elem)

    # Step 15: Parse <advice_rules>/<rule>
    ar_elem = root.find("advice_rules")
    if ar_elem is not None:
        for rule_elem in ar_elem.findall("rule"):
            when_str = rule_elem.get("when", "")
            r.add_advice_rule(
                AdviceRule(
                    when=when_str.split("|") if when_str else [],
                    advice=_text(rule_elem),
                )
            )

    # Step 16: Parse <validation>/<must>
    val_elem = root.find("validation")
    if val_elem is not None:
        for must_elem in val_elem.findall("must"):
            r.validation_musts.append(ValidationMust(description=_text(must_elem)))

    # Step 17: Parse <scoring_guidance>/<mapping>
    sg_elem = root.find("scoring_guidance")
    if sg_elem is not None:
        mapping_elem = sg_elem.find("mapping")
        if mapping_elem is not None:
            r.scoring_guidance = _text(mapping_elem)

    return r


def _parse_criterion(elem: ET.Element) -> Criterion:
    """Parse a <criterion> element into a Criterion dataclass."""
    crit_id = elem.get("id", "")
    name = elem.get("name", "")
    weight = int(elem.get("weight", "0"))
    genre = elem.get("genre")

    anchors: dict[int, str] = {}
    for child in elem:
        tag = child.tag
        m = re.match(r"anchor_(\d+)", tag)
        if m:
            anchors[int(m.group(1))] = _text(child)

    mechanical_rules: list[str] = []
    mr_elem = elem.find("mechanical_rules")
    if mr_elem is not None:
        for rule_elem in mr_elem.findall("rule"):
            mechanical_rules.append(_text(rule_elem))

    uses_patterns: list[str] | None = None
    up_elem = elem.find("uses_patterns")
    if up_elem is not None:
        raw = _text(up_elem)
        if raw:
            uses_patterns = [p.strip() for p in raw.split(",")]

    notes: str | None = None
    notes_elem = elem.find("notes")
    if notes_elem is not None:
        notes = _text(notes_elem)

    return Criterion(
        id=crit_id,
        name=name,
        weight=weight,
        anchors=anchors,
        mechanical_rules=mechanical_rules,
        uses_patterns=uses_patterns,
        genre=genre,
        notes=notes,
    )


def _parse_pattern_library(root: ET.Element) -> PatternLibrary | None:
    """Parse pattern library from multiple XML variants."""
    # Variant 1: <pattern_library> with <list>/<regex> children (ZinsserJudge)
    # Variant 2: <regex_library flags="i"> with <pattern> children (AntiLLMY)
    # Variant 3: <pattern_library> with named group children containing <r>/<m> (ComplianceJudge)

    pl_elem = root.find("pattern_library")
    rl_elem = root.find("regex_library")

    if pl_elem is None and rl_elem is None:
        return None

    lib = PatternLibrary()

    if rl_elem is not None:
        # Variant 2: <regex_library flags="i"> with <pattern id="...">
        lib.flags = rl_elem.get("flags", "")
        for pat_elem in rl_elem.findall("pattern"):
            pat_id = pat_elem.get("id", "")
            lib.entries[pat_id] = _text(pat_elem)
            lib._entry_types[pat_id] = "pattern"
        return lib

    if pl_elem is not None:
        # Check for Variant 1 (direct <list>/<regex> children) vs Variant 3 (named groups)
        has_direct = pl_elem.find("list") is not None or pl_elem.find("regex") is not None

        if has_direct:
            # Variant 1: <list id="...">, <regex id="...">
            for child in pl_elem:
                child_id = child.get("id", "")
                if child.tag == "list":
                    lib.entries[child_id] = _text(child)
                    lib._entry_types[child_id] = "list"
                elif child.tag == "regex":
                    lib.entries[child_id] = _text(child)
                    lib._entry_types[child_id] = "regex"
        else:
            # Variant 3: ComplianceJudge style with named group children
            # e.g., <refusal_regexes>/<r>, <soft_refusal_regexes>/<r>,
            #        <actionability_markers>/<m>
            for group_elem in pl_elem:
                group_name = group_elem.tag
                patterns: list[str] = []
                for child in group_elem:
                    if child.tag in ("r", "m"):
                        # Use raw text (no strip) to preserve regex content exactly
                        text = child.text or ""
                        if text:
                            patterns.append(text)
                if patterns:
                    # Store individual patterns for round-trip fidelity
                    lib._group_patterns[group_name] = patterns
                    # Also store pipe-joined for API use
                    lib.entries[group_name] = "|".join(patterns)
                    lib._entry_types[group_name] = (
                        group_elem.tag + "/" + (group_elem[0].tag if len(group_elem) > 0 else "r")
                    )
                    lib._entry_types[f"_group_{group_name}"] = "group"

        return lib

    return None


def _parse_output_schema(elem: ET.Element) -> OutputSchema:
    """Parse <output_schema> element."""
    schema = OutputSchema()

    # Parse template
    json_template = elem.find("json_template")
    template_elem = elem.find("template")
    if json_template is not None:
        schema.template = _text_raw(json_template)
        schema.format = "json"
    elif template_elem is not None:
        # Handle CDATA or direct text
        text = ""
        if template_elem.text:
            text = template_elem.text.strip()
        schema.template = text
        schema.format = "xml"

    # Parse constraints
    constraints_elem = elem.find("constraints")
    if constraints_elem is not None:
        for child in constraints_elem:
            key = child.tag
            val_text = _text(child)
            # Store booleans as actual bools
            if val_text.lower() in ("true", "false"):
                schema.constraints[key] = val_text.lower() == "true"
            else:
                schema.constraints[key] = val_text

    # Detect format from constraints
    if schema.constraints.get("must_be_json"):
        schema.format = "json"
    elif schema.constraints.get("must_use_xml_tags"):
        schema.format = "xml"

    return schema


def _parse_scoring(elem: ET.Element) -> Scoring:
    """Parse <scoring> element."""
    formula = _text_raw(elem.find("formula"))

    labels: dict[tuple[int, int], str] = {}
    labels_elem = elem.find("labels")
    if labels_elem is not None:
        for label_elem in labels_elem.findall("label"):
            lo = int(label_elem.get("min", "0"))
            hi = int(label_elem.get("max", "0"))
            labels[(lo, hi)] = _text(label_elem)

    inverted = "risk" in formula.lower() or "higher is cleaner" in formula.lower()

    return Scoring(formula=formula, labels=labels, inverted=inverted)


# ── Serialization ──────────────────────────────────────────────────────


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def rubric_to_xml(rubric: Rubric) -> str:
    """Serialize a Rubric to XML string."""
    root = ET.Element("LLM_JUDGE_SPEC")
    root.set("version", rubric.version)
    root.set("name", rubric.name)

    # Step 2: <mission>
    if rubric.mission:
        mission = ET.SubElement(root, "mission")
        mission.text = rubric.mission

    # Step 3: <inputs>
    if rubric.inputs:
        inputs = ET.SubElement(root, "inputs")
        for inp in rubric.inputs:
            field = ET.SubElement(inputs, "field")
            field.set("name", inp.name)
            field.set("required", str(inp.required).lower())
            if inp.description:
                field.text = inp.description

    # Step 4: <what_to_judge>
    if rubric.what_to_judge:
        wtj = ET.SubElement(root, "what_to_judge")
        wtj.text = rubric.what_to_judge

    # Step 5: <definitions>
    if rubric.definitions:
        defs = ET.SubElement(root, "definitions")
        for def_id, def_text in rubric.definitions.items():
            d = ET.SubElement(defs, "def")
            d.set("id", def_id)
            d.text = def_text

    # Step 6: <rubric>
    if rubric.criteria or rubric.disqualifiers:
        rubric_elem = ET.SubElement(root, "rubric")

        for crit in rubric.criteria.values():
            _serialize_criterion(rubric_elem, crit)

        if rubric.disqualifiers:
            dqs = ET.SubElement(rubric_elem, "disqualifiers")
            for dq in rubric.disqualifiers:
                dq_elem = ET.SubElement(dqs, "dq")
                dq_elem.set("id", dq.id)
                dq_elem.text = dq.description

    # Step 7: pattern_library
    if rubric.pattern_library:
        _serialize_pattern_library(root, rubric.pattern_library)

    # Step 8: <decision_logic>
    if rubric.decision_logic:
        dl = ET.SubElement(root, "decision_logic")
        for dr in rubric.decision_logic:
            r = ET.SubElement(dl, "rule")
            r.set("id", dr.id)
            r.text = dr.condition

    # Step 9: <mapping_examples>
    if rubric.mapping_examples:
        me = ET.SubElement(root, "mapping_examples")
        for ex in rubric.mapping_examples:
            ex_elem = ET.SubElement(me, "example")
            ex_elem.set("id", ex.id)
            if ex.user:
                u = ET.SubElement(ex_elem, "user")
                u.text = ex.user
            if ex.assistant:
                a = ET.SubElement(ex_elem, "assistant")
                a.text = ex.assistant
            if ex.verdict:
                v = ET.SubElement(ex_elem, "verdict")
                v.text = ex.verdict

    # Step 10: <output_schema>
    if rubric.output_schema:
        _serialize_output_schema(root, rubric.output_schema)

    # Step 11: <scoring>
    if rubric.scoring:
        _serialize_scoring(root, rubric.scoring)

    # Step 12: <advice_rules>
    if rubric.advice_rules:
        ar = ET.SubElement(root, "advice_rules")
        for adv in rubric.advice_rules:
            r = ET.SubElement(ar, "rule")
            r.set("when", "|".join(adv.when))
            r.text = adv.advice

    # Step 13: <validation>
    if rubric.validation_musts:
        val = ET.SubElement(root, "validation")
        for must in rubric.validation_musts:
            m = ET.SubElement(val, "must")
            m.text = must.description

    # Step 14: <scoring_guidance>
    if rubric.scoring_guidance:
        sg = ET.SubElement(root, "scoring_guidance")
        mapping = ET.SubElement(sg, "mapping")
        mapping.text = rubric.scoring_guidance

    return _pretty_print(root)


def _serialize_criterion(parent: ET.Element, crit: Criterion) -> None:
    """Serialize a Criterion to XML."""
    elem = ET.SubElement(parent, "criterion")
    elem.set("id", crit.id)
    elem.set("name", crit.name)
    if crit.weight:
        elem.set("weight", str(crit.weight))
    if crit.genre:
        elem.set("genre", crit.genre)

    if crit.uses_patterns:
        up = ET.SubElement(elem, "uses_patterns")
        up.text = ",".join(crit.uses_patterns)

    for anchor_key in sorted(crit.anchors.keys()):
        a = ET.SubElement(elem, f"anchor_{anchor_key}")
        a.text = crit.anchors[anchor_key]

    if crit.mechanical_rules:
        mr = ET.SubElement(elem, "mechanical_rules")
        for rule_text in crit.mechanical_rules:
            r = ET.SubElement(mr, "rule")
            r.text = rule_text

    if crit.notes:
        n = ET.SubElement(elem, "notes")
        n.text = crit.notes


def _serialize_pattern_library(root: ET.Element, lib: PatternLibrary) -> None:
    """Serialize PatternLibrary to XML, using original tag types for round-trip fidelity."""
    # Check if any entries are groups (ComplianceJudge variant)
    has_groups = any(k.startswith("_group_") for k in lib._entry_types)

    if lib.flags:
        # Variant 2: <regex_library flags="..."> with <pattern> children
        rl = ET.SubElement(root, "regex_library")
        rl.set("flags", lib.flags)
        for pat_id, pat_text in lib.entries.items():
            p = ET.SubElement(rl, "pattern")
            p.set("id", pat_id)
            p.text = pat_text
    elif has_groups:
        # Variant 3: ComplianceJudge style
        pl = ET.SubElement(root, "pattern_library")
        for entry_id, entry_text in lib.entries.items():
            entry_type = lib._entry_types.get(entry_id, "")
            if "/" in entry_type:
                # This is a group entry like "refusal_regexes/r"
                group_tag = entry_id
                child_tag = entry_type.split("/")[1]
                group_elem = ET.SubElement(pl, group_tag)
                # Use _group_patterns for round-trip fidelity (not split on pipe!)
                if group_tag in lib._group_patterns:
                    for pattern in lib._group_patterns[group_tag]:
                        child = ET.SubElement(group_elem, child_tag)
                        child.text = pattern
                else:
                    # Fallback: single entry (no internal pipes expected)
                    child = ET.SubElement(group_elem, child_tag)
                    child.text = entry_text
    else:
        # Variant 1: <pattern_library> with <list>/<regex> children
        pl = ET.SubElement(root, "pattern_library")
        for entry_id, entry_text in lib.entries.items():
            entry_type = lib._entry_types.get(entry_id, "regex")
            tag = entry_type if entry_type in ("list", "regex") else "regex"
            child = ET.SubElement(pl, tag)
            child.set("id", entry_id)
            child.text = entry_text


def _serialize_output_schema(root: ET.Element, schema: OutputSchema) -> None:
    """Serialize OutputSchema to XML."""
    os_elem = ET.SubElement(root, "output_schema")

    if schema.template:
        if schema.format == "json":
            tmpl = ET.SubElement(os_elem, "json_template")
            tmpl.text = schema.template
        else:
            tmpl = ET.SubElement(os_elem, "template")
            tmpl.text = schema.template

    if schema.constraints:
        cons = ET.SubElement(os_elem, "constraints")
        for key, val in schema.constraints.items():
            c = ET.SubElement(cons, key)
            if isinstance(val, bool):
                c.text = str(val).lower()
            else:
                c.text = str(val)


def _serialize_scoring(root: ET.Element, scoring: Scoring) -> None:
    """Serialize Scoring to XML."""
    s = ET.SubElement(root, "scoring")
    if scoring.formula:
        f = ET.SubElement(s, "formula")
        f.text = scoring.formula

    if scoring.labels:
        labels = ET.SubElement(s, "labels")
        for (lo, hi), label_text in scoring.labels.items():
            lbl = ET.SubElement(labels, "label")
            lbl.set("min", str(lo))
            lbl.set("max", str(hi))
            lbl.text = label_text


def _pretty_print(root: ET.Element) -> str:
    """Pretty-print an ElementTree Element to XML string."""
    rough = ET.tostring(root, encoding="unicode", xml_declaration=False)
    try:
        dom = parseString(rough)
        pretty = dom.toprettyxml(indent="  ")
        # Remove the XML declaration line that minidom adds
        lines = pretty.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        # Remove extra blank lines
        result = "\n".join(line for line in lines if line.strip())
        return result
    except Exception:
        return rough


def constraint_rubric_to_xml(rubric: ConstraintRubric) -> str:
    """Serialize a ConstraintRubric to XML system prompt format."""
    root = ET.Element("ConstraintRubric")
    root.set("name", rubric.name)

    if rubric.instructions:
        inst = ET.SubElement(root, "instructions")
        inst.text = rubric.instructions

    if rubric.output_format:
        fmt = ET.SubElement(root, "output_format")
        fmt.text = rubric.output_format

    if rubric.examples:
        exs = ET.SubElement(root, "examples")
        for ex in rubric.examples:
            e = ET.SubElement(exs, "example")
            inp = ET.SubElement(e, "input")
            inp.text = ex.input
            out = ET.SubElement(e, "output")
            out.text = ex.output

    # Phase 2: metadata-only behaviors tag. Emitted as a space-separated list
    # of canonical values. Runtime never branches on this tag; it is purely
    # documentation for the rubric's declared behavior families. See
    # ``rubrify._behaviors`` for the taxonomy and reference citations.
    if rubric.behaviors:
        beh = ET.SubElement(root, "behaviors")
        beh.text = " ".join(sorted(rubric.behaviors))

    return _pretty_print(root)


def constraint_rubric_from_xml(xml_string: str) -> ConstraintRubric:
    """Parse a ``<ConstraintRubric>`` XML string back into a ConstraintRubric.

    Phase 2 addition. Round-trips the fields emitted by
    :func:`constraint_rubric_to_xml`: name, instructions, output_format,
    examples, and the optional ``<behaviors>`` metadata tag. Absent
    ``<behaviors>`` defaults to an empty frozenset, matching Phase 1
    behaviour.
    """
    from rubrify._types import ICLExample
    from rubrify.rubric import ConstraintRubric

    root = ET.fromstring(xml_string)
    name = root.get("name", "")
    instructions = _text(root.find("instructions"))
    output_format = _text(root.find("output_format"))

    examples: list[ICLExample] = []
    exs_elem = root.find("examples")
    if exs_elem is not None:
        for ex_elem in exs_elem.findall("example"):
            inp_text = _text(ex_elem.find("input"))
            out_text = _text(ex_elem.find("output"))
            examples.append(ICLExample(input=inp_text, output=out_text))

    behaviors_elem = root.find("behaviors")
    if behaviors_elem is not None and behaviors_elem.text:
        tokens = [tok for tok in behaviors_elem.text.replace(",", " ").split() if tok]
        behaviors: frozenset[str] = frozenset(tokens)
    else:
        behaviors = frozenset()

    return ConstraintRubric(
        name=name,
        instructions=instructions,
        output_format=output_format,
        examples=examples,
        behaviors=behaviors,
    )
