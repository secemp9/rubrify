"""XML codec: the researcher's primary surface format for rubric rendering.

Uses xml.etree.ElementTree for proper DOM construction, escaping, and
composability. No string concatenation — every element is a proper node.

Renders rubrics as <LLM_JUDGE_SPEC> documents driven by ConstraintBindings:
  - The binding's SurfaceProjection(codec="xml") is the source of truth
    for each criterion element's attributes
  - Anchor tags use <anchor value="N"> not dynamic tag names
  - All text content is properly escaped via ET

XML gives: rigid hierarchy, attribute metadata, institutional authority,
and tag IDs that cross-reference with output fields.
"""

from __future__ import annotations

# ET for construction (Element, SubElement, indent, tostring) — always safe.
import xml.etree.ElementTree as ET

from rubrify.ir.bundle import RubricBundle
from rubrify.ir.constraints import (
    CharLimitConstraint,
    ConstraintBinding,
    ItemCountConstraint,
    OutputConstraint,
    PrefixSuffixConstraint,
    SurfaceProjection,
    TokenConstraint,
    WordCountConstraint,
)
from rubrify.ir.types import Criterion, OrdinalScale, NumericScale


def render_rubric_xml(bundle: RubricBundle) -> str:
    """Render a locked RubricBundle as an <LLM_JUDGE_SPEC> XML document.

    Bindings drive the criterion rendering — each criterion's XML attributes
    come from its binding's SurfaceProjection(codec="xml"), not from raw
    criterion fields. This closes the triple-layer alignment loop.
    """
    rubric = bundle.rubric
    policy = bundle.surface_policy
    bindings_by_id = {b.criterion_id: b for b in bundle.bindings}

    root = ET.Element("LLM_JUDGE_SPEC")
    root.set("version", rubric.meta.version)
    root.set("name", rubric.meta.name)

    # Mission
    mission = ET.SubElement(root, "mission")
    mission.text = rubric.goal

    # Role (instruction authority block)
    if policy.role:
        role_el = ET.SubElement(root, "role")
        role_el.set("authority", policy.role.authority)
        persona_el = ET.SubElement(role_el, "persona")
        persona_el.text = policy.role.persona
        for obligation in policy.role.obligations:
            ob_el = ET.SubElement(role_el, "obligation")
            ob_el.text = obligation
        for constraint in policy.role.constraints:
            co_el = ET.SubElement(role_el, "constraint")
            co_el.text = constraint

    # Rubric block
    rubric_el = ET.SubElement(root, "rubric")
    for criterion in rubric.criteria:
        binding = bindings_by_id.get(criterion.id)
        _build_criterion_element(rubric_el, criterion, binding)

    # Disqualifiers
    if rubric.disqualifiers:
        dqs_el = ET.SubElement(rubric_el, "disqualifiers")
        for dq in rubric.disqualifiers:
            dq_el = ET.SubElement(dqs_el, "dq")
            dq_el.set("id", dq.id)
            dq_el.text = dq.description

    # Definitions
    if rubric.definitions:
        defs_el = ET.SubElement(root, "definitions")
        for defn in rubric.definitions:
            def_el = ET.SubElement(defs_el, "def")
            def_el.set("id", defn.id)
            def_el.set("term", defn.term)
            def_el.text = defn.description

    # Calibration examples (few-shot)
    if rubric.calibration_examples:
        examples_el = ET.SubElement(root, "mapping_examples")
        for ex in rubric.calibration_examples:
            ex_el = ET.SubElement(examples_el, "example")
            ex_el.set("id", ex.id)
            input_el = ET.SubElement(ex_el, "input")
            input_el.text = ex.input_summary
            verdict_el = ET.SubElement(ex_el, "verdict")
            verdict_el.text = ex.expected_verdict
            if ex.explanation:
                why_el = ET.SubElement(ex_el, "explanation")
                why_el.text = ex.explanation

    # Advice rules
    if rubric.advice_rules:
        advice_el = ET.SubElement(root, "advice_rules")
        for rule in rubric.advice_rules:
            rule_el = ET.SubElement(advice_el, "rule")
            rule_el.text = rule.fix

    # Output schema
    schema_el = ET.SubElement(root, "output_schema")
    tmpl_el = ET.SubElement(schema_el, "json_template")
    tmpl_el.text = _render_json_template(bundle)
    constraints_el = ET.SubElement(schema_el, "constraints")
    _add_text_child(constraints_el, "must_be_json", "true")
    _add_text_child(constraints_el, "no_prose_outside_json", "true")
    if policy.enforce_key_order:
        keys = "score,rationale,evidence,violations,criterion_scores"
        _add_text_child(constraints_el, "key_order", keys)

    # Scoring formula
    scoring_el = ET.SubElement(root, "scoring")
    formula_parts = [f"{c.id}*{c.weight}" for c in rubric.criteria]
    formula_el = ET.SubElement(scoring_el, "formula")
    formula_el.text = f"score = ({' + '.join(formula_parts)}) / total_weight; if any DQ => score=0"

    # Pattern library
    if rubric.patterns:
        patterns_el = ET.SubElement(root, "pattern_library")
        for p in rubric.patterns:
            p_el = ET.SubElement(patterns_el, "pattern")
            p_el.set("id", p.id)
            p_el.set("flags", p.flags)
            p_el.text = p.pattern

    # Output constraints as validation block (triple validation)
    if bundle.output_constraints:
        validation_el = ET.SubElement(root, "validation")
        for constraint in bundle.output_constraints:
            _build_constraint_element(validation_el, constraint)

    # Instructions
    if rubric.instructions:
        instructions_el = ET.SubElement(root, "instructions")
        for instruction in rubric.instructions:
            step_el = ET.SubElement(instructions_el, "step")
            step_el.text = instruction

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def render_criterion_xml(
    criterion: Criterion,
    bundle: RubricBundle,
) -> str:
    """Render a focused document for a single criterion.

    Used when SurfacePolicy.criterion_focus == "focused" to avoid
    sending the entire rubric for every criterion evaluation.
    """
    rubric = bundle.rubric
    policy = bundle.surface_policy
    bindings_by_id = {b.criterion_id: b for b in bundle.bindings}
    binding = bindings_by_id.get(criterion.id)

    root = ET.Element("LLM_JUDGE_SPEC")
    root.set("version", rubric.meta.version)
    root.set("name", rubric.meta.name)
    root.set("focus", criterion.id)

    mission = ET.SubElement(root, "mission")
    mission.text = rubric.goal

    if policy.role:
        role_el = ET.SubElement(root, "role")
        role_el.set("authority", policy.role.authority)
        persona_el = ET.SubElement(role_el, "persona")
        persona_el.text = policy.role.persona

    rubric_el = ET.SubElement(root, "rubric")
    _build_criterion_element(rubric_el, criterion, binding)

    # Relevant disqualifiers only
    for dq in rubric.disqualifiers:
        if not dq.criterion_ids or criterion.id in dq.criterion_ids:
            dqs_el = rubric_el.find("disqualifiers")
            if dqs_el is None:
                dqs_el = ET.SubElement(rubric_el, "disqualifiers")
            dq_el = ET.SubElement(dqs_el, "dq")
            dq_el.set("id", dq.id)
            dq_el.text = dq.description

    # Definitions
    if rubric.definitions:
        defs_el = ET.SubElement(root, "definitions")
        for defn in rubric.definitions:
            def_el = ET.SubElement(defs_el, "def")
            def_el.set("id", defn.id)
            def_el.set("term", defn.term)
            def_el.text = defn.description

    # Calibration examples (few-shot)
    if rubric.calibration_examples:
        examples_el = ET.SubElement(root, "mapping_examples")
        for ex in rubric.calibration_examples:
            ex_el = ET.SubElement(examples_el, "example")
            ex_el.set("id", ex.id)
            input_el = ET.SubElement(ex_el, "input")
            input_el.text = ex.input_summary
            verdict_el = ET.SubElement(ex_el, "verdict")
            verdict_el.text = ex.expected_verdict
            if ex.explanation:
                why_el = ET.SubElement(ex_el, "explanation")
                why_el.text = ex.explanation

    # Focused output schema (derived from the same model as full rendering)
    schema_el = ET.SubElement(root, "output_schema")
    tmpl_el = ET.SubElement(schema_el, "json_template")
    tmpl_el.text = _render_json_template(bundle)
    constraints_el = ET.SubElement(schema_el, "constraints")
    _add_text_child(constraints_el, "must_be_json", "true")
    _add_text_child(constraints_el, "no_prose_outside_json", "true")

    if bundle.output_constraints:
        validation_el = ET.SubElement(root, "validation")
        for constraint in bundle.output_constraints:
            _build_constraint_element(validation_el, constraint)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def render_group_xml(criteria: list[Criterion], bundle: RubricBundle) -> str:
    """Render an XML document for a GROUP of criteria (a subset of the rubric).

    Used by execution strategies that batch criteria into groups rather than
    evaluating them one-at-a-time or all-at-once. Produces a document similar
    to render_rubric_xml but filtered to the specified criteria subset.

    Includes:
      - Root element with rubric name and version
      - Mission/goal text
      - The criteria subset only (not all criteria)
      - Definitions (shared across all criteria)
      - Disqualifiers (only those relevant to the subset)
      - Pattern library (full — patterns can trigger across any criterion)
      - Output schema with json_template for the criteria subset
      - Calibration examples (shared)
      - Output constraints (from bundle.output_constraints)
      - Scoring formula (for the subset)
    """
    rubric = bundle.rubric
    policy = bundle.surface_policy
    bindings_by_id = {b.criterion_id: b for b in bundle.bindings}
    subset_ids = {c.id for c in criteria}

    root = ET.Element("LLM_JUDGE_SPEC")
    root.set("version", rubric.meta.version)
    root.set("name", rubric.meta.name)
    root.set("group", ",".join(c.id for c in criteria))

    # Mission
    mission = ET.SubElement(root, "mission")
    mission.text = rubric.goal

    # Role (instruction authority block)
    if policy.role:
        role_el = ET.SubElement(root, "role")
        role_el.set("authority", policy.role.authority)
        persona_el = ET.SubElement(role_el, "persona")
        persona_el.text = policy.role.persona
        for obligation in policy.role.obligations:
            ob_el = ET.SubElement(role_el, "obligation")
            ob_el.text = obligation
        for constraint in policy.role.constraints:
            co_el = ET.SubElement(role_el, "constraint")
            co_el.text = constraint

    # Rubric block — only the criteria subset
    rubric_el = ET.SubElement(root, "rubric")
    for criterion in criteria:
        binding = bindings_by_id.get(criterion.id)
        _build_criterion_element(rubric_el, criterion, binding)

    # Disqualifiers — only those relevant to the subset
    # A DQ is relevant if it has no criterion_ids (global) or if any of its
    # criterion_ids intersect with the subset.
    if rubric.disqualifiers:
        relevant_dqs = [
            dq for dq in rubric.disqualifiers
            if not dq.criterion_ids or subset_ids & set(dq.criterion_ids)
        ]
        if relevant_dqs:
            dqs_el = ET.SubElement(rubric_el, "disqualifiers")
            for dq in relevant_dqs:
                dq_el = ET.SubElement(dqs_el, "dq")
                dq_el.set("id", dq.id)
                dq_el.text = dq.description

    # Definitions (shared across all criteria)
    if rubric.definitions:
        defs_el = ET.SubElement(root, "definitions")
        for defn in rubric.definitions:
            def_el = ET.SubElement(defs_el, "def")
            def_el.set("id", defn.id)
            def_el.set("term", defn.term)
            def_el.text = defn.description

    # Calibration examples (shared few-shot)
    if rubric.calibration_examples:
        examples_el = ET.SubElement(root, "mapping_examples")
        for ex in rubric.calibration_examples:
            ex_el = ET.SubElement(examples_el, "example")
            ex_el.set("id", ex.id)
            input_el = ET.SubElement(ex_el, "input")
            input_el.text = ex.input_summary
            verdict_el = ET.SubElement(ex_el, "verdict")
            verdict_el.text = ex.expected_verdict
            if ex.explanation:
                why_el = ET.SubElement(ex_el, "explanation")
                why_el.text = ex.explanation

    # Output schema — json_template for the criteria subset only
    schema_el = ET.SubElement(root, "output_schema")
    tmpl_el = ET.SubElement(schema_el, "json_template")
    tmpl_el.text = _render_json_template_for_group(criteria, bundle)
    constraints_el = ET.SubElement(schema_el, "constraints")
    _add_text_child(constraints_el, "must_be_json", "true")
    _add_text_child(constraints_el, "no_prose_outside_json", "true")
    if policy.enforce_key_order:
        keys = "score,rationale,evidence,violations,criterion_scores"
        _add_text_child(constraints_el, "key_order", keys)

    # Scoring formula (for the subset)
    scoring_el = ET.SubElement(root, "scoring")
    formula_parts = [f"{c.id}*{c.weight}" for c in criteria]
    formula_el = ET.SubElement(scoring_el, "formula")
    formula_el.text = f"score = ({' + '.join(formula_parts)}) / total_weight; if any DQ => score=0"

    # Pattern library (full — patterns can trigger across any criterion)
    if rubric.patterns:
        patterns_el = ET.SubElement(root, "pattern_library")
        for p in rubric.patterns:
            p_el = ET.SubElement(patterns_el, "pattern")
            p_el.set("id", p.id)
            p_el.set("flags", p.flags)
            p_el.text = p.pattern

    # Output constraints (triple validation)
    if bundle.output_constraints:
        validation_el = ET.SubElement(root, "validation")
        for constraint in bundle.output_constraints:
            _build_constraint_element(validation_el, constraint)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _render_json_template_for_group(criteria: list[Criterion], bundle: RubricBundle) -> str:
    """Render the expected JSON output template for a criteria subset."""
    from rubrify.codecs.json_codec import generate_judgment_template
    return generate_judgment_template(bundle, criteria=criteria)


def _build_constraint_element(
    parent: ET.Element,
    constraint: OutputConstraint,
) -> ET.Element:
    """Build a typed <must> element with kind-specific attributes.

    Renders both the human-readable description (as text content) and the
    machine-verifiable parameters (as XML attributes) so the LLM sees
    the exact structural requirements alongside the natural language.

    Each constraint variant gets its own structural attributes:
      - prefix_suffix: prefix, suffix
      - word_count:    count, mode
      - char_limit:    limit, mode
      - item_count:    count, mode, delimiter
      - token:         token, must_contain
    Common attributes on all: id, kind, target_field, enforcement.
    """
    must_el = ET.SubElement(parent, "must")

    # Common attributes shared by all constraint variants
    must_el.set("id", constraint.id)
    must_el.set("kind", constraint.kind)
    must_el.set("target_field", constraint.target_field)
    must_el.set("enforcement", constraint.enforcement)

    # Kind-specific structural attributes
    if isinstance(constraint, PrefixSuffixConstraint):
        if constraint.prefix is not None:
            must_el.set("prefix", constraint.prefix)
        if constraint.suffix is not None:
            must_el.set("suffix", constraint.suffix)

    elif isinstance(constraint, WordCountConstraint):
        must_el.set("count", str(constraint.count))
        must_el.set("mode", constraint.mode)

    elif isinstance(constraint, CharLimitConstraint):
        must_el.set("limit", str(constraint.limit))
        must_el.set("mode", constraint.mode)

    elif isinstance(constraint, ItemCountConstraint):
        must_el.set("count", str(constraint.count))
        must_el.set("mode", constraint.mode)
        must_el.set("delimiter", constraint.delimiter)

    elif isinstance(constraint, TokenConstraint):
        must_el.set("token", constraint.token)
        must_el.set("must_contain", str(constraint.must_contain).lower())

    # Description as text content for natural language understanding
    must_el.text = constraint.description

    return must_el


def _build_criterion_element(
    parent: ET.Element,
    criterion: Criterion,
    binding: ConstraintBinding | None,
) -> ET.Element:
    """Build a <criterion> element, driven by the binding's XML projection.

    The binding's SurfaceProjection(codec="xml") is the sole authority for
    element attributes. No fallback to direct criterion fields.
    """
    crit_el = ET.SubElement(parent, "criterion")

    # Binding-driven attributes — the binding IS the authority
    xml_proj = _find_xml_projection(binding) if binding else None
    if xml_proj is None or not xml_proj.attributes:
        raise RuntimeError(
            f"Criterion '{criterion.id}' has no XML projection in its binding. "
            f"This indicates a compiler bug — the bind pass should always generate XML projections."
        )
    for attr_name, attr_val in xml_proj.attributes.items():
        crit_el.set(attr_name, attr_val)

    if criterion.genre:
        crit_el.set("genre", ",".join(criterion.genre))

    # Anchors — proper <anchor value="N"> elements, not dynamic tag names
    scale = criterion.scale
    if isinstance(scale, (OrdinalScale, NumericScale)):
        anchors = getattr(scale, "anchors", None) or []
        for anchor in anchors:
            anchor_el = ET.SubElement(crit_el, "anchor")
            anchor_el.set("value", str(anchor.value))
            anchor_el.text = anchor.description or anchor.label

    # Evidence spec
    if criterion.evidence.required:
        ev_el = ET.SubElement(crit_el, "evidence")
        ev_el.set("required", "true")
        if criterion.evidence.exact_quote:
            ev_el.set("exact_quote", "true")
        ev_el.set("min", str(criterion.evidence.min_items))
        ev_el.set("max", str(criterion.evidence.max_items))

    # Mechanical rules
    for rule in criterion.mechanical_rules:
        rule_el = ET.SubElement(crit_el, "rule")
        rule_el.text = rule

    return crit_el


def _find_xml_projection(binding: ConstraintBinding) -> SurfaceProjection | None:
    """Find the XML-codec projection in a binding's projections list."""
    for proj in binding.projections:
        if proj.codec == "xml":
            return proj
    return None


def _render_json_template(bundle: RubricBundle) -> str:
    """Render the expected JSON output template from the dynamic Pydantic model.

    Single source of truth: derived from the same model used for validation
    and tool construction.
    """
    from rubrify.codecs.json_codec import generate_judgment_template
    return generate_judgment_template(bundle)


def _add_text_child(parent: ET.Element, tag: str, text: str) -> ET.Element:
    """Add a simple text-content child element."""
    el = ET.SubElement(parent, tag)
    el.text = text
    return el


__all__ = [
    "render_criterion_xml",
    "render_group_xml",
    "render_rubric_xml",
]
