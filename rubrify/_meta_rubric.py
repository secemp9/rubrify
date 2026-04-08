"""Meta-rubric system: instruction primitives, generators, and META_EVALUATOR."""

from dataclasses import dataclass as _dataclass

from rubrify._examples import (
    ANTI_SLOP_EXCERPT,
    COMPLIANCE_JUDGE_EXCERPT,
    ZINSSER_V3_EXCERPT,
)
from rubrify._types import (
    Criterion,
    Disqualifier,
    ICLExample,
    Instruction,
    OutputSchema,
    Scoring,
)
from rubrify.rubric import ConstraintRubric, Rubric

# --- Instruction Primitives (one per property in the lattice) ---

INSTRUCTION_PRIMITIVES: dict[str, Instruction] = {
    "mission": Instruction(
        text="Every rubric MUST begin with a <mission> tag containing a single sentence "
        "that names the evaluation domain and output type.",
        property_name="mission",
    ),
    "criteria_structure": Instruction(
        text="Define 3-5 criteria using <criterion id='C{n}' name='...' weight='N'>. "
        "Each criterion MUST have >= 2 <anchor_N> tags with observable, "
        "non-subjective descriptions. Weights should sum to 100 for core criteria.",
        property_name="anchored",
    ),
    "disqualifiers": Instruction(
        text="Define at least 1 <dq> (disqualifier) for auto-fail conditions. "
        "Disqualifiers are binary (fire or don't); do not use soft penalties for major violations.",
        property_name="dq",
    ),
    "output_schema": Instruction(
        text="Include an <output_schema> with a <json_template> showing the exact JSON structure. "
        "Add <constraints> with <must_be_json>true</must_be_json> and "
        "<no_prose_outside_json>true</no_prose_outside_json>.",
        property_name="schema",
    ),
    "scoring": Instruction(
        text="Include a <scoring><formula> that specifies how criterion scores "
        "combine into a total. Include <labels> with <label min='N' max='M'> "
        "mapping score ranges to classifications.",
        property_name="schema",
    ),
    "mechanical": Instruction(
        text="At least one criterion should include <mechanical_rules> with concrete, "
        "regex-checkable or keyword-checkable conditions.",
        property_name="mechanical",
    ),
    "steering": Instruction(
        text="Include at least one context-following anchor in the output schema: "
        "a fixed prefix (e.g., 'BECAUSE:'), exact word count, or key order requirement. "
        "These steering constraints keep the model inside the rubric's behavioral boundaries. "
        "Example: <rationale_anchor>Begin with 'BECAUSE:' and end with '.'; "
        "exactly 35 words.</rationale_anchor>",
        property_name="steering",
    ),
    "mirror": Instruction(
        text="State critical constraints in at least 2 locations: the rubric definition AND "
        "the output schema. If criteria demand a format, the output_schema must also demand "
        "that format.",
        property_name="mirror",
    ),
    "regex_library": Instruction(
        text="Include a <regex_library flags='i'> with named <pattern id='...'> entries. "
        "Each criterion should declare <uses_patterns> referencing pattern IDs from the library.",
        property_name="patterns",
    ),
    "inverted_scoring": Instruction(
        text="Scoring should be INVERTED: higher score = cleaner text. "
        "Define risk = max_score - score. Map risk to severity bands.",
        property_name="inverted",
    ),
    "advice_rules": Instruction(
        text="Include <advice_rules> with "
        "<rule when='pattern_id|pattern_id'>fix instruction</rule> "
        "mapping pattern IDs to concrete fix instructions.",
        property_name="advice",
    ),
    "decision_logic": Instruction(
        text="Include a <decision_logic> section with rules R1, R2, ... "
        "that map criterion values to verdicts (e.g., Yes/Somewhat/No or Pass/Partial/Fail).",
        property_name="decision",
    ),
    "xml_output": Instruction(
        text="Output schema should use XML tags, not JSON. "
        "Include <must_use_xml_tags>true</must_use_xml_tags> and "
        "<no_text_outside_tags>true</no_text_outside_tags>.",
        property_name="schema",
    ),
    "mapping_examples": Instruction(
        text="Include <mapping_examples> with at least 3 <example> elements "
        "showing input scenarios and expected verdicts.",
        property_name="examples",
    ),
    "definitions": Instruction(
        text="Include a <definitions> section with <def id='...'>description</def> entries "
        "for key terms used in the rubric.",
        property_name="",
    ),
}


def compose_instructions(instructions: list[Instruction]) -> str:
    """Compose instruction primitives into a single instruction string."""
    parts = [f"{i + 1}. {inst.text}" for i, inst in enumerate(instructions)]
    header = (
        "Generate a valid <LLM_JUDGE_SPEC> XML rubric from the provided source material. "
        "Follow these structural requirements:\n\n"
    )
    return header + "\n".join(parts)


# --- Property Profiles ---


@_dataclass(frozen=True)
class PropertyProfile:
    """Declares which properties a rubric type requires.

    Used by compose_from_profile() to automatically select the right
    instruction primitives for a generator.
    """

    required_properties: frozenset[str]
    name: str = ""


SCORING_PROFILE = PropertyProfile(
    name="scoring",
    required_properties=frozenset(
        {"mission", "anchored", "dq", "schema", "mechanical", "steering", "mirror"}
    ),
)

DETECTION_PROFILE = PropertyProfile(
    name="detection",
    required_properties=frozenset(
        {
            "mission",
            "anchored",
            "dq",
            "schema",
            "mechanical",
            "steering",
            "mirror",
            "patterns",
            "inverted",
            "advice",
        }
    ),
)

COMPLIANCE_PROFILE = PropertyProfile(
    name="compliance",
    required_properties=frozenset(
        {"mission", "anchored", "dq", "schema", "decision", "examples", "steering"}
    ),
)


def compose_from_profile(
    profile: PropertyProfile,
    primitives: dict[str, Instruction] | None = None,
) -> str:
    """Select instructions matching a property profile and compose them.

    This enables creating generators for new rubric types by declaring
    required properties instead of manually picking instructions.
    """
    if primitives is None:
        primitives = INSTRUCTION_PRIMITIVES
    selected = [
        inst
        for inst in primitives.values()
        if inst.property_name and inst.property_name in profile.required_properties
    ]
    return compose_instructions(selected)


# --- Type-Specific Generators ---

SCORING_GENERATOR = ConstraintRubric(
    name="ScoringRubricGenerator",
    instructions=compose_instructions(
        [
            INSTRUCTION_PRIMITIVES["mission"],
            INSTRUCTION_PRIMITIVES["criteria_structure"],
            INSTRUCTION_PRIMITIVES["disqualifiers"],
            INSTRUCTION_PRIMITIVES["output_schema"],
            INSTRUCTION_PRIMITIVES["scoring"],
            INSTRUCTION_PRIMITIVES["mechanical"],
            INSTRUCTION_PRIMITIVES["steering"],
            INSTRUCTION_PRIMITIVES["mirror"],
        ]
    ),
    output_format="<LLM_JUDGE_SPEC>",
    examples=[ICLExample(input="(scoring rubric example)", output=ZINSSER_V3_EXCERPT)],
)

DETECTION_GENERATOR = ConstraintRubric(
    name="DetectionRubricGenerator",
    instructions=compose_instructions(
        [
            INSTRUCTION_PRIMITIVES["mission"],
            INSTRUCTION_PRIMITIVES["criteria_structure"],
            INSTRUCTION_PRIMITIVES["disqualifiers"],
            INSTRUCTION_PRIMITIVES["output_schema"],
            INSTRUCTION_PRIMITIVES["scoring"],
            INSTRUCTION_PRIMITIVES["regex_library"],
            INSTRUCTION_PRIMITIVES["inverted_scoring"],
            INSTRUCTION_PRIMITIVES["advice_rules"],
            INSTRUCTION_PRIMITIVES["steering"],
            INSTRUCTION_PRIMITIVES["mirror"],
        ]
    ),
    output_format="<LLM_JUDGE_SPEC>",
    examples=[ICLExample(input="(detection rubric example)", output=ANTI_SLOP_EXCERPT)],
)

COMPLIANCE_GENERATOR = ConstraintRubric(
    name="ComplianceRubricGenerator",
    instructions=compose_instructions(
        [
            INSTRUCTION_PRIMITIVES["mission"],
            INSTRUCTION_PRIMITIVES["criteria_structure"],
            INSTRUCTION_PRIMITIVES["disqualifiers"],
            INSTRUCTION_PRIMITIVES["definitions"],
            INSTRUCTION_PRIMITIVES["decision_logic"],
            INSTRUCTION_PRIMITIVES["xml_output"],
            INSTRUCTION_PRIMITIVES["mapping_examples"],
            INSTRUCTION_PRIMITIVES["steering"],
        ]
    ),
    output_format="<LLM_JUDGE_SPEC>",
    examples=[ICLExample(input="(compliance rubric example)", output=COMPLIANCE_JUDGE_EXCERPT)],
)


# --- META_EVALUATOR: scores generated rubrics ---

META_EVALUATOR = Rubric(
    name="MetaRubricEvaluator",
    version="1.0",
    mission="Evaluate whether a rubric specification is well-formed per the formal framework.",
)
META_EVALUATOR.add_criterion(
    Criterion(
        id="C1",
        name="Observable & Anchored Criteria",
        weight=25,
        anchors={
            0: "No criteria or criteria without anchors.",
            1: "Criteria exist but anchors are vague or subjective.",
            2: "Most criteria have 2+ anchors but some are ambiguous.",
            3: "All criteria have 2+ distinct, observable anchors.",
            4: "Anchors include micro-examples or regex references.",
            5: "Every anchor is falsifiable; mechanical rules complement subjective ones.",
        },
        notes="Property lattice: p_anchored, p_criteria. Sufficient condition: S1_anchored.",
    )
)
META_EVALUATOR.add_criterion(
    Criterion(
        id="C2",
        name="Schema-First Output Design",
        weight=25,
        anchors={
            0: "No output schema.",
            1: "Schema exists but doesn't reference criteria or decision points.",
            2: "Schema references some criteria or verdicts; structure unspecified.",
            3: "Schema specifies exact structure (JSON keys or XML tags) with fixed ordering.",
            4: "Schema includes a concrete template (json_template or xml tag set) "
            "with typed fields matching criteria or verdicts.",
            5: "Full alignment: every criterion ID or verdict value appears in the template; "
            "steering constraints present. Equally valid for JSON and XML output formats.",
        },
        notes="Property lattice: p_schema, p_aligned, p_steering. "
        "Sufficient conditions: S2_aligned, S5_steering.",
    )
)
META_EVALUATOR.add_criterion(
    Criterion(
        id="C3",
        name="Mechanical Checkability",
        weight=20,
        anchors={
            0: "No mechanical rules or pattern references.",
            1: "One criterion has vague mechanical guidance.",
            2: "Multiple criteria reference concrete checks.",
            3: "Pattern library or mechanical_rules provide regex/keyword handles.",
            4: "Uses_patterns cross-referencing links criteria to shared pattern library.",
            5: "Comprehensive: pattern library + uses_patterns + mechanical_rules "
            "+ diagnostics in schema.",
        },
        notes="Property lattice: p_mechanical, p_patterns. " "Sufficient condition: S3_mechanical.",
    )
)
META_EVALUATOR.add_criterion(
    Criterion(
        id="C4",
        name="Disqualifiers & Hard Boundaries",
        weight=20,
        anchors={
            0: "No disqualifiers.",
            1: "One vague disqualifier.",
            2: "Disqualifiers exist but overlap with soft penalties.",
            3: "Clear disqualifiers for major failure modes relevant to the rubric's domain; "
            "binary outcomes.",
            4: "Disqualifiers cover domain-relevant failure modes (e.g., empty input, "
            "wrong language, schema violation, task non-applicability).",
            5: "Comprehensive DQ set with explicit pattern triggers or decision rules "
            "where applicable.",
        },
        notes="Property lattice: p_dq. Sufficient condition: S4_disqualifiers.",
    )
)
META_EVALUATOR.add_criterion(
    Criterion(
        id="C5",
        name="Economy & Focus",
        weight=10,
        anchors={
            0: "No criteria or > 10 criteria.",
            1: "1-2 criteria (too few) or 8-10 (bloated).",
            2: "3 criteria or 7 criteria (boundary).",
            3: "4-6 criteria covering distinct dimensions.",
            4: "3-5 criteria with clear separation of concerns.",
            5: "3-5 orthogonal criteria; no redundancy; all weights justified.",
        },
        notes="Property lattice: p_economy. " "Advisory (no corresponding sufficient condition).",
    )
)
META_EVALUATOR.add_disqualifier(Disqualifier("DQ1", "Criteria without any anchors (vague rubric)."))
META_EVALUATOR.add_disqualifier(
    Disqualifier("DQ2", "No output schema defined (prose-only prompt).")
)
META_EVALUATOR.add_disqualifier(
    Disqualifier("DQ3", "Rationale spec allows > 100 words (overlong).")
)
META_EVALUATOR.add_disqualifier(
    Disqualifier("DQ4", "Schema references fields not in criteria (tag-schema mismatch).")
)
META_EVALUATOR.output_schema = OutputSchema(
    format="json",
    template='{"score":0,"class":"","subscores":{},"rationale":"","coaching":[]}',
    constraints={
        "must_be_json": True,
        "no_prose_outside_json": True,
        "key_order": "score,class,subscores,rationale,coaching",
        "rationale_anchor": "Begin with 'BECAUSE:' and end with '.'; exactly 35 words.",
    },
)
META_EVALUATOR.scoring = Scoring(
    formula="Sum weighted C1-C5 (0-5 each). Normalize to 100. "
    "If any DQ: score=0, class='Rejected'.",
    labels={
        (90, 100): "Exemplary rubric",
        (75, 89): "Strong rubric (minor improvements)",
        (60, 74): "Usable rubric (needs refinement)",
        (40, 59): "Weak rubric (significant gaps)",
        (1, 39): "Poor rubric (major revision needed)",
        (0, 0): "Rejected",
    },
)

# Formal mapping: META_EVALUATOR criterion ID -> property predicate names.
# This is the code-level formalization of the framework's correspondence.
META_CRITERION_TO_PROPERTIES: dict[str, list[str]] = {
    "C1": ["anchored", "criteria"],
    "C2": ["schema", "aligned", "steering"],
    "C3": ["mechanical", "patterns"],
    "C4": ["dq"],
    "C5": ["economy"],
}
