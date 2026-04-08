from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rubrify.rubric import Rubric


@dataclass(frozen=True, slots=True)
class PropertyCheck:
    name: str
    predicate: Callable[[Rubric], bool]
    severity: str  # "error" or "warning"
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    is_valid: bool  # All N1-N3 pass
    is_well_formed: bool  # All N + S pass
    errors: tuple[PropertyCheck, ...]  # Failed N checks
    warnings: tuple[PropertyCheck, ...]  # Failed S checks


# --- 16 Property Predicates ---


def p_mission(r: Rubric) -> bool:
    """P1: Has a non-empty, scoped mission."""
    return bool(r.mission and r.mission.strip())


def p_criteria(r: Rubric) -> int:
    """P2: Number of criteria with anchors."""
    return sum(1 for c in r.criteria.values() if c.anchors)


def p_anchored(r: Rubric) -> float:
    """P3: Fraction of criteria with >= 2 distinct anchors."""
    if not r.criteria:
        return 0.0
    return sum(1 for c in r.criteria.values() if len(c.anchors) >= 2) / len(r.criteria)


def p_mechanical(r: Rubric) -> float:
    """P4: Fraction of criteria with mechanical_rules or uses_patterns."""
    if not r.criteria:
        return 0.0
    return sum(1 for c in r.criteria.values() if c.mechanical_rules or c.uses_patterns) / len(
        r.criteria
    )


def p_dq(r: Rubric) -> int:
    """P5: Number of disqualifiers."""
    return len(r.disqualifiers)


def p_schema(r: Rubric) -> bool:
    """P6: Has an output schema with template."""
    return r.output_schema is not None and bool(r.output_schema.template)


def p_aligned(r: Rubric) -> float:
    """P7: Fraction of criteria IDs that appear in output template."""
    if not r.criteria or not r.output_schema or not r.output_schema.template:
        return 0.0
    template = r.output_schema.template
    return sum(1 for cid in r.criteria if cid in template) / len(r.criteria)


def p_steering(r: Rubric) -> int:
    """P8: Number of steering constraints (context-following anchors)."""
    if not r.output_schema:
        return 0
    steering_keys = {
        "rationale_anchor",
        "rationale_style",
        "key_order",
        "fixed_key_order",
        "edits_requirements",
        "advice_style",
    }
    return sum(1 for k in r.output_schema.constraints if k in steering_keys)


def p_mirror(r: Rubric) -> int:
    """P9: Number of constraints stated in >= 2 locations."""
    return len(r.validation_musts)  # validation section = mirror of output_schema


def p_patterns(r: Rubric) -> int:
    """P10: Number of patterns in pattern library."""
    return len(r.pattern_library.entries) if r.pattern_library else 0


def p_examples(r: Rubric) -> int:
    """P11: Number of mapping/ICL examples."""
    return len(r.mapping_examples)


def p_economy(r: Rubric) -> bool:
    """P12: Criteria count in [3,7] range."""
    n = len(r.criteria)
    return n == 0 or 3 <= n <= 7  # 0 criteria is valid for ConstraintRubric-like


def p_inverted(r: Rubric) -> bool:
    """P13: Scoring polarity is inverted."""
    return r.scoring is not None and r.scoring.inverted


def p_decision(r: Rubric) -> bool:
    """P14: Uses decision logic instead of arithmetic scoring."""
    return len(r.decision_logic) > 0


def p_advice(r: Rubric) -> int:
    """P15: Number of advice rules."""
    return len(r.advice_rules)


def p_validation(r: Rubric) -> int:
    """P16: Number of explicit validation constraints."""
    return len(r.validation_musts)


# --- Necessary Conditions (N1-N3): errors if violated ---

NECESSARY: list[PropertyCheck] = [
    PropertyCheck(
        "N1_mission",
        lambda r: p_mission(r),
        "error",
        "Rubric has no mission statement (N1 violation)",
    ),
    PropertyCheck(
        "N2_structure",
        lambda r: bool(r.criteria) or bool(r.decision_logic) or bool(r.mapping_examples),
        "error",
        "Rubric has no criteria, decision logic, or examples (N2 violation)",
    ),
    PropertyCheck(
        "N3_output",
        lambda r: r.output_schema is not None,
        "error",
        "Rubric has no output schema (N3 violation)",
    ),
]


# --- Sufficient Conditions (S1-S6): warnings if violated ---

SUFFICIENT: list[PropertyCheck] = [
    PropertyCheck(
        "S1_anchored",
        lambda r: all(len(c.anchors) >= 2 for c in r.criteria.values()) if r.criteria else True,
        "warning",
        "Not all criteria have >= 2 anchors (S1: stability risk)",
    ),
    PropertyCheck(
        "S2_aligned",
        lambda r: p_aligned(r) >= 0.5 if r.criteria else True,
        "warning",
        "Output schema does not reference majority of criterion IDs (S2: alignment gap)",
    ),
    PropertyCheck(
        "S3_mechanical",
        lambda r: p_mechanical(r) > 0 if r.criteria else True,
        "warning",
        "No criteria have mechanical rules or pattern references (S3: no deterministic handles)",
    ),
    PropertyCheck(
        "S4_disqualifiers",
        lambda r: p_dq(r) >= 1,
        "warning",
        "No disqualifiers defined (S4: no hard failure boundaries)",
    ),
    PropertyCheck(
        "S5_steering",
        lambda r: p_steering(r) >= 1 if r.output_schema else True,
        "warning",
        "No steering constraints in output schema (S5: no context-following anchors for stability)",
    ),
    PropertyCheck(
        "S6_mirror",
        lambda r: p_mirror(r) >= 1 if r.output_schema else True,
        "warning",
        "No policy mirroring (S6: constraints not restated in multiple locations)",
    ),
]


def validate(rubric: Rubric) -> ValidationResult:
    """Validate a rubric against the property lattice.

    Returns ValidationResult with is_valid (N1-N3 pass) and is_well_formed (all pass).
    """
    errors = tuple(c for c in NECESSARY if not c.predicate(rubric))
    warnings = tuple(c for c in SUFFICIENT if not c.predicate(rubric))
    return ValidationResult(
        is_valid=len(errors) == 0,
        is_well_formed=len(errors) == 0 and len(warnings) == 0,
        errors=errors,
        warnings=warnings,
    )


def suggest_fixes(rubric: Rubric) -> list[tuple[PropertyCheck, str]]:
    """For each failed property check, suggest a human-readable fix.

    Returns list of (failed_check, suggestion_text) tuples.
    This is the local (no-LLM) version of the property-to-fix feedback loop.
    """
    result = validate(rubric)
    suggestions: list[tuple[PropertyCheck, str]] = []

    for check in result.errors:
        if check.name == "N1_mission":
            suggestions.append((check, "Add a mission: rubric.mission = 'Evaluate X for Y.'"))
        elif check.name == "N2_structure":
            suggestions.append(
                (check, "Add criteria with anchors, decision rules, or mapping examples.")
            )
        elif check.name == "N3_output":
            suggestions.append(
                (
                    check,
                    "Set rubric.output_schema = OutputSchema(format='json', "
                    'template=\'{"score":0,"rationale":""}\')',
                )
            )

    for check in result.warnings:
        if check.name == "S1_anchored":
            weak = [cid for cid, c in rubric.criteria.items() if len(c.anchors) < 2]
            suggestions.append((check, f"Add >= 2 anchors to: {weak}"))
        elif check.name == "S2_aligned":
            suggestions.append(
                (check, "Update output_schema.template to include criterion IDs in subscores.")
            )
        elif check.name == "S3_mechanical":
            suggestions.append(
                (check, "Add mechanical_rules or uses_patterns to at least one criterion.")
            )
        elif check.name == "S4_disqualifiers":
            suggestions.append(
                (check, "Add: rubric.add_disqualifier(Disqualifier('DQ1', 'Auto-fail condition'))")
            )
        elif check.name == "S5_steering":
            suggestions.append(
                (
                    check,
                    "Add to output_schema.constraints: {'rationale_anchor': "
                    "\"Begin with 'BECAUSE:' and end with '.'; exactly 35 words.\"}",
                )
            )
        elif check.name == "S6_mirror":
            suggestions.append(
                (
                    check,
                    "Add rubric.validation_musts with a ValidationMust restating "
                    "a critical output_schema constraint.",
                )
            )

    return suggestions
