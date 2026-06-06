"""Candidate representation: Rubric <-> dict[str, str] bidirectional mapping.

GEPA's core abstraction is Candidate = dict[str, str]. These two functions
convert between rubrify's structured Rubric and GEPA's flat candidate format.

Structural invariants (criterion IDs, scale types, ranges, groups, disqualifiers,
patterns) are NOT evolvable -- only text and weights are.
"""

from __future__ import annotations

import json
from typing import Any

from rubrify.ir.types import (
    AdviceRule,
    CalibrationExample,
    Criterion,
    Definition,
    NumericScale,
    OrdinalScale,
    Rubric,
    ScaleAnchor,
)
from rubrify.ir.roles import RoleSpec


def rubric_to_candidate(rubric: Rubric, role: RoleSpec | None = None) -> dict[str, str]:
    """Decompose a Rubric (and optional RoleSpec) into GEPA's flat candidate format.

    Each value is a string. Structured sub-components (lists, anchors)
    are serialized as JSON strings so that GEPA's reflection LM can
    reason about them as text while we can deterministically reconstruct
    the typed objects.
    """
    candidate: dict[str, str] = {}

    # Rubric-level text
    candidate["rubric.goal"] = rubric.goal
    if rubric.instructions:
        candidate["rubric.instructions"] = json.dumps(rubric.instructions)

    # Per-criterion text components
    for criterion in rubric.criteria:
        prefix = f"criterion.{criterion.id}"
        candidate[f"{prefix}.description"] = criterion.description

        # Anchors: serialize as JSON list of {value, label, description} dicts.
        # The LM can read and rewrite descriptions; value/label stay fixed.
        scale = criterion.scale
        if isinstance(scale, (OrdinalScale, NumericScale)):
            anchors_data = [
                {"value": a.value, "label": a.label, "description": a.description}
                for a in scale.anchors
            ]
            candidate[f"{prefix}.anchors"] = json.dumps(anchors_data, indent=2)

        candidate[f"{prefix}.weight"] = str(criterion.weight)

    # Definitions
    if rubric.definitions:
        candidate["rubric.definitions"] = json.dumps([
            {"id": d.id, "term": d.term, "description": d.description}
            for d in rubric.definitions
        ], indent=2)

    # Advice rules
    if rubric.advice_rules:
        candidate["rubric.advice_rules"] = json.dumps([
            {"fix": ar.fix}
            for ar in rubric.advice_rules
        ], indent=2)

    # Calibration examples
    if rubric.calibration_examples:
        candidate["rubric.calibration_examples"] = json.dumps([
            {"id": ce.id, "input_summary": ce.input_summary,
             "expected_verdict": ce.expected_verdict, "explanation": ce.explanation}
            for ce in rubric.calibration_examples
        ], indent=2)

    # Role persona and behavioral constraints
    if role is not None:
        candidate["role.persona"] = role.persona
        if role.obligations:
            candidate["role.obligations"] = json.dumps(role.obligations)
        if role.constraints:
            candidate["role.constraints"] = json.dumps(role.constraints)

    return candidate


def candidate_to_rubric(
    candidate: dict[str, str],
    base_rubric: Rubric,
    base_role: RoleSpec | None = None,
) -> tuple[Rubric, RoleSpec | None]:
    """Reconstruct a Rubric from a GEPA candidate dict.

    Uses the base_rubric as the structural template: criterion IDs,
    scale types, groups, disqualifiers, patterns, definitions, etc.
    are preserved from the base. Only the text components that GEPA
    can evolve are overwritten from the candidate dict.

    This ensures we always produce a structurally valid Rubric --
    GEPA cannot accidentally delete a criterion or corrupt a scale type.
    """
    # Rebuild criteria with evolved text
    new_criteria = []
    for base_crit in base_rubric.criteria:
        prefix = f"criterion.{base_crit.id}"

        # Description
        desc = candidate.get(f"{prefix}.description", base_crit.description)

        # Anchors
        scale = base_crit.scale
        anchors_key = f"{prefix}.anchors"
        if anchors_key in candidate and isinstance(scale, (OrdinalScale, NumericScale)):
            anchors_data = json.loads(candidate[anchors_key])
            new_anchors = [
                ScaleAnchor(
                    value=a["value"],
                    label=a["label"],
                    description=a["description"],
                )
                for a in anchors_data
            ]
            if isinstance(scale, OrdinalScale):
                scale = OrdinalScale(anchors=new_anchors)
            else:
                scale = NumericScale(
                    minimum=scale.minimum,
                    maximum=scale.maximum,
                    step=scale.step,
                    anchors=new_anchors,
                )

        # Weight
        weight = float(candidate.get(f"{prefix}.weight", str(base_crit.weight)))

        new_crit = base_crit.model_copy(update={
            "description": desc,
            "scale": scale,
            "weight": weight,
        })
        new_criteria.append(new_crit)

    # Rubric-level fields
    goal = candidate.get("rubric.goal", base_rubric.goal)
    instructions = base_rubric.instructions
    if "rubric.instructions" in candidate:
        instructions = json.loads(candidate["rubric.instructions"])

    # Definitions
    definitions = base_rubric.definitions
    if "rubric.definitions" in candidate:
        defs_data = json.loads(candidate["rubric.definitions"])
        definitions = [Definition(id=d["id"], term=d["term"], description=d["description"]) for d in defs_data]

    # Advice rules
    advice_rules = base_rubric.advice_rules
    if "rubric.advice_rules" in candidate:
        ar_data = json.loads(candidate["rubric.advice_rules"])
        advice_rules = [AdviceRule(fix=ar["fix"]) for ar in ar_data]

    # Calibration examples
    calibration_examples = base_rubric.calibration_examples
    if "rubric.calibration_examples" in candidate:
        ce_data = json.loads(candidate["rubric.calibration_examples"])
        calibration_examples = [
            CalibrationExample(id=ce["id"], input_summary=ce["input_summary"],
                              expected_verdict=ce["expected_verdict"],
                              explanation=ce.get("explanation", ""))
            for ce in ce_data
        ]

    new_rubric = base_rubric.model_copy(update={
        "goal": goal,
        "criteria": new_criteria,
        "instructions": instructions,
        "definitions": definitions,
        "advice_rules": advice_rules,
        "calibration_examples": calibration_examples,
    })

    # Role reconstruction
    new_role = base_role
    if base_role is not None:
        role_updates: dict[str, Any] = {}
        if "role.persona" in candidate:
            role_updates["persona"] = candidate["role.persona"]
        if "role.obligations" in candidate:
            role_updates["obligations"] = json.loads(candidate["role.obligations"])
        if "role.constraints" in candidate:
            role_updates["constraints"] = json.loads(candidate["role.constraints"])
        if role_updates:
            new_role = base_role.model_copy(update=role_updates)

    return new_rubric, new_role


__all__ = [
    "candidate_to_rubric",
    "rubric_to_candidate",
]
