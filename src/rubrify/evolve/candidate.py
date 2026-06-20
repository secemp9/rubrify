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
    CorpusProfile,
    Criterion,
    Definition,
    NumericScale,
    OrdinalScale,
    Rubric,
    ScaleAnchor,
    ScopeSpec,
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

        # Scope specification (interpretation boundary)
        # Always emit keys so GEPA's component selector has stable names.
        scope = criterion.scope
        candidate[f"{prefix}.scope.in_scope"] = json.dumps(scope.in_scope if scope else [])
        candidate[f"{prefix}.scope.out_of_scope"] = json.dumps(scope.out_of_scope if scope else [])

    # Corpus profile — always emit keys for GEPA stability
    cp = rubric.corpus_profile
    candidate["corpus_profile.typical"] = json.dumps(cp.typical_behaviors if cp else [])
    candidate["corpus_profile.atypical"] = json.dumps(cp.atypical_behaviors if cp else [])
    candidate["corpus_profile.quality_axis"] = cp.quality_axis if cp else ""

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

        # Scope
        scope = base_crit.scope
        scope_in_key = f"{prefix}.scope.in_scope"
        scope_out_key = f"{prefix}.scope.out_of_scope"
        if scope_in_key in candidate or scope_out_key in candidate:
            in_scope = json.loads(candidate[scope_in_key]) if scope_in_key in candidate else (scope.in_scope if scope else [])
            out_of_scope = json.loads(candidate[scope_out_key]) if scope_out_key in candidate else (scope.out_of_scope if scope else [])
            scope = ScopeSpec(in_scope=in_scope, out_of_scope=out_of_scope)

        new_crit = base_crit.model_copy(update={
            "description": desc,
            "scale": scale,
            "weight": weight,
            "scope": scope,
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

    # Corpus profile
    corpus_profile = base_rubric.corpus_profile
    if any(k.startswith("corpus_profile.") for k in candidate):
        base_cp = base_rubric.corpus_profile
        typical_raw = candidate.get("corpus_profile.typical")
        atypical_raw = candidate.get("corpus_profile.atypical")
        corpus_profile = CorpusProfile(
            id=base_cp.id if base_cp else "corpus",
            domain=base_cp.domain if base_cp else "",
            typical_behaviors=json.loads(typical_raw) if typical_raw is not None else (base_cp.typical_behaviors if base_cp else []),
            atypical_behaviors=json.loads(atypical_raw) if atypical_raw is not None else (base_cp.atypical_behaviors if base_cp else []),
            quality_axis=candidate.get("corpus_profile.quality_axis", base_cp.quality_axis if base_cp else ""),
        )

    new_rubric = base_rubric.model_copy(update={
        "goal": goal,
        "criteria": new_criteria,
        "instructions": instructions,
        "definitions": definitions,
        "advice_rules": advice_rules,
        "calibration_examples": calibration_examples,
        "corpus_profile": corpus_profile,
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
