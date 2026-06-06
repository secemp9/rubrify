"""RoleSpec — first-class role specification for the judge.

Roleplaying is not cosmetic. It is a form of constraint engineering:
the role assignment defines the model's behavioral space and professional
obligations. A role-play backed by structural constraints (XML, IDs,
decision logic) overrides base behavior more reliably than "you are X."

RoleSpec makes this explicit and auditable.
"""

from __future__ import annotations

from typing import Literal

from rubrify.ir.types import SchemaModel


class RoleSpec(SchemaModel):
    """Role specification for the judge model.

    The role is not a prompt prefix — it is a structural component that
    constrains behavior. The 'obligations' and 'constraints' fields
    define what the model MUST and MUST NOT do, functioning as behavioral
    contracts that the rubric enforces through redundancy.
    """
    id: str
    persona: str
    authority: Literal["absolute", "advisory", "peer"] = "absolute"
    domain: str | None = None
    obligations: list[str] = []
    constraints: list[str] = []


class SurfacePolicy(SchemaModel):
    """Governs how rubrics project to different surface formats.

    The 'policy mirrors' idea: if the task forbids external knowledge,
    declare it thrice — rubric prose, XML structural tag, and JSON
    output check.
    """
    input_codec: Literal["xml"] = "xml"
    output_codec: Literal["json"] = "json"
    role: RoleSpec | None = None
    enforce_key_order: bool = True
    criterion_focus: Literal["full", "focused"] = "full"
    decision_thresholds: list[tuple[float, str]] | None = None
    execution_strategy: Literal["holistic", "grouped", "per_criterion"] = "per_criterion"
    """How the judge dispatches criteria to LLM calls during evaluation.

    - "holistic": one LLM call evaluates all active criteria simultaneously.
      Produces one shared rationale and evidence pool. Matches the original
      rubric architecture where a single output_schema contains all criterion
      scores in one JSON blob.
    - "grouped": one LLM call per CriterionGroup. Criteria within a group are
      evaluated together with shared rationale/evidence. Ungrouped criteria
      fall back to per_criterion behavior.
    - "per_criterion": one LLM call per criterion (current behavior). Maximum
      isolation between criteria, maximum cost.
    """


__all__ = [
    "RoleSpec",
    "SurfacePolicy",
]
