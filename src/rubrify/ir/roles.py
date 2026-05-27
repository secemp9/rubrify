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


class GenreModule(SchemaModel):
    """Genre-conditional module: activates additional criteria when genre matches.

    From ZinsserJudge: G_TRV for travel, G_BUS for business, G_HUM for humor.
    Each genre module has its own criteria that only fire when the input
    genre matches.
    """
    id: str
    genre: list[str]
    criteria_ids: list[str]
    weight_multiplier: float = 1.0
    description: str = ""


__all__ = [
    "GenreModule",
    "RoleSpec",
    "SurfacePolicy",
]
