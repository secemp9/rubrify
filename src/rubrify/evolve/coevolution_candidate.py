"""Co-evolution candidate: multi-artifact <-> dict[str, str] mapping.

Packs the target rubric, proposal gate rubric, reflection templates,
and acceptance parameters into a single GEPA candidate dict with
namespace prefixes. GEPA's round-robin evolves all of them together.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rubrify.ir.roles import RoleSpec
from rubrify.ir.types import Rubric

from rubrify.evolve.candidate import candidate_to_rubric, rubric_to_candidate

PREFIX_TARGET = "target."
PREFIX_GATE = "gate."
PREFIX_REFLECTION = "reflection.template."
PREFIX_ACCEPTANCE = "acceptance."


# -- Template type classification (used for deduplication) --

_TEMPLATE_TYPE_RULES: list[tuple[Callable[[str], bool], str]] = [
    (lambda k: k.endswith(".description") and k.startswith("criterion."), "criterion_description"),
    (lambda k: k.endswith(".anchors") and k.startswith("criterion."), "criterion_anchors"),
    (lambda k: k.endswith(".weight") and k.startswith("criterion."), "criterion_weight"),
    (lambda k: k == "rubric.goal", "rubric_goal"),
    (lambda k: k == "rubric.instructions", "rubric_instructions"),
    (lambda k: k == "rubric.definitions", "rubric_definitions"),
    (lambda k: k == "rubric.advice_rules", "advice_rules"),
    (lambda k: k == "rubric.calibration_examples", "calibration_examples"),
    (lambda k: k.startswith("role."), "role"),
]


@dataclass
class CoEvolutionComponents:
    """All four artifacts reconstructed from a co-evolution candidate."""
    target_rubric: Rubric
    target_role: RoleSpec | None
    gate_rubric: Rubric
    reflection_templates: dict[str, str]  # component_key -> template text
    acceptance_params: dict[str, float]


def _classify_component(key: str) -> str:
    """Map a target component key to its template type name.

    Uses ``_TEMPLATE_TYPE_RULES`` to classify keys like
    ``criterion.C1.description`` -> ``"criterion_description"``.
    Falls back to the key itself if no rule matches.
    """
    for predicate, type_name in _TEMPLATE_TYPE_RULES:
        if predicate(key):
            return type_name
    return key


def _deduplicate_templates(templates: dict[str, str]) -> dict[str, str]:
    """From per-component templates, extract unique type -> template pairs.

    Multiple component keys may share the same template type (e.g. all
    ``criterion.*.description`` share ``criterion_description``).  We
    pick the first template text encountered for each type.
    """
    type_to_template: dict[str, str] = {}
    for component_key, template_text in templates.items():
        type_name = _classify_component(component_key)
        if type_name not in type_to_template:
            type_to_template[type_name] = template_text
    return type_to_template


def _expand_templates(
    candidate: dict[str, str],
    target_keys: set[str] | frozenset[str],
    base_templates: dict[str, str],
) -> dict[str, str]:
    """Rebuild per-component template mapping from deduplicated types in candidate.

    For each component key in ``base_templates``, look up its type,
    find the corresponding ``reflection.template.<type>`` entry in
    the candidate, and map the component key to that template text.
    Falls back to the base template if the type is missing from the candidate.
    """
    expanded: dict[str, str] = {}
    for component_key, base_text in base_templates.items():
        type_name = _classify_component(component_key)
        candidate_key = f"{PREFIX_REFLECTION}{type_name}"
        expanded[component_key] = candidate.get(candidate_key, base_text)
    return expanded


# ── Public API ────────────────────────────────────────────────────


def coevolution_to_candidate(
    target_rubric: Rubric,
    target_role: RoleSpec | None,
    gate_rubric: Rubric,
    reflection_templates: dict[str, str],
    acceptance_params: dict[str, float],
) -> dict[str, str]:
    """Pack four artifacts into one GEPA candidate dict."""
    candidate: dict[str, str] = {}

    # Target rubric (reuses existing rubric_to_candidate)
    target_sub = rubric_to_candidate(target_rubric, target_role)
    for k, v in target_sub.items():
        candidate[f"{PREFIX_TARGET}{k}"] = v

    # Gate rubric (same serialization, different prefix)
    gate_sub = rubric_to_candidate(gate_rubric, None)
    for k, v in gate_sub.items():
        candidate[f"{PREFIX_GATE}{k}"] = v

    # Reflection templates (deduplicated by type)
    type_to_template = _deduplicate_templates(reflection_templates)
    for type_name, template_text in type_to_template.items():
        candidate[f"{PREFIX_REFLECTION}{type_name}"] = template_text

    # Acceptance parameters
    for param_name, param_value in acceptance_params.items():
        candidate[f"{PREFIX_ACCEPTANCE}{param_name}"] = str(param_value)

    return candidate


def candidate_to_coevolution(
    candidate: dict[str, str],
    base_target_rubric: Rubric,
    base_target_role: RoleSpec | None,
    base_gate_rubric: Rubric,
    base_reflection_templates: dict[str, str],
    base_acceptance_params: dict[str, float],
) -> CoEvolutionComponents:
    """Reconstruct four artifacts from one GEPA candidate dict."""

    # Target rubric
    target_sub = {
        k[len(PREFIX_TARGET):]: v
        for k, v in candidate.items()
        if k.startswith(PREFIX_TARGET)
    }
    target_rubric, target_role = candidate_to_rubric(
        target_sub, base_target_rubric, base_target_role
    )

    # Gate rubric
    gate_sub = {
        k[len(PREFIX_GATE):]: v
        for k, v in candidate.items()
        if k.startswith(PREFIX_GATE)
    }
    gate_rubric, _ = candidate_to_rubric(gate_sub, base_gate_rubric, None)

    # Reflection templates: expand from deduplicated types back to per-component
    reflection_templates = _expand_templates(
        candidate, set(target_sub.keys()), base_reflection_templates
    )

    # Acceptance parameters
    acceptance_params: dict[str, float] = {}
    for param_name, default_value in base_acceptance_params.items():
        key = f"{PREFIX_ACCEPTANCE}{param_name}"
        acceptance_params[param_name] = float(candidate.get(key, str(default_value)))

    return CoEvolutionComponents(
        target_rubric=target_rubric,
        target_role=target_role,
        gate_rubric=gate_rubric,
        reflection_templates=reflection_templates,
        acceptance_params=acceptance_params,
    )


__all__ = [
    "PREFIX_ACCEPTANCE",
    "PREFIX_GATE",
    "PREFIX_REFLECTION",
    "PREFIX_TARGET",
    "CoEvolutionComponents",
    "candidate_to_coevolution",
    "coevolution_to_candidate",
]
