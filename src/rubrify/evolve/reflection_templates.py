"""Component-specific reflection prompt templates for rubric evolution.

Each component type gets a specialized template that focuses the reflection
LM on the right diagnostic criteria. GEPA supports dict[str, str] for
reflection_prompt_template, keyed by component name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rubrify.ir.roles import RoleSpec
    from rubrify.ir.types import Rubric


# ── Generic fallback (matches the existing RUBRIC_EVOLUTION_REFLECTION_TEMPLATE) ──

GENERIC_REFLECTION_TEMPLATE = """You are an expert in evaluation rubric design. You are iteratively improving a rubric component to better align automated LLM-judge evaluations with human expert annotations.

## Current Component

The rubric component being optimized:

```
<curr_param>
```

## Evaluation Data

Below are cases where this rubric component was used to evaluate responses. Each case shows:
- The current component value
- How the LLM judge scored using this component
- What the human expert annotated
- Diagnostic feedback about agreement or disagreement

```
<side_info>
```

## Your Task

Analyze the evaluation data systematically:
- **Disagreement patterns**: Where does the judge consistently disagree with humans? What ambiguities in the component text cause this?
- **Anchor clarity**: For anchor descriptions, are the boundaries between score levels clearly delineated? Could a judge confuse adjacent levels?
- **Specificity**: Is the component specific enough to produce consistent judgments, but general enough to apply across diverse responses?
- **Actionability**: Does the component give the judge clear decision criteria, or does it leave room for subjective interpretation?

Propose an improved version that:
1. Resolves identified disagreement patterns
2. Makes boundaries between score levels unambiguous
3. Uses concrete, observable indicators rather than vague qualitative language
4. Preserves any aspects that already achieve human agreement

Provide ONLY the improved component within ``` blocks. If this is a JSON structure (e.g., anchor descriptions), maintain valid JSON format."""


# ── Specialized templates ─────────────────────────────────────────

# Template for criterion descriptions
_CRITERION_DESCRIPTION_TEMPLATE = """You are improving a rubric criterion description to better align LLM judge scoring with human expert annotations.

Current criterion description:
```
<curr_param>
```

Evaluation data showing how this criterion performed:
```
<side_info>
```

Focus on:
- Where did the judge and human disagree? What in the description caused the judge to misinterpret?
- Are there concrete, observable indicators the judge should look for?
- Is the boundary between "good" and "bad" responses clearly defined?
- Would a different judge reading this description arrive at the same scores?

Propose an improved description. Keep it concise. Use observable indicators, not subjective qualities. Provide ONLY the improved text within ``` blocks."""

# Template for criterion anchors (JSON)
_CRITERION_ANCHORS_TEMPLATE = """You are improving the anchor descriptions for a rubric criterion. Anchors define what each score level means.

Current anchors (JSON):
```
<curr_param>
```

Evaluation data showing scoring alignment:
```
<side_info>
```

Focus on:
- At which adjacent score levels did the judge and human disagree most?
- Are the boundaries between levels sharp enough? Can you add distinguishing language?
- Does each anchor describe observable features, not vague impressions?
- Would swapping two anchor descriptions change a reasonable judge's scores?

Propose improved anchors. Maintain the same value/label structure. Only modify descriptions. Return valid JSON within ``` blocks."""

# Template for criterion weights
_CRITERION_WEIGHT_TEMPLATE = """You are adjusting the weight of a rubric criterion. The weight determines how much this criterion contributes to the overall score.

Current weight:
```
<curr_param>
```

Evaluation data with per-criterion performance:
```
<side_info>
```

Focus on:
- Is this criterion reliable (consistent agreement with humans)?
- Is it discriminative (different responses get different scores)?
- Should unreliable criteria have less weight? Should important-but-weak criteria keep weight while their descriptions improve?

Propose an adjusted weight (a single number). Provide ONLY the number within ``` blocks."""

# Template for rubric goal
_RUBRIC_GOAL_TEMPLATE = """You are improving the goal statement of an evaluation rubric. The goal defines what the rubric evaluates.

Current goal:
```
<curr_param>
```

Evaluation data:
```
<side_info>
```

Focus on:
- Does the goal clearly scope what should and should not be evaluated?
- Are criteria-level disagreements caused by the goal being too broad or too narrow?
- Would a judge reading only the goal understand the rubric's intent?

Propose an improved goal. Provide ONLY the improved text within ``` blocks."""

# Template for instructions
_RUBRIC_INSTRUCTIONS_TEMPLATE = """You are improving the step-by-step instructions for an LLM judge using this rubric.

Current instructions (JSON list):
```
<curr_param>
```

Evaluation data:
```
<side_info>
```

Focus on:
- Are the instructions explicit about scoring procedure (read first, then score)?
- Do they tell the judge to score each criterion independently?
- Do they address common errors (inferring intent, averaging adjacent scores, letting one criterion influence another)?

Propose improved instructions. Return a valid JSON list of strings within ``` blocks."""

# Template for definitions
_RUBRIC_DEFINITIONS_TEMPLATE = """You are improving the semantic definitions embedded in this rubric. Definitions ground key terms used in criteria.

Current definitions (JSON):
```
<curr_param>
```

Evaluation data:
```
<side_info>
```

Focus on:
- Are the defined terms used consistently across criteria?
- Do disagreements trace back to ambiguous definitions?
- Can definitions be made more precise without being too narrow?

Propose improved definitions. Return valid JSON within ``` blocks."""

# Template for advice rules
_ADVICE_RULES_TEMPLATE = """You are improving the advice rules in this rubric. Advice rules map detected patterns to concrete fix recommendations.

Current advice rules (JSON):
```
<curr_param>
```

Evaluation data showing which patterns fired and what advice was given:
```
<side_info>
```

Focus on:
- Are triggered advice rules producing actionable, specific fixes?
- Are there disagreements where better advice would have guided the judge differently?
- Are there patterns firing without useful corresponding advice?

Propose improved advice rules. Return valid JSON within ``` blocks."""

# Template for calibration examples
_CALIBRATION_EXAMPLES_TEMPLATE = """You are improving the calibration examples in this rubric. Calibration examples show the judge what correct scoring looks like.

Current calibration examples (JSON):
```
<curr_param>
```

Evaluation data showing alignment:
```
<side_info>
```

Focus on:
- Do existing examples cover the scoring boundary where disagreements cluster?
- Would adding or revising an example resolve a recurring misclassification?
- Do explanations clearly articulate WHY the verdict is correct?

Propose improved calibration examples. Return valid JSON within ``` blocks."""

# Template for role components
_ROLE_TEMPLATE = """You are improving the role specification for the LLM judge.

Current role component:
```
<curr_param>
```

Evaluation data:
```
<side_info>
```

Focus on:
- Does the role persona lead to consistent evaluation behavior?
- Are obligations/constraints clear enough to prevent scoring drift?
- Does the role match the rubric's evaluation domain?

Propose an improved version. Provide ONLY the improved text within ``` blocks."""


# ── Builder ───────────────────────────────────────────────────────

def build_reflection_template_dict(
    rubric: "Rubric",
    role: "RoleSpec | None" = None,
) -> dict[str, str]:
    """Build a component-name-to-template mapping for GEPA's reflection.

    GEPA's ReflectiveMutationProposer accepts reflection_prompt_template
    as either str (one template for all) or dict[str, str] (per-component).
    This function produces the per-component version.
    """
    from rubrify.evolve.candidate import rubric_to_candidate

    templates: dict[str, str] = {}
    candidate_keys = rubric_to_candidate(rubric, role).keys()

    for key in candidate_keys:
        if key.endswith(".description") and key.startswith("criterion."):
            templates[key] = _CRITERION_DESCRIPTION_TEMPLATE
        elif key.endswith(".anchors") and key.startswith("criterion."):
            templates[key] = _CRITERION_ANCHORS_TEMPLATE
        elif key.endswith(".weight") and key.startswith("criterion."):
            templates[key] = _CRITERION_WEIGHT_TEMPLATE
        elif key == "rubric.goal":
            templates[key] = _RUBRIC_GOAL_TEMPLATE
        elif key == "rubric.instructions":
            templates[key] = _RUBRIC_INSTRUCTIONS_TEMPLATE
        elif key == "rubric.definitions":
            templates[key] = _RUBRIC_DEFINITIONS_TEMPLATE
        elif key == "rubric.advice_rules":
            templates[key] = _ADVICE_RULES_TEMPLATE
        elif key == "rubric.calibration_examples":
            templates[key] = _CALIBRATION_EXAMPLES_TEMPLATE
        elif key.startswith("role."):
            templates[key] = _ROLE_TEMPLATE
        else:
            # Fallback to generic
            templates[key] = GENERIC_REFLECTION_TEMPLATE

    return templates


__all__ = [
    "GENERIC_REFLECTION_TEMPLATE",
    "build_reflection_template_dict",
]
