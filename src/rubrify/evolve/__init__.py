"""rubrify.evolve — rubric evolution via GEPA's optimization loop.

Evolves rubric text components (goal, criterion descriptions, anchor
descriptions, weights, role persona, instructions) against a human-annotated
dataset to maximize agreement between automated LLM-judge evaluations
and human expert annotations.

Requires the optional `gepa` dependency:
    pip install rubrify[evolve]  # or: pip install gepa
"""

from rubrify.evolve.coevolution_candidate import CoEvolutionComponents
from rubrify.evolve.evolver import (
    CoEvolutionConfig,
    CoEvolutionResult,
    RubricEvolutionConfig,
    RubricEvolutionResult,
    evolve_rubric,
    evolve_rubric_v3,
)
from rubrify.evolve.proposal_gate import ProposalQualityGate
from rubrify.evolve.types import (
    AnnotatedExample,
    RubricQualityScore,
)

__all__ = [
    "AnnotatedExample",
    "CoEvolutionComponents",
    "CoEvolutionConfig",
    "CoEvolutionResult",
    "ProposalQualityGate",
    "RubricEvolutionConfig",
    "RubricEvolutionResult",
    "RubricQualityScore",
    "evolve_rubric",
    "evolve_rubric_v3",
]
