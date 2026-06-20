"""evolve_rubric: top-level API for rubric evolution via GEPA.

Wires together the RubricEvolverAdapter, GEPA's engine, proposers,
and strategies to evolve a rubric's text components against a
human-annotated dataset.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.utils import MaxMetricCallsStopper

from harn_ai.types import Model as HarnifyModel

from rubrify.ir.constraints import OutputConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy
from rubrify.ir.types import Rubric

from rubrify.evolve.adapter import RubricEvolverAdapter
from rubrify.evolve.candidate import candidate_to_rubric, rubric_to_candidate
from rubrify.evolve.lm_bridge import make_harn_lm
from rubrify.evolve.types import AnnotatedExample

_UNSET = object()  # Sentinel to distinguish "not passed" from explicit values


# ── Reflection prompt template ─────────────────────────────────────

RUBRIC_EVOLUTION_REFLECTION_TEMPLATE = """You are an expert in evaluation rubric design. You are iteratively improving a rubric component to better align automated LLM-judge evaluations with human expert annotations.

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


# ── Result dataclass ───────────────────────────────────────────────

@dataclass
class RubricEvolutionConfig:
    """Configuration for rubric evolution."""
    train_split: float = 0.7
    max_metric_calls: int = 200
    consistency_runs: int = 1
    judge_temperature: float = 0.0
    judge_max_tokens: int = 2048
    reflection_temperature: float = 0.7
    reflection_max_tokens: int = 4096
    reflection_minibatch_size: int = 3
    seed: int = 42
    run_dir: str | None = None
    agreement_weight: float = 0.6
    consistency_weight: float = 0.2
    discrimination_weight: float = 0.2
    rubric_aware_acceptance: bool = False
    acceptance_agreement_tolerance: float = 0.0
    acceptance_discrimination_tolerance: float = 0.05
    acceptance_consistency_tolerance: float = 0.05
    use_specialized_templates: bool = False
    proposal_gate_model: Any = None  # HarnifyModel, use Any to avoid circular import
    proposal_gate_api_key: str | None = None
    proposal_gate_threshold: float = 0.5
    proposal_gate_max_retries: int = 1


@dataclass
class RubricEvolutionResult:
    """Result of rubric evolution."""
    best_rubric: Rubric
    best_role: RoleSpec | None
    seed_rubric: Rubric
    best_candidate: dict[str, str]
    best_score: float
    total_iterations: int
    total_metric_calls: int


# ── Top-level API ──────────────────────────────────────────────────

def evolve_rubric(
    seed_rubric: Rubric,
    annotated_dataset: list[AnnotatedExample],
    judge_model: HarnifyModel,
    reflection_model: HarnifyModel,
    *,
    role: RoleSpec | None = None,
    policy: SurfacePolicy | None = None,
    output_constraints: list[OutputConstraint] | None = None,
    judge_api_key: str | None = None,
    reflection_api_key: str | None = None,
    config: RubricEvolutionConfig | None = None,
    # Legacy kwargs for convenience -- used only when config is None.
    # Raises ValueError if both config and any of these are provided.
    train_split: float | object = _UNSET,
    max_metric_calls: int | object = _UNSET,
    consistency_runs: int | object = _UNSET,
    run_dir: str | None | object = _UNSET,
) -> RubricEvolutionResult:
    """Evolve a rubric using GEPA's reflective prompt evolution.

    This is Mode 1 (granular) rubric evolution: GEPA iteratively mutates the
    rubric's text components (criterion descriptions, anchor descriptions,
    weights, goal, instructions, role, definitions, advice rules, calibration
    examples) guided by structured feedback from rubrify's judge comparing
    against human annotations. Each mutation is evaluated on a training
    minibatch, accepted if improved, then tracked on a validation set with
    Pareto-based candidate selection across multiple objectives (agreement,
    discrimination, consistency).

    Practical guidance for getting improvements:

    - **Dataset size matters most.** 30-50 annotated examples minimum. With
      fewer than ~15 training examples, minibatch optimization overfits to
      surface features that don't generalize to the validation set. Each
      example should have human_scores for every criterion in the rubric.

    - **Disable discrimination at small scale.** Set discrimination_weight=0.0
      when your dataset has fewer than ~10 examples. The normalized entropy
      metric is too noisy with few data points and injects noise into the
      composite score that flips acceptance decisions randomly.

    - **Match minibatch to training size for tiny datasets.** Set
      reflection_minibatch_size equal to your training set size (e.g., 20 if
      you have 30 examples with 70/30 split) to eliminate sampling variance
      between minibatch and full training evaluation.

    - **Budget: 100-500 metric calls for real improvement.** Each GEPA
      iteration costs ~2 metric calls (before + after evaluation). With 20
      evolvable components and round-robin selection, you need at least 40
      iterations (80 calls) to touch each component twice. 200-500 calls
      allows compounding improvements where later mutations build on earlier
      accepted ones.

    Example::

        from harn_ai.models import get_model
        from rubrify.evolve import evolve_rubric, RubricEvolutionConfig

        result = evolve_rubric(
            seed_rubric=my_rubric,
            annotated_dataset=my_50_annotated_examples,
            judge_model=get_model("deepseek", "deepseek-v4-flash"),
            reflection_model=get_model("openai", "gpt-4o"),
            role=my_role,
            config=RubricEvolutionConfig(
                max_metric_calls=300,
                reflection_minibatch_size=5,
                discrimination_weight=0.1,  # low for moderate datasets
            ),
        )
        evolved_rubric = result.best_rubric

    Args:
        seed_rubric: The initial rubric to evolve. Structure (criterion IDs,
            scale types, groups, disqualifiers, patterns) is preserved;
            text components and weights are evolvable.
        annotated_dataset: Human-annotated (response, scores) pairs. Each
            AnnotatedExample must have human_scores keyed by criterion ID.
        judge_model: harn_ai Model for rubrify's Judge (evaluates responses).
        reflection_model: harn_ai Model for GEPA's reflection LM (proposes
            rubric improvements). Can be the same model or a stronger one.
        role: Optional RoleSpec for the judge persona.
        policy: Optional SurfacePolicy for rendering.
        output_constraints: Optional output constraints.
        judge_api_key: API key for judge model. Auto-discovered from env if None.
        reflection_api_key: API key for reflection model.
        config: Full configuration. If None, uses individual kwargs.

    Returns:
        RubricEvolutionResult with the best evolved rubric, score, and history.
    """
    # Resolve config
    _legacy_kwargs = {
        "train_split": train_split,
        "max_metric_calls": max_metric_calls,
        "consistency_runs": consistency_runs,
        "run_dir": run_dir,
    }
    _explicitly_passed = [k for k, v in _legacy_kwargs.items() if v is not _UNSET]

    if config is not None and _explicitly_passed:
        raise ValueError(
            f"Cannot provide both 'config' and individual keyword arguments "
            f"({', '.join(_explicitly_passed)}). Use one or the other."
        )

    if config is None:
        config = RubricEvolutionConfig(
            train_split=train_split if train_split is not _UNSET else 0.7,
            max_metric_calls=max_metric_calls if max_metric_calls is not _UNSET else 200,
            consistency_runs=consistency_runs if consistency_runs is not _UNSET else 1,
            run_dir=run_dir if run_dir is not _UNSET else None,
        )

    # Split dataset into train and validation
    rng = random.Random(config.seed)
    shuffled = list(annotated_dataset)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * config.train_split)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    if not val_data:
        val_data = train_data  # Fallback for tiny datasets

    # Build the adapter
    adapter = RubricEvolverAdapter(
        base_rubric=seed_rubric,
        judge_model=judge_model,
        base_role=role,
        base_policy=policy,
        base_output_constraints=output_constraints,
        judge_api_key=judge_api_key,
        judge_temperature=config.judge_temperature,
        judge_max_tokens=config.judge_max_tokens,
        consistency_runs=config.consistency_runs,
        agreement_weight=config.agreement_weight,
        consistency_weight=config.consistency_weight,
        discrimination_weight=config.discrimination_weight,
    )

    # Build the seed candidate
    seed_candidate = rubric_to_candidate(seed_rubric, role)

    # Build the reflection prompt template (str or dict[str, str])
    if config.use_specialized_templates:
        from rubrify.evolve.reflection_templates import build_reflection_template_dict
        reflection_template: str | dict[str, str] = build_reflection_template_dict(seed_rubric, role)
    else:
        reflection_template = RUBRIC_EVOLUTION_REFLECTION_TEMPLATE

    # Build the reflection LM using harn
    reflection_lm = make_harn_lm(
        reflection_model,
        api_key=reflection_api_key,
        temperature=config.reflection_temperature,
        max_tokens=config.reflection_max_tokens,
    )

    # Build proposal gate if configured
    custom_proposer = None
    if config.proposal_gate_model is not None:
        from rubrify.evolve.proposal_gate import ProposalQualityGate
        from rubrify.evolve.gated_proposer import GatedProposalFn

        gate = ProposalQualityGate(
            gate_model=config.proposal_gate_model,
            gate_api_key=config.proposal_gate_api_key or judge_api_key,
            threshold=config.proposal_gate_threshold,
            max_retries=config.proposal_gate_max_retries,
        )
        custom_proposer = GatedProposalFn(
            reflection_lm=reflection_lm,
            reflection_template=reflection_template,
            gate=gate,
        )

    # Set up GEPA components
    from rubrify.evolve.progress import EvolutionProgress
    logger = EvolutionProgress(total_budget=config.max_metric_calls)
    experiment_tracker = create_experiment_tracker()
    rng_gepa = random.Random(config.seed)

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_data,
        adapter=adapter,
        candidate_selector=ParetoCandidateSelector(rng=rng_gepa),
        module_selector=RoundRobinReflectionComponentSelector(),
        batch_sampler=EpochShuffledBatchSampler(
            minibatch_size=config.reflection_minibatch_size,
            rng=rng_gepa,
        ),
        perfect_score=None,
        skip_perfect_score=False,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        reflection_prompt_template=reflection_template,
        custom_candidate_proposer=custom_proposer,
    )

    # Build acceptance criterion
    if config.rubric_aware_acceptance:
        from rubrify.evolve.acceptance import RubricAwareAcceptance
        acceptance = RubricAwareAcceptance(
            agreement_tolerance=config.acceptance_agreement_tolerance,
            discrimination_tolerance=config.acceptance_discrimination_tolerance,
            consistency_tolerance=config.acceptance_consistency_tolerance,
        )
    else:
        from gepa.strategies.acceptance import StrictImprovementAcceptance
        acceptance = StrictImprovementAcceptance()

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=config.run_dir,
        valset=val_data,
        seed_candidate=seed_candidate,
        perfect_score=None,
        seed=config.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=None,
        frontier_type="hybrid",
        logger=logger,
        experiment_tracker=experiment_tracker,
        stop_callback=MaxMetricCallsStopper(config.max_metric_calls),
        **({} if "acceptance_criterion" not in __import__("inspect").signature(GEPAEngine.__init__).parameters else {"acceptance_criterion": acceptance}),
    )

    with experiment_tracker:
        state = engine.run()

    sys.stderr.write(logger.summary() + "\n")

    gepa_result = GEPAResult.from_state(state, run_dir=config.run_dir, seed=config.seed)

    # Reconstruct the best rubric from the best candidate
    best_candidate = gepa_result.candidates[gepa_result.best_idx]
    best_rubric, best_role = candidate_to_rubric(
        best_candidate, seed_rubric, role
    )

    return RubricEvolutionResult(
        best_rubric=best_rubric,
        best_role=best_role,
        seed_rubric=seed_rubric,
        best_candidate=best_candidate,
        best_score=gepa_result.best_score if hasattr(gepa_result, 'best_score') else state.program_full_scores_val_set[gepa_result.best_idx] if gepa_result.best_idx < len(state.program_full_scores_val_set) else 0.0,
        total_iterations=len(state.program_candidates) - 1,
        total_metric_calls=getattr(state, 'total_num_evals', len(state.program_candidates)),
    )



# ── Mode 3: Co-evolutionary rubric optimization ──────────────────


@dataclass
class CoEvolutionConfig:
    """Configuration for Mode 3 co-evolutionary rubric optimization."""
    train_split: float = 0.7
    max_metric_calls: int = 500   # Higher default: co-evolution has more components
    consistency_runs: int = 1
    judge_temperature: float = 0.0
    judge_max_tokens: int = 2048
    reflection_temperature: float = 0.7
    reflection_max_tokens: int = 4096
    reflection_minibatch_size: int = 3
    seed: int = 42
    run_dir: str | None = None
    agreement_weight: float = 0.6
    consistency_weight: float = 0.2
    discrimination_weight: float = 0.2
    # Meta-component defaults
    meta_degradation_tolerance: float = 0.01
    # Which meta-components to co-evolve (allows partial opt-in)
    evolve_gate: bool = True
    evolve_reflection_templates: bool = True
    evolve_acceptance_params: bool = True
    # Proposal gate model (used both as the gate AND to evaluate gate quality)
    gate_model: Any = None
    gate_api_key: str | None = None
    gate_threshold: float = 0.5
    gate_max_retries: int = 1


@dataclass
class CoEvolutionResult:
    """Result of Mode 3 co-evolutionary rubric optimization."""
    best_rubric: Rubric
    best_role: RoleSpec | None
    seed_rubric: Rubric
    best_candidate: dict[str, str]  # Full co-evolution candidate
    best_target_candidate: dict[str, str]  # Target-only subset
    best_score: float
    total_iterations: int
    total_metric_calls: int
    # Evolved meta-components for inspection/reuse
    evolved_gate_rubric: Rubric
    evolved_reflection_templates: dict[str, str]
    evolved_acceptance_params: dict[str, float]


class _CoEvolutionAcceptance:
    """Accept target mutations on improvement, meta mutations on non-degradation.

    Meta-component mutations (gate, reflection templates, acceptance params)
    have zero immediate effect on the evaluation score because they only
    change the search process, not the target rubric. Requiring strict
    improvement would reject all meta-component mutations. Instead, we
    accept them if the score does not degrade, treating them as neutral
    investments that may pay off in future iterations.

    Since GEPA's acceptance criterion does not know which component was
    mutated, we use a lenient threshold: accept if
    ``sum(new) >= sum(old) - epsilon`` where epsilon is small. This
    effectively accepts neutral meta-mutations while still requiring real
    improvement for target mutations (since those change the evaluation
    score).

    Known limitation
    ~~~~~~~~~~~~~~~~
    The meta-vs-target distinction does NOT actually work as designed.
    ``_get_mutated_component`` cannot reliably detect which component was
    mutated because GEPA proposals do not expose the mutated component
    name in their metadata, and the parent candidate is not accessible
    from the acceptance criterion interface to perform a diff. As a
    result, ``_get_mutated_component`` always returns ``None``, the
    ``is_meta`` flag is always ``False``, and **all** mutations
    (including meta-component mutations) are routed through the strict
    ``_target_acceptance`` path. In practice this means meta-component
    mutations that are score-neutral will be rejected, reducing the
    effectiveness of co-evolution for meta-components.
    """

    def __init__(
        self,
        target_acceptance: Any,
        meta_degradation_tolerance: float = 0.01,
    ):
        self._target_acceptance = target_acceptance
        self._meta_tolerance = meta_degradation_tolerance

    def should_accept(self, proposal: Any, state: Any) -> bool:
        """Decide whether to accept.

        Check if the mutated component is a meta-component by diffing
        the old and new candidates.  If meta, accept on non-degradation.
        If target, delegate to the strict acceptance criterion.
        """
        component_name = self._get_mutated_component(proposal)
        is_meta = (
            component_name is not None
            and not component_name.startswith("target.")
        )

        if is_meta:
            old_sum = sum(proposal.subsample_scores_before or [])
            new_sum = sum(proposal.subsample_scores_after or [])
            return new_sum >= old_sum - self._meta_tolerance
        else:
            return self._target_acceptance.should_accept(proposal, state)

    @staticmethod
    def _get_mutated_component(proposal: Any) -> str | None:
        """Find which component was changed by diffing old/new candidates.

        NOTE: This method always returns None in practice. GEPA proposals
        do not populate ``metadata["mutated_component"]``, and the parent
        candidate is not accessible from the acceptance interface to
        perform a diff. See class docstring "Known limitation" for the
        full impact analysis.
        """
        if not hasattr(proposal, 'candidate') or not hasattr(proposal, 'metadata'):
            return None
        # The metadata may contain the component that was mutated
        if isinstance(proposal.metadata, dict):
            comp = proposal.metadata.get("mutated_component")
            if comp:
                return comp
        # Fallback: diff against parent candidate
        parent_ids = getattr(proposal, 'parent_program_ids', None)
        if parent_ids and hasattr(proposal, 'candidate'):
            # We cannot access the parent candidate from here directly,
            # so fall back to the lenient acceptance for all mutations
            pass
        return None


_META_REFLECTION_TEMPLATE = """You are an expert in meta-optimization for evaluation rubrics. You are improving a meta-component that controls HOW the rubric evolution search operates.

## Current Meta-Component

The meta-component being optimized:

```
<curr_param>
```

## Diagnostic Data

Below are diagnostic records showing how this meta-component relates to the target rubric's performance:

```
<side_info>
```

## Your Task

Analyze the diagnostic data:
- How is this meta-component currently influencing the search?
- Is the current setting too aggressive or too conservative?
- What adjustment would help the evolution process find better rubric configurations?

Propose an improved version. Provide ONLY the improved component within ``` blocks. If numeric, provide just the number. If text, provide the full improved text."""


def evolve_rubric_v3(
    seed_rubric: Rubric,
    annotated_dataset: list[AnnotatedExample],
    judge_model: HarnifyModel,
    reflection_model: HarnifyModel,
    *,
    role: RoleSpec | None = None,
    policy: SurfacePolicy | None = None,
    output_constraints: list[OutputConstraint] | None = None,
    judge_api_key: str | None = None,
    reflection_api_key: str | None = None,
    config: CoEvolutionConfig | None = None,
) -> CoEvolutionResult:
    """Mode 3: Co-evolve target rubric + meta-rubrics in a single GEPA loop.

    In addition to evolving the target rubric's text components (Mode 1),
    this also evolves the proposal quality gate rubric, reflection prompt
    templates, and acceptance parameters.  All four artifact types are
    packed into a single GEPA candidate dict with namespace prefixes and
    optimized via round-robin component selection.

    Args:
        seed_rubric: The initial rubric to evolve.
        annotated_dataset: Human-annotated (response, scores) pairs.
        judge_model: harn_ai Model for rubrify's Judge.
        reflection_model: harn_ai Model for GEPA's reflection LM.
        role: Optional RoleSpec for the judge persona.
        policy: Optional SurfacePolicy for rendering.
        output_constraints: Optional output constraints.
        judge_api_key: API key for judge model.
        reflection_api_key: API key for reflection model.
        config: Full configuration. If None, uses defaults.

    Returns:
        CoEvolutionResult with the best evolved rubric and meta-components.
    """
    from rubrify.evolve.acceptance import RubricAwareAcceptance
    from rubrify.evolve.coevolution_adapter import CoEvolutionAdapter
    from rubrify.evolve.coevolution_candidate import (
        PREFIX_ACCEPTANCE,
        PREFIX_GATE,
        PREFIX_REFLECTION,
        PREFIX_TARGET,
        candidate_to_coevolution,
        coevolution_to_candidate,
    )
    from rubrify.evolve.proposal_gate import make_proposal_quality_rubric
    from rubrify.evolve.reflection_templates import build_reflection_template_dict

    config = config or CoEvolutionConfig()

    # 1. Build seed artifacts
    gate_rubric = make_proposal_quality_rubric()
    reflection_templates = build_reflection_template_dict(seed_rubric, role)
    acceptance_params: dict[str, float] = {
        "agreement_tolerance": 0.0,
        "discrimination_tolerance": 0.05,
        "consistency_tolerance": 0.05,
    }

    # 2. Build co-evolution seed candidate
    seed_candidate = coevolution_to_candidate(
        seed_rubric, role, gate_rubric, reflection_templates, acceptance_params,
    )

    # 3. Identify frozen keys (meta-components the user opted out of evolving)
    frozen_keys: set[str] = set()
    if not config.evolve_gate:
        frozen_keys.update(k for k in seed_candidate if k.startswith(PREFIX_GATE))
    if not config.evolve_reflection_templates:
        frozen_keys.update(k for k in seed_candidate if k.startswith(PREFIX_REFLECTION))
    if not config.evolve_acceptance_params:
        frozen_keys.update(k for k in seed_candidate if k.startswith(PREFIX_ACCEPTANCE))

    # 4. Split dataset into train and validation
    rng = random.Random(config.seed)
    shuffled = list(annotated_dataset)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * config.train_split)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    if not val_data:
        val_data = train_data

    # 5. Build the co-evolution adapter
    adapter = CoEvolutionAdapter(
        base_target_rubric=seed_rubric,
        base_gate_rubric=gate_rubric,
        base_reflection_templates=reflection_templates,
        base_acceptance_params=acceptance_params,
        judge_model=judge_model,
        base_target_role=role,
        base_policy=policy,
        base_output_constraints=output_constraints,
        judge_api_key=judge_api_key,
        judge_temperature=config.judge_temperature,
        judge_max_tokens=config.judge_max_tokens,
        consistency_runs=config.consistency_runs,
        agreement_weight=config.agreement_weight,
        consistency_weight=config.consistency_weight,
        discrimination_weight=config.discrimination_weight,
    )

    # 6. Build co-evolution reflection templates
    #    Target components use the existing specialized templates.
    #    Meta components use the generic meta-reflection template.
    co_reflection_templates: dict[str, str] = {}
    for key in seed_candidate:
        if key in frozen_keys:
            continue
        if key.startswith(PREFIX_TARGET):
            # Re-use the inner reflection template for this target component
            inner_key = key[len(PREFIX_TARGET):]
            co_reflection_templates[key] = reflection_templates.get(
                inner_key, RUBRIC_EVOLUTION_REFLECTION_TEMPLATE
            )
        else:
            # Meta-component: use the generic meta-reflection template
            co_reflection_templates[key] = _META_REFLECTION_TEMPLATE

    # 7. Build reflection LM
    reflection_lm = make_harn_lm(
        reflection_model,
        api_key=reflection_api_key,
        temperature=config.reflection_temperature,
        max_tokens=config.reflection_max_tokens,
    )

    # 8. Set up GEPA components
    from rubrify.evolve.progress import EvolutionProgress
    logger = EvolutionProgress(total_budget=config.max_metric_calls)
    experiment_tracker = create_experiment_tracker()
    rng_gepa = random.Random(config.seed)

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_data,
        adapter=adapter,
        candidate_selector=ParetoCandidateSelector(rng=rng_gepa),
        module_selector=RoundRobinReflectionComponentSelector(),
        batch_sampler=EpochShuffledBatchSampler(
            minibatch_size=config.reflection_minibatch_size,
            rng=rng_gepa,
        ),
        perfect_score=None,
        skip_perfect_score=False,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
        reflection_prompt_template=co_reflection_templates,
    )

    # 9. Build acceptance criterion
    inner_acceptance = RubricAwareAcceptance(
        agreement_tolerance=0.0,
        discrimination_tolerance=0.05,
        consistency_tolerance=0.05,
    )
    acceptance = _CoEvolutionAcceptance(
        target_acceptance=inner_acceptance,
        meta_degradation_tolerance=config.meta_degradation_tolerance,
    )

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=config.run_dir,
        valset=val_data,
        seed_candidate=seed_candidate,
        perfect_score=None,
        seed=config.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=None,
        frontier_type="hybrid",
        logger=logger,
        experiment_tracker=experiment_tracker,
        stop_callback=MaxMetricCallsStopper(config.max_metric_calls),
        **({} if "acceptance_criterion" not in __import__("inspect").signature(GEPAEngine.__init__).parameters else {"acceptance_criterion": acceptance}),
    )

    # 10. Run the engine
    with experiment_tracker:
        state = engine.run()

    sys.stderr.write(logger.summary() + "\n")

    gepa_result = GEPAResult.from_state(state, run_dir=config.run_dir, seed=config.seed)

    # 11. Reconstruct best result
    best_candidate = gepa_result.candidates[gepa_result.best_idx]
    components = candidate_to_coevolution(
        best_candidate,
        seed_rubric,
        role,
        gate_rubric,
        reflection_templates,
        acceptance_params,
    )

    best_target_candidate = {
        k[len(PREFIX_TARGET):]: v
        for k, v in best_candidate.items()
        if k.startswith(PREFIX_TARGET)
    }

    return CoEvolutionResult(
        best_rubric=components.target_rubric,
        best_role=components.target_role,
        seed_rubric=seed_rubric,
        best_candidate=best_candidate,
        best_target_candidate=best_target_candidate,
        best_score=(
            gepa_result.best_score
            if hasattr(gepa_result, 'best_score')
            else (
                state.program_full_scores_val_set[gepa_result.best_idx]
                if gepa_result.best_idx < len(state.program_full_scores_val_set)
                else 0.0
            )
        ),
        total_iterations=len(state.program_candidates) - 1,
        total_metric_calls=getattr(state, 'total_num_evals', len(state.program_candidates)),
        evolved_gate_rubric=components.gate_rubric,
        evolved_reflection_templates=components.reflection_templates,
        evolved_acceptance_params=components.acceptance_params,
    )


__all__ = [
    "CoEvolutionConfig",
    "CoEvolutionResult",
    "RUBRIC_EVOLUTION_REFLECTION_TEMPLATE",
    "RubricEvolutionConfig",
    "RubricEvolutionResult",
    "evolve_rubric",
    "evolve_rubric_v3",
]
