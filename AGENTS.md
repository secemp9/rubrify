# AGENTS.md -- AI Coding Agent Guide for rubrify

This file gives AI coding agents the context needed to work effectively in this codebase. For full user-facing documentation, see README.md.

---

## Project Overview

rubrify is a rubric compiler and judge engine for LLM evaluation. You define structured evaluation rubrics as typed Python objects (the IR layer), compile them into immutable bundles (the compiler), render them as XML prompts (the codecs), and execute criterion-by-criterion LLM-based judgments against text responses (the engine). An optional evolution module uses GEPA to iteratively optimize rubric text components against human-annotated datasets. All LLM access goes through the `harn_ai` library; all models are Pydantic v2 with `extra="forbid"`.

---

## Tech Stack and Constraints

- **Python >= 3.12** (required by pyproject.toml).
- **Pydantic v2 (>= 2.10)** with `SchemaModel` from `harn_ai.types`. SchemaModel sets `extra="forbid"` by default, meaning any unknown field on any model raises `ValidationError`.
- **harn_ai** for LLM access (multi-provider: OpenAI, Anthropic, DeepSeek, Google, local proxies). Provides `Model`, `Context`, `SimpleStreamOptions`, `Tool`, `UserMessage`, `complete_simple()`, `parse_json_with_repair()`, and `get_env_api_key()`.
- **harn_agent** for agent primitives (dependency, but not heavily used in rubrify's own code).
- **defusedxml >= 0.7** for safe XML parsing. Construction uses `xml.etree.ElementTree`; parsing uses `defusedxml.ElementTree`.
- **gepa >= 0.1.0** (optional, for `rubrify[evolve]`). Provides `GEPAAdapter`, `GEPAEngine`, `ReflectiveMutationProposer`, `ParetoCandidateSelector`, `RoundRobinReflectionComponentSelector`, `EpochShuffledBatchSampler`, `MaxMetricCallsStopper`, `AcceptanceCriterion`, `EvaluationBatch`, `GEPAResult`.
- **asyncio throughout the engine**. `execute_criterion()`, `run_judge_loop()`, and `Judge.evaluate()` are all async. The evolution module bridges async-to-sync via `evolve/async_bridge.py`'s `run_async()` utility, which uses `asyncio.run()` when no event loop is running, or dispatches to a `ThreadPoolExecutor` when already inside an async context (e.g. Jupyter, async test harness).
- **Build system**: Hatchling (`hatchling.build`), with `src/rubrify` layout.
- **Workspace sources**: `harn-ai` and `harn-agent` are workspace-local; `gepa` is a path dependency (`../../../gepa`).

---

## Module Map

### `src/rubrify/__init__.py`
Public API surface. Re-exports all core types from ir/, compiler/, codecs/, and engine/. Conditionally imports evolve/ (guarded by `ImportError` for missing gepa).

### ir/ -- Intermediate Representation (Typed Core)

| File | Description | Key Exports |
|---|---|---|
| `ir/__init__.py` | Star-imports from all IR submodules. | -- |
| `ir/types.py` | Scale types (BinaryScale, OrdinalScale, NominalScale, NumericScale), ScaleAnchor, EvidenceSpec, Criterion, CriterionGroup, Disqualifier, Definition, AdviceRule, CalibrationExample, PatternEntry, RubricMeta, Rubric. The Scale union is `Annotated[..., Field(discriminator="kind")]`. Rubric has model validators for unique criterion IDs, valid group refs, and valid disqualifier refs. | `Scale`, `Criterion`, `Rubric`, all scale types |
| `ir/roles.py` | RoleSpec (judge persona with authority, obligations, constraints), SurfacePolicy (codec selection, criterion_focus, decision_thresholds, execution_strategy). `execution_strategy` on SurfacePolicy controls how criteria are dispatched to LLM calls: `"holistic"`, `"grouped"`, or `"per_criterion"` (default). | `RoleSpec`, `SurfacePolicy` |
| `ir/constraints.py` | SurfaceProjection (one codec-specific representation), ConstraintBinding (triple-layer alignment: criterion <-> surface <-> output), AuthorityBlock (instruction vs data marking), OutputConstraint discriminated union of typed variants (PrefixSuffixConstraint, WordCountConstraint, CharLimitConstraint, ItemCountConstraint, TokenConstraint), each with a `check()` method and a `scope` field (`"call"`, `"criterion"`, or `"judgment"`) controlling when the constraint is checked. | `ConstraintBinding`, `SurfaceProjection`, `AuthorityBlock`, `OutputConstraint`, `PrefixSuffixConstraint`, `WordCountConstraint`, `CharLimitConstraint`, `ItemCountConstraint`, `TokenConstraint` |
| `ir/bundle.py` | RubricBundle (frozen Pydantic model, `model_config = ConfigDict(extra="forbid", frozen=True)`). `lock_bundle()` compiles regex patterns from PatternEntry and Disqualifier patterns, fails loudly on invalid regex. | `RubricBundle`, `lock_bundle` |

### compiler/ -- Rubric to RubricBundle Transformation

| File | Description | Key Exports |
|---|---|---|
| `compiler/__init__.py` | Star-imports from compiler.py. | -- |
| `compiler/compiler.py` | `compile_rubric(rubric, *, policy, output_constraints) -> CompilationResult`. Synchronous, pure function. Runs: bind -> authority_blocks -> lock -> audit. `CompilationResult` is a dataclass with `.bundle`, `.issues`, and `.ok` property. | `compile_rubric`, `CompilationResult` |
| `compiler/passes.py` | Individual compiler passes as pure functions: `bind()` (generates ConstraintBinding per criterion with XML and JSON SurfaceProjections), `audit_coverage()`, `audit_projection_completeness()`, `audit_scale_consistency()`, `audit_output_constraints()`. | `bind`, `audit_coverage`, `audit_output_constraints`, `audit_projection_completeness`, `audit_scale_consistency` |

### codecs/ -- Surface Format Rendering and Parsing

| File | Description | Key Exports |
|---|---|---|
| `codecs/__init__.py` | Re-exports from xml_codec and json_codec. | -- |
| `codecs/xml_codec.py` | `render_rubric_xml(bundle) -> str` renders the full `<LLM_JUDGE_SPEC>` XML document. `render_criterion_xml(criterion, bundle) -> str` renders a focused single-criterion document. `render_group_xml(criteria, bundle) -> str` renders a subset document for grouped execution (filtered criteria, relevant disqualifiers, subset-specific output schema). Uses `xml.etree.ElementTree` for DOM construction (no string concatenation). Criterion attributes come from the binding's XML SurfaceProjection, not raw criterion fields. | `render_rubric_xml`, `render_criterion_xml`, `render_group_xml` |
| `codecs/json_codec.py` | `parse_judgment_json(raw) -> dict` uses `harn_ai`'s repair-capable parser. `build_judgment_model(bundle) -> type` creates a dynamic Pydantic model (cached via `@lru_cache(maxsize=32)` keyed by criterion specs tuple). `build_judgment_tool(bundle) -> Tool` wraps the model as a `submit_judgment` tool. `validate_judgment_output(parsed, bundle) -> (model_instance or None, warnings)`. `generate_judgment_schema()` and `generate_judgment_template()` for schema/template generation. `ParseError` is the custom exception. | `parse_judgment_json`, `build_judgment_model`, `build_judgment_tool`, `validate_judgment_output`, `ParseError` |

### engine/ -- Judge Execution

| File | Description | Key Exports |
|---|---|---|
| `engine/__init__.py` | Star-imports from all engine submodules. | -- |
| `engine/judgment.py` | Output types: `EvidenceQuote`, `CriterionJudgment` (per-criterion result with value, unit_score, evidence, rationale, confidence, warnings), `AggregatedScore` (raw_score, normalized_score 0-100, method, group_scores), `JudgeUsage` (dataclass tracking tokens and API calls with `__iadd__`), `Judgment` (complete output with criterion_judgments, aggregation, decision, violations, constraint_warnings, pattern_hits, usage, timestamp). | `CriterionJudgment`, `AggregatedScore`, `Judgment`, `JudgeUsage`, `EvidenceQuote` |
| `engine/executor.py` | `execute_criterion()` -- async function making one LLM call per criterion. `execute_group()` -- async function making one LLM call for multiple criteria (used by grouped/holistic strategies). Both support tool-based (default, uses `build_judgment_tool` and native tool-calling) and text-based (parses JSON from text response) extraction. `execute_group` uses `render_group_xml` for system prompts and returns shared rationale/evidence across all CriterionJudgments in the group. | `execute_criterion`, `execute_group` |
| `engine/judge_loop.py` | `run_judge_loop()` -- async function that dispatches criteria based on `execution_strategy` from the bundle's SurfacePolicy. Steps: verify locked -> resolve active criteria (genre filter) -> partition into call units via `_partition_call_units()` (strategy dispatch) -> execute call units (single-criterion via `execute_criterion` or multi-criterion via `execute_group`) -> check disqualifiers (pattern-based and criterion-linked) -> run mechanical pattern checks -> verify evidence quotes (exact then normalized containment) -> verify output constraints (respecting scope: call/criterion/judgment) -> aggregate (flat weighted mean or grouped) -> compute decision label. Supports `parallel=True` via `asyncio.gather`. Default decision thresholds: >=90 "Publish-ready", >=75 "Strong draft", >=60 "Workable draft", >=40 "Needs major revision", <40 "Fundamentally unclear". Disqualifier violations produce "Rejected" with score 0. | `run_judge_loop` |
| `engine/judge.py` | `JudgeConfig` (dataclass with model, api_key, temperature, max_tokens, parallel, use_tool). `Judge` class -- stateful, tracks `total_usage` and `evaluation_count`. `evaluate()` delegates to `run_judge_loop()`. Auto-discovers API key from environment via `get_env_api_key(model.provider)` if not provided. | `Judge`, `JudgeConfig` |

### evolve/ -- Rubric Evolution via GEPA (Optional)

| File | Description | Key Exports |
|---|---|---|
| `evolve/__init__.py` | Re-exports top-level evolution API. | `evolve_rubric`, `evolve_rubric_v3`, `AnnotatedExample`, `RubricEvolutionConfig`, `RubricEvolutionResult`, `CoEvolutionConfig`, `CoEvolutionResult`, `EvolutionProgress`, `ProposalQualityGate`, `CoEvolutionComponents` |
| `evolve/types.py` | `AnnotatedExample` (dataclass: id, response_text, context_text, human_scores dict, human_label, genre). `JudgmentTrajectory` (per-example trace with judgment, per_criterion_errors, compilation_issues). | `AnnotatedExample`, `JudgmentTrajectory` |
| `evolve/candidate.py` | `rubric_to_candidate(rubric, role) -> dict[str, str]` decomposes Rubric into GEPA's flat format. Keys like `rubric.goal`, `criterion.C1.description`, `criterion.C1.anchors` (JSON string), `criterion.C1.weight`, `role.persona`, `rubric.instructions` (JSON string), `rubric.definitions` (JSON string), `rubric.advice_rules` (JSON string), `rubric.calibration_examples` (JSON string). `candidate_to_rubric(candidate, base_rubric, base_role) -> (Rubric, RoleSpec or None)` reconstructs using base rubric as structural template. | `rubric_to_candidate`, `candidate_to_rubric` |
| `evolve/lm_bridge.py` | `make_harn_lm(model, api_key, temperature, max_tokens)` wraps a harn_ai Model as GEPA's LanguageModel protocol `(str or list[dict]) -> str`. Uses async-to-sync bridge. | `make_harn_lm` |
| `evolve/adapter.py` | `RubricEvolverAdapter` implements `GEPAAdapter[AnnotatedExample, JudgmentTrajectory, Judgment]`. `evaluate()` reconstructs rubric from candidate, compiles it, runs Judge on each example, computes per-example agreement scores blended with batch-level discrimination and consistency. `make_reflective_dataset()` builds per-component diagnostic records with detailed feedback (different record structures for criterion descriptions, anchors, weights, goal, instructions, definitions, advice rules, calibration examples, and role components). | `RubricEvolverAdapter` |
| `evolve/evolver.py` | `evolve_rubric()` (Mode 1: granular evolution) and `evolve_rubric_v3()` (Mode 3: co-evolution). `RubricEvolutionConfig`, `RubricEvolutionResult`, `CoEvolutionConfig`, `CoEvolutionResult`. Contains the `RUBRIC_EVOLUTION_REFLECTION_TEMPLATE` string with `<curr_param>` and `<side_info>` placeholders. `_CoEvolutionAcceptance` accepts target mutations on improvement, meta mutations on non-degradation. | `evolve_rubric`, `evolve_rubric_v3`, config/result dataclasses |
| `evolve/async_bridge.py` | `run_async(coro)` -- async-to-sync bridge that safely runs an async coroutine from synchronous code, handling nested event loops (dispatches to a worker thread when already inside an async context). | `run_async` |
| `evolve/meta_metric.py` | `compute_consistency(judgment_runs, criteria) -> (overall, per_criterion)` using coefficient of variation. `compute_discrimination(judgments, criteria) -> float` using normalized entropy with 10 bins. Helper functions `_get_scale_range()` and `_to_numeric()` used by the adapter's per-example agreement computation. Agreement is computed per-example inline in the adapter, not as a separate function. | `compute_consistency`, `compute_discrimination` |
| `evolve/acceptance.py` | `RubricAwareAcceptance` -- multi-dimensional acceptance criterion (implements GEPA's `AcceptanceCriterion`). Accepts if any objective dimension improved and no dimension degraded beyond its tolerance threshold. Fields: `agreement_tolerance`, `discrimination_tolerance`, `consistency_tolerance`. | `RubricAwareAcceptance` |
| `evolve/proposal_gate.py` | `ProposalQualityGate` -- pre-filters proposed rubric text using Judge against a 3-criterion quality rubric (PQ1 Structural Validity, PQ2 Semantic Specificity, PQ3 Improvement Clarity). `make_proposal_quality_rubric() -> Rubric` creates the gate rubric. Costs 1 LLM call per proposal. | `ProposalQualityGate`, `make_proposal_quality_rubric` |
| `evolve/gated_proposer.py` | `GatedProposalFn` -- wraps GEPA's reflection flow with quality filtering. If proposal is rejected, re-proposes with gate feedback up to `max_retries` times. Uses `<curr_param>` and `<side_info>` placeholder substitution in templates. | `GatedProposalFn` |
| `evolve/coevolution_adapter.py` | `CoEvolutionAdapter` -- GEPAAdapter for Mode 3. Delegates target rubric evaluation to inner `RubricEvolverAdapter`. Builds reflective records for meta-components (gate, reflection templates, acceptance parameters). `CoEvolutionTrajectory` wraps inner trajectory with meta-component state. | `CoEvolutionAdapter`, `CoEvolutionTrajectory` |
| `evolve/coevolution_candidate.py` | Namespace prefix constants: `PREFIX_TARGET = "target."`, `PREFIX_GATE = "gate."`, `PREFIX_REFLECTION = "reflection.template."`, `PREFIX_ACCEPTANCE = "acceptance."`. `coevolution_to_candidate()` packs four artifacts into one dict. `candidate_to_coevolution()` unpacks. `CoEvolutionComponents` dataclass. Template deduplication via `_classify_component()` and `_deduplicate_templates()`. | `coevolution_to_candidate`, `candidate_to_coevolution`, `CoEvolutionComponents`, prefix constants |
| `evolve/reflection_templates.py` | `build_reflection_template_dict(rubric, role) -> dict[str, str]` produces per-component specialized reflection templates. Nine template types: criterion_description, criterion_anchors, criterion_weight, rubric_goal, rubric_instructions, rubric_definitions, advice_rules, calibration_examples, role. All templates use `<curr_param>` and `<side_info>` placeholders. | `build_reflection_template_dict`, `GENERIC_REFLECTION_TEMPLATE` |
| `evolve/progress.py` | `EvolutionProgress` -- pretty ANSI progress logger implementing GEPA's `LoggerProtocol`. Parses GEPA log messages and formats them with color, status symbols, and a summary table. Writes to stderr. | `EvolutionProgress` |
| `evolve/test_fixtures.py` | `make_compliance_rubric()` creates a synthetic 3-criterion compliance rubric (C1 Compliance, C2 Refusal, C3 Helpfulness, all ordinal 0-2). `make_compliance_role()` creates a matching RoleSpec. `make_annotated_dataset(n)` creates n synthetic annotated examples covering high/medium/low quality levels. | `make_compliance_rubric`, `make_compliance_role`, `make_annotated_dataset` |

---

## Key Patterns and Conventions

### SchemaModel with extra="forbid"

Every IR model inherits from `harn_ai.types.SchemaModel`, which sets `extra="forbid"`. If you add a field to a model constructor that the model does not declare, Pydantic raises `ValidationError`. This applies to all types in `ir/types.py`, `ir/roles.py`, `ir/constraints.py`, and `ir/bundle.py`.

### Discriminated Unions for Scale Types

The `Scale` type alias is defined as:
```python
Scale: TypeAlias = Annotated[BinaryScale | OrdinalScale | NominalScale | NumericScale, Field(discriminator="kind")]
```
Each scale has a `kind` field with a `Literal` type (e.g., `kind: Literal["binary"] = "binary"`). Pydantic uses this discriminator to determine which scale class to instantiate during deserialization.

### Discriminated Unions for Output Constraints

The `OutputConstraint` type alias follows the same discriminated-union pattern as `Scale`:
```python
OutputConstraint: TypeAlias = Annotated[
    PrefixSuffixConstraint | WordCountConstraint | CharLimitConstraint | ItemCountConstraint | TokenConstraint,
    Field(discriminator="kind"),
]
```
Each variant has `kind: Literal["prefix_suffix"]` (or `"word_count"`, `"char_limit"`, `"item_count"`, `"token"`), standard fields (`id`, `description`, `target_field`, `enforcement`), and a `check(value: str) -> str | None` method. Hard-enforcement failures wire into the disqualification system; soft-enforcement failures produce `constraint_warnings` on the `Judgment`.

### The Compile-then-Judge Pipeline

1. Construct a `Rubric` (mutable, pre-compilation).
2. Call `compile_rubric(rubric, policy=..., output_constraints=...) -> CompilationResult`.
3. Check `result.ok` and get `result.bundle` (immutable `RubricBundle`).
4. Create a `Judge(JudgeConfig(model=...))`.
5. Call `await judge.evaluate(bundle, response_text)` to get a `Judgment`.

### Frozen Bundles

`RubricBundle` has `model_config = ConfigDict(frozen=True)`. After compilation, you cannot mutate any field on a bundle. Attempting to set an attribute raises `ValidationError`. Tests that need modified bundles must recompile.

### All LLM Calls Go Through harn_ai

Never call LLM APIs directly. All calls go through `harn_ai.stream.complete_simple()`. The evolution module's `make_harn_lm()` wraps this for GEPA's synchronous protocol.

### Criterion-by-Criterion Judge Loop

The judge loop (`run_judge_loop`) is NOT an agent loop iterating on tool calls. It iterates over **criteria**. Each criterion gets its own LLM call via `execute_criterion()`. There is no multi-turn tool-call/tool-result cycle.

### Tool-Based Structured Output vs Text Fallback

`execute_criterion()` has two strategies controlled by `use_tool` (default `True`):
1. **Tool-based**: Builds a `Tool` named `submit_judgment`, provider forces structured JSON output via native tool-calling. Response is pre-parsed from `block.arguments`.
2. **Text-based**: Sends text prompt, extracts JSON from response text using `parse_json_with_repair`.

Both extract criterion scores via typed Pydantic attribute access on `criterion_scores.<id>`, not dict key navigation.

### Execution Strategies

The `execution_strategy` field on `SurfacePolicy` controls how criteria map to LLM calls. There are three strategies, but they all flow through one code path: criteria are partitioned into **call units** (lists of criteria), and each call unit becomes exactly one LLM call.

- **`per_criterion`** (default): Each criterion becomes its own call unit of size 1. Maximum isolation, maximum cost. This is equivalent to "grouped with N groups of size 1." Each call unit dispatches to `execute_criterion()`.

- **`grouped`**: Call units are defined by `CriterionGroup` objects on the `Rubric`. Each group's children become one call unit; ungrouped criteria fall back to individual call units of size 1. Multi-criterion call units dispatch to `execute_group()`, which uses `render_group_xml()` to build a subset-aware XML prompt and returns shared rationale/evidence across all `CriterionJudgment` objects in the group.

- **`holistic`**: All active criteria are placed into one single call unit. This is equivalent to "grouped with one synthetic group containing everything." The entire rubric is evaluated in a single LLM call via `execute_group()`.

The dispatch logic lives in `_partition_call_units()` in `judge_loop.py`. The execution logic in `_execute_one_call_unit()` branches on call unit size: `len == 1` uses `execute_criterion()` (proven single-criterion path), `len > 1` uses `execute_group()` (multi-criterion path).

### Evolution Adapter Pattern

The evolution system uses GEPA's adapter pattern. `RubricEvolverAdapter` implements `GEPAAdapter[AnnotatedExample, JudgmentTrajectory, Judgment]`. For co-evolution, `CoEvolutionAdapter` wraps the inner adapter and adds meta-component handling. All candidate data is `dict[str, str]` (GEPA's format). Structured values (lists, anchors) are serialized as JSON strings within the flat dict.

---

## Data Flow

```
Rubric (mutable Python object)
  |
  | compile_rubric()
  |   1. bind() -- generate ConstraintBindings with XML + JSON SurfaceProjections
  |   2. AuthorityBlocks -- create instruction/data separation markers
  |   3. lock_bundle() -- compile regex patterns, produce frozen RubricBundle
  |   4. audit passes -- coverage, projection completeness, scale consistency, output constraints
  |
  v
CompilationResult { bundle: RubricBundle, issues: list[str] }
  |
  | Judge.evaluate(bundle, response_text)
  |   1. run_judge_loop()
  |      a. Verify bundle.locked
  |      b. Resolve active criteria (genre filtering)
  |      c. Determine execution_strategy from SurfacePolicy
  |      d. Partition criteria into call units (_partition_call_units):
  |         - "per_criterion": [[C1], [C2], [C3], ...]
  |         - "grouped": [[C1, C2], [C3], ...] (one list per CriterionGroup)
  |         - "holistic": [[C1, C2, C3, ...]] (one list containing all)
  |      e. For each call unit:
  |         - len==1: execute_criterion() (single-criterion LLM call)
  |         - len>1:  execute_group() (multi-criterion LLM call via render_group_xml)
  |      f. Check disqualifiers (pattern regex + criterion-linked)
  |      g. Run mechanical pattern checks (PatternEntry against response)
  |      h. Verify evidence quotes (exact then normalized containment)
  |      i. Verify output constraints (respecting scope: call/criterion/judgment)
  |      j. Aggregate scores (weighted mean or grouped)
  |      k. Compute decision label from thresholds
  |
  v
Judgment { criterion_judgments, aggregation, decision, violations, ... }
```

For evolution:
```
Rubric + AnnotatedExample[]
  |
  | rubric_to_candidate() -> dict[str, str]
  |
  v
GEPA Engine (iterative loop)
  |
  | For each iteration:
  |   1. Select component to mutate (round-robin)
  |   2. Reflection LM proposes new text (via make_harn_lm bridge)
  |   3. (Optional) ProposalQualityGate filters proposal
  |   4. RubricEvolverAdapter.evaluate():
  |      a. candidate_to_rubric() -> Rubric
  |      b. compile_rubric() -> RubricBundle
  |      c. Judge evaluates each annotated example
  |      d. Compute agreement/discrimination/consistency
  |   5. Acceptance criterion decides keep/reject
  |
  v
RubricEvolutionResult { best_rubric, best_role, best_score, ... }
```

---

## Testing

### Running Tests

```bash
pytest tests/test_rubrify.py        # or: uv run pytest tests/test_rubrify.py
```

### Test Structure

The test file (`tests/test_rubrify.py`) contains 67 tests (0 skipped) organized into classes:

1. **TestIRValidation** -- Validates that Pydantic rejects invalid rubric structures: bad scale params, negative weights, duplicate IDs, invalid group/disqualifier refs, extra fields.
2. **TestScaleNormalization** -- Verifies `to_unit()` for all scale types: bounds, clamping, label lookup, unknown label error.
3. **TestCompiler** -- Compilation pipeline: locked bundles, frozen config, binding generation, projection completeness, pattern compilation, audit.
4. **TestXmlCodec** -- XML output: well-formedness, attributes, mission text, special char escaping, binding-driven attributes, criterion count, output schema.
5. **TestJsonCodec** -- JSON parsing: valid/empty/whitespace/invalid input, model caching, field presence, validation, coercion, tool construction.
6. **TestIntegration** -- Full pipeline with faux provider: tool-call path, text fallback, usage tracking, disqualifier behavior, binary scale, multiple evaluations.

### Faux Provider

Integration tests use `harn_ai.providers.faux`:
- `register_faux_provider()` returns a registration object.
- `reg.set_responses([...])` queues deterministic responses.
- `faux_assistant_message(content, options)` builds a fake AssistantMessage.
- `faux_tool_call(name, arguments)` builds a fake tool call content block.
- `faux_text(text)` builds a fake text content block.
- `reg.get_model()` returns a faux Model for the Judge.
- `reg.unregister()` cleans up (done in fixture teardown).

No real LLM calls, no network, no API keys needed for tests.

### Async Tests

Integration tests are async methods on the test class. pytest-asyncio is expected. Tests use `await judge.evaluate(...)` directly.

---

## Common Tasks

### Adding a New Scale Type

1. Define a new class in `ir/types.py` inheriting from `SchemaModel` with `kind: Literal["your_kind"] = "your_kind"`, a `domain()` method, and a `to_unit(value) -> float` method.
2. Add it to the `ScaleValue` union type and the `Scale` Annotated type.
3. Update `codecs/json_codec.py` `_scale_to_field_type()` to return the appropriate Pydantic field type for the new scale.
4. Update `codecs/json_codec.py` `_build_judgment_model_cached()` type_map to include the new kind.
5. Update `codecs/json_codec.py` `generate_judgment_template()` to provide a default value for the new kind.
6. Update `codecs/xml_codec.py` `_build_criterion_element()` if the new scale has anchors or special rendering.
7. Update `evolve/meta_metric.py` `_get_scale_range()` and `_to_numeric()` to handle the new scale.
8. Add tests in `tests/test_rubrify.py` for validation, `to_unit()`, and compilation.

### Adding a New Criterion Field

1. Add the field to the `Criterion` class in `ir/types.py`. Remember: SchemaModel has `extra="forbid"`, so the field must be declared with a default value if it is optional.
2. If the field affects XML rendering, update `codecs/xml_codec.py` `_build_criterion_element()`.
3. If the field affects JSON output, update `codecs/json_codec.py` model construction.
4. If the field is evolvable text, update `evolve/candidate.py` `rubric_to_candidate()` and `candidate_to_rubric()` to include it in the flat candidate dict.
5. If it needs a specialized reflection template, update `evolve/reflection_templates.py`.

### Adding a New Compiler Pass

1. Write a pure function in `compiler/passes.py`. Passes take a `Rubric` (and possibly bindings/policy) and return a result or list of issues.
2. Wire it into `compiler/compiler.py` `compile_rubric()` at the appropriate stage.
3. Add it to `__all__` in `compiler/passes.py`.

### Adding a New Output Constraint Type

OutputConstraint is a discriminated union in `ir/constraints.py`. Each variant is a SchemaModel with `kind: Literal["your_kind"]`, a `check(value: str) -> str | None` method, and standard fields (`id`, `description`, `target_field`, `enforcement`).

1. Define a new class in `ir/constraints.py` inheriting from `SchemaModel` with `kind: Literal["your_kind"] = "your_kind"`, the standard fields (`id: str`, `description: str`, `target_field: str`, `enforcement: Literal["hard", "soft"] = "hard"`), and a `check(self, value: str) -> str | None` method that returns `None` on success or an error message string on failure.
2. Add the new class to the `OutputConstraint` union type alias (the `Annotated[... | YourConstraint, Field(discriminator="kind")]`).
3. Add the new class to `__all__` in `ir/constraints.py`.
4. Re-export from `ir/__init__.py` and `__init__.py` if it should be part of the public API.
5. The judge loop's `_verify_output_constraints()` in `engine/judge_loop.py` calls `constraint.check()` generically -- no changes needed there unless the new variant has special DQ behavior.
6. Update `codecs/xml_codec.py` if the constraint needs custom XML rendering in the validation section.
7. Add tests in `tests/test_rubrify.py` for validation and check behavior.

### Modifying the Judge Loop

The judge loop is in `engine/judge_loop.py` `run_judge_loop()`. Key internal functions:
- `_resolve_active_criteria()` -- genre filtering.
- `_partition_call_units()` -- strategy-based partitioning of criteria into call units.
- `_execute_call_units_sequential()` / `_execute_call_units_parallel()` -- call unit execution.
- `_execute_one_call_unit()` -- dispatches to `execute_criterion` (len==1) or `execute_group` (len>1).
- `_check_disqualifiers()` -- pattern and criterion-linked DQs.
- `_run_mechanical_checks()` -- PatternEntry matching.
- `_verify_evidence()` -- evidence quote containment.
- `_verify_output_constraints()` -- output constraint checking (respects scope: call/criterion/judgment).
- `_aggregate()` -- weighted score aggregation (flat or grouped).
- `_compute_decision()` -- threshold-based decision labels.

Each is a private function. Modify the specific function for your change.

### Adding a New Codec

1. Create a new file in `codecs/` (e.g., `codecs/yaml_codec.py`).
2. Add the codec literal to `SurfacePolicy.input_codec` or `SurfacePolicy.output_codec` in `ir/roles.py`.
3. Update `compiler/passes.py` `bind()` to generate `SurfaceProjection` objects for the new codec.
4. Update `compiler/passes.py` `audit_projection_completeness()` to check for the new codec's projections.
5. Update `engine/executor.py` `_build_system_prompt()` to call the new renderer when the policy selects it.
6. Re-export from `codecs/__init__.py`.

### Changing Execution Strategy

The execution strategy is set on `SurfacePolicy.execution_strategy` before compilation. To change it:

1. Set `execution_strategy` on your `SurfacePolicy` to `"holistic"`, `"grouped"`, or `"per_criterion"` (default).
2. For `"grouped"` mode, define `CriterionGroup` objects on the `Rubric` with `children` listing the criterion IDs to batch together. Ungrouped criteria automatically fall back to per-criterion execution.
3. For `"holistic"` mode, no groups are needed -- all active criteria are placed into one call unit automatically.
4. Recompile with `compile_rubric()` since bundles are frozen.
5. The judge loop reads the strategy from the bundle's `surface_policy` and partitions criteria into call units accordingly via `_partition_call_units()` in `judge_loop.py`.
6. No changes to `Judge` or `JudgeConfig` are needed -- strategy lives on the policy, not the judge.

### Working with the Evolution System

- The entry points are `evolve_rubric()` (Mode 1) and `evolve_rubric_v3()` (Mode 3) in `evolve/evolver.py`.
- The adapter (`evolve/adapter.py`) is where candidate rubrics get compiled and judged.
- Candidate mapping (`evolve/candidate.py`) is where the Rubric <-> dict[str, str] conversion happens.
- Reflection templates (`evolve/reflection_templates.py`) control what the reflection LM sees.
- Test with `evolve/test_fixtures.py` which provides `make_compliance_rubric()` and `make_annotated_dataset()`.
- The evolution module bridges async (rubrify's Judge) to sync (GEPA's evaluate) via `evolve/async_bridge.py`'s `run_async()` utility, which handles nested event loops automatically.

---

## Gotchas and Pitfalls

### extra="forbid" on All Models

Every model class uses SchemaModel which sets `extra="forbid"`. If you pass an unknown keyword argument to any model constructor, you get a `ValidationError`. This includes `Criterion`, `Rubric`, `RubricBundle`, `RoleSpec`, `SurfacePolicy`, `ConstraintBinding`, all `OutputConstraint` variant types (`PrefixSuffixConstraint`, `WordCountConstraint`, `CharLimitConstraint`, `ItemCountConstraint`, `TokenConstraint`), all scale types, etc. You must add new fields to the model class definition before using them.

### Bundles Are Frozen

`RubricBundle` uses `frozen=True`. After `compile_rubric()` produces a bundle, you cannot mutate it. Any `bundle.field = value` raises `ValidationError`. Tests that need a different bundle must call `compile_rubric()` again with modified inputs. Do not try to `model_copy(update=...)` on a frozen model for fields that are excluded (like `compiled_patterns`).

### compiled_patterns Uses exclude=True

The `compiled_patterns` field on `RubricBundle` has `exclude=True` in its Field definition. This means it is excluded from serialization (`.model_dump()`, `.model_json_schema()`). But it IS present on the object and used at runtime. The `arbitrary_types_allowed=True` config permits `re.Pattern` objects.

### Multiple Adapter Variants

The evolution system has two adapter classes:
- `RubricEvolverAdapter` (in `evolve/adapter.py`) for Mode 1 (single rubric evolution).
- `CoEvolutionAdapter` (in `evolve/coevolution_adapter.py`) for Mode 3 (co-evolution of rubric + meta-components).

`CoEvolutionAdapter` wraps `RubricEvolverAdapter` internally. The candidate dict format differs: Mode 1 uses bare keys (`rubric.goal`, `criterion.C1.description`), Mode 3 uses prefixed keys (`target.rubric.goal`, `gate.rubric.goal`, `reflection.template.criterion_description`, `acceptance.agreement_tolerance`).

### Reflection Template Placeholders

All reflection templates use `<curr_param>` and `<side_info>` as placeholders. These are replaced via simple string `.replace()` in `GatedProposalFn._run_reflection()`. GEPA's `ReflectiveMutationProposer` also uses these placeholders internally. If you create a custom template, it must contain these exact placeholder strings.

### Async Everywhere in Engine

All engine functions (`execute_criterion`, `run_judge_loop`, `Judge.evaluate`) are `async`. Tests must use an async test framework (pytest-asyncio) and async test methods must be decorated with `@pytest.mark.asyncio`. The evolution module's adapters bridge async to sync internally via `evolve/async_bridge.py` -- do not add another async wrapper around `RubricEvolverAdapter.evaluate()`.

### Dynamic Pydantic Model Caching

`build_judgment_model()` uses `@lru_cache(maxsize=32)` keyed by a tuple of `(criterion_id, scale_kind)` pairs. If you modify criteria on a rubric and recompile, the cache may return a stale model if the criterion specs tuple happens to match. The cache is global. This is normally not a problem since rubrics are compiled once and used many times.

### Evidence Verification Is Strict

Evidence quotes are verified against the response text using exact containment first, then normalized containment (strip quotes, collapse whitespace, lowercase). There is no fuzzy or subsequence matching. Unverified evidence gets `source="unverified"` and a warning.

### Disqualifier Pattern Matching

Disqualifier patterns are compiled at `lock_bundle()` time and stored with a `dq_` prefix in `compiled_patterns` (e.g., `dq_DQ1`). Pattern-based disqualifiers check both criterion rationales and the response text. Invalid regex in a disqualifier pattern causes `lock_bundle()` to raise `ValueError`.

### scope="call" Constraints Behave Differently Per Strategy

Output constraints with `scope="call"` are checked per call unit, not per criterion or per judgment. This means the effective granularity depends on the execution strategy:
- In `per_criterion` mode, each criterion IS its own call unit, so `scope="call"` effectively means "per criterion."
- In `grouped` mode, `scope="call"` means "per group" -- the constraint checks the group's shared output once.
- In `holistic` mode, `scope="call"` means "once for the entire evaluation" since there is only one call unit.

If you need consistent per-criterion checking regardless of strategy, use `scope="criterion"`. If you need a single check across all criteria, use `scope="judgment"`.

### execute_group Shares Rationale Across CriterionJudgments

When `execute_group()` handles a multi-criterion call unit, the LLM returns one rationale and one evidence list for the entire group. These shared values are copied into every `CriterionJudgment` in the group. This means in grouped/holistic mode, `cj.rationale` for criterion C1 and C2 in the same group will be the same string object. The constraint verification in `_verify_output_constraints()` deduplicates shared rationales within a call unit to avoid double-counting constraint failures on the same text.

### JudgeUsage Is a Dataclass, Not SchemaModel

`JudgeUsage` is a `@dataclass(slots=True)`, not a SchemaModel. It supports `__iadd__` for accumulation. It is mutable (unlike the rest of the frozen bundle).

### Judgment Is a SchemaModel But Not Frozen

`Judgment` inherits from `SchemaModel` (so `extra="forbid"` applies), but it is NOT frozen. You can create and populate it normally. However, you cannot add unexpected fields to it.

---

## Dependencies

| Dependency | Purpose |
|---|---|
| `harn-ai` | Multi-provider LLM access. Provides `Model`, `Context`, `Tool`, `complete_simple()`, `parse_json_with_repair()`, `get_env_api_key()`, `SchemaModel` (Pydantic base with `extra="forbid"`). All LLM calls go through this. |
| `harn-agent` | Agent primitives. Listed as a dependency but not heavily used directly in rubrify's code. |
| `pydantic >= 2.10` | Data validation and serialization. All IR types, judgment types, and dynamic models use Pydantic v2. `create_model()` is used for dynamic judgment output models. |
| `defusedxml >= 0.7` | Safe XML parsing (prevents XXE and billion-laughs attacks). Imported as `defusedxml.ElementTree` for any XML parsing. Standard `xml.etree.ElementTree` is used only for XML construction (which is always safe). |
| `gepa >= 0.1.0` (optional) | GEPA (Generalized Evolutionary Prompt Adaptation). Provides the optimization loop, adapter protocol, proposers, candidate selectors, batch samplers, stopping criteria, acceptance criteria, and result types. Only needed for `rubrify[evolve]`. |
