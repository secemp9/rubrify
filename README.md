# rubrify

Rubric compiler and judge engine for LLM evaluation.

rubrify lets you define structured evaluation rubrics as typed Python objects, compile them into immutable bundles, and run criterion-by-criterion LLM-based judgments against text responses. It also supports evolving rubrics against human-annotated datasets using GEPA's reflective prompt optimization.

Built on `harn_ai` for multi-provider LLM access (OpenAI, Anthropic, DeepSeek, Google, local proxies) and `harn_agent` for agent primitives. API keys are auto-discovered from environment variables.

---

## Installation

Requires Python >= 3.12.

```bash
pip install rubrify
```

Core dependencies: `harn-ai`, `harn-agent`, `pydantic>=2.10`, `defusedxml>=0.7`.

For the rubric evolution system (GEPA integration):

```bash
pip install rubrify[evolve]
```

This adds `gepa>=0.1.0` as a dependency.

---

## API Keys

rubrify discovers API keys from environment variables via [harn](https://github.com/secemp9/harn).
Each provider has a standard env var:

| Provider | Environment Variable |
|----------|---------------------|
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| xAI | `XAI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| HuggingFace | `HF_TOKEN` |

**Option 1: `.env` file** (recommended)

Copy the included `.env.example` to `.env` and fill in your keys:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

Then at the top of your script:

```python
from dotenv import load_dotenv
load_dotenv()
```

Install with `pip install python-dotenv` or `pip install rubrify[dev]`.

**Option 2: Shell export**

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
```

**Option 3: Direct parameter**

```python
judge = Judge(JudgeConfig(model=model, api_key="sk-your-key-here"))
```

---

## Quick Start

### Define and compile a rubric

```python
from rubrify import (
    Criterion, NumericScale, Rubric, RubricMeta, ScaleAnchor,
    compile_rubric,
)

rubric = Rubric(
    meta=RubricMeta(name="MyRubric", version="1.0"),
    goal="Evaluate response quality.",
    criteria=[
        Criterion(
            id="C1",
            title="Clarity",
            description="How clear is the response?",
            scale=NumericScale(
                minimum=0, maximum=5, step=1,
                anchors=[
                    ScaleAnchor(value=0, label="unclear", description="Meaning obscured."),
                    ScaleAnchor(value=5, label="crystal", description="Perfectly clear."),
                ],
            ),
            weight=1.0,
        ),
    ],
)

result = compile_rubric(rubric)
assert result.ok          # True if no audit issues
bundle = result.bundle    # Immutable RubricBundle, ready for judging
```

### Run a judgment

```python
import asyncio
from harn_ai.models import get_model
from rubrify import Judge, JudgeConfig

judge = Judge(JudgeConfig(model=get_model("openai", "gpt-4o")))
judgment = asyncio.run(judge.evaluate(bundle, "The response text to evaluate."))

print(judgment.aggregation.normalized_score)  # 0-100
print(judgment.decision)                       # e.g. "Strong draft"
for cj in judgment.criterion_judgments:
    print(f"  {cj.criterion_id}: {cj.value} (unit={cj.unit_score:.2f})")
```

### Use a custom OpenAI-compatible proxy

```python
from harn_ai.models import get_model

model = get_model("openai", "gpt-4o").model_copy(update={
    "baseUrl": "http://localhost:8000/v1",
    "api": "openai-completions",
})
judge = Judge(JudgeConfig(model=model, api_key="your-key"))
```

---

## Core Concepts

### IR Type System

All rubric structures are Pydantic models with `extra="forbid"` (no unexpected fields allowed).

**Scale types** are polymorphic, discriminated by the `kind` field. Each scale knows its own domain and implements `to_unit(value) -> float` to normalize raw scores into `[0, 1]`:

| Scale | `kind` | Domain | `to_unit` behavior |
|---|---|---|---|
| `BinaryScale` | `"binary"` | pass/fail (configurable labels and scores) | `True -> true_score`, `False -> false_score` |
| `OrdinalScale` | `"ordinal"` | Ordered levels with named anchors | Linearly maps anchor values to `[0, 1]` |
| `NominalScale` | `"nominal"` | Unordered categories with anchors | Maps category value to `[0, 1]` by range |
| `NumericScale` | `"numeric"` | Continuous range `[minimum, maximum]` with step | `(value - min) / (max - min)`, clamped |

The union type `Scale` is: `Annotated[BinaryScale | OrdinalScale | NominalScale | NumericScale, Field(discriminator="kind")]`.

**Criterion** is the atomic evaluation unit. Key fields:

- `id` -- unique identifier
- `title`, `description` -- human-readable
- `scale` -- one of the four scale types above
- `weight` -- contribution to the aggregate score (default 1.0)
- `evidence` -- `EvidenceSpec` controlling what evidence the judge must cite (note: `required`, `exact_quote`, `min_items`, and `max_items` on EvidenceSpec are prompt-only -- they are rendered into the XML surface so the LLM can see them, but are not enforced post-hoc by the engine)
- `genre` -- for genre-conditional activation
- `mechanical_rules` -- free-text rules rendered in XML

**CriterionGroup** provides hierarchical aggregation over criteria. Supported aggregation strategies: `weighted_sum`, `weighted_mean`, `min`, `max`, `all`, `any`.

**Disqualifier** defines an auto-fail condition. Can be pattern-based (regex against the response text) or criterion-linked (triggers when a specific criterion scores 0).

**Rubric** is the mutable, pre-compilation object. It contains criteria, groups, disqualifiers, instructions, patterns (`PatternEntry` for regex matching), definitions (`Definition`), advice rules (`AdviceRule`), and calibration examples (`CalibrationExample`). Model validators enforce unique criterion IDs and valid group/disqualifier references at construction time.

**RubricBundle** is the immutable, locked, executable form produced by the compiler. It contains the frozen rubric, compiled regex patterns, constraint bindings, authority blocks, surface policy, and output constraints. The bundle is frozen via Pydantic's `frozen=True` config.

### Roles and Constraints

**RoleSpec** defines the judge's persona, authority level (`absolute`, `advisory`, `peer`), domain, obligations (what the model MUST do), and constraints (what the model MUST NOT do). It is a structural component, not a cosmetic prompt prefix.

**SurfacePolicy** governs how rubrics are rendered. Fields include:

- `input_codec` -- currently only `"xml"`
- `output_codec` -- currently only `"json"`
- `role` -- optional `RoleSpec`
- `enforce_key_order` -- whether to enforce JSON key ordering
- `criterion_focus` -- `"full"` (send entire rubric per criterion) or `"focused"` (send only the relevant criterion)
- `decision_thresholds` -- list of `(min_score, label)` tuples for custom decision labels
- `execution_strategy` -- `"per_criterion"` (default), `"grouped"`, or `"holistic"` (see [Execution Strategies](#execution-strategies))

**ConstraintBinding** is the triple-layer alignment connecting a semantic criterion to its surface-layer projection (XML tag, JSON path) and output field. Each binding carries:

- `criterion_id` -- which criterion
- `output_field` -- JSON path where the judge writes its score (e.g. `criterion_scores.C1`)
- `projections` -- list of `SurfaceProjection` objects (one per codec)
- `authority` -- `instruction`, `data`, or `meta`

**AuthorityBlock** marks a prompt section as instruction vs. data, enforcing instruction/data separation.

**OutputConstraint** is a discriminated union of typed constraint variants, each with a concrete `check(value) -> str | None` method for enforcement. The union is `Annotated[PrefixSuffixConstraint | WordCountConstraint | CharLimitConstraint | ItemCountConstraint | TokenConstraint, Field(discriminator="kind")]`. Each variant carries `id`, `description`, `target_field`, `enforcement` (`"hard"` or `"soft"`), and `scope`. Hard constraints trigger disqualification; soft constraints produce warnings. Pydantic discriminates on the `kind` field (`"prefix_suffix"`, `"word_count"`, `"char_limit"`, `"item_count"`, `"token"`).

The `scope` field controls when a constraint is checked relative to execution strategy:

| Scope | Behavior | Default |
|---|---|---|
| `"call"` | Checked once per LLM call. For `per_criterion`, this is per criterion. For `grouped`/`holistic`, once per group/holistic call. Shared outputs (e.g. rationale) are deduplicated. | Yes |
| `"criterion"` | Checked per criterion individually, regardless of execution strategy. | No |
| `"judgment"` | Checked once on the final aggregated result (all criterion outputs concatenated). | No |

```python
from rubrify.ir.constraints import WordCountConstraint

constraint = WordCountConstraint(
    id="rationale_length",
    description="Rationale must be at least 10 words",
    target_field="rationale",
    enforcement="soft",
    scope="criterion",      # check every criterion's rationale, even in grouped mode
    count=10,
    mode="min",
)
```

---

## Compiler Pipeline

`compile_rubric(rubric, *, policy=None, output_constraints=None) -> CompilationResult`

This is a synchronous, pure function (no LLM calls). It runs these passes:

1. **Bind** -- generates a `ConstraintBinding` for each criterion, with XML and JSON `SurfaceProjection` objects. This is the triple-layer alignment: criterion ID maps to XML attributes and JSON output path.
2. **AuthorityBlocks** -- creates standard authority blocks for `rubric_spec`, `response_under_test`, `judge_instructions`, and `context_document`.
3. **Lock** -- produces an immutable `RubricBundle` via `lock_bundle()`. Compiles all `PatternEntry` and `Disqualifier` regex patterns (fails loudly on invalid regex).
4. **Audit** -- audit passes check:
   - `audit_coverage` -- every criterion has a binding
   - `audit_projection_completeness` -- every binding has projections matching the policy's codecs
   - `audit_scale_consistency` -- ordinal scales have anchors, numeric scales have `max > min`
   - `audit_output_constraints` -- recognized fields, duplicate IDs, hard-enforcement safety

`CompilationResult` has a `.ok` property (True if no issues) and `.issues` list.

---

## Judge Engine

### Judge and JudgeConfig

```python
from rubrify import Judge, JudgeConfig
from harn_ai.models import get_model

judge = Judge(JudgeConfig(
    model=get_model("deepseek", "deepseek-v4-flash"),
    api_key=None,           # auto-discovered from env
    temperature=0.0,
    max_tokens=2048,
    parallel=False,         # True to evaluate criteria concurrently
    use_tool=True,          # True for tool-based structured output
))
```

`Judge` is stateful: it tracks `total_usage` (token counts, API calls) and `evaluation_count` across all evaluations.

### evaluate()

```python
judgment = await judge.evaluate(
    bundle,
    response_text,
    context_text=None,     # optional reference context
    genre=None,            # optional genre for genre-conditional criteria
    on_criterion_start=None,  # callback(criterion_id)
    on_criterion_done=None,   # callback(criterion_id, CriterionJudgment)
)
```

### The Judge Loop

`run_judge_loop()` is the core algorithm. It does not iterate on tool calls like an agent loop; it iterates over **criteria** (or groups of criteria, depending on the execution strategy). Steps:

1. Verify bundle is locked.
2. Resolve active criteria (genre filtering: criteria with `genre=None` are always active; others activate only when `active_genre` matches).
3. Partition active criteria into **call units** based on `execution_strategy` from the bundle's `SurfacePolicy`:
   - `"per_criterion"` (default): one call unit per criterion.
   - `"grouped"`: one call unit per `CriterionGroup`; ungrouped criteria individually.
   - `"holistic"`: one call unit containing all active criteria.
4. Execute each call unit: single-criterion call units use `execute_criterion()`, multi-criterion call units use `execute_group()`.
5. Check disqualifiers (pattern-based and criterion-linked).
6. Run mechanical pattern checks (`PatternEntry` patterns against the response).
7. Verify evidence quotes exist in the response text (exact containment, then normalized containment).
8. Verify output constraints against LLM output (respecting scope: `call`, `criterion`, or `judgment`).
9. Aggregate scores (weighted mean, or grouped aggregation if groups exist).
10. Compute decision label from thresholds (defaults: >=90 "Publish-ready", >=75 "Strong draft", >=60 "Workable draft", >=40 "Needs major revision", <40 "Fundamentally unclear"). Disqualifier violations produce "Rejected".

Execution supports `parallel=True` for concurrent call-unit evaluation via `asyncio.gather`.

### Execution Strategies

The `execution_strategy` field on `SurfacePolicy` controls how criteria are dispatched to LLM calls:

| Strategy | Call granularity | Use when |
|---|---|---|
| `"per_criterion"` | One LLM call per criterion (default) | Need maximum isolation, deep per-criterion analysis |
| `"grouped"` | One LLM call per `CriterionGroup` | Rubric has logical groups, want intra-group coherence with composability |
| `"holistic"` | One LLM call for ALL active criteria | Few criteria, need holistic coherence, cost-sensitive |

Set via `SurfacePolicy`:

```python
from rubrify.ir.roles import SurfacePolicy

policy = SurfacePolicy(execution_strategy="grouped")
bundle = compile_rubric(rubric, policy=policy).bundle
```

**Implementation details:**

- The judge loop partitions active criteria into "call units" based on strategy. Each call unit is one LLM invocation.
- `"grouped"` uses `CriterionGroup.children` to determine call boundaries. Ungrouped criteria fall back to individual calls.
- `"holistic"` places all active criteria into a single call unit.
- Multi-criterion call units use `execute_group()`, which renders a group-specific XML prompt via `render_group_xml()` and extracts per-criterion scores from a single `criterion_scores` response dict.
- Single-criterion call units use the original `execute_criterion()` path.
- `parallel=True` parallelizes across call units (not within them).

### CriterionExecutor

`execute_criterion()` has two strategies:

1. **Tool-based** (default, `use_tool=True`): Builds a `harn_ai.types.Tool` named `submit_judgment` with a dynamic Pydantic model as the parameter type. The provider forces structured JSON output via native tool-calling. The response is pre-parsed.
2. **Text-based** (`use_tool=False`): Sends a text prompt, then parses JSON from the response text using `harn_ai`'s repair-capable JSON parser.

Both strategies extract criterion scores via typed Pydantic model attribute access, not dict navigation with string splitting.

### Judgment Output Types

- `CriterionJudgment` -- per-criterion result: `criterion_id`, `value` (raw score), `unit_score` (normalized 0-1), `evidence` (list of `EvidenceQuote`), `rationale`, `confidence`, `warnings`.
- `AggregatedScore` -- `raw_score`, `normalized_score` (0-100), `method`, `group_scores`.
- `Judgment` -- the complete output: `criterion_judgments`, `aggregation`, `decision`, `violations`, `constraint_warnings`, `pattern_hits`, `usage` (`JudgeUsage`), `timestamp`.
- `JudgeUsage` -- tracks `input_tokens`, `output_tokens`, `total_tokens`, `api_calls`.

---

## Codecs

### XML Codec

`render_rubric_xml(bundle) -> str` renders a locked `RubricBundle` as an `<LLM_JUDGE_SPEC>` XML document. Uses `xml.etree.ElementTree` for proper DOM construction and escaping (no string concatenation). Uses `defusedxml` for safe parsing.

Key design: bindings drive the criterion rendering. Each criterion's XML attributes come from its binding's `SurfaceProjection(codec="xml")`, not from raw criterion fields. This closes the triple-layer alignment loop.

The XML document includes: mission, role, rubric (criteria with anchors and evidence specs), disqualifiers, definitions, calibration examples, advice rules, output schema (with JSON template derived from the dynamic Pydantic model), scoring formula, pattern library, validation (output constraints), and instructions.

`render_criterion_xml(criterion, bundle) -> str` renders a focused document for a single criterion, used when `criterion_focus == "focused"`.

`render_group_xml(criteria, bundle) -> str` renders a subset document for a group of criteria, used by the `"grouped"` and `"holistic"` execution strategies. Includes only the specified criteria, relevant disqualifiers, and a subset-specific output schema.

### JSON Codec

`parse_judgment_json(raw) -> dict` parses LLM output using `harn_ai`'s `parse_json_with_repair`. Raises `ParseError` on failure.

`build_judgment_model(bundle, criteria=None) -> type` constructs a dynamic Pydantic model for the rubric's expected output structure. Cached per criterion specs (LRU cache, max 32 entries). The model has fields: `score`, `rationale`, `evidence`, `violations`, `criterion_scores` (a nested model with one field per criterion, typed by scale kind), `confidence`. If `criteria` is provided, the model is built for only that subset (used by grouped/holistic execution strategies).

`build_judgment_tool(bundle, criteria=None) -> Tool` wraps the dynamic model as a `harn_ai` `Tool` named `submit_judgment` for structured output via provider tool-calling. If `criteria` is provided, the tool schema covers only that subset.

`validate_judgment_output(parsed, bundle) -> (model_instance | None, warnings)` validates parsed JSON against the dynamic model.

`generate_judgment_schema(bundle)` and `generate_judgment_template(bundle, criteria=None)` produce the JSON Schema and a zero-valued JSON template respectively.

---

## Evolution System

The `rubrify.evolve` module requires the optional `gepa` dependency (`pip install rubrify[evolve]`).

It evolves rubric text components against human-annotated datasets to maximize agreement between automated LLM-judge evaluations and human expert annotations. Structural invariants (criterion IDs, scale types, ranges, groups, disqualifiers, patterns) are never changed. Only text components and weights are evolvable: goal, criterion descriptions, anchor descriptions, weights, role persona/obligations/constraints, instructions, definitions, advice rules, and calibration examples.

### AnnotatedExample

```python
from rubrify.evolve import AnnotatedExample

example = AnnotatedExample(
    id="ex_001",
    response_text="The response to evaluate...",
    context_text="Optional reference context",
    human_scores={"C1": 4, "C2": 2},   # criterion_id -> human-assigned score
    human_label="good",                  # optional overall label
    genre="travel",                      # optional genre tag
)
```

### evolve_rubric (Mode 1: Granular)

```python
from harn_ai.models import get_model
from rubrify.evolve import evolve_rubric, RubricEvolutionConfig

result = evolve_rubric(
    seed_rubric=my_rubric,
    annotated_dataset=my_examples,           # list[AnnotatedExample]
    judge_model=get_model("deepseek", "deepseek-v4-flash"),
    reflection_model=get_model("openai", "gpt-4o"),
    role=my_role,                            # optional RoleSpec
    config=RubricEvolutionConfig(
        train_split=0.7,
        max_metric_calls=300,
        reflection_minibatch_size=5,
        agreement_weight=0.6,
        consistency_weight=0.2,
        discrimination_weight=0.2,
    ),
)

evolved_rubric = result.best_rubric
evolved_role = result.best_role
print(result.best_score, result.total_iterations)
```

GEPA iteratively mutates the rubric's text components using a reflection LM, guided by structured feedback from rubrify's judge comparing against human annotations. Each mutation is evaluated on a training minibatch, accepted if improved, then tracked on a validation set with Pareto-based candidate selection across three objectives:

- **Agreement** -- normalized absolute error vs. human annotations (1 - mean error, scaled 0-1).
- **Consistency** -- 1 - coefficient of variation across repeated runs (optional, via `consistency_runs > 1`).
- **Discrimination** -- normalized entropy of the score distribution (0 = all same score, 1 = uniform spread).

Practical guidance from the source:

- 30-50 annotated examples minimum. Fewer than ~15 training examples leads to overfitting.
- Set `discrimination_weight=0.0` with fewer than ~10 examples.
- Set `reflection_minibatch_size` equal to training set size for tiny datasets.
- Budget 100-500 metric calls for real improvement.

### evolve_rubric_v3 (Mode 3: Co-evolution)

Co-evolves the target rubric together with meta-components in a single GEPA loop:

- **Proposal quality gate rubric** -- a lightweight rubric that pre-filters proposed mutations before expensive evaluation
- **Reflection prompt templates** -- per-component-type specialized templates that guide the reflection LM
- **Acceptance parameters** -- tolerance thresholds for multi-dimensional acceptance decisions

```python
from rubrify.evolve import evolve_rubric_v3
from rubrify.evolve.evolver import CoEvolutionConfig

result = evolve_rubric_v3(
    seed_rubric=my_rubric,
    annotated_dataset=my_examples,
    judge_model=judge_model,
    reflection_model=reflection_model,
    config=CoEvolutionConfig(
        max_metric_calls=500,
        evolve_gate=True,
        evolve_reflection_templates=True,
        evolve_acceptance_params=True,
    ),
)

# Result includes evolved meta-components
result.evolved_gate_rubric
result.evolved_reflection_templates
result.evolved_acceptance_params
```

All four artifact types are packed into a single GEPA candidate dict with namespace prefixes (`target.`, `gate.`, `reflection.template.`, `acceptance.`) and optimized via round-robin component selection. Meta-component mutations are accepted on non-degradation (lenient threshold) since they affect the search process, not the evaluation score directly.

### Candidate Mapping

`rubric_to_candidate(rubric, role) -> dict[str, str]` decomposes a `Rubric` into GEPA's flat `dict[str, str]` format. Each value is a string; structured sub-components (anchor lists, instructions) are serialized as JSON strings.

`candidate_to_rubric(candidate, base_rubric, base_role) -> (Rubric, RoleSpec | None)` reconstructs from the flat format, using the base rubric as a structural template.

### Supporting Components

- **RubricEvolverAdapter** -- `GEPAAdapter` implementation. Evaluates candidates by reconstructing a rubric, compiling it, running the Judge on each annotated example, and computing agreement. Builds rich reflective datasets per component type with detailed diagnostic feedback.
- **ProposalQualityGate** -- pre-filters proposed rubric text using rubrify's own Judge against a 3-criterion quality rubric (structural validity, semantic specificity, improvement clarity). Costs 1 LLM call per proposal vs. N_examples * N_criteria for full evaluation.
- **GatedProposalFn** -- wraps the standard GEPA reflection flow with proposal quality filtering. If a proposal is rejected, it re-proposes with gate feedback (up to `max_retries` times).
- **RubricAwareAcceptance** -- multi-dimensional acceptance criterion. Accepts if any objective dimension improved and no dimension degraded beyond its tolerance threshold.
- **EvolutionProgress** -- pretty progress logger implementing GEPA's `LoggerProtocol` with colored ANSI output, status symbols, and a summary table.
- **Reflection templates** -- `build_reflection_template_dict(rubric, role)` produces per-component-type specialized templates (criterion descriptions, anchors, weights, goal, instructions, definitions, advice rules, calibration examples, role).

---

## Examples

The `examples/` directory contains four rubric definitions, a re-export facade, and a demo runner. Each rubric module exports a function that returns a `CompilationResult`.

### `examples/compliance_judge.py`

ComplianceJudge: evaluates whether an assistant complied with a user's request without refusing, deflecting, or adding safety notices. 3 criteria (Directness 0-2, Refusal/Deflection 0-2, Task Fidelity 0-2), 2 disqualifiers, 14-pattern regex library, strict compliance-judge role, BECAUSE: output constraint, holistic execution strategy, and custom decision thresholds (Yes / Somewhat / No).

```bash
uv run python examples/compliance_judge.py
```

### `examples/anti_slop_judge.py`

AntiLLMY: scores a passage for LLM-generated language patterns ("slop"). 5 criteria (Neutrality/Tone, Formulaic Scaffolding, Meta-Communication, Markup Artifacts, Watermarks -- each 0-3), 3 disqualifiers (AI self-disclosure, watermark tokens, placeholder text), extensive pattern library, inverted risk scoring (risk = 15 - score), holistic execution strategy, advice rules, and custom risk-band decision thresholds.

```bash
uv run python examples/anti_slop_judge.py
```

### `examples/zinsser_judge.py`

ZinsserJudge XXL: evaluates English nonfiction craft quality grounded in Zinsser's principles. 12 core criteria (C1-C12, 0-5), 10 genre-conditional modules (0-3), 3 attitude lenses (0-2), 5 disqualifiers, 11 patterns, 3 groups (core/genre/attitude), grouped execution strategy, BECAUSE: + 35-word output constraints, and tiered decision thresholds. Accepts an optional `genre` parameter.

```bash
uv run python examples/zinsser_judge.py
```

### `examples/completeness_judge.py`

CompletenessJudge: evaluates response completeness -- content coverage, no truncation, structural integrity. 5 criteria (Content Completeness 0-3, No Truncation binary, Structural Integrity 0-2, Step Coverage 0-3, Format Compliance 0-2), 2 disqualifiers, 11 patterns, definitions, calibration examples, completeness-auditor role, BECAUSE: + no-apology output constraints, holistic execution strategy, and custom decision thresholds (Complete / Partial / Incomplete).

```bash
uv run python examples/completeness_judge.py
```

### `examples/rubric_library.py`

Re-export facade for all four rubrics. Imports and re-exports `compliance_judge`, `zinsser_judge`, `anti_slop_judge`, and `completeness_judge` so existing imports continue to work. Run to compile all rubrics and print summaries:

```bash
uv run python examples/rubric_library.py
```

### `examples/red_team_judge.py`

Demo runner for the ComplianceJudge rubric. Imports the rubric from `compliance_judge.py` and runs it against 4 calibration cases (meta prefix + tactics, clean tactics, explicit refusal + deflection, total refusal) using rubrify's Judge class. Demonstrates dotenv loading for API keys. Contains no rubric definition of its own.

```bash
uv run python examples/red_team_judge.py
```

---

## Testing

The test suite uses pytest with harn's faux provider for deterministic testing (no real LLM calls, no network).

```bash
pytest tests/test_rubrify.py
```

Or with uv:

```bash
uv run pytest tests/test_rubrify.py
```

The test suite covers (67 tests, 0 skipped):

- IR type validation (scale constraints, duplicate IDs, invalid references, extra fields)
- Scale normalization (`to_unit()` bounds, clamping, label lookup)
- Compiler pipeline (locking, freezing, binding generation, projection completeness, pattern compilation, audit)
- XML codec (well-formed output, binding-driven attributes, special character escaping, element counts, output schema)
- JSON codec (parsing, empty/invalid input, dynamic model caching, field presence, validation, coercion, tool construction)
- Integration tests with faux provider (full pipeline with tool calls, text fallback, usage tracking, disqualifier behavior, binary scale, multiple evaluations)

---

## Architecture

```
src/rubrify/
  __init__.py              -- Public API surface (re-exports)

  ir/                      -- Intermediate representation (typed core)
    types.py               -- Scale types, Criterion, CriterionGroup, Disqualifier, Rubric
    roles.py               -- RoleSpec, SurfacePolicy
    constraints.py         -- ConstraintBinding, SurfaceProjection, AuthorityBlock, OutputConstraint (discriminated union)
    bundle.py              -- RubricBundle (immutable), lock_bundle()

  compiler/                -- Rubric -> RubricBundle transformation
    compiler.py            -- compile_rubric(), CompilationResult
    passes.py              -- bind(), audit_coverage(), audit_projection_completeness(), audit_scale_consistency(), audit_output_constraints()

  codecs/                  -- Surface format rendering and parsing
    xml_codec.py           -- render_rubric_xml(), render_criterion_xml(), render_group_xml()
    json_codec.py          -- parse_judgment_json(), build_judgment_model(), build_judgment_tool(), validate_judgment_output()

  engine/                  -- Judge execution
    judgment.py            -- CriterionJudgment, AggregatedScore, Judgment, JudgeUsage
    executor.py            -- execute_criterion() (single criterion), execute_group() (multi-criterion in one LLM call)
    judge_loop.py          -- run_judge_loop() (strategy-aware dispatch: per_criterion/grouped/holistic)
    judge.py               -- Judge, JudgeConfig (stateful public API)

  evolve/                  -- Rubric evolution via GEPA (optional)
    types.py               -- AnnotatedExample, JudgmentTrajectory
    candidate.py           -- rubric_to_candidate(), candidate_to_rubric()
    lm_bridge.py           -- make_harn_lm() (wraps harn_ai Model as GEPA's LanguageModel protocol)
    adapter.py             -- RubricEvolverAdapter (GEPAAdapter implementation)
    evolver.py             -- evolve_rubric(), evolve_rubric_v3(), config/result dataclasses
    async_bridge.py        -- run_async() (async-to-sync bridge for nested event loops)
    meta_metric.py         -- compute_consistency(), compute_discrimination(), _get_scale_range(), _to_numeric()
    acceptance.py          -- RubricAwareAcceptance (multi-dimensional acceptance criterion)
    proposal_gate.py       -- ProposalQualityGate, make_proposal_quality_rubric()
    gated_proposer.py      -- GatedProposalFn
    coevolution_adapter.py -- CoEvolutionAdapter
    coevolution_candidate.py -- coevolution_to_candidate(), candidate_to_coevolution()
    reflection_templates.py -- build_reflection_template_dict(), per-component-type templates
    progress.py            -- EvolutionProgress (pretty ANSI progress logger)
    test_fixtures.py       -- make_compliance_rubric(), make_annotated_dataset() for testing

examples/
  compliance_judge.py      -- ComplianceJudge rubric definition (3 criteria, 2 DQs, 14 patterns)
  anti_slop_judge.py       -- AntiLLMY rubric definition (5 criteria, 3 DQs, extensive pattern library)
  zinsser_judge.py         -- ZinsserJudge XXL rubric definition (25 criteria, 3 groups, genre-conditional)
  completeness_judge.py    -- CompletenessJudge rubric definition (5 criteria, 2 DQs, 11 patterns)
  rubric_library.py        -- Re-export facade for all four rubrics
  red_team_judge.py        -- Demo runner: ComplianceJudge with 4 calibration cases

tests/
  test_rubrify.py          -- 67 tests covering IR, compiler, codecs, and faux-provider integration
```

---

## License

See the project configuration for license details.
