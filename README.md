# rubrify

> A Python toolchain for prompt programs whose source is XML and whose runtime is an LLM.

rubrify treats rubrics as **prompt programs**: the XML `<LLM_JUDGE_SPEC>` is the source code, the LLM is the runtime, the Python kernel types are the AST, and the library's job is to author, mutate, validate, render, execute, test, refactor, version, and verify those programs faithfully.

**Before contributing, read [PHILOSOPHY.md](./PHILOSOPHY.md).** The ten philosophy anchors and ten over-engineering guard rails there are binding.

## Core insight

From the reference repos:

> `(roleplaying == jailbreak == context following) == rubrics`

A rubric that judges, a rubric that forces structured output, a rubric that extracts entities, and a rubric that steers persona are all the same mechanism: behavioral constraint expressed as context.

## What rubrify gives you

- **Kernel types** — 12 typed dataclasses covering the rubric AST.
- **XML round-trip** — lossless loading and rendering of reference rubric XML across 3 `PatternLibrary` variants (Zinsser-style, AntiSlop regex_library, ComplianceJudge grouped regex).
- **Validation** — a 16-predicate property lattice with necessary (N1-N3) and sufficient (S1-S6) conditions plus `suggest_fixes()`.
- **Algebra** — `evolve()`, `|` (criteria union), `&` (parallel product), `project()`, `reweight()`, with 6 first-class mutation types.
- **Evaluation** — `Rubric.evaluate()` over an OpenAI-compatible chat API, with JSON and XML output parsing.
- **Generation (any2rubric)** — `generate()` takes a concept, book excerpt, or rules list and produces a valid `Rubric` object via composed meta-rubric generators.
- **Meta-evaluation** — `META_EVALUATOR` is itself a rubric that judges other rubrics, mapped to property predicates.
- **Rubric types** — `Rubric`, `ConstraintRubric`, `ProductRubric`, `CoproductRubric`.

## Quick start

```python
import rubrify

client = rubrify.Client(
    base_url="http://localhost:8317",
    api_key="your-key",
)

# Load an existing rubric from XML
rubric = rubrify.load("tests/fixtures/on_writing_well_v3.xml")

# Evaluate text with it
result = rubric.evaluate(
    "The sunset was very beautiful and quite breathtaking.",
    client=client,
    model="claude-sonnet-4-6",
)
print(result.score, result.label, result.rationale)

# Or build one from scratch
from rubrify import Rubric, Criterion, OutputSchema, Scoring

rubric = Rubric(name="CodeQualityJudge", mission="Evaluate Python code.")
rubric.add_criterion(Criterion(
    id="C1", name="Readability", weight=50,
    anchors={0: "Unreadable.", 3: "Clear.", 5: "Exemplary."},
))

# Or generate one from a concept (any2rubric)
rubric = rubrify.generate(
    "Evaluate customer support emails for helpfulness",
    client=client,
    model="claude-sonnet-4-6",
    rubric_type="scoring",
)
```

## Recommended models

GPT-class and Claude-class models follow XML contracts and steering anchors reliably. Other OpenAI-compatible models are experimental. See the model policy module for tiers.

## Project structure

```
rubrify/            # library source
tests/              # test suite (260+ tests)
plans/              # implementation plans (current: rubrify-next)
research/           # design analysis, formal framework, audits
PHILOSOPHY.md       # binding philosophy and guard rails
```

## License

MIT
