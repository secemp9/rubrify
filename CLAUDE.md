# CLAUDE.md

## Project Overview

rubrify is a rubric compiler and judge engine for LLM evaluation. It lets you define structured evaluation rubrics as typed Python objects, compile them into immutable bundles, and run criterion-by-criterion LLM-based judgments against text responses. It also supports evolving rubrics against human-annotated datasets using GEPA's reflective prompt optimization.

## Architecture

The pipeline flows: **IR types** -> **compiler** -> **codecs** -> **engine**.

1. **IR** (`ir/`) -- Pydantic models defining the rubric structure: scales, criteria, groups, disqualifiers, scope specs, corpus profiles, roles, constraints, bundles.
2. **Compiler** (`compiler/`) -- Pure synchronous pipeline: bind projections, create authority blocks, lock into immutable bundle, run audit passes. No LLM calls.
3. **Codecs** (`codecs/`) -- XML rendering (`render_rubric_xml`, `render_criterion_xml`, `render_group_xml`) and JSON parsing/model construction (`build_judgment_model`, `build_judgment_tool`).
4. **Engine** (`engine/`) -- Judge execution: strategy-aware dispatch (`per_criterion`/`grouped`/`holistic`), criterion/group executors, aggregation, disqualifier checking.
5. **Evolve** (`evolve/`, optional) -- GEPA integration for evolving rubric text against human annotations.

## Key Files

```
src/rubrify/
  __init__.py              -- Public API surface, re-exports all user-facing types
  ir/
    types.py               -- Scale types, Criterion, ScopeSpec, CorpusProfile, CriterionGroup, Disqualifier, Rubric
    roles.py               -- RoleSpec, SurfacePolicy
    constraints.py         -- ConstraintBinding, SurfaceProjection, AuthorityBlock, OutputConstraint variants
    bundle.py              -- RubricBundle (frozen), lock_bundle()
  compiler/
    compiler.py            -- compile_rubric(), CompilationResult
    passes.py              -- bind(), audit_coverage(), audit_projection_completeness(), audit_scale_consistency(),
                              audit_output_constraints(), audit_scope_completeness(), audit_hypothesis_neutrality()
  codecs/
    xml_codec.py           -- render_rubric_xml(), render_criterion_xml(), render_group_xml()
    json_codec.py          -- parse_judgment_json(), build_judgment_model(), build_judgment_tool()
  engine/
    judge.py               -- Judge, JudgeConfig
    judge_loop.py          -- run_judge_loop()
    executor.py            -- execute_criterion(), execute_group()
    judgment.py            -- CriterionJudgment, AggregatedScore, Judgment, JudgeUsage
  evolve/
    candidate.py           -- rubric_to_candidate(), candidate_to_rubric() (flat dict mapping for GEPA)
    adapter.py             -- RubricEvolverAdapter (GEPAAdapter implementation)
    evolver.py             -- evolve_rubric(), evolve_rubric_v3()
    reflection_templates.py -- build_reflection_template_dict()
    progress.py            -- EvolutionProgress (ANSI progress logger)
```

## Running Tests

```bash
uv sync --no-sources --extra dev && uv run --no-sources pytest tests/ -v
```

Tests use harn's faux provider -- no real LLM calls, no network access needed.

## Adding a New IR Type

Pattern:
1. Define the model in `ir/types.py` using `SchemaModel` (Pydantic base with `extra="forbid"`)
2. Export it from `ir/__init__.py` and from the top-level `__init__.py`
3. Update compiler passes in `compiler/passes.py` (add audit if needed)
4. Render it in `codecs/xml_codec.py` using `ET.SubElement` (never string concatenation)
5. If evolvable, add to `evolve/candidate.py` in both `rubric_to_candidate()` and `candidate_to_rubric()`

## Key Conventions

- All models use `SchemaModel` -- harn_ai's Pydantic base class with `extra="forbid"` (no unexpected fields)
- XML rendering uses `ET.SubElement`, never string concatenation
- Bundles are frozen (immutable after `lock_bundle()`)
- Tests use harn's faux provider (no real LLM calls)
- Use `uv run --no-sources` for all execution to avoid workspace source resolution issues
- Compiler pipeline is pure/synchronous -- no LLM calls, no async

## Git Conventions

- Tag `v*` triggers PyPI publish via GitHub Actions
- Version is tracked in both `pyproject.toml` and `__init__.py.__version__`
