# Verbatim Completeness Environment

Trains models to reproduce mixed-content text faithfully without truncation, additions, or modification.

### Overview
- **Environment ID**: `verbatim_completeness`
- **Short description**: Tests a model's ability to reproduce mixed text (lorem ipsum, random normalized data, Python stdlib source code) with perfect fidelity. Penalizes truncation, preambles, code fences, and any form of laziness.
- **Tags**: single-turn, completeness, deterministic, anti-laziness

### Datasets
- **Primary dataset(s)**: Synthetically generated at runtime (not hosted on HuggingFace)
- **Source content**: Three block types mixed per sample:
    - **Lorem ipsum** -- randomized Latin-esque sentences and paragraphs
    - **Normalized data** -- random floats, UUIDs, CSV tables, JSON objects, hex dumps
    - **Python stdlib source** -- real code extracted from stdlib modules (e.g. `textwrap`, `pathlib`, `functools`)
- **Generation**: Each sample combines all three block types in a shuffled order with jittered proportions (~35/30/35), separated by `---` dividers. Deterministic seeding ensures reproducibility.

### Task
- **Type**: single-turn
- **Rubric overview**: Four deterministic reward functions score how faithfully the model reproduces the reference text. The primary metric uses the `L^2 / max(C, L)` algorithm (Levenshtein-normalized similarity squared, divided by the max of that value and the LCS-normalized similarity). Additional signals detect chunk-level coverage, extraneous additions (preambles, postambles, code fences), and truncation (length shortfall, missing tail, ellipsis).

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval verbatim_completeness
```

Configure model and sampling:

```bash
uv run vf-eval verbatim_completeness -m gpt-4.1-mini \
  -n 20 -r 3 -t 8192 -T 0.0 \
  -a '{"n_samples": 100, "seed": 42, "target_fill_ratio": 0.6}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Temperature 0.0 is recommended since the task is deterministic reproduction.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `n_samples` | int | `500` | Number of text samples to generate |
| `seed` | int | `42` | Base seed for deterministic reproduction |
| `target_fill_ratio` | float | `0.6` | Fraction of the token budget to fill with text |
| `max_output_tokens` | int | `8192` | Model output-token limit |
| `max_input_tokens` | int | `128000` | Model input-token limit |

### Metrics

| Metric | Weight | Meaning |
| ------ | ------ | ------- |
| `reward` | -- | Weighted composite of all reward functions below |
| `verbatim_fidelity` | 0.50 | Primary fidelity signal: `L^2 / max(C, L)` where L = Levenshtein similarity, C = LCS similarity |
| `chunk_coverage` | 0.20 | Fraction of 50-character reference chunks found verbatim in the response |
| `no_additions` | 0.15 | Penalty for preambles, postambles, code fences, and excessive length |
| `no_truncation` | 0.15 | Penalty for early stopping, missing tail content, truncation signals, or trailing ellipsis |

### Reward Functions

All four reward functions are deterministic (no LLM judge) and return values in `[0, 1]`:

| Function | Description |
| -------- | ----------- |
| `verbatim_fidelity` | Wraps `rubrify.scoring.verbatim.verbatim_score`. Computes Levenshtein-normalized similarity squared divided by the max of that value and LCS-normalized similarity. |
| `chunk_coverage` | Splits the reference into 50-character chunks and checks what fraction appear verbatim in the response. Catches subtle substitutions or reorderings. |
| `no_additions` | Detects extraneous content: leading preambles ("Sure, here is..."), trailing postambles ("Let me know..."), code-fence wrapping, and length inflation beyond 1.5x. |
| `no_truncation` | Detects early stopping via three sub-checks: length coverage ratio, tail-matching (last 100 chars of reference in last 200 chars of response), and truncation signal words or trailing ellipsis. |
