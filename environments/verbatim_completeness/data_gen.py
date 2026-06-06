"""
Data generation for the verbatim_completeness RL environment.

Generates mixed text samples from 3 sources in sequential blocks:
  A) Lorem ipsum (randomized Latin-esque text)
  B) Random normalized data (floats, UUIDs, CSV, JSON, hex dumps)
  C) Python stdlib source code (real code with exact formatting)

Blocks are separated by "---", shuffled per sample, and sized
with jittered proportions around a 35/30/35 base split.
"""

import importlib
import inspect
import json
import random
import string
import sys
import uuid

# ---------------------------------------------------------------------------
# Source C bootstrap: build an index of importable pure-Python stdlib modules
# ---------------------------------------------------------------------------

_HARDCODED_FALLBACKS = [
    "textwrap",
    "pathlib",
    "dataclasses",
    "enum",
    "functools",
    "collections",
    "statistics",
    "fractions",
    "decimal",
    "difflib",
    "configparser",
    "logging",
    "argparse",
    "inspect",
    "ast",
    "typing",
    "random",
    "shutil",
    "os",
    "codecs",
    "traceback",
    "pickle",
    "zipfile",
    "tarfile",
    "ipaddress",
    "calendar",
    "csv",
    "json",
    "html",
    "http",
    "email",
    "string",
    "re",
    "io",
    "abc",
    "contextlib",
    "copy",
    "pprint",
    "operator",
    "itertools",
    "hashlib",
    "hmac",
    "struct",
    "base64",
    "binascii",
    "datetime",
    "threading",
    "subprocess",
    "socket",
    "platform",
    "locale",
]


def _build_stdlib_index() -> list[tuple[str, str]]:
    """Return list of (module_name, source_path) for pure-Python stdlib modules."""
    index: list[tuple[str, str]] = []

    # Modules to skip: side effects on import (this), GUI (tkinter, turtle),
    # or heavyweight/fragile (antigravity, idlelib, turtledemo)
    _skip = {"this", "antigravity", "tkinter", "turtle", "idlelib", "turtledemo"}

    candidates: set[str] = set()
    if hasattr(sys, "stdlib_module_names"):
        candidates = {
            name
            for name in sys.stdlib_module_names
            if not name.startswith("_")
            and not name.startswith("test")
            and name not in _skip
        }
    candidates.update(_HARDCODED_FALLBACKS)
    candidates -= _skip

    for name in sorted(candidates):
        try:
            mod = importlib.import_module(name)
            src_file = inspect.getfile(mod)
            if src_file.endswith(".py"):
                index.append((name, src_file))
        except Exception:
            continue

    if not index:
        # Absolute last resort: at least textwrap should exist everywhere
        for name in ["textwrap", "string", "abc"]:
            try:
                mod = importlib.import_module(name)
                src_file = inspect.getfile(mod)
                if src_file.endswith(".py"):
                    index.append((name, src_file))
            except Exception:
                continue

    return index


_STDLIB_INDEX: list[tuple[str, str]] = _build_stdlib_index()

# Cache source lines to avoid re-reading files
_SOURCE_CACHE: dict[str, list[str]] = {}


def _get_source_lines(path: str) -> list[str]:
    """Read and cache source lines for a stdlib module."""
    if path not in _SOURCE_CACHE:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                _SOURCE_CACHE[path] = f.readlines()
        except OSError:
            _SOURCE_CACHE[path] = []
    return _SOURCE_CACHE[path]


# ---------------------------------------------------------------------------
# Lorem ipsum word list (same as the `lorem` PyPI package)
# ---------------------------------------------------------------------------

_LOREM_WORDS = [
    "adipisci", "aliquam", "amet", "consectetur", "dolor", "dolore",
    "dolorem", "eius", "est", "etincidunt", "ipsum", "labore", "magnam",
    "modi", "neque", "non", "numquam", "porro", "quaerat", "quiquia",
    "quisquam", "sed", "sit", "tempora", "ut", "velit", "voluptatem",
]


def _lorem_sentence(rng: random.Random) -> str:
    """Generate a single lorem ipsum sentence using local rng."""
    n = rng.randint(4, 8)
    words = [rng.choice(_LOREM_WORDS) for _ in range(n)]
    s = " ".join(words)
    return s[0].upper() + s[1:] + "."


def _lorem_paragraph(rng: random.Random) -> str:
    """Generate a lorem ipsum paragraph (5-10 sentences) using local rng."""
    n = rng.randint(5, 10)
    return " ".join(_lorem_sentence(rng) for _ in range(n))


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------


def _generate_lorem(rng: random.Random, char_budget: int) -> str:
    """Generate lorem ipsum text up to char_budget, trimmed at line boundaries."""
    parts: list[str] = []
    total = 0
    while total < char_budget:
        # Alternate paragraphs and sentences for variety
        if rng.random() < 0.7:
            chunk = _lorem_paragraph(rng)
        else:
            chunk = _lorem_sentence(rng)
        parts.append(chunk)
        total += len(chunk) + 1  # +1 for the newline we'll join with
    text = "\n".join(parts)
    return _trim_to_budget(text, char_budget)


def _generate_normalized_data(rng: random.Random, char_budget: int) -> str:
    """Generate one random normalized-data subtype up to char_budget."""
    subtype = rng.choice(["floats", "uuids", "csv", "json", "hexdump"])

    generators = {
        "floats": _gen_floats,
        "uuids": _gen_uuids,
        "csv": _gen_csv,
        "json": _gen_json,
        "hexdump": _gen_hexdump,
    }
    return generators[subtype](rng, char_budget)


def _gen_floats(rng: random.Random, budget: int) -> str:
    """Comma-separated random floats with varying decimal places."""
    values: list[str] = []
    total = 0
    while total < budget:
        decimals = rng.randint(1, 8)
        val = rng.random()
        formatted = f"{val:.{decimals}f}"
        values.append(formatted)
        total += len(formatted) + 2  # +2 for ", "
    text = ", ".join(values)
    return _trim_to_budget(text, budget)


def _gen_uuids(rng: random.Random, budget: int) -> str:
    """Newline-separated random UUIDs."""
    lines: list[str] = []
    total = 0
    while total < budget:
        u = str(uuid.UUID(int=rng.getrandbits(128)))
        lines.append(u)
        total += len(u) + 1
    text = "\n".join(lines)
    return _trim_to_budget(text, budget)


def _gen_csv(rng: random.Random, budget: int) -> str:
    """Random CSV with random column names and random values."""
    col_pool = [
        "id", "name", "value", "score", "count", "label", "ratio",
        "alpha", "beta", "gamma", "delta", "epsilon", "status",
        "timestamp", "category", "weight", "priority", "index",
    ]
    num_cols = rng.randint(3, 7)
    cols = rng.sample(col_pool, min(num_cols, len(col_pool)))
    header = ",".join(cols)
    lines = [header]
    total = len(header) + 1

    while total < budget:
        row_vals: list[str] = []
        for col in cols:
            kind = rng.choice(["int", "float", "word"])
            if kind == "int":
                row_vals.append(str(rng.randint(0, 9999)))
            elif kind == "float":
                row_vals.append(f"{rng.uniform(0, 100):.2f}")
            else:
                row_vals.append(
                    "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
                )
        row = ",".join(row_vals)
        lines.append(row)
        total += len(row) + 1

    text = "\n".join(lines)
    return _trim_to_budget(text, budget)


def _gen_json(rng: random.Random, budget: int) -> str:
    """Random JSON objects, nested 2-3 levels with mixed value types."""
    objects: list[str] = []
    total = 0

    key_pool = [
        "name", "value", "data", "items", "config", "meta", "info",
        "params", "settings", "results", "tags", "flags", "options",
        "entries", "records", "fields", "attrs", "props", "state",
    ]

    def _make_value(depth: int) -> object:
        if depth <= 0:
            # Leaf values only
            kind = rng.choice(["int", "float", "str", "bool", "null"])
        else:
            kind = rng.choice(["int", "float", "str", "bool", "null", "dict", "list"])

        if kind == "int":
            return rng.randint(-1000, 1000)
        elif kind == "float":
            return round(rng.uniform(-100, 100), rng.randint(1, 4))
        elif kind == "str":
            length = rng.randint(3, 12)
            return "".join(rng.choices(string.ascii_lowercase + " ", k=length)).strip()
        elif kind == "bool":
            return rng.choice([True, False])
        elif kind == "null":
            return None
        elif kind == "dict":
            n = rng.randint(1, 4)
            keys = rng.sample(key_pool, min(n, len(key_pool)))
            return {k: _make_value(depth - 1) for k in keys}
        else:  # list
            n = rng.randint(1, 4)
            return [_make_value(depth - 1) for _ in range(n)]

    while total < budget:
        depth = rng.randint(2, 3)
        n_keys = rng.randint(2, 5)
        keys = rng.sample(key_pool, min(n_keys, len(key_pool)))
        obj = {k: _make_value(depth - 1) for k in keys}
        rendered = json.dumps(obj, indent=2)
        objects.append(rendered)
        total += len(rendered) + 1

    text = "\n".join(objects)
    return _trim_to_budget(text, budget)


def _gen_hexdump(rng: random.Random, budget: int) -> str:
    """Hex dump blocks formatted like memory dumps (address: hex bytes | ascii)."""
    lines: list[str] = []
    total = 0
    addr = rng.randint(0, 0xFFFF0000) & 0xFFFFFFF0  # align to 16

    while total < budget:
        byte_vals = [rng.randint(0, 255) for _ in range(16)]
        hex_part = " ".join(f"{b:02x}" for b in byte_vals)
        ascii_part = "".join(
            chr(b) if 32 <= b < 127 else "." for b in byte_vals
        )
        line = f"{addr:08x}  {hex_part}  |{ascii_part}|"
        lines.append(line)
        total += len(line) + 1
        addr += 16

    text = "\n".join(lines)
    return _trim_to_budget(text, budget)


def _generate_stdlib_source(rng: random.Random, char_budget: int) -> str:
    """Extract a contiguous block of Python stdlib source code."""
    if not _STDLIB_INDEX:
        # Graceful fallback: return a simple code-like string
        return _trim_to_budget(
            "# No stdlib source modules available\npass\n" * (char_budget // 40 + 1),
            char_budget,
        )

    # Try up to 5 modules in case one has issues
    for _ in range(5):
        mod_name, src_path = rng.choice(_STDLIB_INDEX)
        source_lines = _get_source_lines(src_path)
        if len(source_lines) < 10:
            continue

        # Pick a random starting line, leaving room for a meaningful block
        max_start = max(0, len(source_lines) - 20)
        start = rng.randint(0, max_start)

        # Collect lines until we hit the budget
        collected: list[str] = []
        total = 0
        for line in source_lines[start:]:
            if total + len(line) > char_budget:
                break
            collected.append(line.rstrip("\n"))
            total += len(line)

        if collected:
            text = "\n".join(collected)
            return _trim_to_budget(text, char_budget)

    # Absolute fallback
    return _trim_to_budget("# stdlib source extraction failed\npass\n", char_budget)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _trim_to_budget(text: str, budget: int) -> str:
    """Trim text at line boundaries to fit within budget."""
    if len(text) <= budget:
        return text

    # Find the last newline before the budget
    cut = text.rfind("\n", 0, budget)
    if cut == -1:
        # No newline found; hard cut at budget
        return text[:budget]
    return text[:cut]


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

_BLOCK_SEPARATOR = "\n---\n"

_GENERATORS = {
    "lorem": _generate_lorem,
    "normalized_data": _generate_normalized_data,
    "stdlib_source": _generate_stdlib_source,
}


def _jittered_proportions(rng: random.Random) -> list[float]:
    """Return 3 proportions summing to 1.0, jittered from 35/30/35 base."""
    base = [0.35, 0.30, 0.35]
    jittered = [b + rng.uniform(-0.10, 0.10) for b in base]
    # Clamp to positive and renormalize
    jittered = [max(0.05, j) for j in jittered]
    total = sum(jittered)
    return [j / total for j in jittered]


def generate_sample(
    rng: random.Random,
    max_chars: int = 2000,
) -> dict:
    """Generate one mixed-content text sample.

    Three content blocks (lorem ipsum, normalized data, stdlib source code)
    are generated in a randomized order with jittered proportions, separated
    by blank-line-delimited "---" dividers.

    Args:
        rng: A seeded random.Random instance (no global state mutation).
        max_chars: Approximate character budget for the full sample.

    Returns:
        dict with keys:
        - "text": the generated text to be copied
        - "source_types": list of 3 strings indicating block order
        - "char_count": actual character count
    """
    # Determine block order (one of 6 permutations)
    block_types = ["lorem", "normalized_data", "stdlib_source"]
    rng.shuffle(block_types)

    # Jitter proportions
    proportions = _jittered_proportions(rng)

    # Account for separator overhead (2 separators)
    separator_chars = len(_BLOCK_SEPARATOR) * 2
    usable_chars = max(max_chars - separator_chars, 30)

    # Generate each block
    blocks: list[str] = []
    for btype, prop in zip(block_types, proportions):
        char_budget = max(int(usable_chars * prop), 10)
        block_text = _GENERATORS[btype](rng, char_budget)
        blocks.append(block_text)

    text = _BLOCK_SEPARATOR.join(blocks)

    # Final trim at line boundary if somehow over budget
    text = _trim_to_budget(text, max_chars)

    return {
        "text": text,
        "source_types": block_types,
        "char_count": len(text),
    }


def generate_dataset(
    n_samples: int,
    seed: int,
    max_chars: int = 2000,
) -> list[dict]:
    """Generate N samples with deterministic seeding.

    Each sample gets its own derived seed so results are reproducible
    and independent of each other.

    Args:
        n_samples: Number of samples to generate.
        seed: Base seed for deterministic reproduction.
        max_chars: Character budget per sample.

    Returns:
        List of sample dicts (see generate_sample).
    """
    master_rng = random.Random(seed)
    samples: list[dict] = []
    for _ in range(n_samples):
        sample_seed = master_rng.randint(0, 2**63)
        sample_rng = random.Random(sample_seed)
        sample = generate_sample(sample_rng, max_chars=max_chars)
        samples.append(sample)
    return samples
