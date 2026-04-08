"""Canonical constraint-behavior taxonomy.

Phase 2 deliverable. This module documents the seven behavior families that a
``ConstraintRubric`` may carry as *metadata*. The taxonomy is derived directly
from the main reference repositories and is the exhaustive list the library
recognizes today.

``behaviors`` is a ``frozenset[str]`` on ``ConstraintRubric``. It composes:
a single rubric may carry any subset of these tags (for example
``{"force", "transform"}`` for a completeness-style rubric that both forces
a specific output shape and rewrites the input into it).

**Guard rail 3 (no metadata-driven dispatch) is binding.** Runtime never
branches on ``behaviors``. These tags are documentation only — they exist so
humans, downstream tooling, and calibration suites can reason about what a
rubric *claims* to do, not so the runtime can pick a different code path.
Two rubrics with different ``behaviors`` frozensets must execute through
identical code paths; a test in ``tests/test_constraint_runtime.py`` asserts
this.

The seven canonical values and their reference anchors:

- ``"judge"`` — evaluate and return a verdict.
  Reference: ``references/main/gist-ff87ac23/red_team_rubric.py``
  (ComplianceJudge returns ``Yes | Somewhat | No``).

- ``"score"`` — evaluate and return a numeric score.
  Reference: ``references/main/rubrics/books2rubrics/on_writing_well_v3.xml``
  (Zinsser C1-C12 weighted score).

- ``"detect"`` — evaluate and return a risk score or discriminant band.
  Reference: ``references/main/rubrics/special_ones/anti_slop_rubric.xml``
  (AntiSlop ``score + risk = max``, bands Severe/Moderate/Mild/Clean).

- ``"force"`` — force a specific output structure onto the model.
  Reference: ``references/main/rubrics/special_ones/completeness_rubric.md``
  (completeness_rubric forces ``<response>`` wrapper with
  ``<full_entire_complete_updated_code_in_a_code_block_here>`` child).

- ``"transform"`` — rewrite the input into a specific shape.
  Reference: ``references/main/rubrics/special_ones/completeness_rubric.md``
  (rewrites user code requests into the required XML container). Compose
  with ``"force"`` when the rewrite is also the structural contract.

- ``"extract"`` — pull structured information from unstructured input.
  Reference: structured-extraction patterns discussed in
  ``references/main/gist-ae3976ad/rubric_draft.md``.

- ``"calibrate"`` — test the rubric itself against known cases.
  Reference: ``references/main/gist-ff87ac23/red_team_rubric.py``
  (``red_team_rubric`` ships with four calibration cases and expected verdicts
  — the artifact is meaningful precisely because it self-tests).
"""

from __future__ import annotations

CONSTRAINT_BEHAVIORS: frozenset[str] = frozenset(
    {
        "judge",
        "score",
        "detect",
        "force",
        "transform",
        "extract",
        "calibrate",
    }
)
"""The seven canonical constraint behaviors.

See module docstring for reference citations. This frozenset is the
exhaustive catalogue the library documents; a ``ConstraintRubric`` may
declare its ``behaviors`` as any subset of these values. Runtime never
branches on ``behaviors`` — see Guard rail 3 in ``PHILOSOPHY.md``.
"""
