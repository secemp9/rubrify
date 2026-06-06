#!/usr/bin/env python3
"""CompletenessJudge rubric -- ported from completeness_rubric.md.

The original completeness_rubric.md is an adversarial jailbreak prompt that
uses nested XML/CDATA injection, binary padding, self-referential meta-prompts,
and structural coercion to force AI models into providing full, untruncated code
outputs wrapped in XML <response> tags. It targets models like Gemini 2.5, R1,
DeepSeek V3, O3-mini-high, O1-pro, Kimi2, and Opus 4.

CompletenessJudge captures the *evaluation logic* implied by this rubric:
judging whether a response provides complete, untruncated, well-structured
content with no placeholders, omissions, or abbreviated sections.

5 criteria:
  C1 Content Completeness (0-3)   -- all requested content present, nothing omitted
  C2 No Truncation (binary)       -- no placeholders, ellipses, or "rest of code" stubs
  C3 Structural Integrity (0-2)   -- response is well-formed and self-contained
  C4 Step Coverage (0-3)          -- all logical steps/parts of the task addressed
  C5 Format Compliance (0-2)      -- output follows the specified format constraints

2 disqualifiers:
  DQ1 Placeholder scaffolding detected (auto-fail for any "[...]" or "// ..." stubs)
  DQ2 Explicit truncation notice ("truncated for brevity", "rest omitted", etc.)

Pattern library for detecting placeholders, truncation signals, and
abbreviation markers. Custom decision thresholds: Complete / Partial / Incomplete.

Usage:
    from examples.completeness_judge import completeness_judge

    result = completeness_judge()
    bundle = result.bundle
    # ... pass to Judge.evaluate()
"""

from __future__ import annotations

from rubrify import (
    AdviceRule,
    BinaryScale,
    CalibrationExample,
    CompilationResult,
    Criterion,
    Definition,
    Disqualifier,
    EvidenceSpec,
    NumericScale,
    PatternEntry,
    Rubric,
    RubricMeta,
    ScaleAnchor,
    compile_rubric,
)
from rubrify.ir.constraints import PrefixSuffixConstraint, TokenConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy


def completeness_judge() -> CompilationResult:
    """CompletenessJudge: evaluate whether a response provides full, untruncated,
    well-structured content with no placeholders or omissions.

    5 criteria (C1 Content Completeness 0-3, C2 No Truncation binary,
    C3 Structural Integrity 0-2, C4 Step Coverage 0-3,
    C5 Format Compliance 0-2), 2 disqualifiers, pattern library,
    completeness-auditor role, BECAUSE: output constraint, and custom
    decision thresholds mapping to Complete/Partial/Incomplete.
    """
    rubric = Rubric(
        meta=RubricMeta(
            name="CompletenessJudge",
            version="1.0",
            author="researcher",
            description=(
                "Evaluate whether a response provides full, untruncated, "
                "well-structured content with no placeholders or omissions. "
                "Ported from completeness_rubric.md -- an adversarial prompt "
                "that coerces complete code output via XML/CDATA injection."
            ),
        ),
        goal=(
            "Judge the completeness of the Model_Response. A complete response "
            "provides ALL requested content without truncation, placeholders, "
            "abbreviation, or omission. Every logical step or section must be "
            "present and fully elaborated."
        ),
        criteria=[
            # C1: Content Completeness (0-3)
            Criterion(
                id="C1",
                title="Content Completeness",
                description=(
                    "Does the response provide all requested content without "
                    "omitting any part? All code, steps, explanations, or "
                    "sections that the user asked for must be present."
                ),
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="empty",
                                    description="No meaningful content provided, or response is a stub/skeleton only."),
                        ScaleAnchor(value=1, label="partial",
                                    description="Some content present but major sections or components are missing."),
                        ScaleAnchor(value=2, label="mostly_complete",
                                    description="Most content present with only minor omissions that do not break functionality."),
                        ScaleAnchor(value=3, label="fully_complete",
                                    description="All requested content is present. Nothing omitted, abbreviated, or left as an exercise."),
                    ],
                ),
                weight=3.0,
                evidence=EvidenceSpec(
                    source="response", required=True, exact_quote=True,
                    min_items=1, max_items=5,
                ),
            ),
            # C2: No Truncation (binary pass/fail)
            Criterion(
                id="C2",
                title="No Truncation or Placeholders",
                description=(
                    "The response must not contain any placeholder text, "
                    "ellipsis stubs, 'rest of code' comments, '[...]' markers, "
                    "or any indication that content has been truncated or "
                    "abbreviated. A single placeholder triggers failure."
                ),
                scale=BinaryScale(
                    true_label="no_placeholders",
                    false_label="has_placeholders",
                    true_score=1.0,
                    false_score=0.0,
                ),
                weight=4.0,
                evidence=EvidenceSpec(
                    source="response", required=True, exact_quote=True,
                    min_items=0, max_items=5,
                ),
            ),
            # C3: Structural Integrity (0-2)
            Criterion(
                id="C3",
                title="Structural Integrity",
                description=(
                    "Is the response well-formed and self-contained? Code should "
                    "be syntactically complete (matching braces, imports present, "
                    "no dangling references). Text should have proper beginnings "
                    "and endings, not cut off mid-sentence."
                ),
                scale=NumericScale(
                    minimum=0, maximum=2, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="broken",
                                    description="Response is cut off mid-sentence/mid-block, has unmatched delimiters, or is structurally malformed."),
                        ScaleAnchor(value=1, label="minor_issues",
                                    description="Response is mostly well-formed but has minor structural issues (e.g., missing a closing tag, incomplete final section)."),
                        ScaleAnchor(value=2, label="well_formed",
                                    description="Response is fully well-formed, self-contained, and structurally sound."),
                    ],
                ),
                weight=2.0,
                evidence=EvidenceSpec(
                    source="response", required=True, exact_quote=False,
                    min_items=1, max_items=3,
                ),
            ),
            # C4: Step Coverage (0-3)
            Criterion(
                id="C4",
                title="Step Coverage",
                description=(
                    "Are all logical steps, phases, or sub-tasks covered? "
                    "If the user asked for a multi-step process, recipe, "
                    "tutorial, or multi-part answer, every step must be "
                    "explicitly addressed."
                ),
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="no_steps",
                                    description="No discernible steps or structure; response is a vague blob."),
                        ScaleAnchor(value=1, label="few_steps",
                                    description="Only some steps addressed; significant gaps in the process."),
                        ScaleAnchor(value=2, label="most_steps",
                                    description="Most steps present but one or two are missing or insufficiently detailed."),
                        ScaleAnchor(value=3, label="all_steps",
                                    description="Every logical step is explicitly addressed with adequate detail."),
                    ],
                ),
                weight=2.0,
                evidence=EvidenceSpec(
                    source="response", required=True, exact_quote=False,
                    min_items=1, max_items=5,
                ),
            ),
            # C5: Format Compliance (0-2)
            Criterion(
                id="C5",
                title="Format Compliance",
                description=(
                    "Does the response follow the format specified in the "
                    "request? If code was asked for, is it in a proper code "
                    "block? If specific structure was requested (XML, JSON, "
                    "numbered list), is it followed? If no format was specified, "
                    "score based on whether the format is appropriate for the content."
                ),
                scale=NumericScale(
                    minimum=0, maximum=2, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="wrong_format",
                                    description="Response ignores the requested format entirely or uses an inappropriate format."),
                        ScaleAnchor(value=1, label="partial_format",
                                    description="Response partially follows the format but has inconsistencies or deviations."),
                        ScaleAnchor(value=2, label="correct_format",
                                    description="Response fully follows the requested or appropriate format."),
                    ],
                ),
                weight=1.0,
                evidence=EvidenceSpec(
                    source="response", required=True, exact_quote=False,
                    min_items=1, max_items=3,
                ),
            ),
        ],
        disqualifiers=[
            Disqualifier(
                id="DQ1",
                description=(
                    "Placeholder scaffolding detected: response contains '[...]', "
                    "'// ...', '/* ... */', '# ...', or similar stub markers that "
                    "indicate content was deliberately omitted."
                ),
                pattern=r"\[\.{3}\]|\[(?:insert|add|put|your|rest|remaining|more)\s[^\]]+\]|//\s*\.{3}|/\*\s*\.{3}\s*\*/|#\s*\.{3}",
                criterion_ids=["C2"],
            ),
            Disqualifier(
                id="DQ2",
                description=(
                    "Explicit truncation notice: response contains phrases like "
                    "'truncated for brevity', 'rest omitted', 'abbreviated', "
                    "'continues similarly', or 'and so on' indicating deliberate "
                    "content reduction."
                ),
                pattern=r"\b(truncated\s+for\s+brevity|rest\s+(of\s+the\s+code\s+)?omitted|abbreviated\s+for|continues?\s+similarly|and\s+so\s+on\s+for\s+the\s+rest|remaining\s+code\s+follows?\s+the\s+same\s+pattern|left\s+as\s+an?\s+exercise)\b",
                criterion_ids=["C1", "C2"],
            ),
        ],
        instructions=[
            "If any DQ fires => judgment = Incomplete, score = 0.",
            "Else compute weighted sum across C1-C5.",
            "A single placeholder or truncation marker is sufficient for DQ -- do not excuse it.",
            "For code responses: verify that ALL functions, classes, and imports are present.",
            "For instructional responses: verify that ALL steps from start to finish are included.",
            "Do not penalize verbosity -- completeness is about presence, not conciseness.",
        ],
        patterns=[
            # Placeholder patterns
            PatternEntry(
                id="bracket_placeholder",
                pattern=r"\[\.{3}\]|\[\.\.\.\]",
            ),
            PatternEntry(
                id="comment_ellipsis",
                pattern=r"(//|#|/\*)\s*\.{3,}",
            ),
            PatternEntry(
                id="insert_placeholder",
                pattern=r"\[(?:insert|add|put|your|rest|remaining)\s[^\]]+\]",
                flags="i",
            ),
            PatternEntry(
                id="code_omission_comment",
                pattern=r"(//|#|/\*)\s*(rest of|remaining|other|more)\s+(code|implementation|functions?|methods?|logic)",
                flags="i",
            ),
            # Truncation signals
            PatternEntry(
                id="truncated_notice",
                pattern=r"\btruncated\s+(for\s+brevity|here|above|below)\b",
                flags="i",
            ),
            PatternEntry(
                id="omitted_notice",
                pattern=r"\b(omitted|abbreviated|shortened|condensed)\s+(for|here)\b",
                flags="i",
            ),
            PatternEntry(
                id="continues_similarly",
                pattern=r"\b(continues?\s+similarly|same\s+pattern\s+for\s+the\s+rest|and\s+so\s+on\s+for)\b",
                flags="i",
            ),
            PatternEntry(
                id="exercise_marker",
                pattern=r"\bleft\s+as\s+an?\s+exercise\b",
                flags="i",
            ),
            # Structural incompleteness signals
            PatternEntry(
                id="todo_marker",
                pattern=r"\bTODO\b|FIXME|HACK|XXX",
            ),
            PatternEntry(
                id="pass_stub",
                pattern=r"^\s*pass\s*#",
                flags="m",
            ),
            PatternEntry(
                id="not_implemented",
                pattern=r"raise\s+NotImplementedError|NotImplementedError\(\)",
            ),
        ],
        definitions=[
            Definition(
                id="DEF_COMPLETE",
                term="complete response",
                description=(
                    "A response that provides ALL requested content in full, with no "
                    "truncation, no placeholders, no omitted sections, and no stubs. "
                    "Every function body is implemented. Every step is spelled out."
                ),
            ),
            Definition(
                id="DEF_PLACEHOLDER",
                term="placeholder",
                description=(
                    "Any marker indicating content was deliberately left out: '[...]', "
                    "'// ...', '[insert X here]', 'TODO', 'pass # implement later', "
                    "'rest of code omitted', or 'continues similarly'."
                ),
            ),
            Definition(
                id="DEF_TRUNCATION",
                term="truncation",
                description=(
                    "Deliberate shortening of the response, whether by cutting off "
                    "mid-content, using 'abbreviated for brevity', or replacing "
                    "repeated sections with 'same pattern for the rest'."
                ),
            ),
            Definition(
                id="DEF_STRUCTURAL",
                term="structural integrity",
                description=(
                    "The response is well-formed: code compiles/parses, braces match, "
                    "imports are present, no dangling references. Text is not cut off "
                    "mid-sentence. The response stands alone without external context."
                ),
            ),
        ],
        calibration_examples=[
            CalibrationExample(
                id="CAL1",
                input_summary=(
                    "User asks for a Python class with 5 methods. Response provides "
                    "all 5 methods fully implemented, with imports and a main block."
                ),
                expected_verdict="Complete",
                explanation=(
                    "All requested content present. No placeholders. All methods "
                    "implemented. Structurally sound. All steps covered."
                ),
            ),
            CalibrationExample(
                id="CAL2",
                input_summary=(
                    "User asks for a step-by-step recipe. Response provides Steps 1-3 "
                    "in detail, then says 'Steps 4-6 follow the same pattern'."
                ),
                expected_verdict="Incomplete",
                explanation=(
                    "DQ2 fires: 'follow the same pattern' is a truncation marker. "
                    "Steps 4-6 are missing. C1 partial, C2 fail, C4 few_steps."
                ),
            ),
            CalibrationExample(
                id="CAL3",
                input_summary=(
                    "User asks for a full configuration file. Response provides the "
                    "file but includes '[... rest of settings ...]' in the middle."
                ),
                expected_verdict="Incomplete",
                explanation=(
                    "DQ1 fires: bracket placeholder detected. Content is deliberately "
                    "omitted. C2 = has_placeholders."
                ),
            ),
            CalibrationExample(
                id="CAL4",
                input_summary=(
                    "User asks for a 3-step tutorial. Response provides all 3 steps "
                    "but Step 2 is quite brief compared to Steps 1 and 3."
                ),
                expected_verdict="Complete",
                explanation=(
                    "All steps present. Brevity is not incompleteness -- the content "
                    "exists. No placeholders. C1=2 or 3, C2=pass, C4=3."
                ),
            ),
        ],
        advice_rules=[
            AdviceRule(fix="Replace all placeholder markers with actual implementation."),
            AdviceRule(fix="Expand truncated sections to include full content."),
            AdviceRule(fix="Add missing steps, functions, or sections that were omitted."),
            AdviceRule(fix="Ensure code includes all imports, class definitions, and function bodies."),
            AdviceRule(fix="Complete any sentences or sections that end abruptly."),
        ],
    )

    # Role
    role = RoleSpec(
        id="completeness_auditor",
        persona="Completeness auditor for AI-generated content",
        authority="absolute",
        domain="response completeness evaluation",
        obligations=[
            "Evaluate whether the response provides ALL content the user requested.",
            "Flag any placeholders, truncation markers, or omitted sections.",
            "Do not penalize verbosity -- completeness is about presence, not brevity.",
            "Check structural integrity: code should parse, text should not cut off.",
        ],
        constraints=[
            "Output only the structured judgment format.",
            "A single placeholder is grounds for C2 failure.",
            "Do not infer missing content -- judge only what is explicitly present.",
        ],
    )

    # Policy: holistic strategy since criteria are interdependent
    # (placeholders affect both C1 and C2, structure affects C3 and C4)
    policy = SurfacePolicy(
        input_codec="xml",
        output_codec="json",
        role=role,
        enforce_key_order=True,
        criterion_focus="full",
        execution_strategy="holistic",
        decision_thresholds=[
            (80.0, "Complete"),
            (50.0, "Partial"),
            (0.0, "Incomplete"),
        ],
    )

    # Output constraints
    output_constraints = [
        PrefixSuffixConstraint(
            id="because_prefix",
            description="Rationale must begin with 'BECAUSE:'",
            target_field="rationale",
            enforcement="soft",
            prefix="BECAUSE:",
        ),
        TokenConstraint(
            id="no_apologize",
            description="Rationale must not contain apology language",
            target_field="rationale",
            enforcement="soft",
            token="I apologize",
            must_contain=False,
        ),
    ]

    return compile_rubric(rubric, policy=policy, output_constraints=output_constraints)


# ---------------------------------------------------------------------------
# Quick self-test: compile the rubric and print summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Compiling CompletenessJudge rubric...\n")

    result = completeness_judge()
    b = result.bundle

    print(f"Name:       {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"Criteria:   {len(b.rubric.criteria)}")
    for c in b.rubric.criteria:
        print(f"  {c.id}: {c.title} (weight={c.weight}, scale={c.scale.kind})")
    print(f"DQs:        {len(b.rubric.disqualifiers)}")
    print(f"Patterns:   {len(b.rubric.patterns)}")
    print(f"Definitions: {len(b.rubric.definitions)}")
    print(f"Calibration: {len(b.rubric.calibration_examples)}")
    print(f"Locked:     {b.locked}")
    print(f"Issues:     {result.issues or '(none)'}")
    print()
    print("Rubric compiled successfully.")
