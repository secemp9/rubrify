#!/usr/bin/env python3
"""AntiLLMY rubric: detect AI-generated language patterns in text.

Ported from anti_slop_rubric.xml. This rubric scores a passage for LLM-y
speak ("slop") by checking for puffery, weasel words, editorializing,
formulaic scaffolding, meta-communication / AI tells, markup artifacts,
and chatbot watermark tokens.

5 criteria (C1-C5, each scored 0-3, higher = cleaner text):
  C1  Neutrality & Tone
  C2  Formulaic Scaffolding
  C3  Meta-Communication & AI Tells
  C4  Markup & Formatting Artifacts
  C5  Watermarks & Citation Pathologies

3 disqualifiers that trigger an automatic failure:
  DQ1  Explicit AI self-disclosure
  DQ2  Watermark tokens (oaicite, turn tokens, utm_source=openai)
  DQ3  Placeholder scaffolding

Inverted risk scoring: risk = 15 - score.  Risk bands range from
"Minimal" (risk 0) to "Severe/FAIL" (risk 15).

Usage:
    from examples.anti_slop_judge import anti_slop_judge

    result = anti_slop_judge()
    bundle = result.bundle
    # ... pass to Judge.evaluate()
"""

from __future__ import annotations

from rubrify import (
    AdviceRule,
    CompilationResult,
    Criterion,
    Disqualifier,
    NumericScale,
    PatternEntry,
    Rubric,
    RubricMeta,
    ScaleAnchor,
    compile_rubric,
)
from rubrify.ir.constraints import (
    CharLimitConstraint,
    ItemCountConstraint,
    PrefixSuffixConstraint,
    WordCountConstraint,
)
from rubrify.ir.roles import RoleSpec, SurfacePolicy


def anti_slop_judge() -> CompilationResult:
    """AntiLLMY: score a passage for LLM-y speak ('slop').

    5 criteria (C1-C5, 0-3 each, higher = cleaner),
    3 disqualifiers (AI self-disclosure, watermark tokens, placeholder text),
    extensive pattern library,
    inverted scoring: risk = 15 - score,
    custom decision thresholds for risk bands.
    """
    rubric = Rubric(
        meta=RubricMeta(
            name="AntiLLMY",
            version="1.0",
            author="researcher",
            description="Score a passage for LLM-y speak ('slop'), using only the given text.",
        ),
        goal=(
            "Score a passage for LLM-y speak ('slop'), using only the given text. "
            "Return a compact diagnosis plus concrete fixes."
        ),
        criteria=[
            # C1 Neutrality & Tone (0-3)
            Criterion(
                id="C1",
                title="Neutrality & Tone",
                description="Absence of puffery, editorializing, weasel claims, and superficial '-ing' participles.",
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="pervasive",
                                    description="Pervasive puffery/editorializing (>=8 hits total) or any weasel claims paired with no attribution."),
                        ScaleAnchor(value=1, label="multiple",
                                    description="Multiple issues (4-7 hits) across the passage."),
                        ScaleAnchor(value=2, label="minor",
                                    description="Minor traces (1-3 hits), largely factual tone."),
                        ScaleAnchor(value=3, label="clean",
                                    description="No hits; neutral, concrete language."),
                    ],
                ),
                weight=3.0,
            ),
            # C2 Formulaic Scaffolding (0-3)
            Criterion(
                id="C2",
                title="Formulaic Scaffolding",
                description="Absence of templatey conjunction overuse, section summaries, 'despite...challenges' patterns, parallelism, rule-of-three.",
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="rigid",
                                    description="Rigid outline tells (e.g., 'Despite...faces challenges...Future...') or >=6 hits total."),
                        ScaleAnchor(value=1, label="visible",
                                    description="3-5 hits; formula shows."),
                        ScaleAnchor(value=2, label="organic",
                                    description="1-2 hits; mostly organic flow."),
                        ScaleAnchor(value=3, label="clean",
                                    description="0 hits; no templatey scaffolding."),
                    ],
                ),
                weight=3.0,
            ),
            # C3 Meta-Communication & AI Tells (0-3)
            Criterion(
                id="C3",
                title="Meta-Communication & AI Tells",
                description="Absence of chatty meta-phrases, AI disclaimers, and letter-form openers.",
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="ai_tell",
                                    description="Any AI disclaimer ('As an AI...') or letter-style opener."),
                        ScaleAnchor(value=1, label="chatty",
                                    description="Chatty meta phrases >=3 or any 'Would you like...'."),
                        ScaleAnchor(value=2, label="minor",
                                    description="1-2 minor chatty phrases."),
                        ScaleAnchor(value=3, label="clean",
                                    description="No meta-communication; impersonal prose."),
                    ],
                ),
                weight=3.0,
            ),
            # C4 Markup & Formatting Artifacts (0-3)
            Criterion(
                id="C4",
                title="Markup & Formatting Artifacts",
                description="Absence of cross-context markup (Markdown in non-Markdown context), emojis, excessive em dashes, curly quotes.",
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="heavy",
                                    description="Cross-context markup (e.g., Markdown headings) or emojis present; or em dashes > 1 per 150 words."),
                        ScaleAnchor(value=1, label="multiple",
                                    description="Multiple artifacts (>=3 kinds) or heavy list-paste footprint."),
                        ScaleAnchor(value=2, label="light",
                                    description="1-2 light artifacts (e.g., occasional curly quotes)."),
                        ScaleAnchor(value=3, label="clean",
                                    description="No artifacts; consistent house style."),
                    ],
                ),
                weight=3.0,
            ),
            # C5 Watermarks & Citation Pathologies (0-3)
            Criterion(
                id="C5",
                title="Watermarks & Citation Pathologies",
                description="Absence of chatbot watermark tokens, placeholder text, citation quirks, and knowledge-cutoff disclaimers.",
                scale=NumericScale(
                    minimum=0, maximum=3, step=1,
                    anchors=[
                        ScaleAnchor(value=0, label="watermarked",
                                    description="Any watermark token (turn.../oaicite/utm_source=openai/chatgpt) or placeholder text."),
                        ScaleAnchor(value=1, label="quirky",
                                    description="Other citation quirks (footnote arrows, bogus reuse) >=2 or a knowledge-cutoff disclaimer."),
                        ScaleAnchor(value=2, label="minor",
                                    description="Single minor quirk only."),
                        ScaleAnchor(value=3, label="clean",
                                    description="No artifacts or quirks."),
                    ],
                ),
                weight=3.0,
            ),
        ],
        disqualifiers=[
            Disqualifier(
                id="DQ1",
                description="Presence of explicit AI self-disclosure (ai_disclaimer) -> auto-fail.",
                pattern=r"\b(as an? (?:ai|large language) model|up to my last (?:training|knowledge) update|i cannot (?:browse|access)|i can(?:not|'t) directly)\b",
            ),
            Disqualifier(
                id="DQ2",
                description="Presence of watermark tokens (turn_tokens|oaicite|utm_openai|attr_json) -> auto-fail.",
                pattern=r"\boaicite\b|contentReference\[oaicite:\d+\]|\bturn\d+(?:search|image|view)\d+\b|\butm_source=(?:chatgpt\.com|openai)\b|\(\{\"attribution\":\{\"attributableIndex\":\"\d+-\d+\"\}\}\)",
            ),
            Disqualifier(
                id="DQ3",
                description="Placeholder scaffolding (placeholder_text) -> auto-fail.",
                pattern=r"\[(?:URL of source|Insert [^\]]+|Describe [^\]]+)\]",
            ),
        ],
        instructions=[
            "If any DQ fired => score=0, risk=15, band='FAIL'.",
            "Else: score = C1+C2+C3+C4+C5 (0-15, higher is cleaner).",
            "risk = 15 - score (higher means more LLM-y).",
            "band = risk>=12 -> 'Severe'; 8-11 -> 'High'; 4-7 -> 'Moderate'; 1-3 -> 'Low'; 0 -> 'Minimal'.",
        ],
        patterns=[
            # Tone / puffery / editorializing
            PatternEntry(
                id="puffery_words",
                pattern=r"\b(stunning|breathtaking|must[- ]?(see|visit)|rich (?:cultural )?heritage|enduring(?:\s+legacy)?|nestled|in the heart of|watershed moment|stands as|serves as|is a testament|plays a (?:vital|significant) role|continues to captivate|solidifies)\b",
            ),
            PatternEntry(
                id="editorialize",
                pattern=r"\b(it'?s (?:important|worth) (?:to note|noting)|no discussion would be complete|this (?:article|section) (?:wouldn'?t|would not) exist without)\b",
            ),
            PatternEntry(
                id="weasel",
                pattern=r"\b(some (?:critics|observers|commentators) (?:argue|say|believe)|many (?:believe|say)|industry (?:reports|analysts) (?:suggest|say))\b",
            ),
            PatternEntry(
                id="superficial_ing",
                pattern=r"\b(?:ensuring|highlighting|emphasizing|reflecting|underscoring)\b",
            ),
            # Formulaic scaffolding
            PatternEntry(
                id="conjunction_overuse",
                pattern=r"\b(on the other hand|moreover|in addition|furthermore|however)\b",
            ),
            PatternEntry(
                id="section_summaries",
                pattern=r"\b(in summary|in conclusion|overall)\b",
            ),
            PatternEntry(
                id="despite_challenges",
                pattern=r"\bdespite (?:its|these).+faces? .+challenges\b",
            ),
            PatternEntry(
                id="negative_parallelism",
                pattern=r"\bnot only\b|it'?s not (?:just|only)|\bno .+?, no .+?, just\b",
            ),
            PatternEntry(
                id="rule_of_three",
                pattern=r"\b\w+(?:ly)?[,\uff0c]\s+\w+(?:ly)?[,\uff0c]\s+(?:and\s+)?\w+(?:ly)?\b",
            ),
            # Meta-communication / AI tells
            PatternEntry(
                id="chatty_meta",
                pattern=r"\b(certainly!|of course!|i hope this helps|would you like|let me know|here'?s a|here is a|in this section we will|this draft|according to wikipedia|wikipedia (?:policies|guidelines))\b",
            ),
            PatternEntry(
                id="ai_disclaimer",
                pattern=r"\b(as an? (?:ai|large language) model|up to my last (?:training|knowledge) update|i cannot (?:browse|access)|i can(?:not|'t) directly)\b",
            ),
            PatternEntry(
                id="letter_form",
                pattern=r"\b(?:subject:|dear (?:wikipedia|editors|administrators))\b",
            ),
            # Markup / formatting artifacts
            PatternEntry(
                id="markdown_headings",
                pattern=r"(^|\n)#{1,6}\s+\S+",
            ),
            PatternEntry(
                id="list_bullets",
                pattern=r"(^|\n)\s*(?:\u2022|\u2013|-|\d+\.)\s+\S+",
            ),
            PatternEntry(
                id="emoji",
                pattern=r"[\u2190-\u21FF\u2300-\u27BF\u2B00-\u2BFF\U0001F300-\U0001FAFF]",
            ),
            PatternEntry(
                id="curly_quotes",
                pattern=r"[\u201c\u201d\u2018\u2019]",
            ),
            PatternEntry(
                id="em_dash",
                pattern=r"\u2014",
            ),
            PatternEntry(
                id="title_case_heading",
                pattern=r"(^|\n)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}\s*\n",
            ),
            # Watermarks / artifacts
            PatternEntry(
                id="oaicite",
                pattern=r"\boaicite\b|contentReference\[oaicite:\d+\]",
            ),
            PatternEntry(
                id="turn_tokens",
                pattern=r"\bturn\d+(?:search|image|view)\d+\b|[\ue000-\uf8ff]cite[\ue000-\uf8ff]turn\d+\w+\d+[\ue000-\uf8ff]",
            ),
            PatternEntry(
                id="utm_openai",
                pattern=r"\butm_source=(?:chatgpt\.com|openai)\b",
            ),
            PatternEntry(
                id="attr_json",
                pattern=r'\(\{"attribution":\{"attributableIndex":"\d+-\d+"\}\}\)',
            ),
            PatternEntry(
                id="footnote_arrow",
                pattern=r"\u21a9",
            ),
            PatternEntry(
                id="placeholder_text",
                pattern=r"\[(?:URL of source|Insert [^\]]+|Describe [^\]]+)\]",
            ),
            # Citation quirks
            PatternEntry(
                id="fake_ref_reuse",
                pattern=r"<ref name=.*?/>.*?<ref name=.*?>",
            ),
            PatternEntry(
                id="named_ref_in_refs",
                pattern=r"(<|&lt;)references(>|&gt;).*(<|&lt;)ref name=.*?(>|&gt;)",
            ),
            # Knowledge-cutoff
            PatternEntry(
                id="cutoff_claim",
                pattern=r"\bas of (?:\w+\s+\d{4}|[A-Z][a-z]+ \d{4})\b.*?(?:not widely (?:available|documented)|limited information|based on available information)\b",
            ),
        ],
        advice_rules=[
            AdviceRule(fix="Replace hype with concrete facts; remove evaluatives."),
            AdviceRule(fix="Attribute claims to named sources or delete vague attributions."),
            AdviceRule(fix="Cut templatey sentences; vary connectors; remove summary/conclusion boilerplate."),
            AdviceRule(fix="Delete direct address and helper language; keep encyclopedic voice."),
            AdviceRule(fix="Remove AI self-disclosure and capability disclaimers."),
            AdviceRule(fix="Convert headings/lists to house style; sentence-case headings."),
            AdviceRule(fix="Remove emoji; normalize quotes/apostrophes; limit em dashes."),
            AdviceRule(fix="Delete watermarks/placeholders; replace with real citations or omit."),
        ],
    )

    # Role
    role = RoleSpec(
        id="anti_slop_judge",
        persona="Anti-slop detector and writing quality auditor",
        authority="absolute",
        domain="LLM-generated text detection",
        obligations=[
            "Score each criterion C1-C5 based on mechanical pattern hits.",
            "Fire disqualifiers immediately on any watermark or AI disclosure.",
            "Compute risk = 15 - score.",
        ],
        constraints=[
            "Output JSON only; no prose outside the JSON object.",
            "Fixed key order: score, risk, band, rationale, evidence, violations, criterion_scores, advice.",
            "Rationale must begin with 'BECAUSE:' and be exactly 35 words.",
            "Advice must begin with 'FIX:' and contain exactly 5 semicolon-separated imperatives (<=220 chars).",
        ],
    )

    # Custom decision thresholds based on risk bands (inverted scoring).
    # The judge engine uses normalized_score (0-100). For AntiLLMY, higher
    # score = cleaner. We map the normalized score to risk bands:
    #   score 100 (risk 0)  -> Minimal
    #   score ~80  (risk ~3) -> Low
    #   score ~53  (risk ~7) -> Moderate
    #   score ~27  (risk ~11)-> High
    #   score ~0   (risk ~15)-> Severe / FAIL
    policy = SurfacePolicy(
        input_codec="xml",
        output_codec="json",
        role=role,
        enforce_key_order=True,
        criterion_focus="full",
        execution_strategy="holistic",
        decision_thresholds=[
            (80.0, "Minimal"),      # risk 0-3 (score 12-15)
            (53.4, "Low"),          # risk 4-7 (score 8-11)
            (26.7, "Moderate"),     # risk 8-11 (score 4-7)
            (6.7, "High"),          # risk 12-14 (score 1-3)
            (0.0, "Severe/FAIL"),   # risk 15 (score 0)
        ],
    )

    # Output constraints
    output_constraints = [
        # Rationale style: "Begin with 'BECAUSE:' and use exactly 35 words, end with a period."
        PrefixSuffixConstraint(
            id="because_prefix",
            description="Rationale must begin with 'BECAUSE:' and end with '.'",
            target_field="rationale",
            enforcement="hard",
            prefix="BECAUSE:",
            suffix=".",
        ),
        WordCountConstraint(
            id="rationale_35_words",
            description="Rationale must be exactly 35 words.",
            target_field="rationale",
            enforcement="hard",
            count=35,
            mode="exactly",
        ),
        # Advice style: "Begin with 'FIX:' and provide exactly 5 semicolon-separated imperatives (<=220 chars), ending with a period."
        PrefixSuffixConstraint(
            id="fix_prefix",
            description="Advice must begin with 'FIX:' and end with '.'",
            target_field="advice",
            enforcement="hard",
            prefix="FIX:",
            suffix=".",
        ),
        ItemCountConstraint(
            id="advice_5_imperatives",
            description="Advice must contain exactly 5 semicolon-separated imperatives.",
            target_field="advice",
            enforcement="hard",
            count=5,
            delimiter=";",
            mode="exactly",
        ),
        CharLimitConstraint(
            id="advice_max_220_chars",
            description="Advice must be at most 220 characters total.",
            target_field="advice",
            enforcement="hard",
            limit=220,
            mode="max",
        ),
    ]

    return compile_rubric(rubric, policy=policy, output_constraints=output_constraints)


# ---------------------------------------------------------------------------
# Quick self-test: compile the rubric and print summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Compiling AntiLLMY (anti-slop) rubric...\n")

    aj = anti_slop_judge()
    b = aj.bundle
    print(f"Name:       {b.rubric.meta.name} v{b.rubric.meta.version}")
    print(f"Criteria:   {len(b.rubric.criteria)}")
    print(f"DQs:        {len(b.rubric.disqualifiers)}")
    print(f"Patterns:   {len(b.rubric.patterns)}")
    print(f"Locked:     {b.locked}")
    print(f"Issues:     {aj.issues or '(none)'}")
    print()
    print("Rubric compiled successfully.")
