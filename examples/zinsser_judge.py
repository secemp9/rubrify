#!/usr/bin/env python3
"""ZinsserJudge rubric -- ported from On Writing Well (William Zinsser).

A writing quality judge with genre-conditional criteria, grounded in
Zinsser's principles of simplicity, clarity, unity, strong verbs, clean
usage, structure, voice, and credibility.  The original XML rubric
(on_writing_well_v3.xml) defines three tiers of evaluation:

  12 core criteria  (C1-C12, 0-5 each, weights sum to 100)
      Clarity, Economy, Unity, Leads/Endings, Verbs/Nouns, Usage,
      Modifiers, Rhythm, Paragraphing, Credibility, Audience, Revision.

  10 genre modules  (G_NFL..G_HUM, 0-3 each, genre-conditional)
      Nonfiction-as-Literature, Explainer, Business, Interview, Travel,
      Memoir, Sci/Tech, Sports, Arts, Humor.

  3 attitude lenses (A_VOX, A_CONF, A_DEC, 0-2 each, always active)
      Voice, Enjoyment/Confidence, Writer's Decisions.

  5 disqualifiers, an extensive pattern library for mechanical diagnostics,
  BECAUSE: + 35-word + 3 BEFORE->AFTER output constraints, and tiered
  decision thresholds (Publish-ready .. Fundamentally unclear).

Usage:
    from examples.zinsser_judge import zinsser_judge

    result = zinsser_judge()            # no genre filter
    result = zinsser_judge("travel")    # travel genre modules active
    bundle = result.bundle
    # ... pass to Judge.evaluate()
"""

from __future__ import annotations

from rubrify import (
    CompilationResult,
    Criterion,
    CriterionGroup,
    Disqualifier,
    NumericScale,
    PatternEntry,
    Rubric,
    RubricMeta,
    ScaleAnchor,
    compile_rubric,
)
from rubrify.ir.constraints import PrefixSuffixConstraint, WordCountConstraint
from rubrify.ir.roles import RoleSpec, SurfacePolicy


# ---------------------------------------------------------------------------
# Helper builders (module-level so they can be reused / tested independently)
# ---------------------------------------------------------------------------

def _core_criterion(
    cid: str,
    name: str,
    weight: float,
    anchors: list[str],
    mechanical_rules: list[str] | None = None,
) -> Criterion:
    """Build a 0-5 core criterion with labelled anchors."""
    return Criterion(
        id=cid,
        title=name,
        description=name,
        scale=NumericScale(
            minimum=0, maximum=5, step=1,
            anchors=[
                ScaleAnchor(value=i, label=f"anchor_{i}", description=desc)
                for i, desc in enumerate(anchors)
            ],
        ),
        weight=weight,
        mechanical_rules=mechanical_rules or [],
    )


def _genre_criterion(
    cid: str,
    name: str,
    weight: float,
    genres: list[str],
    anchors: list[str],
    mechanical_rules: list[str] | None = None,
) -> Criterion:
    """Build a 0-3 genre-conditional criterion."""
    return Criterion(
        id=cid,
        title=name,
        description=name,
        scale=NumericScale(
            minimum=0, maximum=3, step=1,
            anchors=[
                ScaleAnchor(value=i, label=f"anchor_{i}", description=desc)
                for i, desc in enumerate(anchors)
            ],
        ),
        weight=weight,
        genre=genres,
        mechanical_rules=mechanical_rules or [],
    )


def _attitude_criterion(
    cid: str,
    name: str,
    weight: float,
    anchors: list[str],
) -> Criterion:
    """Build a 0-2 attitude (process-lens) criterion."""
    return Criterion(
        id=cid,
        title=name,
        description=name,
        scale=NumericScale(
            minimum=0, maximum=2, step=1,
            anchors=[
                ScaleAnchor(value=i, label=f"anchor_{i}", description=desc)
                for i, desc in enumerate(anchors)
            ],
        ),
        weight=weight,
    )


# ---------------------------------------------------------------------------
# Main rubric builder
# ---------------------------------------------------------------------------

def zinsser_judge(genre: str | None = None) -> CompilationResult:
    """ZinsserJudge XXL: evaluate English nonfiction for craft quality
    and reader usefulness, grounded in William Zinsser's principles.

    12 core criteria (C1-C12, 0-5, weighted to sum=100),
    10 genre modules (G_NFL..G_HUM, 0-3, genre-conditional),
    3 attitude lenses (A_VOX, A_CONF, A_DEC, 0-2),
    5 disqualifiers, extensive pattern library,
    BECAUSE: + 35-word + 3 BEFORE->AFTER output constraints,
    and tiered decision thresholds.
    """

    # ---- Core criteria C1-C12 (0-5, weights sum to ~100) ----
    core_criteria = [
        _core_criterion("C1", "Clarity & Simplicity", 13, [
            "Muddy, effortful; meaning obscured.",
            "Frequent haze; key ideas buried.",
            "Mostly clear with rough spots.",
            "Clear; only minor fog.",
            "Clean, simple throughout.",
            "Lean, lucid, respectful of reader's time.",
        ], ["Prefer short, precise words over inflated diction (e.g., now > at this point in time)."]),

        _core_criterion("C2", "Economy & Anti-Clutter", 10, [
            "Windy; filler and jargon dominate.",
            "Heavy clutter; many needless words.",
            "Some clutter; trimming needed.",
            "Mostly tight; occasional puff.",
            "Spare, vigorous; few puff spots.",
            "Every word works; brisk pace.",
        ], [
            "Flag clutter phrases: at this point in time; due to the fact that; in order to; going forward; in terms of; the fact that.",
            "Flag journalese/buzzwords: famed; upcoming; greats; notables; leverage(=use); synergy; impactful; value-add; utilize; capacity planning adds objectivity.",
        ]),

        _core_criterion("C3", "Unity & Focus (Pronoun/Tense/Tone)", 10, [
            "Drift; no controlling idea; tense/person lurches.",
            "Weak throughline; tangents abound.",
            "Mostly on topic; some detours.",
            "Clear throughline; stable voice; minor detours.",
            "Strong unity; choices feel inevitable.",
            "Exemplary cohesion; one main point governs all.",
        ], ["Stabilize person/tense/mood; shifts must be motivated."]),

        _core_criterion("C4", "Structure: Leads & Endings", 11, [
            "Lead dull/confusing; ending fizzles or summarizes lifelessly.",
            "Generic lead; perfunctory 'In conclusion...'.",
            "Workmanlike bookends; limited pull.",
            "Engaging lead; competent landing.",
            "Fresh promise up front; earned last line (snap/echo).",
            "Memorable opening + curtain line; momentum sustained.",
        ], [
            "Flag cliche leads: future archaeologist; visitor from Mars; 'One day not long ago...'; 'have-in-common' lead.",
            "Flag endings that merely summarize; prefer apt, specific final image/implication.",
        ]),

        _core_criterion("C5", "Verbs, Nouns & Sentence Energy", 8, [
            "Flabby verbs; abstract noun piles; passive dominates.",
            "Frequent passives/concept-noun chains.",
            "Mixed energy; avoidable passives.",
            "Mostly active; concrete diction.",
            "Strong verbs + concrete nouns drive rhythm.",
            "Vivid, precise diction; prose has life.",
        ], [
            r"Passive proxy: \b(be|been|being|is|are|was|were)\b\s+\w+ed\b(?:\s+by\b)?",
            "Concept-noun chains: >=3 adjacent nouns (creeping nounism).",
        ]),

        _core_criterion("C6", "Usage & Word Choice (Sense over Fetish)", 8, [
            "Careless usage; cliches/malapropisms.",
            "Many cliches; imprecision.",
            "Some dull phrasing; slips.",
            "Mostly precise/fresh; few slips.",
            "Consistently exact, idiomatic.",
            "Original yet natural; zero pedantry.",
        ], [
            "Don't penalize split infinitives or terminal prepositions when natural.",
            "That/Which rule-of-thumb: 'which' nonrestrictive (comma); otherwise prefer 'that'.",
            "Journalese list: famed; host(v); enthuse; emote; beef up; put teeth into; upcoming, etc.",
        ]),

        _core_criterion("C7", "Modifiers, Qualifiers & Overstatement", 6, [
            "Modifiers smother meaning; hype and hyperbole.",
            "Many weak qualifiers; overstatement.",
            "Occasional over-modifying.",
            "Generally restrained.",
            "Lean, necessary modifiers; proportionate claims.",
            "Exemplary restraint; precision under control.",
        ], [
            "Hedges: a bit; a little; sort of; kind of; pretty much; really; very; quite; rather; somewhat.",
            "Adverb density: >8 '-ly' per 100 words nudges C7 down.",
            "Overstatement flags: literally (for emphasis); 'ten 747s in my brain'.",
        ]),

        _core_criterion("C8", "Rhythm, Cadence & Read-Aloud Flow", 7, [
            "Monotone; tangles at aloud read.",
            "Clunky rhythm; breathless chains.",
            "Mixed cadence; some tongue-trippers.",
            "Readable aloud; varied sentence lengths.",
            "Good musical line; strategic short punch.",
            "Beautiful ear; begs to be read aloud.",
        ], ["Flag sentences >40 words with heavy nesting for split."]),

        _core_criterion("C9", "Paragraphing & Transitions", 6, [
            "Walls of text; choppy fragments; missing pivots.",
            "Weak transitions; uneven para size.",
            "Serviceable flow; lumps remain.",
            "Clear paragraph logic; mood-changers cue shifts.",
            "Strong para rhythm; lucid signposting.",
            "Exemplary flow; reader never lost.",
        ], [
            "Flag paragraphs >180-200 words ('wall') and runs of >=5 single-sentence paragraphs ('choppy').",
            "Mood-changers: but; yet; however; nevertheless; still; instead; thus; therefore; meanwhile; now; later; subsequently.",
        ]),

        _core_criterion("C10", "Credibility, Evidence & Fairness", 7, [
            "Overstatement, shaky facts, misrepresentation.",
            "Assertions sans support; quote abuse.",
            "Some support; occasional overreach.",
            "Fair paraphrase/quotes; proportionate claims.",
            "Evidence marshalled well; factual care obvious.",
            "Scrupulous fairness and specificity; trust earned.",
        ], [
            "Use provided context strictly when present; avoid sensationalism; don't fabricate or infer quotes.",
        ]),

        _core_criterion("C11", "Audience, Inclusivity & Tone", 4, [
            "Condescension; stereotypes/dated terms.",
            "Tone mismatch; insider lingo unexplained.",
            "Generally aware; a few misfires.",
            "Purpose & audience aligned.",
            "Inclusive language; jargon translated.",
            "Reader empathy; guidance without lecturing.",
        ], [
            "Flag: chairman, fireman, policeman, stewardess, coed, poetess; prefer neutral forms.",
            "Detect 'man-hours', 'manned' etc.; suggest neutral alternatives.",
        ]),

        _core_criterion("C12", "Revision Discipline & Anti-Perfectionism", 11, [
            "No sign of pruning; first-draft sprawl.",
            "Little trimming; redundancies remain.",
            "Some pruning; repetition persists.",
            "Clear evidence of editing; repetition reduced.",
            "Read-aloud polish; brackets-and-cut habits evident.",
            "Exemplary revision; sentences feel inevitable; avoids 'tyranny of the final product.'",
        ], [
            "Reward signs of read-aloud, 'quickest fix' cuts, and courage to ship; flag perfectionism that kills clarity.",
        ]),
    ]

    # ---- Genre modules (0-3 each, genre-conditional) ----
    genre_criteria = [
        _genre_criterion("G_NFL", "Nonfiction-as-Literature (Persona + Reporting)", 5,
                         ["nonfiction_literature"], [
                             "Snobbery or mere pose; no reported life.",
                             "Voice present; thin reporting/context.",
                             "Reported specifics + persona in fair balance.",
                             "Vigorous reporting + humane voice; context enlarges the piece.",
                         ]),
        _genre_criterion("G_EXP", "Explainer Pyramid & Sequence", 6,
                         ["explainer", "science_tech", "academic"], [
                             "Jargon wall; no sequence.",
                             "Pieces explained; lacks linear build.",
                             "Starts with key fact; steps mostly logical.",
                             "Bottom-up clarity: one narrow fact -> broader implications; precise analogies; human anchor.",
                         ]),
        _genre_criterion("G_BUS", "Business Plain English & Humanity", 6,
                         ["business", "email"], [
                             "Institutional fog; 'capacity planning adds objectivity.'",
                             "Some humanity; jargon survives.",
                             "Clear, human, respectful; actions/owners/dates.",
                             "Warm, precise, truthful; institution 'warmed up' by a real 'I'.",
                         ]),
        _genre_criterion("G_INT", "Interview Craft & Ethical Quotes", 6,
                         ["interview", "news"], [
                             "Quote dump or distortion/fabrication.",
                             "Weak attributions; context missing.",
                             "Clean attributions; quotes advance ideas; fair trims.",
                             "Deft weaving; 'one person talking to the reader'; ethical edits; vivid specifics.",
                         ], [
                             "Favor quote-first sentences; avoid 'X said that he ...' filler. Don't give away whole plot; don't over-summarize.",
                         ]),
        _genre_criterion("G_TRV", "Travel Specificity & Anti-Travelese", 6,
                         ["travel"], [
                             "Brochure voice; quaint/dappled/roseate; canals are 'romantic'.",
                             "Some detail; many stock phrases.",
                             "Concrete, sensory detail; fresh angle; significant detail only.",
                             "Clear idea of the place; no travelese; selective detail; avoid plot-dump.",
                         ]),
        _genre_criterion("G_MEM", "Memoir Focus, Specifics & Honesty", 6,
                         ["memoir"], [
                             "Diary sprawl; therapy on page.",
                             "Generalities; few specifics; no frame.",
                             "Narrow frame + concrete detail; honest but crafted.",
                             "Selective window; resonant specifics; ethical restraint; avoids self-pity.",
                         ]),
        _genre_criterion("G_SCI", "Sci/Tech Accuracy & Accessibility", 5,
                         ["science_tech"], [
                             "Shaky accuracy; upside-down logic.",
                             "Basics present; heavy jargon.",
                             "Clear sequence; apt analogies; defined terms.",
                             "Complex made simple without dumbing down; human element.",
                         ]),
        _genre_criterion("G_SPO", "Sports: Clarity, Cliches & Human Scale", 5,
                         ["sports"], [
                             "Cliche deluge; stats swamp; writer is the story.",
                             "Over-ornamented lingo; late score reveal.",
                             "Result up front; stats serve story; minimal cliches.",
                             "Fresh verbs; human scale; context beyond the game; avoids self-importance.",
                         ]),
        _genre_criterion("G_ART", "Arts: Context, Criteria & Spoiler Control", 6,
                         ["arts", "column_opinion"], [
                             "Plot dump/snark; ecstatic adjectives; spoilers.",
                             "Opinion with thin criteria/examples.",
                             "Criteria + precise examples; restrained adjectives; minimal spoilers.",
                             "Context in tradition; quotes/details; judgments that travel.",
                         ]),
        _genre_criterion("G_HUM", "Humor Device & Target Ethics", 5,
                         ["humor"], [
                             "Jokes strain; cruelty at the wrong target.",
                             "Device unstable; punchlines telegraphed.",
                             "Consistent device (satire/parody/irony); surprise; restraint.",
                             "Serious point via controlled device; humane target; economy.",
                         ]),
    ]

    # ---- Attitude lenses (0-2 each, always active) ----
    attitude_criteria = [
        _attitude_criterion("A_VOX", "Sound of Your Voice (Natural 'I')", 2, [
            "Stiff institutional voice; afraid of 'I' where apt.",
            "Voice peeks through.",
            "Natural, honest voice; 'I' used judiciously; reader hears a person.",
        ]),
        _attitude_criterion("A_CONF", "Enjoyment, Fear & Confidence", 2, [
            "Fear on the page (throat-clearing, heavy hedging).",
            "Intermittent stiffness.",
            "Relaxed, confident; writes to please self first; reader benefits.",
        ]),
        _attitude_criterion("A_DEC", "A Writer's Decisions", 2, [
            "Unmotivated choices (tense/person/angle).",
            "Some conscious choices evident.",
            "Decisions visible and apt (scope, frame, angle).",
        ]),
    ]

    all_criteria = core_criteria + genre_criteria + attitude_criteria

    # ---- Disqualifiers ----
    disqualifiers = [
        Disqualifier(id="DQ1", description="Safety/policy violation (hate, incitement, etc.)."),
        Disqualifier(id="DQ2", description="Plagiarism/unattributed copying of distinctive language."),
        Disqualifier(id="DQ3", description="Fabrication contradicting provided context (if present)."),
        Disqualifier(id="DQ4", description="Meta/refusal ('As an AI...') or non-schema output.",
                     pattern=r"\bAs an AI\b"),
        Disqualifier(id="DQ5", description="Schema constraint violated (non-JSON, wrong key order, rationale != 35 words)."),
    ]

    # ---- Pattern library ----
    patterns = [
        PatternEntry(id="hedges",
                     pattern=r"\b(a bit|a little|sort of|kind of|pretty much|really|very|quite|rather|somewhat)\b"),
        PatternEntry(id="travelese",
                     pattern=r"\b(quaint|dappled|roseate|nestled|breathtaking|hidden gem|must-see|vibrant|bustling|cozy)\b"),
        PatternEntry(id="journalese",
                     pattern=r"\b(famed|upcoming|greats|notables|host(?:ed|ing)?|enthuse|emote|beef up|put teeth into|impactful|value(?:-| )add|utilize)\b"),
        PatternEntry(id="sports_cliches",
                     pattern=r"\b(southpaw|portsider|circuit clout|twirler|horsehide|knotting the count|twin killing|gridiron great|pigskin|bleacherites)\b"),
        PatternEntry(id="adverb_ly",
                     pattern=r"\b\w+ly\b"),
        PatternEntry(id="passive_proxy",
                     pattern=r"\b(be|been|being|is|are|was|were)\b\s+\w+ed\b(?:\s+by\b)?"),
        PatternEntry(id="concept_chain",
                     pattern=r"\b(\w+)(\s+\w+){2,}\b"),
        PatternEntry(id="which_wo_comma",
                     pattern=r"\bwhich\b(?!\s*,)"),
        PatternEntry(id="exclamation",
                     pattern=r"!"),
        PatternEntry(id="semicolon",
                     pattern=r";"),
        PatternEntry(id="throat_clearing_leads",
                     pattern=r"(When some future archaeologist|If a creature from Mars|One day not long ago|What did .* have in common\?)"),
    ]

    # ---- Groups (core vs genre vs attitude) ----
    core_ids = [f"C{i}" for i in range(1, 13)]
    genre_ids = ["G_NFL", "G_EXP", "G_BUS", "G_INT", "G_TRV", "G_MEM", "G_SCI", "G_SPO", "G_ART", "G_HUM"]
    attitude_ids = ["A_VOX", "A_CONF", "A_DEC"]

    groups = [
        CriterionGroup(
            id="core",
            title="Core Craft Criteria",
            children=core_ids,
            aggregation="weighted_mean",
            weight=100.0,
        ),
        CriterionGroup(
            id="genre",
            title="Genre/Method Modules",
            children=genre_ids,
            aggregation="weighted_mean",
            weight=20.0,
        ),
        CriterionGroup(
            id="attitude",
            title="Attitude Lenses",
            children=attitude_ids,
            aggregation="weighted_mean",
            weight=6.0,
        ),
    ]

    rubric = Rubric(
        meta=RubricMeta(
            name="ZinsserJudge-XXL",
            version="3.0",
            author="researcher",
            description=(
                "Evaluate English nonfiction for craft quality and reader "
                "usefulness, grounded in Zinsser's principles from On Writing Well."
            ),
        ),
        goal=(
            "Ground judgments in Zinsser's principles (simplicity, clarity, unity, "
            "strong verbs, clean usage, structure, voice, credibility), methods "
            "(unity; leads/endings; sentence/paragraph craft), forms (interview, "
            "travel, memoir, science/tech, business, sports, arts, humor), and "
            "attitudes (voice, enjoyment/confidence, anti-perfectionism, sound decisions). "
            "Enforce deterministic JSON output."
        ),
        criteria=all_criteria,
        groups=groups,
        disqualifiers=disqualifiers,
        instructions=[
            "Sum weighted core (C1-C12) + active genre modules (G_*) + attitude lenses (A_*).",
            "Normalize to 100 for class assignment.",
            "If any DQ fires => score=0, class='Rejected'.",
        ],
        patterns=patterns,
    )

    # ---- Role ----
    role = RoleSpec(
        id="zinsser_judge",
        persona="William Zinsser-informed writing coach and evaluator",
        authority="absolute",
        domain="English nonfiction craft evaluation",
        obligations=[
            "Score every core criterion C1-C12.",
            "Activate genre modules only when genre matches.",
            "Always score attitude lenses A_VOX, A_CONF, A_DEC.",
            "Provide exactly 3 BEFORE->AFTER edit pairs.",
            "Quote >= 2 exact phrases from the text as evidence.",
        ],
        constraints=[
            "Output JSON only; no prose outside the JSON object.",
            "Fixed key order: score, class, verdict, subscores, rationale, evidence, actions, diagnostics, violations, meta.",
            "Rationale must be exactly 35 words, prefixed with 'BECAUSE:'.",
        ],
    )

    # ---- Surface policy with Zinsser decision thresholds ----
    policy = SurfacePolicy(
        input_codec="xml",
        output_codec="json",
        role=role,
        enforce_key_order=True,
        criterion_focus="full",
        execution_strategy="grouped",
        decision_thresholds=[
            (90.0, "Publish-ready"),
            (75.0, "Strong draft (light copyedit)"),
            (60.0, "Workable draft (developmental edits)"),
            (40.0, "Needs major revision"),
            (0.0, "Fundamentally unclear"),
        ],
    )

    # ---- Output constraints ----
    output_constraints = [
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
    ]

    return compile_rubric(rubric, policy=policy, output_constraints=output_constraints)


# ---------------------------------------------------------------------------
# Quick self-test: compile the rubric and print summaries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Compiling ZinsserJudge rubric...\n")

    # Without genre
    result = zinsser_judge()
    b = result.bundle
    active_core = [c for c in b.rubric.criteria if c.genre is None]
    genre_crit = [c for c in b.rubric.criteria if c.genre is not None]
    print(f"Name:           {b.rubric.meta.name}")
    print(f"Version:        {b.rubric.meta.version}")
    print(f"Core criteria:  {len(active_core)}")
    print(f"Genre criteria: {len(genre_crit)}")
    print(f"Disqualifiers:  {len(b.rubric.disqualifiers)}")
    print(f"Patterns:       {len(b.rubric.patterns)}")
    print(f"Groups:         {[g.id for g in b.rubric.groups]}")
    print(f"Locked:         {b.locked}")
    print(f"Issues:         {result.issues or '(none)'}")
    print()

    # With genre="travel"
    result_travel = zinsser_judge(genre="travel")
    b2 = result_travel.bundle
    print(f"(with genre='travel')")
    print(f"  Locked:       {b2.locked}")
    print(f"  Issues:       {result_travel.issues or '(none)'}")
