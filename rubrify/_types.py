from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Criterion:
    id: str  # "C1", "G_SCI", "A_VOX"
    name: str  # "Clarity & Simplicity"
    weight: int = 0  # integer weight (sum to 100 for core)
    anchors: dict[int, str] = field(default_factory=dict)  # {0: "worst", ..., 5: "best"}
    mechanical_rules: list[str] = field(default_factory=list)
    uses_patterns: list[str] | None = None  # ["puffery_words", "weasel"]
    genre: str | None = None  # "science_tech" or "business,email"
    notes: str | None = None

    @property
    def scale(self) -> tuple[int, int]:
        """Infer score range from anchor keys. E.g., (0, 5) for 6 anchors."""
        if not self.anchors:
            return (0, 0)
        return (min(self.anchors), max(self.anchors))


@dataclass(slots=True)
class Disqualifier:
    id: str  # "DQ1"
    description: str  # "Code contains hardcoded credentials."


@dataclass(slots=True)
class OutputSchema:
    format: str = "json"  # "json" or "xml"
    template: str = ""  # JSON template string or XML CDATA template
    constraints: dict[str, str | bool] = field(default_factory=dict)
    # constraints keys: must_be_json, no_prose_outside_json, key_order,
    #   rationale_anchor, edits_requirements, evidence_rule, diagnostic_rule,
    #   must_use_xml_tags, no_text_outside_tags, allowed_judgements, etc.


@dataclass(slots=True)
class Scoring:
    formula: str = ""  # Free prose (model interprets)
    labels: dict[tuple[int, int], str] = field(default_factory=dict)  # {(90,100): "Publish-ready"}
    inverted: bool = False  # True for detection rubrics (higher = cleaner)

    @staticmethod
    def weighted_sum(criteria_ids: list[str], dq_behavior: str = "score=0") -> str:
        """Helper: generate a standard weighted-sum formula string."""
        ids = ", ".join(criteria_ids)
        return f"Sum weighted ({ids}). Normalize to 100. If any DQ: {dq_behavior}."

    @staticmethod
    def inverted_sum(criteria_ids: list[str], max_score: int) -> str:
        """Helper: generate an inverted scoring formula string."""
        ids = "+".join(criteria_ids)
        return (
            f"score = {ids} (0-{max_score}, higher is cleaner). "
            f"risk = {max_score} - score. "
            f"band = f(risk)."
        )


@dataclass(slots=True)
class PatternLibrary:
    entries: dict[str, str] = field(default_factory=dict)  # {id: regex_pattern_string}
    flags: str = ""  # "i" for case-insensitive
    _entry_types: dict[str, str] = field(default_factory=dict)  # {id: "list"|"regex"|"pattern"}
    _group_patterns: dict[str, list[str]] = field(
        default_factory=dict
    )  # Variant 3: individual patterns per group
    # Unified: handles both <pattern_library> (<list>/<regex>) and <regex_library> (<pattern>)
    # All entries are stored as regex strings. <list> pipe-delimited content stored as-is.

    def add(self, pattern_id: str, pattern: str) -> None:
        self.entries[pattern_id] = pattern
        if pattern_id not in self._entry_types:
            self._entry_types[pattern_id] = "regex"

    def add_list(self, list_id: str, pipe_delimited: str) -> None:
        """Add a pipe-delimited word/phrase list (stored as regex alternation)."""
        self.entries[list_id] = pipe_delimited
        self._entry_types[list_id] = "list"


@dataclass(slots=True)
class DecisionRule:
    id: str  # "R1"
    condition: str  # "If any DQ applies => Judgement = No."


@dataclass(slots=True)
class AdviceRule:
    when: list[str]  # ["puffery_words", "editorialize"]
    advice: str  # "Replace hype with concrete facts."


@dataclass(slots=True)
class MappingExample:
    id: str  # "E1"
    user: str | None = None  # user turn text
    assistant: str | None = None  # assistant response text
    verdict: str = ""  # "Yes (meta-prefix neutral; direct fulfillment)."


@dataclass(slots=True)
class ValidationMust:
    description: str  # "Output JSON only in the exact key order."


@dataclass(slots=True)
class InputField:
    name: str  # "candidate_text"
    required: bool = False
    description: str = ""


@dataclass(slots=True)
class Instruction:
    """Instruction primitive for ConstraintRubric and meta-rubric generators.

    Each instruction corresponds to a property predicate from _properties.py.
    The property_name field enables profile-based instruction composition.
    """

    text: str
    property_name: str = ""


@dataclass(slots=True)
class ICLExample:
    """In-context learning example for ConstraintRubric."""

    input: str
    output: str
