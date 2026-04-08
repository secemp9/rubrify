__version__ = "0.1.0"

from rubrify._meta_rubric import (
    COMPLIANCE_GENERATOR,
    COMPLIANCE_PROFILE,
    DETECTION_GENERATOR,
    DETECTION_PROFILE,
    META_CRITERION_TO_PROPERTIES,
    META_EVALUATOR,
    SCORING_GENERATOR,
    SCORING_PROFILE,
    PropertyProfile,
    compose_from_profile,
)
from rubrify._mutations import (
    AddCriterion,
    AddDisqualifier,
    AddPattern,
    AddSteeringConstraint,
    AdjustWeight,
    RemoveCriterion,
    RubricMutation,
)
from rubrify._properties import ValidationResult, suggest_fixes, validate
from rubrify._types import (
    AdviceRule,
    Criterion,
    DecisionRule,
    Disqualifier,
    ICLExample,
    InputField,
    Instruction,
    MappingExample,
    OutputSchema,
    PatternLibrary,
    Scoring,
    ValidationMust,
)
from rubrify.client import ChatClient, Client
from rubrify.generate import generate, refine
from rubrify.input_render import (
    CandidateTextRenderer,
    ConversationJudgeRenderer,
    InputRenderer,
    PassthroughRenderer,
    TemplateRenderer,
    validate_payload,
)
from rubrify.repair import (
    RepairResult,
    attempt_schema_repair,
    extract_json_candidate,
    extract_xml_candidate,
)
from rubrify.result import EvaluationResult, EvaluationTrace
from rubrify.rubric import ConstraintRubric, CoproductRubric, ProductRubric, Rubric


def load(path: str) -> Rubric:
    """Load a rubric from an XML file path."""
    from pathlib import Path

    from rubrify.xml_io import rubric_from_xml

    xml_string = Path(path).read_text(encoding="utf-8")
    return rubric_from_xml(xml_string)


def loads(xml_string: str) -> Rubric:
    """Load a rubric from an XML string."""
    from rubrify.xml_io import rubric_from_xml

    return rubric_from_xml(xml_string)


__all__ = [
    "load",
    "loads",
    "validate",
    "suggest_fixes",
    "generate",
    "refine",
    "Rubric",
    "ConstraintRubric",
    "ProductRubric",
    "CoproductRubric",
    "Criterion",
    "Disqualifier",
    "OutputSchema",
    "Scoring",
    "PatternLibrary",
    "DecisionRule",
    "AdviceRule",
    "MappingExample",
    "ValidationMust",
    "InputField",
    "Instruction",
    "ICLExample",
    "EvaluationResult",
    "EvaluationTrace",
    "ValidationResult",
    "AddCriterion",
    "RemoveCriterion",
    "AdjustWeight",
    "AddPattern",
    "AddDisqualifier",
    "AddSteeringConstraint",
    "RubricMutation",
    "Client",
    "ChatClient",
    "META_EVALUATOR",
    "SCORING_GENERATOR",
    "DETECTION_GENERATOR",
    "COMPLIANCE_GENERATOR",
    "PropertyProfile",
    "SCORING_PROFILE",
    "DETECTION_PROFILE",
    "COMPLIANCE_PROFILE",
    "compose_from_profile",
    "META_CRITERION_TO_PROPERTIES",
    # Phase 1: input renderers
    "InputRenderer",
    "CandidateTextRenderer",
    "ConversationJudgeRenderer",
    "TemplateRenderer",
    "PassthroughRenderer",
    "validate_payload",
    # Phase 1: repair layer
    "RepairResult",
    "extract_json_candidate",
    "extract_xml_candidate",
    "attempt_schema_repair",
]
