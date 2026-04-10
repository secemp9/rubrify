__version__ = "0.1.0"

from rubrify._behaviors import CONSTRAINT_BEHAVIORS
from rubrify._calibration_suites import (
    ANTI_SLOP_DISCRIMINANT_SUITE,
    COMPLETENESS_FORCING_SUITE,
    COMPLIANCE_JUDGE_SUITE,
)
from rubrify._examples import (
    COMPLETENESS_EXAMPLE,
    EXTRACTION_EXAMPLE,
    TRANSFORM_EXAMPLE,
)
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
    AddMappingExample,
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
from rubrify.calibration import (
    CalibrationCase,
    CalibrationReport,
    CalibrationResult,
    assert_calibration,
    calibration_to_mutations,
    run_calibration_suite,
    run_meta_evaluator_self_calibration,
    summarize_report,
)
from rubrify.client import AnthropicClient, ChatClient, Client, OpenAIClient, OpenRouterClient
from rubrify.generate import (
    generate,
    generate_classifier,
    generate_constraint,
    generate_detector,
    generate_evaluator,
    generate_from_examples,
    generate_transformer,
    refine,
)
from rubrify.improve import ImproveReport, default_advice_extractor, improve_text
from rubrify.input_render import (
    CandidateTextRenderer,
    ConversationJudgeRenderer,
    InputRenderer,
    PassthroughRenderer,
    TemplateRenderer,
    validate_payload,
)
from rubrify.provenance import RefinementReport, RefinementStep, RubricProvenance
from rubrify.repair import (
    RepairResult,
    attempt_schema_repair,
    extract_json_candidate,
    extract_xml_candidate,
)
from rubrify.result import ConstraintResult, EvaluationResult, EvaluationTrace
from rubrify.rubric import ConditionalRubric, ConstraintRubric, ParallelRubric, Rubric


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
    "ParallelRubric",
    "ConditionalRubric",
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
    "ConstraintResult",
    "ValidationResult",
    "AddCriterion",
    "RemoveCriterion",
    "AdjustWeight",
    "AddPattern",
    "AddDisqualifier",
    "AddSteeringConstraint",
    "AddMappingExample",
    "RubricMutation",
    "Client",
    "ChatClient",
    "OpenRouterClient",
    "OpenAIClient",
    "AnthropicClient",
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
    # Phase 2: constraint runtime
    "CONSTRAINT_BEHAVIORS",
    "COMPLETENESS_EXAMPLE",
    "EXTRACTION_EXAMPLE",
    "TRANSFORM_EXAMPLE",
    # Phase 3: calibration as unit testing
    "CalibrationCase",
    "CalibrationResult",
    "CalibrationReport",
    "run_calibration_suite",
    "assert_calibration",
    "summarize_report",
    "run_meta_evaluator_self_calibration",
    "COMPLIANCE_JUDGE_SUITE",
    "ANTI_SLOP_DISCRIMINANT_SUITE",
    "COMPLETENESS_FORCING_SUITE",
    # Phase 4: provenance and lineage
    "RefinementStep",
    "RubricProvenance",
    "RefinementReport",
    "calibration_to_mutations",
    "improve_text",
    "ImproveReport",
    "default_advice_extractor",
    # Phase 5: behavior-oriented generation helpers
    "generate_evaluator",
    "generate_detector",
    "generate_classifier",
    "generate_constraint",
    "generate_transformer",
    "generate_from_examples",
]
