"""rubrify — rubric compiler and judge engine for LLM evaluation.

Built on harn_ai for multi-provider LLM access. Works with any model
harn supports: OpenAI, Anthropic, DeepSeek, Google, local proxies, etc.
API keys are auto-discovered from environment variables.

Quick start (hosted provider):

    from harn_ai.models import get_model
    from rubrify import compile_rubric, Judge, JudgeConfig

    result = compile_rubric(rubric)
    judge = Judge(JudgeConfig(model=get_model("deepseek", "deepseek-v4-flash")))
    judgment = await judge.evaluate(result.bundle, "response text")

With a custom OpenAI-compatible proxy (e.g., local vLLM, llama.cpp, LiteLLM):

    from harn_ai.models import get_model

    model = get_model("openai", "gpt-4o").model_copy(update={
        "baseUrl": "http://localhost:8000/v1",
        "api": "openai-completions",
    })
    judge = Judge(JudgeConfig(model=model, api_key="your-key"))
"""

from rubrify.compiler.compiler import compile_rubric, CompilationResult  # noqa: F401
from rubrify.engine.judge import Judge, JudgeConfig  # noqa: F401
from rubrify.engine.judgment import (  # noqa: F401
    AggregatedScore,
    CriterionJudgment,
    EvidenceQuote,
    Judgment,
    JudgeUsage,
)
from rubrify.engine.executor import LLMApiError  # noqa: F401
from rubrify.engine.judge_loop import run_judge_loop  # noqa: F401
from rubrify.ir.bundle import RubricBundle, lock_bundle  # noqa: F401
from rubrify.ir.types import (  # noqa: F401
    AdviceRule, BinaryScale, CalibrationExample, Criterion,
    CriterionGroup, Definition, Disqualifier,
    EvidenceSpec, NominalScale, NumericScale, OrdinalScale,
    PatternEntry, Rubric, RubricMeta, Scale, ScaleAnchor,
)
from rubrify.ir.constraints import (  # noqa: F401
    AuthorityBlock, CharLimitConstraint, ConstraintBinding,
    ItemCountConstraint, OutputConstraint, PrefixSuffixConstraint,
    TokenConstraint, WordCountConstraint,
)
from rubrify.ir.roles import RoleSpec, SurfacePolicy  # noqa: F401

try:
    from rubrify.evolve import (  # noqa: F401
        evolve_rubric, AnnotatedExample, RubricEvolutionResult, RubricEvolutionConfig,
    )
except ImportError:
    pass

__version__ = "0.1.5"
