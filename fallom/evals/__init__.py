"""
Fallom Evals - Run LLM evaluations locally using G-Eval with LLM as a Judge.

Evaluate production outputs or compare different models on your dataset.
Results are uploaded to Fallom dashboard for visualization.


"""

# Types
from .types import (
    MetricName,
    MetricInput,
    DatasetInput,
    ModelResponse,
    ModelCallable,
    DatasetItem,
    Model,
    EvalResult,
    CustomMetric,
    Golden,
    LLMTestCase,
    AVAILABLE_METRICS,
)

# Prompts
from .prompts import METRIC_PROMPTS

# Core functions
from .core import (
    init,
    evaluate,
    compare_models,
    DEFAULT_JUDGE_MODEL,
)

# Helper functions and classes
from .helpers import (
    create_openai_model,
    create_custom_model,
    create_model_from_callable,
    custom_metric,
    dataset_from_traces,
    dataset_from_fallom,
    EvaluationDataset,
)

__all__ = [
    # Types
    "MetricName",
    "MetricInput",
    "DatasetInput",
    "ModelResponse",
    "ModelCallable",
    "DatasetItem",
    "Model",
    "EvalResult",
    "CustomMetric",
    "Golden",
    "LLMTestCase",
    "AVAILABLE_METRICS",
    # Prompts
    "METRIC_PROMPTS",
    # Core
    "init",
    "evaluate",
    "compare_models",
    "DEFAULT_JUDGE_MODEL",
    # Helpers & Classes
    "create_openai_model",
    "create_custom_model",
    "create_model_from_callable",
    "custom_metric",
    "dataset_from_traces",
    "dataset_from_fallom",
    "EvaluationDataset",
]

