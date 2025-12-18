"""
Fallom Evals - Run LLM evaluations locally using G-Eval with LLM as a Judge.

Evaluate production outputs or compare different models on your dataset.
Results are uploaded to Fallom dashboard for visualization.

Example:
    from fallom import evals

    # Initialize
    evals.init(api_key="your-fallom-key")

    # Create dataset from your data
    dataset = [
        evals.DatasetItem(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
            system_message="You are a helpful assistant."
        ),
    ]

    # Evaluate production outputs
    results = evals.evaluate(
        dataset=dataset,
        metrics=["answer_relevancy", "faithfulness", "completeness"]
    )

    # Compare with other models
    comparison = evals.compare_models(
        dataset=dataset,
        models=["anthropic/claude-3-5-sonnet", "google/gemini-2.0-flash"],
        metrics=["answer_relevancy", "faithfulness"]
    )
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

# Helper functions
from .helpers import (
    create_openai_model,
    create_custom_model,
    create_model_from_callable,
    custom_metric,
    dataset_from_traces,
    dataset_from_fallom,
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
    "AVAILABLE_METRICS",
    # Prompts
    "METRIC_PROMPTS",
    # Core
    "init",
    "evaluate",
    "compare_models",
    "DEFAULT_JUDGE_MODEL",
    # Helpers
    "create_openai_model",
    "create_custom_model",
    "create_model_from_callable",
    "custom_metric",
    "dataset_from_traces",
    "dataset_from_fallom",
]

