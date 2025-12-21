"""Core evaluation functions."""

import os
import json
import time
import requests
from typing import Optional, List, Dict, Any, Union

from .types import (
    MetricInput,
    DatasetInput,
    DatasetItem,
    Model,
    EvalResult,
    CustomMetric,
    LLMTestCase,
    AVAILABLE_METRICS,
)
from .prompts import METRIC_PROMPTS, build_g_eval_prompt
from .helpers import dataset_from_fallom

# Type alias for evaluate() input
TestCaseInput = Union[List[LLMTestCase], List[DatasetItem], str]

# Module state
_api_key: Optional[str] = None
_base_url: str = "https://app.fallom.com"
_initialized: bool = False

# Default judge model (via OpenRouter)
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def init(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> None:
    """
    Initialize Fallom evals.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to https://app.fallom.com

    Example:
        from fallom import evals
        evals.init(api_key="your-api-key")
    """
    global _api_key, _base_url, _initialized

    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = base_url or os.environ.get("FALLOM_BASE_URL", "https://app.fallom.com")

    if not _api_key:
        raise ValueError(
            "No API key provided. Set FALLOM_API_KEY environment variable "
            "or pass api_key parameter."
        )

    _initialized = True


def _run_g_eval(
    metric: MetricInput,
    input_text: str,
    output_text: str,
    system_message: Optional[str],
    judge_model: str
) -> tuple:
    """
    Run G-Eval for a single metric using OpenRouter.

    G-Eval uses chain-of-thought prompting where the LLM:
    1. Follows evaluation steps
    2. Provides reasoning
    3. Gives a final score

    Returns:
        tuple: (score, reasoning)
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable required for evaluations."
        )

    # Get metric config - either from built-in or custom metric
    if isinstance(metric, CustomMetric):
        criteria = metric.criteria
        steps = metric.steps
    else:
        metric_config = METRIC_PROMPTS[metric]
        criteria = metric_config["criteria"]
        steps = metric_config["steps"]

    prompt = build_g_eval_prompt(criteria, steps, system_message, input_text, output_text)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    result = json.loads(data["choices"][0]["message"]["content"])
    return result["score"], result["overall_reasoning"]


def _resolve_dataset(dataset_input: DatasetInput) -> List[DatasetItem]:
    """Resolve dataset input - either use directly or fetch from Fallom."""
    if isinstance(dataset_input, str):
        # It's a dataset key - fetch from Fallom
        return dataset_from_fallom(
            dataset_input,
            _api_key=_api_key,
            _base_url=_base_url,
            _initialized=_initialized
        )
    return dataset_input


def _call_model_openrouter(
    model_slug: str,
    messages: List[Dict],
    kwargs: Dict
) -> Dict[str, Any]:
    """Call a model via OpenRouter."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required for model comparison")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_slug,
            "messages": messages,
            **kwargs
        },
        timeout=120
    )
    response.raise_for_status()
    data = response.json()

    return {
        "content": data["choices"][0]["message"]["content"],
        "tokens_in": data.get("usage", {}).get("prompt_tokens"),
        "tokens_out": data.get("usage", {}).get("completion_tokens"),
        "cost": data.get("usage", {}).get("total_cost")
    }


def evaluate(
    dataset: Optional[DatasetInput] = None,
    metrics: Optional[List[MetricInput]] = None,
    judge_model: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    verbose: bool = True,
    test_cases: Optional[List[LLMTestCase]] = None,
    _skip_upload: bool = False
) -> List[EvalResult]:
    """
    Evaluate outputs against specified metrics using G-Eval.

    Results are automatically uploaded to Fallom dashboard.

    Args:
        dataset: Either a list of DatasetItem OR a string (dataset key to fetch from Fallom)
        metrics: List of metrics to run (built-in or custom). Default: all built-in metrics
        judge_model: Model to use as judge via OpenRouter (default: openai/gpt-4o-mini)
        name: Name for this evaluation run (auto-generated if not provided)
        description: Optional description
        verbose: Print progress
        test_cases: List of LLMTestCase (alternative to dataset - for custom LLM pipeline outputs)

    Returns:
        List of EvalResult with scores for each item

    """
    # Handle test_cases input (convert to DatasetItem format)
    if test_cases is not None:
        resolved_dataset = [
            DatasetItem(
                input=tc.input,
                output=tc.actual_output,
                system_message=tc.system_message,
                metadata=tc.metadata
            )
            for tc in test_cases
        ]
    elif dataset is not None:
        # Resolve dataset - fetch from Fallom if it's a string
        resolved_dataset = _resolve_dataset(dataset)
    else:
        raise ValueError("Either 'dataset' or 'test_cases' must be provided")

    # Use default judge model if not specified
    judge_model = judge_model or DEFAULT_JUDGE_MODEL

    if metrics is None:
        metrics = list(AVAILABLE_METRICS)

    # Validate built-in metrics (custom metrics don't need validation)
    for m in metrics:
        if isinstance(m, str) and m not in AVAILABLE_METRICS:
            raise ValueError(f"Invalid metric: {m}. Available: {AVAILABLE_METRICS}. Or use CustomMetric for custom metrics.")

    results = []

    for i, item in enumerate(resolved_dataset):
        if verbose:
            print(f"Evaluating item {i+1}/{len(resolved_dataset)}...")

        result = EvalResult(
            input=item.input,
            output=item.output,
            system_message=item.system_message,
            model="production",
            is_production=True,
            reasoning={}
        )

        # Run each metric
        for metric in metrics:
            # Get metric name for display and storage
            metric_name = metric.name if isinstance(metric, CustomMetric) else metric

            if verbose:
                print(f"  Running {metric_name}...")

            try:
                score, reasoning = _run_g_eval(
                    metric=metric,
                    input_text=item.input,
                    output_text=item.output,
                    system_message=item.system_message,
                    judge_model=judge_model
                )

                setattr(result, metric_name, score)
                result.reasoning[metric_name] = reasoning
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                setattr(result, metric_name, None)
                result.reasoning[metric_name] = f"Error: {str(e)}"

        results.append(result)

    if verbose:
        _print_summary(results, metrics)

    # Auto-upload to Fallom (unless called from compare_models)
    if not _skip_upload:
        if _initialized:
            run_name = name or f"Production Eval {time.strftime('%Y-%m-%d %H:%M')}"
            _upload_results(results, run_name, description, judge_model, verbose)
        elif verbose:
            print("\n⚠️  Fallom not initialized - results not uploaded. Call evals.init() to enable auto-upload.")

    return results


def compare_models(
    dataset: DatasetInput,
    models: List[Union[str, Model]],
    metrics: Optional[List[MetricInput]] = None,
    judge_model: Optional[str] = None,
    include_production: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[EvalResult]]:
    """
    Compare multiple models on the same dataset.

    Results are automatically uploaded to Fallom dashboard.

    Args:
        dataset: Either a list of DatasetItem OR a string (dataset key to fetch from Fallom)
        models: List of models to test. Each can be:
            - A string (model slug for OpenRouter, e.g., "anthropic/claude-3-5-sonnet")
            - A Model instance (for custom/fine-tuned models)
        metrics: List of metrics to run (built-in or custom)
        judge_model: Model to use as judge via OpenRouter (default: openai/gpt-4o-mini)
        include_production: Include production outputs in comparison
        model_kwargs: Additional kwargs for OpenRouter model calls (temperature, max_tokens, etc.)
        name: Name for this evaluation run (auto-generated if not provided)
        description: Optional description
        verbose: Print progress

    Returns:
        Dict mapping model name to list of EvalResults

    """
    # Resolve dataset - fetch from Fallom if it's a string
    resolved_dataset = _resolve_dataset(dataset)

    # Use default judge model if not specified
    judge_model = judge_model or DEFAULT_JUDGE_MODEL

    if metrics is None:
        metrics = list(AVAILABLE_METRICS)

    model_kwargs = model_kwargs or {}
    results: Dict[str, List[EvalResult]] = {}

    # Evaluate production outputs first
    if include_production:
        if verbose:
            print("\n=== Evaluating Production Outputs ===")
        results["production"] = evaluate(
            dataset=resolved_dataset,
            metrics=metrics,
            judge_model=judge_model,
            verbose=verbose,
            _skip_upload=True
        )

    # Run each model
    for model_input in models:
        # Normalize to Model object
        if isinstance(model_input, str):
            model = Model(name=model_input, call_fn=None)
        else:
            model = model_input

        if verbose:
            print(f"\n=== Testing Model: {model.name} ===")

        model_results = []

        for i, item in enumerate(resolved_dataset):
            if verbose:
                print(f"Item {i+1}/{len(resolved_dataset)}: Generating output...")

            # Generate output from model
            start = time.time()

            messages = []
            if item.system_message:
                messages.append({"role": "system", "content": item.system_message})
            messages.append({"role": "user", "content": item.input})

            try:
                # Call the model - either custom function or OpenRouter
                if model.call_fn is not None:
                    response = model.call_fn(messages)
                else:
                    response = _call_model_openrouter(model.name, messages, model_kwargs)

                latency_ms = int((time.time() - start) * 1000)
                output = response["content"]

                # Create result with generation metadata
                result = EvalResult(
                    input=item.input,
                    output=output,
                    system_message=item.system_message,
                    model=model.name,
                    is_production=False,
                    reasoning={},
                    latency_ms=latency_ms,
                    tokens_in=response.get("tokens_in"),
                    tokens_out=response.get("tokens_out"),
                    cost=response.get("cost")
                )

                # Run metrics
                for metric in metrics:
                    metric_name = metric.name if isinstance(metric, CustomMetric) else metric
                    if verbose:
                        print(f"  Running {metric_name}...")

                    try:
                        score, reasoning = _run_g_eval(
                            metric=metric,
                            input_text=item.input,
                            output_text=output,
                            system_message=item.system_message,
                            judge_model=judge_model
                        )

                        setattr(result, metric_name, score)
                        result.reasoning[metric_name] = reasoning
                    except Exception as e:
                        if verbose:
                            print(f"    Error: {e}")
                        setattr(result, metric_name, None)
                        result.reasoning[metric_name] = f"Error: {str(e)}"

                model_results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  Error generating output: {e}")
                # Create error result
                result = EvalResult(
                    input=item.input,
                    output=f"Error: {str(e)}",
                    system_message=item.system_message,
                    model=model.name,
                    is_production=False,
                    reasoning={"error": str(e)}
                )
                model_results.append(result)

        results[model.name] = model_results

    if verbose:
        _print_comparison_summary(results, metrics)

    # Auto-upload to Fallom
    if _initialized:
        run_name = name or f"Model Comparison {time.strftime('%Y-%m-%d %H:%M')}"
        _upload_results(results, run_name, description, judge_model, verbose)
    elif verbose:
        print("\n⚠️  Fallom not initialized - results not uploaded. Call evals.init() to enable auto-upload.")

    return results


def _print_summary(results: List[EvalResult], metrics: List[MetricInput]) -> None:
    """Print evaluation summary."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)

    for metric in metrics:
        metric_name = metric.name if isinstance(metric, CustomMetric) else metric
        scores = [getattr(r, metric_name, None) for r in results if getattr(r, metric_name, None) is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{metric_name}: {avg:.1%} avg")


def _print_comparison_summary(
    results: Dict[str, List[EvalResult]],
    metrics: List[MetricInput]
) -> None:
    """Print model comparison summary."""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    # Header
    header = f"{'Model':<30}"
    for metric in metrics:
        metric_name = metric.name if isinstance(metric, CustomMetric) else metric
        header += f"{metric_name[:12]:<15}"
    print(header)
    print("-" * 70)

    # Scores for each model
    for model, model_results in results.items():
        row = f"{model:<30}"
        for metric in metrics:
            metric_name = metric.name if isinstance(metric, CustomMetric) else metric
            scores = [getattr(r, metric_name, None) for r in model_results if getattr(r, metric_name, None) is not None]
            if scores:
                avg = sum(scores) / len(scores)
                row += f"{avg:.1%}{'':>9}"
            else:
                row += f"{'N/A':<15}"
        print(row)


def _upload_results(
    results: List[EvalResult] | Dict[str, List[EvalResult]],
    name: str,
    description: Optional[str],
    judge_model: str,
    verbose: bool = True
) -> str:
    """Internal function to upload results to Fallom."""
    # Normalize results format
    if isinstance(results, list):
        all_results = results
    else:
        all_results = []
        for model_results in results.values():
            all_results.extend(model_results)

    # Calculate dataset size (unique input+system_message combinations)
    unique_items = set((r.input, r.system_message or "") for r in all_results)

    # Prepare payload
    payload = {
        "name": name,
        "description": description,
        "dataset_size": len(unique_items),
        "judge_model": judge_model,
        "results": [
            {
                "input": r.input,
                "system_message": r.system_message,
                "model": r.model,
                "output": r.output,
                "is_production": r.is_production,
                "answer_relevancy": r.answer_relevancy,
                "hallucination": r.hallucination,
                "toxicity": r.toxicity,
                "faithfulness": r.faithfulness,
                "completeness": r.completeness,
                "reasoning": r.reasoning,
                "latency_ms": r.latency_ms,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost": r.cost
            }
            for r in all_results
        ]
    }

    try:
        # Upload to Fallom
        response = requests.post(
            f"{_base_url}/api/sdk-evals",
            headers={
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        run_id = data["run_id"]
        dashboard_url = f"{_base_url}/evals/{run_id}"

        if verbose:
            print(f"\n✅ Results uploaded to Fallom! View at: {dashboard_url}")
        return dashboard_url
    except Exception as e:
        if verbose:
            print(f"\n⚠️  Failed to upload results: {e}")
        return ""

