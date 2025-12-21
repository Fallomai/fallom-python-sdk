"""Helper functions for creating models and datasets."""

import os
import requests
from typing import Optional, List, Dict, Any

from .types import Model, ModelCallable, ModelResponse, DatasetItem, CustomMetric


def create_openai_model(
    model_id: str,
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Model:
    """
    Create a Model using OpenAI directly (for fine-tuned models or Azure OpenAI).

    Args:
        model_id: The OpenAI model ID (e.g., "gpt-4o" or "ft:gpt-4o-2024-08-06:org::id")
        name: Display name for the model (defaults to model_id)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: Custom base URL (for Azure OpenAI or proxies)
        temperature: Temperature for generation
        max_tokens: Max tokens for generation

    Returns:
        Model instance that can be used in compare_models()

    """
    def call_fn(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for create_openai_model(). "
                "Install with: pip install openai"
            )

        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs
        )

        return {
            "content": response.choices[0].message.content or "",
            "tokens_in": response.usage.prompt_tokens if response.usage else None,
            "tokens_out": response.usage.completion_tokens if response.usage else None,
            "cost": None
        }

    return Model(name=name or model_id, call_fn=call_fn)


def create_custom_model(
    name: str,
    endpoint: str,
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    model_field: str = "model",
    model_value: Optional[str] = None,
    **kwargs
) -> Model:
    """
    Create a Model for any OpenAI-compatible API endpoint.

    Works with self-hosted models (vLLM, Ollama, LMStudio, etc.), custom endpoints,
    or any service that follows the OpenAI chat completions API format.

    Args:
        name: Display name for the model
        endpoint: Full URL to the chat completions endpoint
        api_key: API key (passed as Bearer token)
        headers: Additional headers to include
        model_field: Name of the model field in the request (default: "model")
        model_value: Value for the model field (defaults to name)
        **kwargs: Additional fields to include in every request

    Returns:
        A Model instance

    """
    def call_fn(messages: List[Dict[str, str]]) -> ModelResponse:
        request_headers = {"Content-Type": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            request_headers.update(headers)

        payload = {
            model_field: model_value or name,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            endpoint,
            headers=request_headers,
            json=payload,
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

    return Model(name=name, call_fn=call_fn)


def create_model_from_callable(
    name: str,
    call_fn: ModelCallable
) -> Model:
    """
    Create a Model from any callable function.

    This is the most flexible option - you provide a function that takes
    messages and returns a response.

    Args:
        name: Display name for the model
        call_fn: Function that takes messages list and returns response dict
                 Messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
                 Response format: {"content": "...", "tokens_in": N, "tokens_out": N, "cost": N}

    Returns:
        A Model instance

    """
    return Model(name=name, call_fn=call_fn)


def custom_metric(
    name: str,
    criteria: str,
    steps: List[str]
) -> CustomMetric:
    """
    Create a custom evaluation metric using G-Eval.

    Args:
        name: Unique identifier for the metric (e.g., "brand_alignment")
        criteria: Description of what the metric evaluates
        steps: List of evaluation steps for the LLM judge to follow

    Returns:
        A CustomMetric instance

    """
    return CustomMetric(name=name, criteria=criteria, steps=steps)


def dataset_from_traces(traces: List[Dict]) -> List[DatasetItem]:
    """
    Create a dataset from Fallom trace data.

    Args:
        traces: List of trace dicts with attributes

    Returns:
        List of DatasetItem ready for evaluation

    """
    items = []

    for trace in traces:
        attrs = trace.get("attributes", {})
        if not attrs:
            continue

        # Extract input (last user message)
        input_text = ""
        for i in range(100):
            role = attrs.get(f"gen_ai.prompt.{i}.role")
            if role is None:
                break
            if role == "user":
                input_text = attrs.get(f"gen_ai.prompt.{i}.content", "")

        # Extract output
        output_text = attrs.get("gen_ai.completion.0.content", "")

        # Extract system message
        system_message = None
        if attrs.get("gen_ai.prompt.0.role") == "system":
            system_message = attrs.get("gen_ai.prompt.0.content")

        if input_text and output_text:
            items.append(DatasetItem(
                input=input_text,
                output=output_text,
                system_message=system_message
            ))

    return items


def dataset_from_fallom(
    dataset_key: str,
    version: Optional[int] = None,
    *,
    _api_key: Optional[str] = None,
    _base_url: Optional[str] = None,
    _initialized: bool = False
) -> List[DatasetItem]:
    """
    Fetch a dataset stored in Fallom by its key.

    Args:
        dataset_key: The unique key of the dataset (e.g., "customer-support-qa")
        version: Specific version number to fetch. If None, fetches the latest version.

    Returns:
        List of DatasetItem ready for evaluation

    Note:
        This function requires init() to be called first, or the internal
        _api_key, _base_url, and _initialized parameters to be passed.
    """
    # Import here to avoid circular dependency
    from . import core

    api_key = _api_key or core._api_key
    base_url = _base_url or core._base_url
    initialized = _initialized or core._initialized

    if not initialized:
        raise RuntimeError("Fallom evals not initialized. Call evals.init() first.")

    # Build URL
    url = f"{base_url}/api/datasets/{dataset_key}"
    params = {}
    if version is not None:
        params["version"] = str(version)

    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        params=params,
        timeout=30
    )

    if response.status_code == 404:
        raise ValueError(f"Dataset '{dataset_key}' not found")
    elif response.status_code == 403:
        raise ValueError(f"Access denied to dataset '{dataset_key}'")

    response.raise_for_status()
    data = response.json()

    # Convert to DatasetItem list
    items = []
    for entry in data.get("entries", []):
        items.append(DatasetItem(
            input=entry["input"],
            output=entry["output"],
            system_message=entry.get("systemMessage"),
            metadata=entry.get("metadata")
        ))

    dataset_name = data.get("dataset", {}).get("name", dataset_key)
    version_num = data.get("version", {}).get("version", "latest")
    print(f"✓ Loaded dataset '{dataset_name}' (version {version_num}) with {len(items)} entries")

    return items


class EvaluationDataset:
    """
    A dataset for evaluation that supports pulling from Fallom and adding test cases.
    
    This provides a workflow where you:
    1. Pull a dataset (goldens) from Fallom
    2. Run your own LLM pipeline on each golden to generate outputs
    3. Add the results as test cases
    4. Evaluate the test cases
    
    """
    
    def __init__(self):
        from .types import Golden, LLMTestCase
        self._goldens: List[Golden] = []
        self._test_cases: List[LLMTestCase] = []
        self._dataset_key: Optional[str] = None
        self._dataset_name: Optional[str] = None
        self._version: Optional[int] = None
    
    @property
    def goldens(self) -> List["Golden"]:
        """List of golden records (inputs with optional expected outputs)."""
        from .types import Golden
        return self._goldens
    
    @property
    def test_cases(self) -> List["LLMTestCase"]:
        """List of test cases (inputs with actual outputs from your LLM)."""
        from .types import LLMTestCase
        return self._test_cases
    
    @property
    def dataset_key(self) -> Optional[str]:
        """The Fallom dataset key if pulled from Fallom."""
        return self._dataset_key
    
    def pull(
        self, 
        alias: str, 
        version: Optional[int] = None
    ) -> "EvaluationDataset":
        """
        Pull a dataset from Fallom.
        
        Args:
            alias: The dataset key/alias in Fallom
            version: Specific version to pull (default: latest)
            
        Returns:
            Self for chaining
            
        """
        from .types import Golden
        from . import core
        
        if not core._initialized:
            raise RuntimeError("Fallom evals not initialized. Call evals.init() first.")
        
        # Fetch from Fallom
        url = f"{core._base_url}/api/datasets/{alias}"
        params = {"include_entries": "true"}
        if version is not None:
            params["version"] = str(version)
        
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {core._api_key}",
                "Content-Type": "application/json"
            },
            params=params,
            timeout=30
        )
        
        if response.status_code == 404:
            raise ValueError(f"Dataset '{alias}' not found")
        elif response.status_code == 403:
            raise ValueError(f"Access denied to dataset '{alias}'")
        
        response.raise_for_status()
        data = response.json()
        
        # Store metadata
        self._dataset_key = alias
        self._dataset_name = data.get("dataset", {}).get("name", alias)
        self._version = data.get("version", {}).get("version")
        
        # Convert entries to goldens
        self._goldens = []
        for entry in data.get("entries", []):
            self._goldens.append(Golden(
                input=entry.get("input", ""),
                expected_output=entry.get("output"),
                system_message=entry.get("systemMessage"),
                metadata=entry.get("metadata")
            ))
        
        print(f"✓ Pulled dataset '{self._dataset_name}' (version {self._version}) with {len(self._goldens)} goldens")
        return self
    
    def add_golden(self, golden: "Golden") -> "EvaluationDataset":
        """
        Add a golden record manually.
        
        Args:
            golden: A Golden instance
            
        Returns:
            Self for chaining
        """
        self._goldens.append(golden)
        return self
    
    def add_goldens(self, goldens: List["Golden"]) -> "EvaluationDataset":
        """
        Add multiple golden records.
        
        Args:
            goldens: List of Golden instances
            
        Returns:
            Self for chaining
        """
        self._goldens.extend(goldens)
        return self
    
    def add_test_case(self, test_case: "LLMTestCase") -> "EvaluationDataset":
        """
        Add a test case with actual LLM output.
        
        Args:
            test_case: An LLMTestCase instance
            
        Returns:
            Self for chaining
        """
        self._test_cases.append(test_case)
        return self
    
    def add_test_cases(self, test_cases: List["LLMTestCase"]) -> "EvaluationDataset":
        """
        Add multiple test cases.
        
        Args:
            test_cases: List of LLMTestCase instances
            
        Returns:
            Self for chaining
        """
        self._test_cases.extend(test_cases)
        return self
    
    def generate_test_cases(
        self,
        llm_app: ModelCallable,
        include_context: bool = False
    ) -> "EvaluationDataset":
        """
        Automatically generate test cases by running all goldens through your LLM app.
        
        Args:
            llm_app: A callable that takes messages and returns response dict
            include_context: Whether to include context from the response metadata
            
        Returns:
            Self for chaining

        """
        from .types import LLMTestCase
        
        print(f"Generating test cases for {len(self._goldens)} goldens...")
        
        for i, golden in enumerate(self._goldens):
            # Build messages
            messages = []
            if golden.system_message:
                messages.append({"role": "system", "content": golden.system_message})
            messages.append({"role": "user", "content": golden.input})
            
            # Call the LLM app
            response = llm_app(messages)
            
            # Create test case
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=response.get("content", ""),
                expected_output=golden.expected_output,
                system_message=golden.system_message,
                context=response.get("context") if include_context else golden.context,
                metadata=golden.metadata
            )
            self._test_cases.append(test_case)
            
            print(f"  [{i+1}/{len(self._goldens)}] Generated output for: {golden.input[:50]}...")
        
        print(f"✓ Generated {len(self._test_cases)} test cases")
        return self
    
    def clear_test_cases(self) -> "EvaluationDataset":
        """Clear all test cases (useful for re-running with different LLM)."""
        self._test_cases = []
        return self
    
    def __len__(self) -> int:
        """Return the number of goldens."""
        return len(self._goldens)
    
    def __repr__(self) -> str:
        return f"EvaluationDataset(goldens={len(self._goldens)}, test_cases={len(self._test_cases)}, key={self._dataset_key})"

