# Fallom SDK

Model A/B testing, prompt management, and tracing for LLM applications. Zero latency, production-ready.

## Installation

```bash
pip install fallom

# With auto-instrumentation for your LLM provider:
pip install fallom opentelemetry-instrumentation-openai
pip install fallom opentelemetry-instrumentation-anthropic
```

## Quick Start

```python
# ⚠️ IMPORTANT: Import and initialize Fallom BEFORE importing OpenAI!
import fallom
fallom.init(api_key="your-api-key")

# NOW import OpenAI (after instrumentation is set up)
from openai import OpenAI
client = OpenAI()

# Set default session context for tracing
fallom.trace.set_session("my-agent", session_id)

# All LLM calls are now automatically traced!
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

> ⚠️ **Import Order Matters!** Auto-instrumentation hooks into libraries when they're imported. You must call `fallom.init()` BEFORE importing `openai`, `anthropic`, etc.

## Model A/B Testing

Run A/B tests on models with zero latency. Same session always gets same model (sticky assignment).

```python
from fallom import models

# Get assigned model for this session
model = models.get("summarizer-config", session_id)
# Returns: "gpt-4o" or "claude-3-5-sonnet" based on your config weights

agent = Agent(model=model)
agent.run(message)
```

### Version Pinning

Pin to a specific config version, or use latest (default):

```python
# Use latest version (default)
model = models.get("my-config", session_id)

# Pin to specific version
model = models.get("my-config", session_id, version=2)
```

### Fallback for Resilience

Always provide a fallback so your app works even if Fallom is down:

```python
model = models.get(
    "my-config", 
    session_id, 
    fallback="gpt-4o-mini"  # Used if config not found or Fallom unreachable
)
```

**Resilience guarantees:**
- Short timeouts (1-2 seconds max)
- Background config sync (never blocks your requests)
- Graceful degradation (returns fallback on any error)
- Your app is never impacted by Fallom being down

## Prompt Management

Manage prompts centrally and A/B test them with zero latency.

### Basic Prompt Retrieval

```python
from fallom import prompts

# Get a managed prompt (with template variables)
prompt = prompts.get("onboarding", variables={
    "user_name": "John",
    "company": "Acme"
})

# Use the prompt with any LLM
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user}
    ]
)
```

The `prompt` object contains:
- `key`: The prompt key
- `version`: The prompt version
- `system`: The system prompt (with variables replaced)
- `user`: The user template (with variables replaced)

### Prompt A/B Testing

Run experiments on different prompt versions:

```python
from fallom import prompts

# Get prompt from A/B test (sticky assignment based on session_id)
prompt = prompts.get_ab("onboarding-test", session_id, variables={
    "user_name": "John"
})

# prompt.ab_test_key and prompt.variant_index are set
# for analytics in your dashboard
```

### Version Pinning

```python
# Use latest version (default)
prompt = prompts.get("my-prompt")

# Pin to specific version
prompt = prompts.get("my-prompt", version=2)
```

### Automatic Trace Tagging

When you call `prompts.get()` or `prompts.get_ab()`, the next LLM call is automatically tagged with the prompt information. This allows you to see which prompts are used in your traces without any extra code.

```python
# Get prompt - sets up auto-tagging for next LLM call
prompt = prompts.get("onboarding", variables={"user_name": "John"})

# This call is automatically tagged with prompt_key, prompt_version, etc.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user}
    ]
)
```

## Tracing

Auto-capture all LLM calls with OpenTelemetry instrumentation.

> ⚠️ **Important:** Auto-tracing only works with supported LLM SDKs (OpenAI, Anthropic, etc.) - not raw HTTP requests. If you're using an OpenAI-compatible API like OpenRouter, LiteLLM, or a self-hosted model, use the OpenAI SDK with a custom `base_url`:
>
> ```python
> from openai import OpenAI
> 
> # OpenRouter, LiteLLM, vLLM, etc.
> client = OpenAI(
>     base_url="https://openrouter.ai/api/v1",  # or your provider's URL
>     api_key="your-provider-key"
> )
> 
> # Now this call will be auto-traced!
> response = client.chat.completions.create(model="gpt-4o", messages=[...])
> ```

### Automatic Tracing

```python
# Step 1: Import and init Fallom FIRST
import fallom
fallom.init()

# Step 2: Import OpenAI AFTER init
from openai import OpenAI
client = OpenAI()

# Set session context
fallom.trace.set_session("my-agent", session_id)

# Step 3: All LLM calls automatically traced with:
# - Model, tokens, latency
# - Prompts and completions
# - Your config_key and session_id
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

### Custom Metrics

Record business metrics that OTEL can't capture automatically:

```python
from fallom import trace

# Record custom metrics for this session
trace.span({
    "outlier_score": 0.8,
    "user_satisfaction": 4,
    "conversion": True
})

# Or explicitly specify session (for batch jobs)
trace.span(
    {"outlier_score": 0.8},
    config_key="my-agent",
    session_id="user123-convo456"
)
```

## Multiple A/B Tests in One Workflow

If you have multiple LLM calls and only want to A/B test some of them:

```python
import fallom

fallom.init(api_key="your-api-key")

# Set default context for all tracing
fallom.trace.set_session("my-agent", session_id)

# ... regular LLM calls (traced as "my-agent") ...

# A/B test a specific call - models.get() auto-updates context
model = models.get("summarizer-test", session_id, fallback="gpt-4o")
result = summarizer.run(model=model)

# Reset context back to default
fallom.trace.set_session("my-agent", session_id)

# ... more regular LLM calls (traced as "my-agent") ...
```

## Configuration

### Environment Variables

```bash
FALLOM_API_KEY=your-api-key
FALLOM_TRACES_URL=https://traces.fallom.com
FALLOM_CONFIGS_URL=https://configs.fallom.com
FALLOM_PROMPTS_URL=https://prompts.fallom.com
```

### Initialization Options

```python
fallom.init(
    api_key="your-api-key",      # Or use FALLOM_API_KEY env var
    # URLs default to production, override for local dev:
    # traces_url="http://localhost:3002",
    # configs_url="http://localhost:3003",
    # prompts_url="http://localhost:3004",
    capture_content=True         # Set False for privacy mode
)
```

### Privacy Mode

For companies with strict data policies, disable prompt/completion capture:

```python
# Via parameter
fallom.init(capture_content=False)

# Or via environment variable
# FALLOM_CAPTURE_CONTENT=false
```

In privacy mode, Fallom still tracks:
- ✅ Model used
- ✅ Token counts
- ✅ Latency
- ✅ Session/config context
- ✅ Prompt key/version (metadata only)
- ❌ Prompt content (not captured)
- ❌ Completion content (not captured)

## API Reference

### `fallom.init(api_key?, base_url?, capture_content?)`
Initialize the SDK. Call this before importing LLM libraries for auto-instrumentation.

- `capture_content`: Whether to capture prompt/completion text (default: `True`)

### `fallom.models.get(config_key, session_id, version?, fallback?) -> str`
Get model assignment for a session.
- `config_key`: Your config name from the dashboard
- `session_id`: Unique session/conversation ID (sticky assignment)
- `version`: Pin to specific version (default: latest)
- `fallback`: Model to return if anything fails

### `fallom.prompts.get(prompt_key, variables?, version?) -> PromptResult`
Get a managed prompt.
- `prompt_key`: Your prompt key from the dashboard
- `variables`: Dict of template variables (e.g., `{"user_name": "John"}`)
- `version`: Pin to specific version (default: latest)
- Returns: `PromptResult` with `key`, `version`, `system`, `user`

### `fallom.prompts.get_ab(ab_test_key, session_id, variables?) -> PromptResult`
Get a prompt from an A/B test (sticky assignment).
- `ab_test_key`: Your A/B test key from the dashboard
- `session_id`: Unique session/conversation ID (for sticky assignment)
- `variables`: Dict of template variables
- Returns: `PromptResult` with `key`, `version`, `system`, `user`, `ab_test_key`, `variant_index`

### `fallom.trace.set_session(config_key, session_id)`
Set trace context. All subsequent LLM calls will be tagged with this config_key and session_id.

### `fallom.trace.clear_session()`
Clear trace context.

### `fallom.trace.span(data, config_key?, session_id?)`
Record custom business metrics.
- `data`: Dict of metrics to record
- `config_key`: Optional if `set_session()` was called
- `session_id`: Optional if `set_session()` was called

## Supported LLM Providers

Auto-instrumentation available for:
- OpenAI (+ OpenAI-compatible APIs: OpenRouter, LiteLLM, vLLM, Ollama, etc.)
- Anthropic
- Cohere
- AWS Bedrock
- Google Generative AI
- Mistral AI
- LangChain
- Replicate
- Vertex AI

Install the corresponding `opentelemetry-instrumentation-*` package for your provider.

**Note:** You must use the official SDK for your provider. Raw HTTP requests (e.g., `requests.post()`) will not be traced. For OpenAI-compatible APIs, use the OpenAI SDK with a custom `base_url`.

## Testing

Run the test suite:

```bash
cd sdk/python-sdk
pip install pytest
pytest tests/ -v
```

## Deploying

To publish a new version to PyPI:

```bash
cd sdk/python-sdk

# Update version in pyproject.toml
# Then:
pip install build twine
python -m build
twine upload dist/*
```

## Examples

See the `examples/` folder for complete examples:
- `basic_usage.py` - Simple A/B testing
- `tracing_only.py` - Just tracing, no A/B testing
- `models_only.py` - Just A/B testing, no tracing
- `batch_job.py` - Recording metrics in batch jobs

## License

MIT
