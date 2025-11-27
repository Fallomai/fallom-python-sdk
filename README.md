# Fallom SDK

Model A/B testing and tracing for LLM applications. Zero latency, production-ready.

## Installation

```bash
pip install fallom

# With auto-instrumentation for your LLM provider:
pip install fallom opentelemetry-instrumentation-openai
pip install fallom opentelemetry-instrumentation-anthropic
```

## Quick Start

```python
import fallom

# Initialize FIRST - before importing your LLM libraries
fallom.init(api_key="your-api-key")

# Set default session context for tracing
fallom.trace.set_session("my-agent", session_id)

# All LLM calls are now automatically traced!
response = openai.chat.completions.create(...)
```

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

## Tracing

Auto-capture all LLM calls with OpenTelemetry instrumentation.

### Automatic Tracing

```python
import fallom

# Initialize before importing LLM libraries
fallom.init()

# Set session context
fallom.trace.set_session("my-agent", session_id)

# All LLM calls automatically traced with:
# - Model, tokens, latency
# - Prompts and completions
# - Your config_key and session_id
response = openai.chat.completions.create(model="gpt-4o", messages=[...])
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

fallom.init()

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
FALLOM_BASE_URL=https://spans.fallom.com  # or http://localhost:8001 for local dev
```

### Initialization Options

```python
fallom.init(
    api_key="your-api-key",      # Or use FALLOM_API_KEY env var
    base_url="https://spans.fallom.com",  # Or use FALLOM_BASE_URL env var
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
- OpenAI
- Anthropic
- Cohere
- AWS Bedrock
- Google Generative AI
- Mistral AI
- LangChain
- Replicate
- Vertex AI

Install the corresponding `opentelemetry-instrumentation-*` package for your provider.

## Examples

See the `examples/` folder for complete examples:
- `basic_usage.py` - Simple A/B testing
- `tracing_only.py` - Just tracing, no A/B testing
- `models_only.py` - Just A/B testing, no tracing
- `batch_job.py` - Recording metrics in batch jobs

## License

MIT
