# Fallom SDK

Model A/B testing, prompt management, and tracing for LLM applications. Zero latency, production-ready.

## Installation

```bash
pip install fallom
```

## Quick Start

```python
import fallom
from openai import OpenAI

# Initialize Fallom once at app startup
fallom.init(api_key="your-api-key")

# Create a session for this conversation/request
session = fallom.session(
    config_key="my-app",
    session_id="session-123",
    customer_id="user-456",  # optional
)

# Wrap your LLM client
openai = session.wrap_openai(OpenAI())

# All LLM calls are now automatically traced!
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Providers

Wrap any of these LLM clients:

```python
# OpenAI
openai = session.wrap_openai(OpenAI())

# Anthropic
anthropic = session.wrap_anthropic(Anthropic())

# Google AI
import google.generativeai as genai
genai.configure(api_key="...")
model = genai.GenerativeModel("gemini-1.5-flash")
gemini = session.wrap_google_ai(model)

# OpenRouter (uses OpenAI SDK)
openrouter = session.wrap_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="your-openrouter-key",
    )
)
```

## Model A/B Testing

Run A/B tests on models with zero latency. Same session always gets same model (sticky assignment).

```python
from fallom import models

# Get assigned model for this session
model = models.get("summarizer-config", session_id)
# Returns: "gpt-4o" or "claude-3-5-sonnet" based on your config weights
```

Or use the session's `get_model()` method:

```python
session = fallom.session(
    config_key="summarizer-config",
    session_id=session_id,
)

# Get model for this session's config
model = await session.get_model(fallback="gpt-4o-mini")
```

### Version Pinning

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

### User Targeting

Override weighted distribution for specific users or segments:

```python
model = models.get(
    "my-config",
    session_id,
    fallback="gpt-4o-mini",
    customer_id="user-123",  # For individual targeting
    context={                 # For rule-based targeting
        "plan": "enterprise",
        "region": "us-west"
    }
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
response = openai.chat.completions.create(
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

### Automatic Trace Tagging

When you call `prompts.get()` or `prompts.get_ab()`, the next LLM call is automatically tagged with the prompt information:

```python
# Get prompt - sets up auto-tagging for next LLM call
prompt = prompts.get("onboarding", variables={"user_name": "John"})

# This call is automatically tagged with prompt_key, prompt_version, etc.
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user}
    ]
)
```

## Session Context

Sessions group related LLM calls together (e.g., a conversation or agent run):

```python
session = fallom.session(
    config_key="my-agent",       # Groups traces in dashboard
    session_id="session-123",    # Conversation/request ID
    customer_id="user-456",      # Optional: end-user identifier
    metadata={                   # Optional: custom key-value metadata
        "deployment": "dedicated",
        "request_type": "transcript",
    },
    tags=["production", "high-priority"],  # Optional: simple string tags
)
```

### Concurrent Sessions

Sessions are isolated - safe for concurrent requests:

```python
import concurrent.futures

def handle_request(user_id: str, conversation_id: str):
    session = fallom.session(
        config_key="my-agent",
        session_id=conversation_id,
        customer_id=user_id,
    )

    openai = session.wrap_openai(OpenAI())

    # This session's context is isolated
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )

# Safe to run concurrently!
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(handle_request, "user-1", "conv-1"),
        executor.submit(handle_request, "user-2", "conv-2"),
    ]
```

## Custom Metrics

Record business metrics that can't be captured automatically:

```python
from fallom import trace

# Record custom metrics (requires session context via set_session)
trace.set_session("my-agent", session_id)
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

## Configuration

### Environment Variables

```bash
FALLOM_API_KEY=your-api-key
FALLOM_TRACES_URL=https://traces.fallom.com
FALLOM_CONFIGS_URL=https://configs.fallom.com
FALLOM_PROMPTS_URL=https://prompts.fallom.com
FALLOM_CAPTURE_CONTENT=true  # Set to false for privacy mode
```

### Initialization Options

```python
fallom.init(
    api_key="your-api-key",       # Or use FALLOM_API_KEY env var
    traces_url="...",             # Override traces endpoint
    configs_url="...",            # Override configs endpoint
    prompts_url="...",            # Override prompts endpoint
    capture_content=True,         # Set False for privacy mode
    debug=False,                  # Enable debug logging
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

### `fallom.init(api_key?, traces_url?, configs_url?, prompts_url?, capture_content?, debug?)`
Initialize the SDK. Call this once at app startup.

### `fallom.session(config_key, session_id, customer_id?, metadata?, tags?) -> FallomSession`
Create a session for tracing.
- `config_key`: Your config name from the dashboard
- `session_id`: Unique session/conversation ID
- `customer_id`: Optional user identifier
- `metadata`: Optional dict of custom metadata
- `tags`: Optional list of string tags

### `FallomSession.wrap_openai(client) -> client`
Wrap an OpenAI client for automatic tracing.

### `FallomSession.wrap_anthropic(client) -> client`
Wrap an Anthropic client for automatic tracing.

### `FallomSession.wrap_google_ai(model) -> model`
Wrap a Google AI GenerativeModel for automatic tracing.

### `FallomSession.get_model(config_key?, fallback?, version?) -> str`
Get model assignment for this session.

### `fallom.models.get(config_key, session_id, version?, fallback?, customer_id?, context?) -> str`
Get model assignment for a session.

### `fallom.prompts.get(prompt_key, variables?, version?) -> PromptResult`
Get a managed prompt.

### `fallom.prompts.get_ab(ab_test_key, session_id, variables?) -> PromptResult`
Get a prompt from an A/B test (sticky assignment).

### `fallom.trace.set_session(config_key, session_id, customer_id?, metadata?, tags?)`
Set trace context (legacy API for backwards compatibility).

### `fallom.trace.span(data, config_key?, session_id?)`
Record custom business metrics.

## Legacy API

For backwards compatibility, you can still use the global `set_session()` API with auto-instrumentation:

```python
import fallom
fallom.init()

from openai import OpenAI
client = OpenAI()

fallom.trace.set_session("my-agent", session_id)

# Calls are traced if opentelemetry instrumentation is installed
response = client.chat.completions.create(...)
```

However, we recommend using the new session-based API for:
- Better isolation in concurrent environments
- Explicit wrapping (no import order dependencies)
- Clearer code structure

## Testing

Run the test suite:

```bash
cd sdk/python-sdk
pip install pytest
pytest tests/ -v
```

## License

MIT
