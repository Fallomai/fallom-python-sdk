# CLAUDE.md - Python SDK (fallom)

Python SDK for Fallom - model A/B testing, prompt management, and tracing for LLM applications.

## Deployment Rules

**NEVER push without explicit permission.** Pushes can trigger PyPI publishes.

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode
pip install -e .

# Run tests
pytest tests/
```

## Publishing

```bash
# 1. Update version in pyproject.toml
# 2. Build
python -m build

# 3. Upload to PyPI
twine upload dist/*
```

## Architecture

The SDK uses session-based tracing with manual wrapping:

- **Session pattern**: Each request/conversation creates a session
- **Wrapping**: Wraps OpenAI, Anthropic, Google AI clients
- **Zero-latency A/B testing**: Model assignment happens locally
- **Thread-safe**: Multiple sessions can run in parallel

## Code Style

### Keep APIs Wide

Design APIs to be flexible. Match the TypeScript SDK patterns where possible.

```python
# Good - flexible
session.wrap_openai(client)

# Avoid - too restrictive
session.wrap_openai(client, strict_mode=True, version="v1")
```

### DRY and Simple

- Keep code DRY - extract shared logic
- Prefer simple, readable implementations
- Files over 500 lines likely need refactoring

## Testing with Integration Tests

After making SDK changes, test against the monorepo integration tests:

```bash
# 1. Install SDK in editable mode
pip install -e .

# 2. Run integration tests from the monorepo
cd ../../fallom-monorepo/python-integration-tests
doppler run -- pytest
```

Iterate on SDK changes until integration tests pass.

## Key Files

| File | Purpose |
|------|---------|
| `fallom/__init__.py` | Main entry point |
| `fallom/session.py` | Session management |
| `fallom/wrappers/` | Provider wrappers (OpenAI, Anthropic, etc.) |
| `fallom/models.py` | Model A/B testing |
| `fallom/prompts.py` | Prompt management |
