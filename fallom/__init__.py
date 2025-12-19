"""
Fallom - Model A/B testing, prompt management, tracing, and evals for LLM applications.

Usage:
    import fallom
    from openai import OpenAI

    # Initialize Fallom once at app startup
    fallom.init(api_key="your-api-key")

    # Create a session for this conversation/request
    session = fallom.session(
        config_key="my-app",
        session_id="session-123",
        customer_id="user-456"
    )

    # Wrap your LLM client
    openai = session.wrap_openai(OpenAI())

    # All LLM calls are now automatically traced!
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Model A/B testing
    model_name = await session.get_model(fallback="gpt-4o-mini")

    # Prompt management
    from fallom import prompts
    prompt = prompts.get("onboarding", variables={"user_name": "John"})

    # Run evaluations
    from fallom import evals
    evals.init()
    dataset = [
        evals.DatasetItem(input="...", output="...", system_message="...")
    ]
    results = evals.evaluate(dataset, metrics=["answer_relevancy", "faithfulness"])
    evals.upload_results(results, name="My Eval Run")
"""

from fallom import trace
from fallom import models
from fallom import prompts
from fallom import evals

# Re-export session from trace for convenience
from fallom.trace import session, FallomSession

__version__ = "0.2.0"


def init(
    api_key: str = None,
    traces_url: str = None,
    configs_url: str = None,
    prompts_url: str = None,
    capture_content: bool = True,
    debug: bool = False
):
    """
    Initialize Fallom trace, models, and prompts at once.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        traces_url: Traces API URL. Defaults to FALLOM_TRACES_URL or https://traces.fallom.com
        configs_url: Configs API URL. Defaults to FALLOM_CONFIGS_URL or https://configs.fallom.com
        prompts_url: Prompts API URL. Defaults to FALLOM_PROMPTS_URL or https://prompts.fallom.com
        capture_content: Whether to capture prompt/completion content in traces.
                        Set to False for privacy/compliance (only metadata is stored).
                        Defaults to True. Also respects FALLOM_CAPTURE_CONTENT env var.
        debug: Enable debug logging. Defaults to False.

    Example:
        import fallom

        # Basic initialization
        fallom.init()

        # For local development:
        fallom.init(
            traces_url="http://localhost:3002",
            configs_url="http://localhost:3003",
            prompts_url="http://localhost:3004"
        )

        # Privacy mode (no prompts/completions stored):
        fallom.init(capture_content=False)

        # With debug logging:
        fallom.init(debug=True)
    """
    import os

    _traces_url = traces_url or os.environ.get("FALLOM_TRACES_URL", "https://traces.fallom.com")
    _configs_url = configs_url or os.environ.get("FALLOM_CONFIGS_URL", "https://configs.fallom.com")
    _prompts_url = prompts_url or os.environ.get("FALLOM_PROMPTS_URL", "https://prompts.fallom.com")

    trace.init(api_key=api_key, base_url=_traces_url, capture_content=capture_content, debug=debug)
    models.init(api_key=api_key, base_url=_configs_url)
    prompts.init(api_key=api_key, base_url=_prompts_url)


def shutdown():
    """Shutdown Fallom gracefully."""
    trace.shutdown()
