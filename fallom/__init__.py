"""
Fallom - Model A/B testing, prompt management, and tracing for LLM applications.

Usage:
    import fallom
    from fallom import trace, models, prompts

    fallom.init()

    # Model A/B testing
    model = models.get("linkedin-agent", session_id)
    
    # Prompt management (with auto-trace tagging)
    prompt = prompts.get("onboarding", {"user_name": "John"})
    
    # Or prompt A/B testing
    prompt = prompts.get_ab("onboarding-test", session_id, {"user_name": "John"})

    # Use with any LLM
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user}
        ]
    )  # Automatically traced with session + prompt info

    # Later, add custom metrics
    trace.span({"outlier_score": 0.8})
"""

from fallom import trace
from fallom import models
from fallom import prompts

__version__ = "0.1.0"


def init(api_key: str = None, base_url: str = None, capture_content: bool = True):
    """
    Initialize trace, models, and prompts at once.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to FALLOM_BASE_URL env var, or https://spans.fallom.com
        capture_content: Whether to capture prompt/completion content in traces.
                        Set to False for privacy/compliance (only metadata is stored).
                        Defaults to True. Also respects FALLOM_CAPTURE_CONTENT env var.

    Example:
        import fallom
        fallom.init()
        
        # For local development:
        fallom.init(base_url="http://localhost:8001")
        
        # Privacy mode (no prompts/completions stored):
        fallom.init(capture_content=False)
    """
    import os
    base_url = base_url or os.environ.get("FALLOM_BASE_URL", "https://spans.fallom.com")
    trace.init(api_key=api_key, base_url=base_url, capture_content=capture_content)
    models.init(api_key=api_key, base_url=base_url)
    prompts.init(api_key=api_key, base_url=base_url)

