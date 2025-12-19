"""
Fallom tracing module.

Provides session-scoped tracing for LLM calls with automatic
capture of tokens, costs, latency, and content.

Usage:
    import fallom

    # Initialize once at app startup
    fallom.init(api_key="your-api-key")

    # Create a session for each conversation/request
    session = fallom.session(
        config_key="my-app",
        session_id="session-123",
        customer_id="user-456"
    )

    # Wrap your LLM client
    openai = session.wrap_openai(OpenAI())

    # All calls are automatically traced
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )

Legacy Usage (auto-instrumentation):
    The SDK also supports auto-instrumentation for backwards compatibility.
    Import and initialize Fallom BEFORE importing your LLM library.

    import fallom
    fallom.init()

    from openai import OpenAI
    client = OpenAI()

    fallom.trace.set_session("my-app", "session-123")
    # All calls are traced
"""

from .core import (
    init,
    shutdown,
    send_trace,
    send_trace_sync,
    is_initialized,
    should_capture_content,
    get_api_key,
    get_base_url,
    is_debug_mode,
)

from .session import session, FallomSession

from .types import (
    SessionContext,
    SessionOptions,
    TraceContext,
    TraceData,
    WaterfallTimings,
)

from .wrappers import (
    wrap_openai,
    wrap_anthropic,
    wrap_google_ai,
)

# Legacy support: set_session, get_session, clear_session, span
# These use a global context variable for backwards compatibility
import contextvars
from typing import Optional, Dict, Any

_legacy_session_context = contextvars.ContextVar("fallom_legacy_session", default=None)


def set_session(
    config_key: str,
    session_id: str,
    customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
):
    """
    Set the current session context (legacy API).

    All subsequent LLM calls in this thread/async context will be
    automatically tagged with this config_key, session_id, and customer_id.

    For new code, prefer using fallom.session() which is more explicit
    and thread-safe.

    Args:
        config_key: Your config name (e.g., "linkedin-agent")
        session_id: Your session/conversation ID
        customer_id: Optional customer/user identifier for analytics
        metadata: Optional metadata dict
        tags: Optional list of tags

    Example:
        fallom.trace.set_session("linkedin-agent", session_id, customer_id="user_123")
        agent.run(message)  # Automatically traced with session + customer
    """
    ctx = {
        "config_key": config_key,
        "session_id": session_id,
    }
    if customer_id:
        ctx["customer_id"] = customer_id
    if metadata:
        ctx["metadata"] = metadata
    if tags:
        ctx["tags"] = tags
    _legacy_session_context.set(ctx)


def get_session() -> Optional[Dict[str, Any]]:
    """Get current session context, if any (legacy API)."""
    return _legacy_session_context.get()


def clear_session():
    """Clear session context (legacy API)."""
    _legacy_session_context.set(None)


def span(
    data: Dict[str, Any],
    config_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Record custom business metrics. Latest value per field wins.

    Use this for metrics that can't be captured automatically:
    - Outlier scores
    - Engagement metrics
    - Conversion rates
    - Any business-specific outcome

    Args:
        data: Dict of metrics to record
        config_key: Config name (optional if set_session was called)
        session_id: Session ID (optional if set_session was called)

    Examples:
        # If session context is set:
        trace.span({"outlier_score": 0.8, "engagement": 42})

        # Or explicitly:
        trace.span(
            {"outlier_score": 0.8},
            config_key="linkedin-agent",
            session_id="user123-convo456"
        )
    """
    import threading
    import requests
    from .core import get_api_key, get_base_url, is_initialized

    if not is_initialized():
        raise RuntimeError("Fallom not initialized. Call fallom.init() first.")

    # Use context if config_key/session_id not provided
    ctx = _legacy_session_context.get()
    config_key = config_key or (ctx and ctx.get("config_key"))
    session_id = session_id or (ctx and ctx.get("session_id"))

    if not config_key or not session_id:
        raise ValueError(
            "No session context. Either call set_session() first, "
            "or pass config_key and session_id explicitly."
        )

    # Send async to not block
    def _send_span():
        try:
            requests.post(
                f"{get_base_url()}/spans",
                headers={"Authorization": f"Bearer {get_api_key()}"},
                json={
                    "config_key": config_key,
                    "session_id": session_id,
                    "data": data
                },
                timeout=5
            )
        except Exception:
            pass  # Fail silently

    threading.Thread(target=_send_span, daemon=True).start()


__all__ = [
    # Core functions
    "init",
    "shutdown",
    "send_trace",
    "send_trace_sync",
    "is_initialized",
    "should_capture_content",
    "get_api_key",
    "get_base_url",
    "is_debug_mode",
    # Session API
    "session",
    "FallomSession",
    # Types
    "SessionContext",
    "SessionOptions",
    "TraceContext",
    "TraceData",
    "WaterfallTimings",
    # Wrappers
    "wrap_openai",
    "wrap_anthropic",
    "wrap_google_ai",
    # Legacy API
    "set_session",
    "get_session",
    "clear_session",
    "span",
]

