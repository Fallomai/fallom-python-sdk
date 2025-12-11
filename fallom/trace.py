"""
Fallom tracing module.

Auto-instruments all LLM calls via OTEL and groups them by session.
Also supports custom spans for business metrics.
"""
import os
import contextvars
import threading
import requests
from typing import Optional, Dict, Any

# OTEL imports
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_api_key: str = None
_base_url: str = None
_initialized: bool = False
_capture_content: bool = True
_session_context = contextvars.ContextVar("fallom_session", default=None)


class FallomSpanProcessor(SpanProcessor):
    """
    Custom SpanProcessor that injects fallom context into every span.
    This ensures all auto-instrumented LLM calls get our config_key, session_id,
    and prompt context (if prompts.get() or prompts.get_ab() was called).
    """
    
    def on_start(self, span: ReadableSpan, parent_context=None):
        """Called when a span starts - inject our context as attributes."""
        # Inject session context
        ctx = _session_context.get()
        if ctx:
            span.set_attribute("fallom.config_key", ctx.get("config_key", ""))
            span.set_attribute("fallom.session_id", ctx.get("session_id", ""))
            if ctx.get("customer_id"):
                span.set_attribute("fallom.customer_id", ctx["customer_id"])
        
        # Inject prompt context (one-shot - clears after use)
        try:
            from fallom import prompts
            prompt_ctx = prompts.get_prompt_context()
            if prompt_ctx:
                span.set_attribute("fallom.prompt_key", prompt_ctx.get("prompt_key", ""))
                span.set_attribute("fallom.prompt_version", prompt_ctx.get("prompt_version", 0))
                if prompt_ctx.get("ab_test_key"):
                    span.set_attribute("fallom.prompt_ab_test", prompt_ctx.get("ab_test_key", ""))
                    span.set_attribute("fallom.prompt_variant", prompt_ctx.get("variant_index", 0))
                # Clear after injection (one-shot)
                prompts.clear_prompt_context()
        except ImportError:
            pass  # prompts module not available
    
    def on_end(self, span: ReadableSpan):
        """Called when a span ends."""
        pass
    
    def shutdown(self):
        """Shutdown the processor."""
        pass
    
    def force_flush(self, timeout_millis: int = 30000):
        """Force flush any pending spans."""
        return True


def init(api_key: str = None, base_url: str = None, capture_content: bool = True):
    """
    Initialize Fallom tracing. Auto-instruments all LLM calls.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to FALLOM_BASE_URL env var, or https://spans.fallom.com
        capture_content: Whether to capture prompt/completion content in traces.
                        Set to False for privacy/compliance. Defaults to True.
                        Also respects FALLOM_CAPTURE_CONTENT env var ("true"/"false").

    Example:
        from fallom import trace
        
        # Normal usage (captures everything)
        trace.init()

        # Privacy mode (no prompts/completions stored)
        trace.init(capture_content=False)

        trace.set_session("my-agent", session_id)
        agent.run(message)  # Automatically traced
    """
    global _api_key, _base_url, _initialized, _capture_content

    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = base_url or os.environ.get("FALLOM_TRACES_URL", os.environ.get("FALLOM_BASE_URL", "https://traces.fallom.com"))
    
    # Check env var for capture_content (explicit param takes precedence)
    env_capture = os.environ.get("FALLOM_CAPTURE_CONTENT", "").lower()
    if env_capture in ("false", "0", "no"):
        _capture_content = False
    else:
        _capture_content = capture_content
    if not _api_key:
        raise ValueError(
            "No API key provided. Set FALLOM_API_KEY environment variable "
            "or pass api_key parameter."
        )

    _initialized = True

    # Set up OTEL tracer pointing to our endpoint
    provider = TracerProvider()
    
    # Add our custom processor FIRST to inject fallom context into every span
    provider.add_span_processor(FallomSpanProcessor())
    
    # Then add the exporter processor
    exporter = OTLPSpanExporter(
        endpoint=f"{_base_url}/v1/traces",
        headers={"Authorization": f"Bearer {_api_key}"}
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    # Auto-instrument all supported LLM libraries
    _auto_instrument()


def _auto_instrument():
    """
    Automatically instrument all supported LLM/agent libraries.
    Silently skips libraries that aren't installed.
    Respects _capture_content setting for privacy.
    """
    instrumentors = [
        # LLM SDKs
        ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
        ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
        ("opentelemetry.instrumentation.cohere", "CohereInstrumentor"),
        ("opentelemetry.instrumentation.bedrock", "BedrockInstrumentor"),
        ("opentelemetry.instrumentation.google_generativeai", "GoogleGenerativeAiInstrumentor"),
        ("opentelemetry.instrumentation.mistralai", "MistralAiInstrumentor"),
        ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
        ("opentelemetry.instrumentation.replicate", "ReplicateInstrumentor"),
        ("opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor"),
        # HTTP clients (for libraries like agno that use requests/httpx internally)
        ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
    ]

    for module_name, class_name in instrumentors:
        try:
            module = __import__(module_name, fromlist=[class_name])
            instrumentor_class = getattr(module, class_name)
            instrumentor = instrumentor_class()
            
            # Try to pass capture_content setting
            # Different instrumentors use different param names
            try:
                instrumentor.instrument(capture_content=_capture_content)
            except TypeError:
                # Fallback: try enrich_token_usage pattern (some instrumentors)
                try:
                    instrumentor.instrument(
                        enrich_assistant=_capture_content,
                        enrich_token_usage=True
                    )
                except TypeError:
                    # Last resort: no content control available
                    instrumentor.instrument()
        except ImportError:
            pass  # Library not installed, skip
        except Exception:
            pass  # Instrumentation failed, skip silently


def set_session(config_key: str, session_id: str, customer_id: str = None):
    """
    Set the current session context.

    All subsequent LLM calls in this thread/async context will be
    automatically tagged with this config_key, session_id, and customer_id.

    Args:
        config_key: Your config name (e.g., "linkedin-agent")
        session_id: Your session/conversation ID
        customer_id: Optional customer/user identifier for analytics

    Example:
        trace.set_session("linkedin-agent", session_id, customer_id="user_123")
        agent.run(message)  # Automatically traced with session + customer
    """
    ctx = {"config_key": config_key, "session_id": session_id}
    if customer_id:
        ctx["customer_id"] = customer_id
    _session_context.set(ctx)
    # FallomSpanProcessor will inject these into all spans automatically


def get_session() -> Optional[Dict[str, str]]:
    """Get current session context, if any."""
    return _session_context.get()


def clear_session():
    """Clear session context. Call at end of request if needed."""
    _session_context.set(None)


def span(data: dict, config_key: str = None, session_id: str = None):
    """
    Record custom business metrics. Latest value per field wins.

    Use this for metrics that OTEL can't capture automatically:
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

        # In a batch job (no context):
        for session in sessions:
            trace.span(
                {"outlier_score": calculate_score(session)},
                config_key="linkedin-agent",
                session_id=session.id
            )
    """
    if not _initialized:
        raise RuntimeError("Fallom not initialized. Call trace.init() first.")

    # Use context if config_key/session_id not provided
    ctx = _session_context.get()
    config_key = config_key or (ctx and ctx.get("config_key"))
    session_id = session_id or (ctx and ctx.get("session_id"))

    if not config_key or not session_id:
        raise ValueError(
            "No session context. Either call set_session() first, "
            "or pass config_key and session_id explicitly."
        )

    # Send async to not block
    threading.Thread(
        target=_send_span,
        args=(config_key, session_id, data),
        daemon=True
    ).start()


def _send_span(config_key: str, session_id: str, data: dict):
    """Send span to backend (runs in background thread)."""
    try:
        requests.post(
            f"{_base_url}/spans",
            headers={"Authorization": f"Bearer {_api_key}"},
            json={
                "config_key": config_key,
                "session_id": session_id,
                "data": data
            },
            timeout=5
        )
    except Exception:
        pass  # Fail silently, don't crash user's code

