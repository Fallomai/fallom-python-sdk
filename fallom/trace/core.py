"""
Core Fallom tracing functionality.

Handles initialization and trace sending.
Session management is now handled by FallomSession.
"""
import os
import json
import threading
import contextvars
from typing import Optional, Dict, Any

import requests

from .types import TraceContext, TraceData

# =============================================================================
# Module State
# =============================================================================

_trace_context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    "fallom_trace_context", default=None
)
_fallback_trace_context: Optional[TraceContext] = None

_api_key: Optional[str] = None
_base_url: str = "https://traces.fallom.com"
_initialized: bool = False
_capture_content: bool = True
_debug_mode: bool = False

# =============================================================================
# Logging
# =============================================================================

def _log(*args):
    """Print debug message if debug mode is enabled."""
    if _debug_mode:
        print("[Fallom]", *args)


# =============================================================================
# State Accessors (for use by wrappers and session)
# =============================================================================

def get_trace_context_storage() -> contextvars.ContextVar[Optional[TraceContext]]:
    """Get the trace context storage."""
    return _trace_context


def get_fallback_trace_context() -> Optional[TraceContext]:
    """Get the fallback trace context."""
    return _fallback_trace_context


def is_initialized() -> bool:
    """Check if Fallom is initialized."""
    return _initialized


def should_capture_content() -> bool:
    """Check if content capture is enabled."""
    return _capture_content


def get_api_key() -> Optional[str]:
    """Get the API key."""
    return _api_key


def get_base_url() -> str:
    """Get the base URL."""
    return _base_url


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return _debug_mode


# =============================================================================
# Initialization
# =============================================================================

def init(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    capture_content: bool = True,
    debug: bool = False
) -> None:
    """
    Initialize Fallom tracing.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to https://traces.fallom.com
        capture_content: Whether to capture prompt/completion content.
        debug: Enable debug logging.

    Example:
        import fallom
        
        fallom.init(api_key=os.environ["FALLOM_API_KEY"])
        
        session = fallom.session(
            config_key="my-agent",
            session_id="session-123",
        )
        
        openai = session.wrap_openai(OpenAI())
        response = await openai.chat.completions.create(...)
    """
    global _api_key, _base_url, _initialized, _capture_content, _debug_mode
    
    if _initialized:
        return
    
    _debug_mode = debug
    
    _log("🚀 Initializing Fallom tracing...")
    
    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = (
        base_url or 
        os.environ.get("FALLOM_TRACES_URL") or 
        os.environ.get("FALLOM_BASE_URL") or 
        "https://traces.fallom.com"
    )
    
    # Check env var for capture_content
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
    
    _log(f"📡 Exporter URL: {_base_url}/v1/traces")
    _log("✅ SDK initialized")


def shutdown() -> None:
    """Shutdown the tracing SDK gracefully."""
    global _initialized
    _initialized = False


# =============================================================================
# Trace Sending
# =============================================================================

def send_trace(trace: TraceData) -> None:
    """
    Send a trace to the Fallom API.
    Used internally by wrappers.
    Runs in background thread to avoid blocking.
    """
    threading.Thread(
        target=_send_trace_sync,
        args=(trace,),
        daemon=True
    ).start()


def _send_trace_sync(trace: TraceData) -> None:
    """Synchronously send a trace (runs in background thread)."""
    url = f"{_base_url}/v1/traces"
    _log(f"📤 Sending trace to: {url}")
    _log(f"   Session: {trace.session_id}, Config: {trace.config_key}")
    
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            },
            json=trace.to_dict(),
            timeout=5
        )
        
        if not response.ok:
            _log(f"❌ Trace send failed: {response.status_code} {response.text}")
        else:
            _log(f"✅ Trace sent: {trace.name} {trace.model}")
    except Exception as e:
        _log(f"❌ Trace send error: {e}")


def send_trace_sync(trace: TraceData) -> None:
    """
    Send a trace synchronously (blocking).
    Use this when you need to ensure the trace is sent before continuing.
    """
    _send_trace_sync(trace)

