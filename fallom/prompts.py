"""
Fallom prompts module.

Provides prompt management and A/B testing.
Zero latency on get() - uses local cache + template interpolation.

Design principles:
- Never block user's app if Fallom is down
- Very short timeouts (1-2 seconds max)
- Always return usable prompt (fallback if needed)
- Background sync keeps prompts fresh
- Auto-tag next OTEL span with prompt context
"""
import os
import re
import hashlib
import threading
import time
import requests
import contextvars
from dataclasses import dataclass
from typing import Dict, Optional, Any

_api_key: str = None
_base_url: str = None
_prompt_cache: Dict[str, dict] = {}  # key -> {versions: {v: content}, current: int}
_prompt_ab_cache: Dict[str, dict] = {}  # key -> {versions: {v: variants}, current: int}
_initialized: bool = False
_sync_thread: threading.Thread = None

# Prompt context for auto-tagging next OTEL span
_prompt_context = contextvars.ContextVar("fallom_prompt", default=None)

# Short timeouts - we'd rather return fallback than add latency
_SYNC_TIMEOUT = 2  # seconds


@dataclass
class PromptResult:
    """Result from prompts.get() or prompts.get_ab()."""
    key: str
    version: int
    system: str  # System prompt with variables replaced
    user: str    # User template with variables replaced
    ab_test_key: Optional[str] = None
    variant_index: Optional[int] = None


def init(api_key: str = None, base_url: str = None):
    """
    Initialize Fallom prompts.

    This is optional - get() will auto-init if needed.
    Non-blocking: starts background fetch immediately.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to FALLOM_BASE_URL env var.
    """
    global _api_key, _base_url, _initialized, _sync_thread

    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = base_url or os.environ.get("FALLOM_PROMPTS_URL", os.environ.get("FALLOM_BASE_URL", "https://prompts.fallom.com"))
    _initialized = True

    if not _api_key:
        return  # No API key - get() will raise or return fallback

    # Start background fetch immediately (non-blocking)
    threading.Thread(target=_fetch_all, daemon=True).start()

    # Start background sync thread for periodic refresh
    if _sync_thread is None or not _sync_thread.is_alive():
        _sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        _sync_thread.start()


def _ensure_init():
    """Auto-initialize if not already done."""
    if not _initialized:
        try:
            init()
        except Exception:
            pass


def _fetch_all():
    """Fetch both prompts and prompt A/B tests."""
    _fetch_prompts()
    _fetch_prompt_ab_tests()


def _fetch_prompts(timeout: float = _SYNC_TIMEOUT):
    """Fetch all prompts for this API key."""
    global _prompt_cache
    if not _api_key:
        return
    try:
        resp = requests.get(
            f"{_base_url}/prompts",
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout
        )
        if resp.ok:
            prompts = resp.json().get("prompts", [])
            for p in prompts:
                key = p["key"]
                version = p.get("version", 1)
                if key not in _prompt_cache:
                    _prompt_cache[key] = {"versions": {}, "current": None}
                _prompt_cache[key]["versions"][version] = {
                    "system_prompt": p["system_prompt"],
                    "user_template": p["user_template"]
                }
                _prompt_cache[key]["current"] = version
    except Exception:
        pass  # Keep using cached - don't crash


def _fetch_prompt_ab_tests(timeout: float = _SYNC_TIMEOUT):
    """Fetch all prompt A/B tests for this API key."""
    global _prompt_ab_cache
    if not _api_key:
        return
    try:
        resp = requests.get(
            f"{_base_url}/prompt-ab-tests",
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout
        )
        if resp.ok:
            tests = resp.json().get("prompt_ab_tests", [])
            for t in tests:
                key = t["key"]
                version = t.get("version", 1)
                if key not in _prompt_ab_cache:
                    _prompt_ab_cache[key] = {"versions": {}, "current": None}
                _prompt_ab_cache[key]["versions"][version] = {
                    "variants": t["variants"]
                }
                _prompt_ab_cache[key]["current"] = version
    except Exception:
        pass


def _sync_loop():
    """Background thread that syncs prompts every 30 seconds."""
    while True:
        time.sleep(30)
        try:
            _fetch_all()
        except Exception:
            pass


def _replace_variables(template: str, variables: Dict[str, Any]) -> str:
    """Replace {{variable}} placeholders in template."""
    if not variables:
        return template
    
    def replacer(match):
        var_name = match.group(1).strip()
        return str(variables.get(var_name, match.group(0)))
    
    return re.sub(r'\{\{(\s*\w+\s*)\}\}', replacer, template)


def _set_prompt_context(
    prompt_key: str,
    prompt_version: int,
    ab_test_key: Optional[str] = None,
    variant_index: Optional[int] = None
):
    """Set prompt context for next OTEL span."""
    _prompt_context.set({
        "prompt_key": prompt_key,
        "prompt_version": prompt_version,
        "ab_test_key": ab_test_key,
        "variant_index": variant_index
    })


def get_prompt_context() -> Optional[dict]:
    """Get current prompt context (for FallomSpanProcessor)."""
    return _prompt_context.get()


def clear_prompt_context():
    """Clear prompt context after it's been used."""
    _prompt_context.set(None)


def get(
    prompt_key: str,
    variables: Dict[str, Any] = None,
    version: Optional[int] = None
) -> PromptResult:
    """
    Get a prompt (non-A/B).

    This is zero latency - uses local cache + string interpolation.
    No network call on the hot path.

    Also sets prompt context for next OTEL span auto-tagging.

    Args:
        prompt_key: Your prompt key (e.g., "onboarding")
        variables: Dict of template variables (e.g., {"user_name": "John"})
        version: Pin to specific version. None = current (default)

    Returns:
        PromptResult with system and user messages (variables replaced)

    Raises:
        ValueError: If prompt not found

    Examples:
        prompt = prompts.get("onboarding", {"user_name": "John"})
        
        # Use with OpenAI
        openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}
            ]
        )
    """
    _ensure_init()

    # Get from cache
    prompt_data = _prompt_cache.get(prompt_key)

    # If not in cache, try fetching
    if not prompt_data:
        _fetch_prompts(timeout=_SYNC_TIMEOUT)
        prompt_data = _prompt_cache.get(prompt_key)

    if not prompt_data:
        raise ValueError(
            f"Prompt '{prompt_key}' not found. "
            "Check that it exists in your Fallom dashboard."
        )

    # Get specific version or current
    target_version = version if version is not None else prompt_data["current"]
    content = prompt_data["versions"].get(target_version)

    if not content:
        raise ValueError(
            f"Prompt '{prompt_key}' version {target_version} not found."
        )

    # Replace variables
    system = _replace_variables(content["system_prompt"], variables or {})
    user = _replace_variables(content["user_template"], variables or {})

    # Set context for next OTEL span
    _set_prompt_context(prompt_key, target_version)

    return PromptResult(
        key=prompt_key,
        version=target_version,
        system=system,
        user=user
    )


def get_ab(
    ab_test_key: str,
    session_id: str,
    variables: Dict[str, Any] = None
) -> PromptResult:
    """
    Get a prompt from an A/B test.

    Uses session_id hash for deterministic, sticky assignment.
    Same session always gets same variant.

    Also sets prompt context for next OTEL span auto-tagging.

    Args:
        ab_test_key: Your A/B test key (e.g., "onboarding-experiment")
        session_id: Your session/conversation ID (for sticky assignment)
        variables: Dict of template variables (e.g., {"user_name": "John"})

    Returns:
        PromptResult with system and user messages (variables replaced)

    Raises:
        ValueError: If A/B test or selected prompt not found

    Examples:
        prompt = prompts.get_ab("onboarding-test", session_id, {"user_name": "John"})
        
        # Same usage as prompts.get()
        openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}
            ]
        )
    """
    _ensure_init()

    # Get A/B test from cache
    ab_data = _prompt_ab_cache.get(ab_test_key)

    # If not in cache, try fetching
    if not ab_data:
        _fetch_prompt_ab_tests(timeout=_SYNC_TIMEOUT)
        ab_data = _prompt_ab_cache.get(ab_test_key)

    if not ab_data:
        raise ValueError(
            f"Prompt A/B test '{ab_test_key}' not found. "
            "Check that it exists in your Fallom dashboard."
        )

    # Get current version
    current_version = ab_data["current"]
    version_data = ab_data["versions"].get(current_version)

    if not version_data:
        raise ValueError(
            f"Prompt A/B test '{ab_test_key}' has no current version."
        )

    variants = version_data["variants"]

    # Deterministic assignment from session_id hash (same as models)
    hash_bytes = hashlib.md5(session_id.encode()).digest()
    hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 1_000_000

    # Walk through variants by weight
    cumulative = 0.0
    selected_variant = variants[-1]  # Fallback to last
    selected_index = len(variants) - 1

    for i, v in enumerate(variants):
        cumulative += float(v["weight"]) * 10000
        if hash_val < cumulative:
            selected_variant = v
            selected_index = i
            break

    # Get the actual prompt content
    prompt_key = selected_variant["prompt_key"]
    prompt_version = selected_variant.get("prompt_version")  # None = use current

    # Fetch prompt content
    prompt_data = _prompt_cache.get(prompt_key)
    if not prompt_data:
        _fetch_prompts(timeout=_SYNC_TIMEOUT)
        prompt_data = _prompt_cache.get(prompt_key)

    if not prompt_data:
        raise ValueError(
            f"Prompt '{prompt_key}' (from A/B test '{ab_test_key}') not found."
        )

    # Get specific version or current
    target_version = prompt_version if prompt_version is not None else prompt_data["current"]
    content = prompt_data["versions"].get(target_version)

    if not content:
        raise ValueError(
            f"Prompt '{prompt_key}' version {target_version} not found."
        )

    # Replace variables
    system = _replace_variables(content["system_prompt"], variables or {})
    user = _replace_variables(content["user_template"], variables or {})

    # Set context for next OTEL span
    _set_prompt_context(prompt_key, target_version, ab_test_key, selected_index)

    return PromptResult(
        key=prompt_key,
        version=target_version,
        system=system,
        user=user,
        ab_test_key=ab_test_key,
        variant_index=selected_index
    )

