"""
Fallom models module.

Provides model A/B testing with versioned configs.
Zero latency on get() - uses local hash + cached config.

Design principles:
- Never block user's app if Fallom is down
- Very short timeouts (1-2 seconds max)
- Always return a usable model (fallback if needed)
- Background sync keeps configs fresh
"""
import os
import hashlib
import threading
import time
import requests
from typing import Dict, Optional

_api_key: str = None
_base_url: str = None
_config_cache: Dict[str, dict] = {}  # key -> {version -> config}
_initialized: bool = False
_sync_thread: threading.Thread = None

# Short timeouts - we'd rather return fallback than add latency
_INIT_TIMEOUT = 2  # seconds - initial fetch
_SYNC_TIMEOUT = 2  # seconds - background sync
_RECORD_TIMEOUT = 1  # seconds - recording sessions


def init(api_key: str = None, base_url: str = None):
    """
    Initialize Fallom models.

    This is optional - get() will auto-init if needed.
    Non-blocking: starts background config fetch immediately.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to FALLOM_BASE_URL env var, or https://spans.fallom.com

    Example:
        from fallom import models
        models.init()
    """
    global _api_key, _base_url, _initialized, _sync_thread

    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = base_url or os.environ.get("FALLOM_CONFIGS_URL", os.environ.get("FALLOM_BASE_URL", "https://configs.fallom.com"))
    _initialized = True
    
    if not _api_key:
        return  # No API key - get() will return fallback

    # Start background fetch immediately (non-blocking)
    # This way if there's any work before the first get() call, configs might be ready
    threading.Thread(target=_fetch_configs, daemon=True).start()

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


_debug_mode = False

def _log(msg: str):
    """Print debug message if debug mode is enabled."""
    if _debug_mode:
        print(f"[Fallom] {msg}")


def _fetch_configs(timeout: float = _SYNC_TIMEOUT):
    """Fetch all configs (latest versions) for this API key."""
    global _config_cache
    if not _api_key:
        _log("_fetch_configs: No API key, skipping")
        return
    try:
        _log(f"Fetching configs from {_base_url}/configs")
        resp = requests.get(
            f"{_base_url}/configs",
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout
        )
        _log(f"Response status: {resp.status_code}")
        if resp.ok:
            configs = resp.json().get("configs", [])
            _log(f"Got {len(configs)} configs: {[c.get('key') for c in configs]}")
            for c in configs:
                key = c["key"]
                version = c.get("version", 1)
                variants = c.get("variants", [])
                _log(f"Config '{key}' v{version}: {variants}")
                # Store by key, with version info
                if key not in _config_cache:
                    _config_cache[key] = {"versions": {}, "latest": None}
                _config_cache[key]["versions"][version] = c
                # Track latest version (the one returned by /configs is always the current/latest)
                _config_cache[key]["latest"] = version
        else:
            _log(f"Fetch failed: {resp.text}")
    except Exception as e:
        _log(f"Fetch exception: {e}")
        pass  # Keep using cached configs - don't crash


def _fetch_specific_version(config_key: str, version: int, timeout: float = _SYNC_TIMEOUT) -> Optional[dict]:
    """Fetch a specific version of a config. Used when version pinning."""
    if not _api_key:
        return None
    try:
        resp = requests.get(
            f"{_base_url}/configs/{config_key}/version/{version}",
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=timeout
        )
        if resp.ok:
            config = resp.json()
            # Cache it
            if config_key not in _config_cache:
                _config_cache[config_key] = {"versions": {}, "latest": None}
            _config_cache[config_key]["versions"][version] = config
            return config
    except Exception:
        pass
    return None


def _sync_loop():
    """Background thread that syncs configs every 30 seconds."""
    while True:
        time.sleep(30)
        try:
            _fetch_configs()
        except Exception:
            pass


def get(
    config_key: str, 
    session_id: str, 
    version: Optional[int] = None,
    fallback: Optional[str] = None,
    debug: bool = False
) -> str:
    """
    Get model assignment for a session.

    This is zero latency - uses local hash computation + cached config.
    No network call on the hot path.

    Same session_id always returns same model (sticky assignment).

    Also automatically sets trace context, so all subsequent LLM calls
    are tagged with this session.

    Args:
        config_key: Your config name (e.g., "linkedin-agent")
        session_id: Your session/conversation ID (must be consistent)
        version: Pin to specific version (1, 2, etc). None = latest (default)
        fallback: Model to return if config not found or Fallom is down.
                  If not provided and config fails, raises ValueError.

    Returns:
        Model string (e.g., "claude-opus", "gpt-4o")

    Raises:
        ValueError: If config not found AND no fallback provided

    Examples:
        # Basic usage (latest version)
        model = models.get("linkedin-agent", session_id)

        # Pin to specific version
        model = models.get("linkedin-agent", session_id, version=2)

        # With fallback for resilience
        model = models.get("linkedin-agent", session_id, fallback="gpt-4o-mini")
    """
    global _debug_mode
    _debug_mode = debug
    
    _ensure_init()
    _log(f"get() called: config_key={config_key}, session_id={session_id}, fallback={fallback}")

    # Try to get config
    try:
        config_data = _config_cache.get(config_key)
        _log(f"Cache lookup for '{config_key}': {'found' if config_data else 'not found'}")
        
        # If not in cache, try fetching (handles cold start / first call)
        if not config_data:
            _log("Not in cache, fetching...")
            _fetch_configs(timeout=_SYNC_TIMEOUT)  # Quick blocking fetch
            config_data = _config_cache.get(config_key)
            _log(f"After fetch, cache lookup: {'found' if config_data else 'still not found'}")
        
        if not config_data:
            _log(f"Config not found, using fallback: {fallback}")
            if fallback:
                print(f"[Fallom WARNING] Config '{config_key}' not found, using fallback model: {fallback}")
                return _return_with_trace(config_key, session_id, fallback, version=0)
            raise ValueError(
                f"Config '{config_key}' not found. "
                "Check that it exists in your Fallom dashboard."
            )

        # Get specific version or latest
        if version is not None:
            # User wants a specific version
            config = config_data["versions"].get(version)
            if not config:
                # Not in cache - try fetching it (short timeout)
                config = _fetch_specific_version(config_key, version, timeout=_SYNC_TIMEOUT)
            if not config:
                # Still not found - use fallback or error
                if fallback:
                    print(f"[Fallom WARNING] Config '{config_key}' version {version} not found, using fallback: {fallback}")
                    return _return_with_trace(config_key, session_id, fallback, version=0)
                raise ValueError(
                    f"Config '{config_key}' version {version} not found."
                )
            target_version = version
        else:
            # Use latest (cached, zero latency)
            target_version = config_data["latest"]
            config = config_data["versions"].get(target_version)
            if not config:
                if fallback:
                    print(f"[Fallom WARNING] Config '{config_key}' has no cached version, using fallback: {fallback}")
                    return _return_with_trace(config_key, session_id, fallback, version=0)
                raise ValueError(
                    f"Config '{config_key}' has no cached version."
                )

        variants_raw = config["variants"]
        config_version = config.get("version", target_version)
        
        # Handle both list and dict formats for variants
        # List: [{"model": "x", "weight": 50}, ...]
        # Dict: {"control": {"model": "x", "weight": 50}, ...}
        if isinstance(variants_raw, dict):
            variants = list(variants_raw.values())
        else:
            variants = variants_raw
        
        _log(f"Config found! Version: {config_version}, Variants: {variants}")

        # Deterministic assignment from session_id hash
        # Same session_id always gets same model (sticky)
        # Using 1M buckets for 0.01% granularity (good for apps with millions of users)
        hash_bytes = hashlib.md5(session_id.encode()).digest()
        hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 1_000_000
        _log(f"Session hash: {hash_val} (out of 1,000,000)")

        # Walk through variants by weight
        # Weights are percentages (0-100) with decimal support (e.g., 0.01 for 0.01%)
        cumulative = 0.0
        assigned_model = variants[-1]["model"]  # Fallback to last
        for v in variants:
            # Convert percentage to per-million (e.g., 50% -> 500000, 0.01% -> 100)
            old_cumulative = cumulative
            cumulative += float(v["weight"]) * 10000
            _log(f"Variant {v['model']}: weight={v['weight']}%, range={old_cumulative}-{cumulative}, hash={hash_val}, match={hash_val < cumulative}")
            if hash_val < cumulative:
                assigned_model = v["model"]
                break

        _log(f"✅ Assigned model: {assigned_model}")
        return _return_with_trace(config_key, session_id, assigned_model, config_version)

    except ValueError:
        raise  # Re-raise ValueErrors (config not found)
    except Exception as e:
        # Any other error - return fallback if provided
        if fallback:
            print(f"[Fallom WARNING] Error getting model for '{config_key}': {e}. Using fallback: {fallback}")
            return _return_with_trace(config_key, session_id, fallback, version=0)
        raise


def _return_with_trace(config_key: str, session_id: str, model: str, version: int) -> str:
    """Set trace context and record session, then return model."""
    # Auto-set trace context so subsequent calls are tagged
    try:
        from fallom import trace
        trace.set_session(config_key, session_id)
    except Exception:
        pass  # Tracing might not be initialized, that's ok

    # Record session async (non-blocking)
    if version > 0:  # Don't record fallback usage
        threading.Thread(
            target=_record_session,
            args=(config_key, version, session_id, model),
            daemon=True
        ).start()

    return model


def _record_session(config_key: str, version: int, session_id: str, model: str):
    """Record session assignment to backend (runs in background thread)."""
    if not _api_key:
        return
    try:
        resp = requests.post(
            f"{_base_url}/sessions",
            headers={"Authorization": f"Bearer {_api_key}"},
            json={
                "config_key": config_key,
                "config_version": version,
                "session_id": session_id,
                "assigned_model": model
            },
            timeout=_RECORD_TIMEOUT
        )
    except Exception:
        pass  # Fail silently - never impact user's app

