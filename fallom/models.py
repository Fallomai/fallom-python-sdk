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
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass

_api_key: Optional[str] = None
_base_url: str = "https://configs.fallom.com"
_config_cache: Dict[str, dict] = {}  # key -> {versions: {v: config}, latest: int}
_initialized: bool = False
_sync_thread: Optional[threading.Thread] = None
_debug_mode: bool = False

# Short timeouts - we'd rather return fallback than add latency
_INIT_TIMEOUT = 2  # seconds - initial fetch
_SYNC_TIMEOUT = 2  # seconds - background sync
_RECORD_TIMEOUT = 1  # seconds - recording sessions


@dataclass
class IndividualTarget:
    """Individual target - assigns specific users to specific variants."""
    field: str
    value: str
    variant_index: int


@dataclass
class TargetingCondition:
    """Condition for rule-based targeting."""
    field: str
    operator: str  # "eq", "neq", "in", "nin", "contains", "startsWith", "endsWith"
    value: Union[str, List[str]]


@dataclass
class TargetingRule:
    """Rule-based targeting with conditions."""
    conditions: List[TargetingCondition]
    variant_index: int


@dataclass
class Targeting:
    """Targeting configuration for user-level model assignment."""
    individual_targets: Optional[List[IndividualTarget]] = None
    rules: Optional[List[TargetingRule]] = None
    enabled: bool = True


def _log(msg: str):
    """Print debug message if debug mode is enabled."""
    if _debug_mode:
        print(f"[Fallom] {msg}")


def init(api_key: str = None, base_url: str = None):
    """
    Initialize Fallom models.

    This is optional - get() will auto-init if needed.
    Non-blocking: starts background config fetch immediately.

    Args:
        api_key: Your Fallom API key. Defaults to FALLOM_API_KEY env var.
        base_url: API base URL. Defaults to https://configs.fallom.com

    Example:
        from fallom import models
        models.init()
    """
    global _api_key, _base_url, _initialized, _sync_thread

    _api_key = api_key or os.environ.get("FALLOM_API_KEY")
    _base_url = base_url or os.environ.get("FALLOM_CONFIGS_URL",
                                            os.environ.get("FALLOM_BASE_URL", "https://configs.fallom.com"))
    _initialized = True

    if not _api_key:
        return  # No API key - get() will return fallback

    # Start background fetch immediately (non-blocking)
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
                # Track latest version
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


def _evaluate_targeting(
    targeting: Optional[Dict[str, Any]],
    customer_id: Optional[str],
    context: Optional[Dict[str, str]]
) -> Optional[int]:
    """
    Evaluate targeting rules to find a matching variant.
    Returns the variant index if matched, or None if no match.
    """
    if not targeting or targeting.get("enabled") is False:
        return None

    # Build the evaluation context from customer_id and context
    eval_context: Dict[str, str] = {}
    if context:
        eval_context.update(context)
    if customer_id:
        eval_context["customerId"] = customer_id

    _log(f"Evaluating targeting with context: {eval_context}")

    # 1. Check individual targets first (exact match)
    individual_targets = targeting.get("individualTargets", [])
    for target in individual_targets:
        field = target.get("field")
        value = target.get("value")
        variant_index = target.get("variantIndex")
        field_value = eval_context.get(field)
        if field_value == value:
            _log(f"Individual target matched: {field}={value} -> variant {variant_index}")
            return variant_index

    # 2. Check rule-based targeting
    rules = targeting.get("rules", [])
    for rule in rules:
        conditions = rule.get("conditions", [])
        all_match = True

        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            field_value = eval_context.get(field)

            if field_value is None:
                all_match = False
                break

            match = False
            if operator == "eq":
                match = field_value == value
            elif operator == "neq":
                match = field_value != value
            elif operator == "in":
                match = isinstance(value, list) and field_value in value
            elif operator == "nin":
                match = isinstance(value, list) and field_value not in value
            elif operator == "contains":
                match = isinstance(value, str) and value in field_value
            elif operator == "startsWith":
                match = isinstance(value, str) and field_value.startswith(value)
            elif operator == "endsWith":
                match = isinstance(value, str) and field_value.endswith(value)

            if not match:
                all_match = False
                break

        if all_match:
            variant_index = rule.get("variantIndex")
            _log(f"Rule matched: {conditions} -> variant {variant_index}")
            return variant_index

    _log("No targeting rules matched, falling back to weighted random")
    return None


def get(
    config_key: str,
    session_id: str,
    version: Optional[int] = None,
    fallback: Optional[str] = None,
    customer_id: Optional[str] = None,
    context: Optional[Dict[str, str]] = None,
    debug: bool = False
) -> str:
    """
    Get model assignment for a session.

    This is zero latency - uses local hash computation + cached config.
    No network call on the hot path.

    Same session_id always returns same model (sticky assignment).

    Args:
        config_key: Your config name (e.g., "linkedin-agent")
        session_id: Your session/conversation ID (must be consistent)
        version: Pin to specific version (1, 2, etc). None = latest (default)
        fallback: Model to return if config not found or Fallom is down.
                  If not provided and config fails, raises ValueError.
        customer_id: User ID for individual targeting (e.g., "user-123")
        context: Additional context for rule-based targeting (e.g., {"plan": "enterprise"})
        debug: Enable debug logging

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

        # With targeting
        model = models.get(
            "linkedin-agent",
            session_id,
            customer_id="user-123",
            context={"plan": "enterprise"}
        )
    """
    global _debug_mode
    _debug_mode = debug

    _ensure_init()
    _log(f"get() called: config_key={config_key}, session_id={session_id}, fallback={fallback}")

    try:
        config_data = _config_cache.get(config_key)
        _log(f"Cache lookup for '{config_key}': {'found' if config_data else 'not found'}")

        # If not in cache, try fetching (handles cold start / first call)
        if not config_data:
            _log("Not in cache, fetching...")
            _fetch_configs(timeout=_SYNC_TIMEOUT)
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
            config = config_data["versions"].get(version)
            if not config:
                config = _fetch_specific_version(config_key, version, timeout=_SYNC_TIMEOUT)
            if not config:
                if fallback:
                    print(f"[Fallom WARNING] Config '{config_key}' version {version} not found, using fallback: {fallback}")
                    return _return_with_trace(config_key, session_id, fallback, version=0)
                raise ValueError(
                    f"Config '{config_key}' version {version} not found."
                )
            target_version = version
        else:
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
        if isinstance(variants_raw, dict):
            variants = list(variants_raw.values())
        else:
            variants = variants_raw

        _log(f"Config found! Version: {config_version}, Variants: {variants}")

        # 1. First, try targeting rules (if customer_id or context provided)
        targeting = config.get("targeting")
        targeted_variant_index = _evaluate_targeting(targeting, customer_id, context)
        if targeted_variant_index is not None and targeted_variant_index < len(variants):
            assigned_model = variants[targeted_variant_index]["model"]
            _log(f"✅ Assigned model via targeting: {assigned_model}")
            return _return_with_trace(config_key, session_id, assigned_model, config_version)

        # 2. Fall back to deterministic assignment from session_id hash
        hash_bytes = hashlib.md5(session_id.encode()).digest()
        hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 1_000_000
        _log(f"Session hash: {hash_val} (out of 1,000,000)")

        # Walk through variants by weight
        cumulative = 0.0
        assigned_model = variants[-1]["model"]  # Fallback to last
        for v in variants:
            old_cumulative = cumulative
            cumulative += float(v["weight"]) * 10000
            _log(f"Variant {v['model']}: weight={v['weight']}%, range={old_cumulative}-{cumulative}, hash={hash_val}, match={hash_val < cumulative}")
            if hash_val < cumulative:
                assigned_model = v["model"]
                break

        _log(f"✅ Assigned model via weighted random: {assigned_model}")
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
        requests.post(
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
