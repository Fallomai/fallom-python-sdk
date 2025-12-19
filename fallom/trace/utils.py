"""
Utility functions for Fallom tracing.
"""
import os
import secrets
from datetime import datetime, timezone


def generate_hex_id(length: int = 32) -> str:
    """Generate a random hex ID of the specified length."""
    return secrets.token_hex(length // 2)


def iso_now() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_to_iso(timestamp_ms: int) -> str:
    """Convert millisecond timestamp to ISO 8601 string."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.isoformat()

