"""
FallomSession - Session-scoped tracing for concurrent-safe operations.
"""
from typing import Optional, Dict, Any, List, TypeVar

from .types import SessionContext, SessionOptions
from .wrappers.openai import wrap_openai
from .wrappers.anthropic import wrap_anthropic
from .wrappers.google_ai import wrap_google_ai

T = TypeVar("T")


class FallomSession:
    """
    A session-scoped Fallom instance.

    All wrappers created from this session automatically use the session context,
    making them safe for concurrent operations without global state issues.

    Example:
        session = fallom.session(
            config_key="my-app",
            session_id="session-123",
            customer_id="user-456"
        )

        # All calls use the session context
        openai = session.wrap_openai(OpenAI())
        response = openai.chat.completions.create(...)

        # Or wrap Anthropic
        anthropic = session.wrap_anthropic(Anthropic())
    """

    def __init__(
        self,
        config_key: str,
        session_id: str,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Create a new Fallom session.

        Args:
            config_key: Your config name (e.g., "linkedin-agent")
            session_id: Your session/conversation ID
            customer_id: Optional customer/user identifier for analytics
            metadata: Custom key-value metadata for filtering/grouping
            tags: Simple string tags for quick filtering
        """
        self._ctx = SessionContext(
            config_key=config_key,
            session_id=session_id,
            customer_id=customer_id,
            metadata=metadata,
            tags=tags,
        )

    def get_context(self) -> SessionContext:
        """Get the session context."""
        return SessionContext(
            config_key=self._ctx.config_key,
            session_id=self._ctx.session_id,
            customer_id=self._ctx.customer_id,
            metadata=self._ctx.metadata.copy() if self._ctx.metadata else None,
            tags=self._ctx.tags.copy() if self._ctx.tags else None,
        )

    async def get_model(
        self,
        config_key: Optional[str] = None,
        fallback: Optional[str] = None,
        version: Optional[int] = None,
    ) -> str:
        """
        Get model assignment for this session (A/B testing).

        Args:
            config_key: Config key to use (defaults to session config_key)
            fallback: Fallback model if config not found
            version: Pin to specific config version

        Returns:
            Model string (e.g., "gpt-4o", "claude-3-opus")
        """
        from fallom import models

        config = config_key or self._ctx.config_key
        return models.get(
            config,
            self._ctx.session_id,
            version=version,
            fallback=fallback,
        )

    def wrap_openai(self, client: T) -> T:
        """
        Wrap an OpenAI client for automatic tracing.

        Args:
            client: An OpenAI client instance

        Returns:
            The wrapped client with tracing enabled
        """
        return wrap_openai(client, self._ctx)

    def wrap_anthropic(self, client: T) -> T:
        """
        Wrap an Anthropic client for automatic tracing.

        Args:
            client: An Anthropic client instance

        Returns:
            The wrapped client with tracing enabled
        """
        return wrap_anthropic(client, self._ctx)

    def wrap_google_ai(self, model: T) -> T:
        """
        Wrap a Google AI model for automatic tracing.

        Args:
            model: A Google GenerativeModel instance

        Returns:
            The wrapped model with tracing enabled
        """
        return wrap_google_ai(model, self._ctx)


def session(
    config_key: str,
    session_id: str,
    customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> FallomSession:
    """
    Create a session-scoped Fallom instance.

    Args:
        config_key: Your config name (e.g., "linkedin-agent")
        session_id: Your session/conversation ID
        customer_id: Optional customer/user identifier for analytics
        metadata: Custom key-value metadata for filtering/grouping
        tags: Simple string tags for quick filtering

    Returns:
        A FallomSession instance with the specified context

    Example:
        session = fallom.session(
            config_key="my-app",
            session_id="session-123",
            customer_id="user-456",
            tags=["production", "high-priority"]
        )

        openai = session.wrap_openai(OpenAI())
    """
    return FallomSession(
        config_key=config_key,
        session_id=session_id,
        customer_id=customer_id,
        metadata=metadata,
        tags=tags,
    )

