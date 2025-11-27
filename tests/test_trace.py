"""
Tests for fallom.trace module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set API key before importing
os.environ["FALLOM_API_KEY"] = "test-api-key"


class TestTraceInit:
    """Tests for trace.init()"""

    def test_init_requires_api_key(self):
        """Should raise error when no API key provided."""
        # Clear env var
        with patch.dict(os.environ, {}, clear=True):
            from fallom import trace
            # Reset initialized state
            trace._initialized = False
            trace._api_key = None

            with pytest.raises(ValueError, match="No API key provided"):
                trace.init()

    def test_init_with_env_var(self):
        """Should use FALLOM_API_KEY env var."""
        with patch.dict(os.environ, {"FALLOM_API_KEY": "env-api-key"}):
            from fallom import trace
            trace._initialized = False
            trace._api_key = None

            with patch("opentelemetry.trace.set_tracer_provider"):
                trace.init()

            assert trace._api_key == "env-api-key"
            assert trace._initialized is True

    def test_init_with_explicit_key(self):
        """Should use explicitly provided API key."""
        from fallom import trace
        trace._initialized = False
        trace._api_key = None

        with patch("opentelemetry.trace.set_tracer_provider"):
            trace.init(api_key="explicit-key")

        assert trace._api_key == "explicit-key"

    def test_init_with_custom_base_url(self):
        """Should use custom base URL."""
        from fallom import trace
        trace._initialized = False

        with patch("opentelemetry.trace.set_tracer_provider"):
            trace.init(api_key="test-key", base_url="https://custom.api.com")

        assert trace._base_url == "https://custom.api.com"


class TestSetSession:
    """Tests for trace.set_session()"""

    def test_set_session_stores_context(self):
        """Should store config_key and session_id in context."""
        from fallom import trace

        trace.set_session("my-agent", "session-123")

        ctx = trace.get_session()
        assert ctx["config_key"] == "my-agent"
        assert ctx["session_id"] == "session-123"

    def test_clear_session(self):
        """Should clear session context."""
        from fallom import trace

        trace.set_session("my-agent", "session-123")
        trace.clear_session()

        assert trace.get_session() is None


class TestSpan:
    """Tests for trace.span()"""

    def test_span_requires_initialization(self):
        """Should raise error if not initialized."""
        from fallom import trace
        trace._initialized = False

        with pytest.raises(RuntimeError, match="not initialized"):
            trace.span({"score": 0.8})

    def test_span_uses_context(self):
        """Should use session context if no explicit args."""
        from fallom import trace
        trace._initialized = True
        trace.set_session("my-agent", "session-123")

        with patch.object(trace, "_send_span") as mock_send:
            with patch("threading.Thread") as mock_thread:
                mock_thread.return_value.start = MagicMock()
                trace.span({"score": 0.8})

                # Check thread was started with correct args
                mock_thread.assert_called_once()
                call_args = mock_thread.call_args
                assert call_args.kwargs["args"] == ("my-agent", "session-123", {"score": 0.8})

    def test_span_explicit_args(self):
        """Should use explicit config_key and session_id."""
        from fallom import trace
        trace._initialized = True

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            trace.span(
                {"score": 0.8},
                config_key="explicit-agent",
                session_id="explicit-session"
            )

            call_args = mock_thread.call_args
            assert call_args.kwargs["args"] == (
                "explicit-agent",
                "explicit-session",
                {"score": 0.8}
            )

    def test_span_requires_session(self):
        """Should raise error if no session context or explicit args."""
        from fallom import trace
        trace._initialized = True
        trace.clear_session()

        with pytest.raises(ValueError, match="No session context"):
            trace.span({"score": 0.8})


class TestSendSpan:
    """Tests for _send_span()"""

    def test_send_span_makes_request(self):
        """Should POST to /spans endpoint."""
        from fallom import trace
        trace._api_key = "test-key"
        trace._base_url = "https://api.test.com"

        with patch("requests.post") as mock_post:
            mock_post.return_value.ok = True
            trace._send_span("my-agent", "session-123", {"score": 0.8})

            mock_post.assert_called_once_with(
                "https://api.test.com/spans",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "config_key": "my-agent",
                    "session_id": "session-123",
                    "data": {"score": 0.8}
                },
                timeout=5
            )

    def test_send_span_fails_silently(self):
        """Should not raise on network errors."""
        from fallom import trace
        trace._api_key = "test-key"
        trace._base_url = "https://api.test.com"

        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")
            # Should not raise
            trace._send_span("my-agent", "session-123", {"score": 0.8})

