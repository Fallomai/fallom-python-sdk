"""
Tests for fallom.models module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set API key before importing
os.environ["FALLOM_API_KEY"] = "test-api-key"


class TestModelsInit:
    """Tests for models.init()"""

    def test_init_without_api_key_allows_fallback(self):
        """Should allow init without API key (for fallback-only mode)."""
        with patch.dict(os.environ, {}, clear=True):
            from fallom import models
            models._initialized = False
            models._api_key = None

            # Should NOT raise - allows fallback mode
            models.init()
            assert models._initialized is True
            assert models._api_key is None

    def test_init_fetches_configs(self):
        """Should fetch configs on init."""
        from fallom import models
        models._initialized = False
        models._api_key = None
        models._config_cache = {}

        with patch("requests.get") as mock_get:
            mock_get.return_value.ok = True
            mock_get.return_value.json.return_value = {
                "configs": [
                    {"key": "my-agent", "version": 1, "variants": [{"model": "gpt-4", "weight": 100}]}
                ]
            }
            models.init(api_key="test-key")

            assert models._initialized is True
            assert "my-agent" in models._config_cache


class TestModelsGet:
    """Tests for models.get()"""

    def setup_method(self):
        """Set up test fixtures."""
        from fallom import models
        models._initialized = True
        models._api_key = "test-key"
        models._base_url = "https://api.test.com"
        models._config_cache = {
            "my-agent": {
                "versions": {
                    1: {
                        "key": "my-agent",
                        "version": 1,
                        "variants": [
                            {"model": "claude-opus", "weight": 80},
                            {"model": "gpt-4o", "weight": 20}
                        ]
                    }
                },
                "latest": 1
            },
            "single-model": {
                "versions": {
                    1: {
                        "key": "single-model",
                        "version": 1,
                        "variants": [
                            {"model": "gpt-4", "weight": 100}
                        ]
                    }
                },
                "latest": 1
            }
        }

    def test_get_returns_model(self):
        """Should return a model string."""
        from fallom import models

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            model = models.get("single-model", "session-123")

        assert model == "gpt-4"

    def test_get_is_deterministic(self):
        """Same session_id should always return same model."""
        from fallom import models

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()

            # Call multiple times with same session_id
            results = [
                models.get("my-agent", "session-xyz")
                for _ in range(10)
            ]

        # All results should be the same
        assert len(set(results)) == 1

    def test_get_different_sessions_can_differ(self):
        """Different session_ids can get different models."""
        from fallom import models

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()

            # Try many different sessions
            results = set()
            for i in range(100):
                model = models.get("my-agent", f"session-{i}")
                results.add(model)

        # With 80/20 split and 100 sessions, we should see both models
        assert len(results) == 2
        assert "claude-opus" in results
        assert "gpt-4o" in results

    def test_get_raises_for_unknown_config(self):
        """Should raise ValueError for unknown config."""
        from fallom import models

        with pytest.raises(ValueError, match="not found"):
            models.get("unknown-agent", "session-123")

    def test_get_sets_trace_context(self):
        """Should call trace.set_session()."""
        from fallom import models, trace

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            with patch.object(trace, "set_session") as mock_set_session:
                models.get("single-model", "session-123")

                mock_set_session.assert_called_once_with("single-model", "session-123")

    def test_get_records_session(self):
        """Should record session assignment in background."""
        from fallom import models

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            models.get("single-model", "session-123")

            # Find the call that records the session
            calls = [c for c in mock_thread.call_args_list if "_record_session" in str(c)]
            assert len(calls) == 1


class TestStickyAssignment:
    """Tests for sticky model assignment behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        from fallom import models
        models._initialized = True
        models._api_key = "test-key"
        models._base_url = "https://api.test.com"
        models._config_cache = {
            "test-agent": {
                "versions": {
                    1: {
                        "key": "test-agent",
                        "version": 1,
                        "variants": [
                            {"model": "model-a", "weight": 50},
                            {"model": "model-b", "weight": 50}
                        ]
                    }
                },
                "latest": 1
            }
        }

    def test_sticky_across_calls(self):
        """Same session should get same model across multiple calls."""
        from fallom import models

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()

            session_id = "user-123-convo-456"
            first_model = models.get("test-agent", session_id)

            # Call many more times
            for _ in range(50):
                model = models.get("test-agent", session_id)
                assert model == first_model

    def test_deterministic_hash(self):
        """Hash should be deterministic and produce consistent results."""
        import hashlib

        session_id = "test-session"
        hash_bytes = hashlib.md5(session_id.encode()).digest()
        hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 10000

        # Same input should always produce same hash
        for _ in range(10):
            new_hash = hashlib.md5(session_id.encode()).digest()
            new_val = int.from_bytes(new_hash[:4], byteorder='big') % 10000
            assert new_val == hash_val


class TestRecordSession:
    """Tests for _record_session()"""

    def test_record_session_makes_request(self):
        """Should POST to /sessions endpoint."""
        from fallom import models
        models._api_key = "test-key"
        models._base_url = "https://api.test.com"

        with patch("requests.post") as mock_post:
            mock_post.return_value.ok = True
            models._record_session("my-agent", 1, "session-123", "gpt-4")

            mock_post.assert_called_once_with(
                "https://api.test.com/sessions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "config_key": "my-agent",
                    "config_version": 1,
                    "session_id": "session-123",
                    "assigned_model": "gpt-4"
                },
                timeout=models._RECORD_TIMEOUT
            )

    def test_record_session_fails_silently(self):
        """Should not raise on network errors."""
        from fallom import models
        models._api_key = "test-key"
        models._base_url = "https://api.test.com"

        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")
            # Should not raise
            models._record_session("my-agent", 1, "session-123", "gpt-4")

