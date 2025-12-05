"""
Tests for fallom.prompts module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set API key before importing
os.environ["FALLOM_API_KEY"] = "test-api-key"


class TestVariableReplacement:
    """Tests for template variable replacement."""

    def test_replace_single_variable(self):
        """Should replace a single {{variable}}."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{name}}!",
            {"name": "John"}
        )
        assert result == "Hello John!"

    def test_replace_multiple_variables(self):
        """Should replace multiple variables."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{name}}, you are {{age}} years old.",
            {"name": "John", "age": 25}
        )
        assert result == "Hello John, you are 25 years old."

    def test_replace_with_spaces(self):
        """Should handle spaces in variable names."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{ name }}!",
            {"name": "John"}
        )
        assert result == "Hello John!"

    def test_missing_variable_unchanged(self):
        """Missing variables should be left as-is."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{name}}, {{missing}}!",
            {"name": "John"}
        )
        assert result == "Hello John, {{missing}}!"

    def test_empty_variables(self):
        """Should handle empty variables dict."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{name}}!",
            {}
        )
        assert result == "Hello {{name}}!"

    def test_none_variables(self):
        """Should handle None variables."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Hello {{name}}!",
            None
        )
        assert result == "Hello {{name}}!"

    def test_numeric_variable(self):
        """Should convert numeric variables to string."""
        from fallom import prompts

        result = prompts._replace_variables(
            "Count: {{count}}",
            {"count": 42}
        )
        assert result == "Count: 42"


class TestPromptsGet:
    """Tests for prompts.get()"""

    def setup_method(self):
        """Set up test fixtures."""
        from fallom import prompts
        prompts._initialized = True
        prompts._api_key = "test-key"
        prompts._base_url = "https://api.test.com"
        prompts._prompt_cache = {
            "onboarding": {
                "versions": {
                    1: {
                        "system_prompt": "You are a helpful assistant.",
                        "user_template": "Help {{user_name}} get started."
                    },
                    2: {
                        "system_prompt": "You are a concise assistant.",
                        "user_template": "Briefly help {{user_name}}."
                    }
                },
                "current": 2
            }
        }

    def test_get_returns_prompt_result(self):
        """Should return a PromptResult dataclass."""
        from fallom import prompts

        result = prompts.get("onboarding", {"user_name": "John"})

        assert result.key == "onboarding"
        assert result.version == 2
        assert result.system == "You are a concise assistant."
        assert result.user == "Briefly help John."

    def test_get_uses_current_version(self):
        """Should use current version by default."""
        from fallom import prompts

        result = prompts.get("onboarding")

        assert result.version == 2

    def test_get_with_specific_version(self):
        """Should use specific version when provided."""
        from fallom import prompts

        result = prompts.get("onboarding", version=1)

        assert result.version == 1
        assert result.system == "You are a helpful assistant."

    def test_get_raises_for_unknown_prompt(self):
        """Should raise ValueError for unknown prompt."""
        from fallom import prompts

        with pytest.raises(ValueError, match="not found"):
            prompts.get("unknown-prompt")

    def test_get_sets_prompt_context(self):
        """Should set prompt context for OTEL span tagging."""
        from fallom import prompts

        prompts.get("onboarding", {"user_name": "John"})

        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["prompt_key"] == "onboarding"
        assert ctx["prompt_version"] == 2
        assert ctx["ab_test_key"] is None

    def test_get_from_cache_no_network(self):
        """Should use cache without making network calls."""
        from fallom import prompts

        with patch("fallom.prompts.requests.get") as mock_get:
            prompts.get("onboarding")

            # Should not have made any HTTP requests
            mock_get.assert_not_called()


class TestPromptsGetAB:
    """Tests for prompts.get_ab()"""

    def setup_method(self):
        """Set up test fixtures."""
        from fallom import prompts
        prompts._initialized = True
        prompts._api_key = "test-key"
        prompts._base_url = "https://api.test.com"

        # Set up prompt cache
        prompts._prompt_cache = {
            "prompt-a": {
                "versions": {
                    1: {
                        "system_prompt": "System A",
                        "user_template": "User A: {{input}}"
                    }
                },
                "current": 1
            },
            "prompt-b": {
                "versions": {
                    1: {
                        "system_prompt": "System B",
                        "user_template": "User B: {{input}}"
                    }
                },
                "current": 1
            }
        }

        # Set up A/B test cache
        prompts._prompt_ab_cache = {
            "onboarding-test": {
                "versions": {
                    1: {
                        "variants": [
                            {"prompt_key": "prompt-a", "prompt_version": None, "weight": 50},
                            {"prompt_key": "prompt-b", "prompt_version": None, "weight": 50}
                        ]
                    }
                },
                "current": 1
            },
            "single-variant": {
                "versions": {
                    1: {
                        "variants": [
                            {"prompt_key": "prompt-a", "prompt_version": 1, "weight": 100}
                        ]
                    }
                },
                "current": 1
            }
        }

    def test_get_ab_returns_prompt_result(self):
        """Should return a PromptResult with A/B info."""
        from fallom import prompts

        result = prompts.get_ab("single-variant", "session-123", {"input": "test"})

        assert result.key == "prompt-a"
        assert result.ab_test_key == "single-variant"
        assert result.variant_index == 0
        assert result.user == "User A: test"

    def test_get_ab_is_deterministic(self):
        """Same session_id should always return same variant."""
        from fallom import prompts

        # Call multiple times with same session_id
        results = [
            prompts.get_ab("onboarding-test", "session-xyz", {"input": "test"})
            for _ in range(10)
        ]

        # All results should have same key
        keys = [r.key for r in results]
        assert len(set(keys)) == 1

    def test_get_ab_different_sessions_can_differ(self):
        """Different session_ids can get different variants."""
        from fallom import prompts

        # Try many different sessions
        results = set()
        for i in range(100):
            result = prompts.get_ab("onboarding-test", f"session-{i}", {"input": "test"})
            results.add(result.key)

        # With 50/50 split and 100 sessions, we should see both prompts
        assert len(results) == 2
        assert "prompt-a" in results
        assert "prompt-b" in results

    def test_get_ab_sets_prompt_context(self):
        """Should set prompt context with A/B info."""
        from fallom import prompts

        prompts.get_ab("onboarding-test", "session-123", {"input": "test"})

        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["ab_test_key"] == "onboarding-test"
        assert ctx["variant_index"] is not None

    def test_get_ab_raises_for_unknown_test(self):
        """Should raise ValueError for unknown A/B test."""
        from fallom import prompts

        with pytest.raises(ValueError, match="not found"):
            prompts.get_ab("unknown-test", "session-123")


class TestPromptContext:
    """Tests for prompt context (OTEL span tagging)."""

    def test_set_and_get_context(self):
        """Should be able to set and get prompt context."""
        from fallom import prompts

        prompts._set_prompt_context("my-prompt", 1, "my-ab-test", 0)

        ctx = prompts.get_prompt_context()
        assert ctx["prompt_key"] == "my-prompt"
        assert ctx["prompt_version"] == 1
        assert ctx["ab_test_key"] == "my-ab-test"
        assert ctx["variant_index"] == 0

    def test_clear_context(self):
        """Should be able to clear prompt context."""
        from fallom import prompts

        prompts._set_prompt_context("my-prompt", 1)
        prompts.clear_prompt_context()

        assert prompts.get_prompt_context() is None

    def test_context_is_thread_local(self):
        """Context should be isolated per thread/async context."""
        import threading
        from fallom import prompts

        results = []

        def thread_func(name):
            prompts._set_prompt_context(f"prompt-{name}", 1)
            # Small delay to interleave
            import time
            time.sleep(0.01)
            ctx = prompts.get_prompt_context()
            results.append((name, ctx["prompt_key"]))

        threads = [
            threading.Thread(target=thread_func, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have its own context
        for name, prompt_key in results:
            assert prompt_key == f"prompt-{name}"


class TestFetchPrompts:
    """Tests for _fetch_prompts()"""

    def test_fetch_populates_cache(self):
        """Should populate cache from API response."""
        from fallom import prompts
        prompts._api_key = "test-key"
        prompts._base_url = "https://api.test.com"
        prompts._prompt_cache = {}

        with patch("fallom.prompts.requests.get") as mock_get:
            mock_get.return_value.ok = True
            mock_get.return_value.json.return_value = {
                "prompts": [{
                    "key": "new-prompt",
                    "version": 1,
                    "system_prompt": "System",
                    "user_template": "User"
                }]
            }

            prompts._fetch_prompts()

            assert "new-prompt" in prompts._prompt_cache
            assert prompts._prompt_cache["new-prompt"]["current"] == 1

    def test_fetch_fails_silently(self):
        """Should not raise on network errors."""
        from fallom import prompts
        prompts._api_key = "test-key"
        prompts._base_url = "https://api.test.com"

        with patch("fallom.prompts.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            # Should not raise
            prompts._fetch_prompts()


class TestDeterministicHash:
    """Tests for deterministic session-based selection."""

    def test_hash_is_deterministic(self):
        """Hash should produce same result for same input."""
        import hashlib

        session_id = "test-session-12345"

        # Compute hash multiple times
        results = []
        for _ in range(10):
            hash_bytes = hashlib.md5(session_id.encode()).digest()
            hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 1_000_000
            results.append(hash_val)

        # All should be same
        assert len(set(results)) == 1

    def test_hash_distributes_evenly(self):
        """Hash should distribute relatively evenly."""
        import hashlib

        # Generate 10000 different session IDs
        buckets = {"low": 0, "high": 0}
        for i in range(10000):
            session_id = f"session-{i}"
            hash_bytes = hashlib.md5(session_id.encode()).digest()
            hash_val = int.from_bytes(hash_bytes[:4], byteorder='big') % 1_000_000

            if hash_val < 500_000:
                buckets["low"] += 1
            else:
                buckets["high"] += 1

        # Should be roughly 50/50 (allow 5% tolerance)
        assert 4500 < buckets["low"] < 5500
        assert 4500 < buckets["high"] < 5500

