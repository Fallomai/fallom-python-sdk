"""
Integration tests for fallom.prompts module.

Run with: 
    FALLOM_API_KEY=your-api-key pytest tests/test_integration.py -v

Or set env vars in your shell:
    export FALLOM_API_KEY=your-api-key
    export FALLOM_BASE_URL=https://prompts.fallom.com  # optional

Configurable test data (set via env vars):
    FALLOM_TEST_PROMPT_KEY - prompt key for prompts.get() tests
    FALLOM_TEST_PROMPT_AB_KEY - A/B test key for prompts.get_ab() tests
"""
import os
import pytest

# Validate required env vars
if not os.environ.get("FALLOM_API_KEY"):
    pytest.skip("FALLOM_API_KEY environment variable required", allow_module_level=True)

# Set default base URL if not provided
os.environ.setdefault("FALLOM_BASE_URL", "https://prompts.fallom.com")

# Test data configuration (matches TypeScript integration tests)
TEST_PROMPT_KEY = os.environ.get("FALLOM_TEST_PROMPT_KEY", "email-writer")
TEST_PROMPT_AB_KEY = os.environ.get("FALLOM_TEST_PROMPT_AB_KEY", "integration-test")


class TestPromptsGetIntegration:
    """Integration tests for prompts.get()"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize prompts module before each test."""
        from fallom import prompts
        prompts._initialized = False  # Reset state
        prompts._prompt_cache = {}
        prompts._prompt_ab_cache = {}
        prompts.init()
        # Synchronously fetch data (don't rely on background thread)
        prompts._fetch_prompts()
        prompts._fetch_prompt_ab_tests()
        yield
        prompts.clear_prompt_context()

    def test_get_basic_prompt(self):
        """Should fetch a prompt from the server."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY)
        
        assert result.key == TEST_PROMPT_KEY
        assert result.version >= 1
        # At least one of system/user should be present
        assert result.system is not None or result.user is not None

    def test_get_with_variables(self):
        """Should replace {{variables}} in templates."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {
            "user_name": "Alice",
            "company": "TestCorp",
            "conversation_history": "User: Hello",
            "user_message": "How are you?"
        })
        
        # Should return a valid prompt
        assert result.key == TEST_PROMPT_KEY

    def test_get_specific_version(self):
        """Should fetch a specific version when requested."""
        from fallom import prompts
        
        # Get the latest version first, then request it specifically
        latest = prompts.get(TEST_PROMPT_KEY)
        result = prompts.get(TEST_PROMPT_KEY, version=latest.version)
        
        assert result.version == latest.version
        assert result.key == TEST_PROMPT_KEY

    def test_get_unknown_prompt_raises(self):
        """Should raise ValueError for non-existent prompt."""
        from fallom import prompts
        
        with pytest.raises(ValueError, match="not found"):
            prompts.get("this-prompt-does-not-exist-xyz")

    def test_get_sets_prompt_context(self):
        """Should set context for OTEL span tagging."""
        from fallom import prompts
        
        prompts.get(TEST_PROMPT_KEY)
        
        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["prompt_key"] == TEST_PROMPT_KEY
        assert ctx["prompt_version"] >= 1
        assert ctx["ab_test_key"] is None

    def test_get_empty_variables(self):
        """Should handle empty variables dict."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {})
        
        assert result.key == TEST_PROMPT_KEY
        # Should still work, unreplaced variables stay as-is

    def test_get_extra_variables_ignored(self):
        """Should ignore variables not in template."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {
            "user_name": "Bob",
            "unused_var": "should be ignored",
            "another_unused": 12345
        })
        
        assert result.key == TEST_PROMPT_KEY
        # Should not crash


class TestPromptsGetABIntegration:
    """Integration tests for prompts.get_ab()"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize prompts module before each test."""
        from fallom import prompts
        prompts._initialized = False
        prompts._prompt_cache = {}
        prompts._prompt_ab_cache = {}
        prompts.init()
        # Synchronously fetch data
        prompts._fetch_prompts()
        prompts._fetch_prompt_ab_tests()
        yield
        prompts.clear_prompt_context()

    def test_get_ab_basic(self):
        """Should fetch a prompt from an A/B test."""
        from fallom import prompts
        
        result = prompts.get_ab(TEST_PROMPT_AB_KEY, "session-123")
        
        assert result.ab_test_key == TEST_PROMPT_AB_KEY
        assert result.variant_index in [0, 1]
        assert result.key is not None

    def test_get_ab_is_deterministic(self):
        """Same session_id should always return same variant."""
        from fallom import prompts
        
        session_id = "deterministic-test-session-999"
        
        results = [
            prompts.get_ab(TEST_PROMPT_AB_KEY, session_id)
            for _ in range(10)
        ]
        
        # All should return same prompt key
        keys = [r.key for r in results]
        assert len(set(keys)) == 1, f"Expected same key for all, got: {keys}"
        
        # All should return same variant index
        variants = [r.variant_index for r in results]
        assert len(set(variants)) == 1

    def test_get_ab_different_sessions_distribute(self):
        """Different sessions should distribute across variants."""
        from fallom import prompts
        
        results = {}
        for i in range(100):
            result = prompts.get_ab(TEST_PROMPT_AB_KEY, f"distribution-test-{i}")
            # Use variant_index since multiple variants may use same prompt key
            variant = result.variant_index
            results[variant] = results.get(variant, 0) + 1
        
        # With 50/50 split, we should see both variants
        assert len(results) == 2, f"Expected 2 variants, got: {results}"
        
        # Each should have at least 20% (allowing for hash distribution variance)
        for variant, count in results.items():
            assert count >= 20, f"Variant {variant} only got {count}/100 sessions"

    def test_get_ab_with_variables(self):
        """Should replace variables in A/B test prompts."""
        from fallom import prompts
        
        result = prompts.get_ab(TEST_PROMPT_AB_KEY, "session-456", {
            "user_name": "Charlie",
            "conversation_history": "User: Hello",
            "user_message": "How are you?"
        })
        
        # Should return a valid prompt
        assert result.key is not None

    def test_get_ab_unknown_test_raises(self):
        """Should raise ValueError for non-existent A/B test."""
        from fallom import prompts
        
        with pytest.raises(ValueError, match="not found"):
            prompts.get_ab("this-ab-test-does-not-exist-xyz", "session-1")

    def test_get_ab_sets_prompt_context(self):
        """Should set context with A/B test info."""
        from fallom import prompts
        
        prompts.get_ab(TEST_PROMPT_AB_KEY, "session-789")
        
        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["ab_test_key"] == TEST_PROMPT_AB_KEY
        assert ctx["variant_index"] is not None
        assert ctx["prompt_key"] is not None


class TestPromptsCaching:
    """Integration tests for caching behavior."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fallom import prompts
        prompts._initialized = False
        prompts._prompt_cache = {}
        prompts._prompt_ab_cache = {}
        prompts.init()
        prompts._fetch_prompts()
        prompts._fetch_prompt_ab_tests()
        yield

    def test_second_get_uses_cache(self):
        """Second get() should use cache (faster)."""
        from fallom import prompts
        import time
        
        # First call - may hit network
        start1 = time.time()
        prompts.get(TEST_PROMPT_KEY)
        time1 = time.time() - start1
        
        # Second call - should use cache
        start2 = time.time()
        prompts.get(TEST_PROMPT_KEY)
        time2 = time.time() - start2
        
        # Cache should be much faster (or at least not slower)
        # We're just verifying it doesn't crash, timing is informational
        print(f"\nFirst call: {time1*1000:.2f}ms, Second call: {time2*1000:.2f}ms")

    def test_cache_persists_across_calls(self):
        """Cache should persist for multiple get() calls."""
        from fallom import prompts
        
        result1 = prompts.get(TEST_PROMPT_KEY)
        result2 = prompts.get(TEST_PROMPT_KEY)
        result3 = prompts.get(TEST_PROMPT_KEY)
        
        assert result1.key == result2.key == result3.key
        assert result1.version == result2.version == result3.version


class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fallom import prompts
        prompts._initialized = False
        prompts._prompt_cache = {}
        prompts._prompt_ab_cache = {}
        prompts.init()
        prompts._fetch_prompts()
        prompts._fetch_prompt_ab_tests()
        yield
        prompts.clear_prompt_context()

    def test_special_characters_in_variables(self):
        """Should handle special characters in variable values."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {
            "user_name": "O'Brien <script>alert('xss')</script>",
            "company": "Test & Co. \"quoted\""
        })
        
        # Should not crash
        assert result.key == TEST_PROMPT_KEY

    def test_unicode_in_variables(self):
        """Should handle unicode in variable values."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {
            "user_name": "日本語ユーザー",
            "company": "Ñoño Corp 🚀"
        })
        
        assert result.key == TEST_PROMPT_KEY

    def test_very_long_session_id(self):
        """Should handle very long session IDs."""
        from fallom import prompts
        
        long_session = "session-" + "x" * 10000
        
        result = prompts.get_ab(TEST_PROMPT_AB_KEY, long_session)
        
        assert result.ab_test_key == TEST_PROMPT_AB_KEY

    def test_empty_session_id(self):
        """Should handle empty session ID."""
        from fallom import prompts
        
        result = prompts.get_ab(TEST_PROMPT_AB_KEY, "")
        
        # Empty string should still hash deterministically
        assert result.ab_test_key == TEST_PROMPT_AB_KEY

    def test_numeric_variable_values(self):
        """Should convert numeric values to strings."""
        from fallom import prompts
        
        result = prompts.get(TEST_PROMPT_KEY, {
            "count": 42,
            "price": 19.99,
            "negative": -100
        })
        
        assert result.key == TEST_PROMPT_KEY


if __name__ == "__main__":
    print("Run: pytest tests/test_integration.py -v")

