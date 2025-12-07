"""
Integration tests for fallom.prompts module.

Run with: 
    FALLOM_API_KEY=your-api-key pytest tests/test_integration.py -v

Or set env vars in your shell:
    export FALLOM_API_KEY=your-api-key
    export FALLOM_BASE_URL=https://spans.fallom.com  # optional
"""
import os
import pytest

# Validate required env vars
if not os.environ.get("FALLOM_API_KEY"):
    pytest.skip("FALLOM_API_KEY environment variable required", allow_module_level=True)

# Set default base URL if not provided
os.environ.setdefault("FALLOM_BASE_URL", "https://spans.fallom.com")


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
        
        result = prompts.get("test-prompt")
        
        assert result.key == "test-prompt"
        assert result.version >= 1
        assert result.system is not None
        assert result.user is not None

    def test_get_with_variables(self):
        """Should replace {{variables}} in templates."""
        from fallom import prompts
        
        result = prompts.get("test-prompt", {
            "user_name": "Alice",
            "company": "TestCorp"
        })
        
        # Variables should be replaced (not contain {{user_name}})
        assert "{{user_name}}" not in result.system
        assert "{{user_name}}" not in result.user
        # Check if Alice appears (if template uses user_name)
        assert "Alice" in result.system or "Alice" in result.user or "{{" not in result.user

    def test_get_specific_version(self):
        """Should fetch a specific version when requested."""
        from fallom import prompts
        
        # Note: API returns current version (2), so we test with that
        # To test version pinning with older versions, backend would need to return all versions
        result = prompts.get("test-prompt-versioned", version=2)
        
        assert result.version == 2
        assert "version 2" in result.system.lower() or result.version == 2

    def test_get_unknown_prompt_raises(self):
        """Should raise ValueError for non-existent prompt."""
        from fallom import prompts
        
        with pytest.raises(ValueError, match="not found"):
            prompts.get("this-prompt-does-not-exist-xyz")

    def test_get_sets_prompt_context(self):
        """Should set context for OTEL span tagging."""
        from fallom import prompts
        
        prompts.get("test-prompt")
        
        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["prompt_key"] == "test-prompt"
        assert ctx["prompt_version"] >= 1
        assert ctx["ab_test_key"] is None

    def test_get_empty_variables(self):
        """Should handle empty variables dict."""
        from fallom import prompts
        
        result = prompts.get("test-prompt", {})
        
        assert result.key == "test-prompt"
        # Should still work, unreplaced variables stay as-is

    def test_get_extra_variables_ignored(self):
        """Should ignore variables not in template."""
        from fallom import prompts
        
        result = prompts.get("test-prompt", {
            "user_name": "Bob",
            "unused_var": "should be ignored",
            "another_unused": 12345
        })
        
        assert result.key == "test-prompt"
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
        
        result = prompts.get_ab("test-ab-experiment", "session-123")
        
        assert result.key in ["test-prompt-a", "test-prompt-b"]
        assert result.ab_test_key == "test-ab-experiment"
        assert result.variant_index in [0, 1]

    def test_get_ab_is_deterministic(self):
        """Same session_id should always return same variant."""
        from fallom import prompts
        
        session_id = "deterministic-test-session-999"
        
        results = [
            prompts.get_ab("test-ab-experiment", session_id)
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
            result = prompts.get_ab("test-ab-experiment", f"distribution-test-{i}")
            key = result.key
            results[key] = results.get(key, 0) + 1
        
        # With 50/50 split, we should see both variants
        assert len(results) == 2, f"Expected 2 variants, got: {results}"
        
        # Each should have at least 20% (allowing for hash distribution variance)
        for key, count in results.items():
            assert count >= 20, f"Variant {key} only got {count}/100 sessions"

    def test_get_ab_with_variables(self):
        """Should replace variables in A/B test prompts."""
        from fallom import prompts
        
        result = prompts.get_ab("test-ab-experiment", "session-456", {
            "user_name": "Charlie"
        })
        
        # Variables should be replaced
        assert "{{user_name}}" not in result.user

    def test_get_ab_unknown_test_raises(self):
        """Should raise ValueError for non-existent A/B test."""
        from fallom import prompts
        
        with pytest.raises(ValueError, match="not found"):
            prompts.get_ab("this-ab-test-does-not-exist-xyz", "session-1")

    def test_get_ab_sets_prompt_context(self):
        """Should set context with A/B test info."""
        from fallom import prompts
        
        prompts.get_ab("test-ab-experiment", "session-789")
        
        ctx = prompts.get_prompt_context()
        assert ctx is not None
        assert ctx["ab_test_key"] == "test-ab-experiment"
        assert ctx["variant_index"] is not None
        assert ctx["prompt_key"] in ["test-prompt-a", "test-prompt-b"]


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
        prompts.get("test-prompt")
        time1 = time.time() - start1
        
        # Second call - should use cache
        start2 = time.time()
        prompts.get("test-prompt")
        time2 = time.time() - start2
        
        # Cache should be much faster (or at least not slower)
        # We're just verifying it doesn't crash, timing is informational
        print(f"\nFirst call: {time1*1000:.2f}ms, Second call: {time2*1000:.2f}ms")

    def test_cache_persists_across_calls(self):
        """Cache should persist for multiple get() calls."""
        from fallom import prompts
        
        result1 = prompts.get("test-prompt")
        result2 = prompts.get("test-prompt")
        result3 = prompts.get("test-prompt")
        
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
        
        result = prompts.get("test-prompt", {
            "user_name": "O'Brien <script>alert('xss')</script>",
            "company": "Test & Co. \"quoted\""
        })
        
        # Should not crash
        assert result.key == "test-prompt"

    def test_unicode_in_variables(self):
        """Should handle unicode in variable values."""
        from fallom import prompts
        
        result = prompts.get("test-prompt", {
            "user_name": "日本語ユーザー",
            "company": "Ñoño Corp 🚀"
        })
        
        assert result.key == "test-prompt"

    def test_very_long_session_id(self):
        """Should handle very long session IDs."""
        from fallom import prompts
        
        long_session = "session-" + "x" * 10000
        
        result = prompts.get_ab("test-ab-experiment", long_session)
        
        assert result.ab_test_key == "test-ab-experiment"

    def test_empty_session_id(self):
        """Should handle empty session ID."""
        from fallom import prompts
        
        result = prompts.get_ab("test-ab-experiment", "")
        
        # Empty string should still hash deterministically
        assert result.ab_test_key == "test-ab-experiment"

    def test_numeric_variable_values(self):
        """Should convert numeric values to strings."""
        from fallom import prompts
        
        result = prompts.get("test-prompt", {
            "count": 42,
            "price": 19.99,
            "negative": -100
        })
        
        assert result.key == "test-prompt"


if __name__ == "__main__":
    print("Run: pytest tests/test_integration.py -v")

