"""
Example: Using only model A/B testing (minimal tracing).

Use this when you just want to A/B test models and don't need
full OTEL tracing (though basic session tracking still happens).
"""
import os
from fallom import models

# Set your API key
os.environ["FALLOM_API_KEY"] = "your-api-key"

# Initialize just models (tracing won't be fully set up)
models.init()


def handle_message(user_id: str, convo_id: str, message: str):
    """Handle an incoming message with A/B tested model."""
    session_id = f"{user_id}-{convo_id}"
    
    # Get model assignment - same session always gets same model
    model = models.get("my-agent", session_id)
    
    print(f"Using model: {model}")
    
    # Use the assigned model in your agent
    # 
    # if model.startswith("claude"):
    #     # Use Anthropic
    #     client = anthropic.Anthropic()
    #     response = client.messages.create(model=model, ...)
    # elif model.startswith("gpt"):
    #     # Use OpenAI
    #     client = OpenAI()
    #     response = client.chat.completions.create(model=model, ...)
    
    return f"Response from {model}"


def check_assignment_consistency():
    """
    Demonstrate that the same session always gets the same model.
    """
    session_id = "user_test-convo_test"
    
    # Call 10 times - should always return the same model
    assignments = []
    for i in range(10):
        model = models.get("my-agent", session_id)
        assignments.append(model)
    
    # All assignments should be identical
    assert len(set(assignments)) == 1, "Assignment should be sticky!"
    print(f"All 10 calls returned: {assignments[0]} ✓")


if __name__ == "__main__":
    # Basic usage
    handle_message("user_123", "convo_456", "Hello!")
    
    # Verify sticky assignment
    check_assignment_consistency()

