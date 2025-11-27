"""
Example: Using only tracing (no A/B testing).

Use this when you want to trace your LLM calls but aren't 
doing model A/B testing yet.
"""
import os
from fallom import trace

# Set your API key
os.environ["FALLOM_API_KEY"] = "your-api-key"

# Initialize just tracing
trace.init()


def handle_message(user_id: str, convo_id: str, message: str):
    """Handle an incoming message with tracing."""
    session_id = f"{user_id}-{convo_id}"
    
    # Set trace context - all LLM calls will be grouped by this session
    trace.set_session("my-agent", session_id)
    
    # Your agent code here - all LLM calls automatically traced
    # 
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": message}]
    # )
    
    return "Response"


def with_custom_metrics(user_id: str, convo_id: str, message: str):
    """Handle a message and add custom business metrics."""
    session_id = f"{user_id}-{convo_id}"
    
    trace.set_session("my-agent", session_id)
    
    # Your agent code...
    result = "Agent response here"
    
    # Add custom metrics after processing
    # These get aggregated in the dashboard
    trace.span({
        "response_quality": 0.85,
        "user_satisfaction": 4.5,
        "was_helpful": True
    })
    
    return result


if __name__ == "__main__":
    # Basic tracing
    handle_message("user_123", "convo_456", "Hello!")
    
    # With custom metrics
    with_custom_metrics("user_123", "convo_789", "Help me with something")
    
    print("Messages handled with tracing!")

