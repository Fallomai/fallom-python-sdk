"""
Basic example: Using both tracing and model A/B testing.

This shows the typical usage pattern where you:
1. Initialize Fallom
2. Get a model assignment for each session
3. Let auto-instrumentation capture all LLM calls
"""
import os
import fallom
from fallom import models

# Set your API key (or use environment variable)
os.environ["FALLOM_API_KEY"] = "your-api-key"

# Initialize everything (tracing + models)
fallom.init()


def handle_message(user_id: str, convo_id: str, message: str):
    """
    Handle an incoming message.
    
    This is your typical message handler in an agent/chatbot application.
    """
    # Create a unique session ID for this conversation
    session_id = f"{user_id}-{convo_id}"
    
    # Get model assignment (zero latency - uses local hash + cached config)
    # This also automatically sets the trace context
    model = models.get("my-agent", session_id)
    
    print(f"Session {session_id} assigned to model: {model}")
    
    # Your agent code here - all LLM calls are automatically traced!
    # Example with OpenAI:
    # 
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": message}]
    # )
    # return response.choices[0].message.content
    
    # Example with Anthropic:
    #
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": message}]
    # )
    # return response.content[0].text
    
    return f"Response using {model}"


# Simulate some conversations
if __name__ == "__main__":
    # Same user, same conversation = same model (sticky)
    response1 = handle_message("user_123", "convo_456", "Hello!")
    response2 = handle_message("user_123", "convo_456", "How are you?")
    
    # Different conversation = might get different model (A/B test)
    response3 = handle_message("user_123", "convo_789", "New conversation")
    
    print("\nAll responses:")
    print(f"  1: {response1}")
    print(f"  2: {response2}")
    print(f"  3: {response3}")

