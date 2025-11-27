"""
Example: Adding custom metrics in a batch job.

Use this pattern when you want to compute and attach metrics
outside of the normal request flow, like in a nightly job.

Common use cases:
- Computing outlier scores
- Calculating engagement metrics
- Adding quality scores from human review
- Attaching conversion data
"""
import os
from fallom import trace

# Set your API key
os.environ["FALLOM_API_KEY"] = "your-api-key"

# Initialize tracing
trace.init()


def nightly_outlier_detection():
    """
    Example: Run this as a cron job every night to compute
    outlier scores for yesterday's sessions.
    """
    # In reality, you'd fetch this from your database
    sessions = [
        {"session_id": "user_123-convo_456", "outlier_score": 0.82},
        {"session_id": "user_123-convo_789", "outlier_score": 0.15},
        {"session_id": "user_456-convo_001", "outlier_score": 0.65},
        {"session_id": "user_789-convo_002", "outlier_score": 0.91},
    ]
    
    for session in sessions:
        # Attach outlier score to the session
        # This will show up in dashboard aggregations
        trace.span(
            {"outlier_score": session["outlier_score"]},
            config_key="my-agent",
            session_id=session["session_id"]
        )
    
    print(f"Updated {len(sessions)} sessions with outlier scores")


def engagement_metrics_update():
    """
    Example: Update sessions with engagement metrics
    computed from downstream analytics.
    """
    # Fetch from your analytics system
    engagement_data = [
        {
            "session_id": "user_123-convo_456",
            "messages_sent": 12,
            "session_duration_minutes": 8.5,
            "returned_within_24h": True
        },
        {
            "session_id": "user_456-convo_001",
            "messages_sent": 3,
            "session_duration_minutes": 1.2,
            "returned_within_24h": False
        },
    ]
    
    for data in engagement_data:
        session_id = data.pop("session_id")
        trace.span(
            data,
            config_key="my-agent",
            session_id=session_id
        )
    
    print(f"Updated {len(engagement_data)} sessions with engagement metrics")


def human_review_scores():
    """
    Example: Attach human review scores to sessions
    after manual QA review.
    """
    reviewed_sessions = [
        {"session_id": "user_123-convo_456", "quality_score": 4, "was_accurate": True},
        {"session_id": "user_789-convo_002", "quality_score": 2, "was_accurate": False},
    ]
    
    for review in reviewed_sessions:
        session_id = review.pop("session_id")
        trace.span(
            review,
            config_key="my-agent",
            session_id=session_id
        )
    
    print(f"Updated {len(reviewed_sessions)} sessions with human review scores")


def conversion_tracking():
    """
    Example: Track which sessions led to conversions
    (signups, purchases, etc.)
    """
    # From your conversion tracking system
    conversions = [
        {"session_id": "user_456-convo_001", "converted": True, "revenue": 49.99},
        {"session_id": "user_789-convo_002", "converted": False, "revenue": 0},
    ]
    
    for conv in conversions:
        session_id = conv.pop("session_id")
        trace.span(
            conv,
            config_key="my-agent",
            session_id=session_id
        )
    
    print(f"Updated {len(conversions)} sessions with conversion data")


if __name__ == "__main__":
    print("Running batch metric updates...\n")
    
    nightly_outlier_detection()
    engagement_metrics_update()
    human_review_scores()
    conversion_tracking()
    
    print("\nAll batch jobs complete!")
    print("Check your Fallom dashboard to see aggregated metrics by model.")

