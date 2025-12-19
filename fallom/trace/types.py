"""
Type definitions for Fallom tracing module.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class SessionContext:
    """Session context for grouping traces."""
    config_key: str
    session_id: str
    customer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


@dataclass 
class TraceContext:
    """Trace context for linking spans together."""
    trace_id: str
    parent_span_id: Optional[str] = None


@dataclass
class WaterfallTimings:
    """
    Detailed timing breakdown for waterfall visualization.
    All times are in milliseconds relative to request_start (which is 0).
    """
    request_start: int = 0
    request_end: int = 0
    response_end: int = 0
    first_token_time: Optional[int] = None
    steps: Optional[List[Dict]] = None


@dataclass
class TraceData:
    """
    Data structure for a trace sent to the Fallom API.
    
    SDK sends minimal structured data + raw attributes.
    Microservice extracts tokens, costs, previews, etc. from attributes.
    """
    # Required identifiers
    config_key: str
    session_id: str
    trace_id: str
    span_id: str
    
    # Basic span info
    name: str
    start_time: str
    end_time: str
    duration_ms: int
    status: str  # "OK" or "ERROR"
    
    # Optional identifiers
    customer_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    # Model info
    kind: Optional[str] = None
    model: Optional[str] = None
    error_message: Optional[str] = None
    
    # Streaming info
    time_to_first_token_ms: Optional[int] = None
    is_streaming: Optional[bool] = None
    
    # Custom metadata and tags for filtering
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    # Raw data container - microservice parses everything from here
    attributes: Optional[Dict[str, Any]] = None
    
    # Prompt management
    prompt_key: Optional[str] = None
    prompt_version: Optional[int] = None
    prompt_ab_test_key: Optional[str] = None
    prompt_variant_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "config_key": self.config_key,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }
        
        # Add optional fields only if they have values
        if self.customer_id is not None:
            result["customer_id"] = self.customer_id
        if self.parent_span_id is not None:
            result["parent_span_id"] = self.parent_span_id
        if self.kind is not None:
            result["kind"] = self.kind
        if self.model is not None:
            result["model"] = self.model
        if self.error_message is not None:
            result["error_message"] = self.error_message
        if self.time_to_first_token_ms is not None:
            result["time_to_first_token_ms"] = self.time_to_first_token_ms
        if self.is_streaming is not None:
            result["is_streaming"] = self.is_streaming
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.tags is not None:
            result["tags"] = self.tags
        if self.attributes is not None:
            result["attributes"] = self.attributes
        if self.prompt_key is not None:
            result["prompt_key"] = self.prompt_key
        if self.prompt_version is not None:
            result["prompt_version"] = self.prompt_version
        if self.prompt_ab_test_key is not None:
            result["prompt_ab_test_key"] = self.prompt_ab_test_key
        if self.prompt_variant_index is not None:
            result["prompt_variant_index"] = self.prompt_variant_index
            
        return result


@dataclass
class SessionOptions:
    """Options for creating a Fallom session."""
    config_key: str
    session_id: str
    customer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

