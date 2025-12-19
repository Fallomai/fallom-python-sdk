"""
Google AI SDK Wrapper

SDK is "dumb" - just captures raw request/response and sends to microservice.
All parsing/extraction happens server-side for easier maintenance.
"""
import json
import time
from typing import Any, TypeVar
from functools import wraps

from ..core import (
    get_trace_context_storage,
    get_fallback_trace_context,
    is_initialized,
    should_capture_content,
    send_trace,
)
from ..utils import generate_hex_id, timestamp_to_iso
from ..types import SessionContext, TraceData

T = TypeVar("T")


def wrap_google_ai(model: T, session_ctx: SessionContext) -> T:
    """
    Wrap a Google AI model to automatically trace generateContent calls.
    
    Args:
        model: A Google AI GenerativeModel instance
        session_ctx: Session context for the trace
        
    Returns:
        The wrapped model (same type as input)
    """
    original_generate_content = model.generate_content
    ctx = session_ctx
    
    @wraps(original_generate_content)
    def traced_generate_content(*args, **kwargs):
        if not is_initialized():
            return original_generate_content(*args, **kwargs)
        
        trace_ctx_storage = get_trace_context_storage()
        trace_ctx = trace_ctx_storage.get() or get_fallback_trace_context()
        trace_id = trace_ctx.trace_id if trace_ctx else generate_hex_id(32)
        span_id = generate_hex_id(16)
        parent_span_id = trace_ctx.parent_span_id if trace_ctx else None
        
        request = args[0] if args else kwargs.get("contents")
        start_time = int(time.time() * 1000)
        capture_content = should_capture_content()
        
        try:
            response = original_generate_content(*args, **kwargs)
            end_time = int(time.time() * 1000)
            
            # SDK is dumb - just send raw data
            attributes = {
                "fallom.sdk_version": "2",
                "fallom.method": "generateContent",
            }
            
            # Extract function calls from candidates
            candidates = response.candidates if response.candidates else []
            function_calls = []
            
            for candidate in candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append({
                                "name": part.function_call.name,
                                "arguments": dict(part.function_call.args) if part.function_call.args else {},
                            })
            
            if capture_content:
                # Serialize request (handle different input types)
                try:
                    if isinstance(request, str):
                        request_data = request
                    elif isinstance(request, list):
                        request_data = [str(r) for r in request]
                    else:
                        request_data = str(request)
                except:
                    request_data = str(request)
                
                attributes["fallom.raw.request"] = json.dumps(request_data)
                
                response_data = {
                    "text": response.text if hasattr(response, "text") else None,
                    "candidates": len(candidates),
                    "finishReason": candidates[0].finish_reason.name if candidates and candidates[0].finish_reason else None,
                }
                
                if function_calls:
                    response_data["toolCalls"] = function_calls
                
                attributes["fallom.raw.response"] = json.dumps(response_data)
            
            # Extract usage metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                attributes["fallom.raw.usage"] = json.dumps({
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count,
                })
            
            # Build waterfall timing data
            waterfall_timings = {
                "requestStart": 0,
                "requestEnd": end_time - start_time,
                "responseEnd": end_time - start_time,
                "totalDurationMs": end_time - start_time,
                "toolCalls": [
                    {"name": fc["name"], "callTime": 0}
                    for fc in function_calls
                ],
            }
            attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
            
            # Get prompt context if set
            try:
                from fallom.prompts import get_prompt_context
                prompt_ctx = get_prompt_context()
            except ImportError:
                prompt_ctx = None
            
            # Try to get model name
            model_name = getattr(model, "model_name", None) or getattr(model, "_model_name", None) or "gemini"
            
            trace_data = TraceData(
                config_key=ctx.config_key,
                session_id=ctx.session_id,
                customer_id=ctx.customer_id,
                metadata=ctx.metadata,
                tags=ctx.tags,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name="generateContent",
                kind="llm",
                model=model_name,
                start_time=timestamp_to_iso(start_time),
                end_time=timestamp_to_iso(end_time),
                duration_ms=end_time - start_time,
                status="OK",
                attributes=attributes,
                prompt_key=prompt_ctx.get("prompt_key") if prompt_ctx else None,
                prompt_version=prompt_ctx.get("prompt_version") if prompt_ctx else None,
                prompt_ab_test_key=prompt_ctx.get("ab_test_key") if prompt_ctx else None,
                prompt_variant_index=prompt_ctx.get("variant_index") if prompt_ctx else None,
            )
            
            send_trace(trace_data)
            
            return response
            
        except Exception as error:
            end_time = int(time.time() * 1000)
            
            model_name = getattr(model, "model_name", None) or getattr(model, "_model_name", None) or "gemini"
            
            trace_data = TraceData(
                config_key=ctx.config_key,
                session_id=ctx.session_id,
                customer_id=ctx.customer_id,
                metadata=ctx.metadata,
                tags=ctx.tags,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name="generateContent",
                kind="llm",
                model=model_name,
                start_time=timestamp_to_iso(start_time),
                end_time=timestamp_to_iso(end_time),
                duration_ms=end_time - start_time,
                status="ERROR",
                error_message=str(error),
                attributes={
                    "fallom.sdk_version": "2",
                    "fallom.method": "generateContent",
                },
            )
            
            send_trace(trace_data)
            
            raise
    
    # Replace the method on the model
    model.generate_content = traced_generate_content
    
    return model

