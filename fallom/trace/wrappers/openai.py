"""
OpenAI SDK Wrapper

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


def wrap_openai(client: T, session_ctx: SessionContext) -> T:
    """
    Wrap an OpenAI client to automatically trace all chat completions.
    
    Args:
        client: An OpenAI client instance
        session_ctx: Session context for the trace
        
    Returns:
        The wrapped client (same type as input)
    """
    original_create = client.chat.completions.create
    ctx = session_ctx
    
    @wraps(original_create)
    def traced_create(*args, **kwargs):
        if not is_initialized():
            return original_create(*args, **kwargs)
        
        trace_ctx_storage = get_trace_context_storage()
        trace_ctx = trace_ctx_storage.get() or get_fallback_trace_context()
        trace_id = trace_ctx.trace_id if trace_ctx else generate_hex_id(32)
        span_id = generate_hex_id(16)
        parent_span_id = trace_ctx.parent_span_id if trace_ctx else None
        
        params = kwargs if kwargs else (args[0] if args else {})
        start_time = int(time.time() * 1000)
        capture_content = should_capture_content()
        
        try:
            response = original_create(*args, **kwargs)
            end_time = int(time.time() * 1000)
            
            # SDK is dumb - just send raw data
            attributes = {
                "fallom.sdk_version": "2",
                "fallom.method": "chat.completions.create",
            }
            
            if capture_content:
                # Send ALL request data - microservice extracts what it needs
                attributes["fallom.raw.request"] = json.dumps({
                    "messages": params.get("messages"),
                    "model": params.get("model"),
                    "tools": params.get("tools"),
                    "tool_choice": params.get("tool_choice"),
                    "functions": params.get("functions"),
                    "function_call": params.get("function_call"),
                })
                
                # Send ALL response data including tool calls
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice else None
                
                response_data = {
                    "text": message.content if message else None,
                    "finishReason": choice.finish_reason if choice else None,
                    "responseId": response.id,
                    "model": response.model,
                }
                
                # Tool calls
                if message and message.tool_calls:
                    response_data["toolCalls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                
                if message and hasattr(message, "function_call") and message.function_call:
                    response_data["functionCall"] = {
                        "name": message.function_call.name,
                        "arguments": message.function_call.arguments,
                    }
                
                attributes["fallom.raw.response"] = json.dumps(response_data)
            
            if response.usage:
                attributes["fallom.raw.usage"] = json.dumps({
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                })
            
            # Build waterfall timing data
            waterfall_timings = {
                "requestStart": 0,
                "requestEnd": end_time - start_time,
                "responseEnd": end_time - start_time,
                "totalDurationMs": end_time - start_time,
            }
            
            # Add tool calls to timing
            choice = response.choices[0] if response.choices else None
            if choice and choice.message and choice.message.tool_calls:
                waterfall_timings["toolCalls"] = [
                    {"id": tc.id, "name": tc.function.name, "callTime": 0}
                    for tc in choice.message.tool_calls
                ]
            
            attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
            
            # Get prompt context if set (one-shot, clears after read)
            try:
                from fallom.prompts import get_prompt_context
                prompt_ctx = get_prompt_context()
            except ImportError:
                prompt_ctx = None
            
            trace_data = TraceData(
                config_key=ctx.config_key,
                session_id=ctx.session_id,
                customer_id=ctx.customer_id,
                metadata=ctx.metadata,
                tags=ctx.tags,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name="chat.completions.create",
                kind="llm",
                model=response.model or params.get("model"),
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
            
            trace_data = TraceData(
                config_key=ctx.config_key,
                session_id=ctx.session_id,
                customer_id=ctx.customer_id,
                metadata=ctx.metadata,
                tags=ctx.tags,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name="chat.completions.create",
                kind="llm",
                model=params.get("model"),
                start_time=timestamp_to_iso(start_time),
                end_time=timestamp_to_iso(end_time),
                duration_ms=end_time - start_time,
                status="ERROR",
                error_message=str(error),
                attributes={
                    "fallom.sdk_version": "2",
                    "fallom.method": "chat.completions.create",
                },
            )
            
            send_trace(trace_data)
            
            raise
    
    # Replace the method on the client
    client.chat.completions.create = traced_create
    
    return client

