"""
OpenAI SDK Wrapper

SDK is "dumb" - just captures raw request/response and sends to microservice.
All parsing/extraction happens server-side for easier maintenance.

Supports:
- Chat Completions API (client.chat.completions.create)
- Responses API (client.responses.create)
- Both sync (OpenAI) and async (AsyncOpenAI) clients
"""
import asyncio
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


def _create_trace_data(
    ctx: SessionContext,
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    method_name: str,
    model: str,
    start_time: int,
    end_time: int,
    status: str,
    attributes: dict,
    error_message: str = None,
) -> TraceData:
    """Helper to create TraceData object."""
    # Get prompt context if set (one-shot, clears after read)
    try:
        from fallom.prompts import get_prompt_context
        prompt_ctx = get_prompt_context()
    except ImportError:
        prompt_ctx = None

    return TraceData(
        config_key=ctx.config_key,
        session_id=ctx.session_id,
        customer_id=ctx.customer_id,
        metadata=ctx.metadata,
        tags=ctx.tags,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=method_name,
        kind="llm",
        model=model,
        start_time=timestamp_to_iso(start_time),
        end_time=timestamp_to_iso(end_time),
        duration_ms=end_time - start_time,
        status=status,
        error_message=error_message,
        attributes=attributes,
        prompt_key=prompt_ctx.get("prompt_key") if prompt_ctx else None,
        prompt_version=prompt_ctx.get("prompt_version") if prompt_ctx else None,
        prompt_ab_test_key=prompt_ctx.get("ab_test_key") if prompt_ctx else None,
        prompt_variant_index=prompt_ctx.get("variant_index") if prompt_ctx else None,
    )


def _build_chat_completions_attributes(params: dict, response: Any, capture_content: bool) -> dict:
    """Build attributes for chat.completions.create calls."""
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
    choice = response.choices[0] if response.choices else None
    if choice and choice.message and choice.message.tool_calls:
        tool_calls_timing = [
            {"id": tc.id, "name": tc.function.name, "callTime": 0}
            for tc in choice.message.tool_calls
        ]
    else:
        tool_calls_timing = None

    return attributes, tool_calls_timing


def _build_responses_attributes(params: dict, response: Any, capture_content: bool) -> dict:
    """Build attributes for responses.create calls (OpenAI Responses API)."""
    attributes = {
        "fallom.sdk_version": "2",
        "fallom.method": "responses.create",
    }

    if capture_content:
        # Build request data from Responses API params
        request_data = {
            "model": params.get("model"),
            "input": params.get("input"),
            "instructions": params.get("instructions"),
            "tools": params.get("tools"),
        }
        # Include other optional params if present
        for key in ["temperature", "max_output_tokens", "top_p", "store", "metadata"]:
            if key in params:
                request_data[key] = params.get(key)

        attributes["fallom.raw.request"] = json.dumps(request_data)

        # Build response data from Responses API response
        # The response structure is different from Chat Completions
        response_data = {
            "responseId": response.id if hasattr(response, "id") else None,
            "model": response.model if hasattr(response, "model") else None,
            "status": response.status if hasattr(response, "status") else None,
        }

        # Extract output content
        if hasattr(response, "output") and response.output:
            output_items = []
            for item in response.output:
                item_data = {"type": getattr(item, "type", None)}
                if hasattr(item, "content"):
                    # Handle content items (text, etc.)
                    content_parts = []
                    for content in item.content if item.content else []:
                        if hasattr(content, "text"):
                            content_parts.append({"type": "text", "text": content.text})
                        elif hasattr(content, "type"):
                            content_parts.append({"type": content.type})
                    item_data["content"] = content_parts
                if hasattr(item, "name"):
                    item_data["name"] = item.name
                if hasattr(item, "arguments"):
                    item_data["arguments"] = item.arguments
                if hasattr(item, "call_id"):
                    item_data["call_id"] = item.call_id
                output_items.append(item_data)
            response_data["output"] = output_items

            # Extract text for convenience (first text content found)
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text") and content.text:
                            response_data["text"] = content.text
                            break
                    if "text" in response_data:
                        break

        attributes["fallom.raw.response"] = json.dumps(response_data)

    # Handle usage data
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        usage_data = {}
        if hasattr(usage, "input_tokens"):
            usage_data["prompt_tokens"] = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            usage_data["completion_tokens"] = usage.output_tokens
        if hasattr(usage, "total_tokens"):
            usage_data["total_tokens"] = usage.total_tokens
        elif "prompt_tokens" in usage_data and "completion_tokens" in usage_data:
            usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
        
        if usage_data:
            attributes["fallom.raw.usage"] = json.dumps(usage_data)

    return attributes, None  # No tool calls timing for responses API (handled differently)


def wrap_openai(client: T, session_ctx: SessionContext) -> T:
    """
    Wrap an OpenAI client to automatically trace all API calls.
    
    Supports:
    - Chat Completions API: client.chat.completions.create
    - Responses API: client.responses.create
    - Both sync (OpenAI) and async (AsyncOpenAI) clients
    
    Args:
        client: An OpenAI client instance (sync or async)
        session_ctx: Session context for the trace
        
    Returns:
        The wrapped client (same type as input)
    """
    ctx = session_ctx
    
    # Check if this is an async client
    is_async = asyncio.iscoroutinefunction(getattr(client.chat.completions, "create", None))
    
    # Wrap chat.completions.create
    _wrap_chat_completions(client, ctx, is_async)
    
    # Wrap responses.create if available (newer OpenAI SDK versions)
    if hasattr(client, "responses") and hasattr(client.responses, "create"):
        _wrap_responses(client, ctx, is_async)
    
    return client


def _wrap_chat_completions(client: T, ctx: SessionContext, is_async: bool) -> None:
    """Wrap the chat.completions.create method."""
    original_create = client.chat.completions.create
    
    if is_async:
        @wraps(original_create)
        async def traced_create_async(*args, **kwargs):
            if not is_initialized():
                return await original_create(*args, **kwargs)
            
            trace_ctx_storage = get_trace_context_storage()
            trace_ctx = trace_ctx_storage.get() or get_fallback_trace_context()
            trace_id = trace_ctx.trace_id if trace_ctx else generate_hex_id(32)
            span_id = generate_hex_id(16)
            parent_span_id = trace_ctx.parent_span_id if trace_ctx else None
            
            params = kwargs if kwargs else (args[0] if args else {})
            start_time = int(time.time() * 1000)
            capture_content = should_capture_content()
            
            try:
                response = await original_create(*args, **kwargs)
                end_time = int(time.time() * 1000)
                
                attributes, tool_calls_timing = _build_chat_completions_attributes(
                    params, response, capture_content
                )
                
                # Build waterfall timing data
                waterfall_timings = {
                    "requestStart": 0,
                    "requestEnd": end_time - start_time,
                    "responseEnd": end_time - start_time,
                    "totalDurationMs": end_time - start_time,
                }
                if tool_calls_timing:
                    waterfall_timings["toolCalls"] = tool_calls_timing
                attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="chat.completions.create",
                    model=response.model or params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="OK",
                    attributes=attributes,
                )
                
                send_trace(trace_data)
                return response
                
            except Exception as error:
                end_time = int(time.time() * 1000)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="chat.completions.create",
                    model=params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="ERROR",
                    error_message=str(error),
                    attributes={
                        "fallom.sdk_version": "2",
                        "fallom.method": "chat.completions.create",
                    },
                )
                
                send_trace(trace_data)
                raise
        
        client.chat.completions.create = traced_create_async
    else:
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
                
                attributes, tool_calls_timing = _build_chat_completions_attributes(
                    params, response, capture_content
                )
                
                # Build waterfall timing data
                waterfall_timings = {
                    "requestStart": 0,
                    "requestEnd": end_time - start_time,
                    "responseEnd": end_time - start_time,
                    "totalDurationMs": end_time - start_time,
                }
                if tool_calls_timing:
                    waterfall_timings["toolCalls"] = tool_calls_timing
                attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="chat.completions.create",
                    model=response.model or params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="OK",
                    attributes=attributes,
                )
                
                send_trace(trace_data)
                return response
                
            except Exception as error:
                end_time = int(time.time() * 1000)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="chat.completions.create",
                    model=params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="ERROR",
                    error_message=str(error),
                    attributes={
                        "fallom.sdk_version": "2",
                        "fallom.method": "chat.completions.create",
                    },
                )
                
                send_trace(trace_data)
                raise
        
        client.chat.completions.create = traced_create


def _wrap_responses(client: T, ctx: SessionContext, is_async: bool) -> None:
    """Wrap the responses.create method (OpenAI Responses API)."""
    original_create = client.responses.create
    
    if is_async:
        @wraps(original_create)
        async def traced_responses_create_async(*args, **kwargs):
            if not is_initialized():
                return await original_create(*args, **kwargs)
            
            trace_ctx_storage = get_trace_context_storage()
            trace_ctx = trace_ctx_storage.get() or get_fallback_trace_context()
            trace_id = trace_ctx.trace_id if trace_ctx else generate_hex_id(32)
            span_id = generate_hex_id(16)
            parent_span_id = trace_ctx.parent_span_id if trace_ctx else None
            
            params = kwargs if kwargs else (args[0] if args else {})
            start_time = int(time.time() * 1000)
            capture_content = should_capture_content()
            
            try:
                response = await original_create(*args, **kwargs)
                end_time = int(time.time() * 1000)
                
                attributes, _ = _build_responses_attributes(params, response, capture_content)
                
                # Build waterfall timing data
                waterfall_timings = {
                    "requestStart": 0,
                    "requestEnd": end_time - start_time,
                    "responseEnd": end_time - start_time,
                    "totalDurationMs": end_time - start_time,
                }
                attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
                
                # Get model from response or params
                model = getattr(response, "model", None) or params.get("model")
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="responses.create",
                    model=model,
                    start_time=start_time,
                    end_time=end_time,
                    status="OK",
                    attributes=attributes,
                )
                
                send_trace(trace_data)
                return response
                
            except Exception as error:
                end_time = int(time.time() * 1000)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="responses.create",
                    model=params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="ERROR",
                    error_message=str(error),
                    attributes={
                        "fallom.sdk_version": "2",
                        "fallom.method": "responses.create",
                    },
                )
                
                send_trace(trace_data)
                raise
        
        client.responses.create = traced_responses_create_async
    else:
        @wraps(original_create)
        def traced_responses_create(*args, **kwargs):
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
                
                attributes, _ = _build_responses_attributes(params, response, capture_content)
                
                # Build waterfall timing data
                waterfall_timings = {
                    "requestStart": 0,
                    "requestEnd": end_time - start_time,
                    "responseEnd": end_time - start_time,
                    "totalDurationMs": end_time - start_time,
                }
                attributes["fallom.raw.timings"] = json.dumps(waterfall_timings)
                
                # Get model from response or params
                model = getattr(response, "model", None) or params.get("model")
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="responses.create",
                    model=model,
                    start_time=start_time,
                    end_time=end_time,
                    status="OK",
                    attributes=attributes,
                )
                
                send_trace(trace_data)
                return response
                
            except Exception as error:
                end_time = int(time.time() * 1000)
                
                trace_data = _create_trace_data(
                    ctx=ctx,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    method_name="responses.create",
                    model=params.get("model"),
                    start_time=start_time,
                    end_time=end_time,
                    status="ERROR",
                    error_message=str(error),
                    attributes={
                        "fallom.sdk_version": "2",
                        "fallom.method": "responses.create",
                    },
                )
                
                send_trace(trace_data)
                raise
        
        client.responses.create = traced_responses_create
