"""
LangChain Callback Handler for Fallom Tracing.

This module provides a LangChain-compatible callback handler that automatically traces all LLM calls, chain executions, tool usages, and agent actions to Fallom.

Usage:
    import fallom
    from fallom.trace.wrappers.langchain import FallomCallbackHandler
    
    fallom.init(api_key="your-api-key")
    
    handler = FallomCallbackHandler(
        config_key="my-langchain-app",
        session_id="session-123",
    )
    
    # Use with any LangChain component
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    
    # Or pass to invoke/run calls
    chain.invoke({"input": "Hello"}, config={"callbacks": [handler]})
"""
import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from ..session import FallomSession

from ..core import (
    is_initialized,
    should_capture_content,
    send_trace,
    is_debug_mode,
)
from ..utils import generate_hex_id, timestamp_to_iso
from ..types import SessionContext, TraceData


def _log(*args):
    """Print debug message if debug mode is enabled."""
    if is_debug_mode():
        print("[Fallom LangChain]", *args)


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Serialize LangChain messages to dicts."""
    result = []
    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            result.append({
                "role": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", ""),
            })
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"content": str(msg)})
    return result


def _extract_model_from_serialized(serialized: Dict[str, Any]) -> Optional[str]:
    """Extract model name from serialized LangChain component."""
    kwargs = serialized.get("kwargs", {})
    
    model = (
        kwargs.get("model") or
        kwargs.get("model_name") or
        serialized.get("id", [None])[-1]
    )
    
    return model


# Lazy-loaded handler class
_handler_class = None


def _get_handler_class():
    """Lazily import LangChain and create the handler class."""
    global _handler_class
    
    if _handler_class is not None:
        return _handler_class
    
    # Lazy import - only when actually used
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.outputs import LLMResult, ChatGeneration, Generation
    except ImportError:
        raise ImportError(
            "LangChain is not installed. Install with: pip install fallom[langchain]"
        )
    
    class _FallomCallbackHandler(BaseCallbackHandler):
        """
        LangChain callback handler that sends traces to Fallom.
        
        This handler captures:
        - LLM calls (start, end, errors)
        - Chain executions (start, end, errors)
        - Tool/function calls (start, end, errors)
        - Agent actions
        - Retriever queries
        
        Example:
            import fallom
            from fallom.trace.wrappers.langchain import FallomCallbackHandler
            
            fallom.init(api_key="your-api-key")
            
            handler = FallomCallbackHandler(
                config_key="my-app",
                session_id="user-123-conversation-456",
            )
            
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
            response = llm.invoke("What is the capital of France?")
        """
        
        def __init__(
            self,
            config_key: str,
            session_id: str,
            customer_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
        ):
            """
            Initialize the Fallom callback handler.
            
            Args:
                config_key: Your config name from the Fallom dashboard
                session_id: Unique session/conversation ID
                customer_id: Optional user identifier for analytics
                metadata: Optional custom metadata for filtering
                tags: Optional string tags for quick filtering
            """
            super().__init__()
            
            self._ctx = SessionContext(
                config_key=config_key,
                session_id=session_id,
                customer_id=customer_id,
                metadata=metadata,
                tags=tags,
            )
            
            self._trace_id = generate_hex_id(32)
            self._span_stack: Dict[str, Dict[str, Any]] = {}
            
            _log(f"Initialized handler for session: {session_id}")
        
        def _get_run_id_str(self, run_id: UUID) -> str:
            return str(run_id)
        
        def _get_parent_span_id(self, parent_run_id: Optional[UUID]) -> Optional[str]:
            if parent_run_id is None:
                return None
            parent_key = self._get_run_id_str(parent_run_id)
            parent_info = self._span_stack.get(parent_key)
            return parent_info.get("span_id") if parent_info else None
        
        def _start_span(
            self,
            run_id: UUID,
            parent_run_id: Optional[UUID],
            name: str,
            kind: str,
            inputs: Optional[Dict[str, Any]] = None,
            model: Optional[str] = None,
        ) -> str:
            span_id = generate_hex_id(16)
            start_time = int(time.time() * 1000)
            
            self._span_stack[self._get_run_id_str(run_id)] = {
                "span_id": span_id,
                "start_time": start_time,
                "name": name,
                "kind": kind,
                "inputs": inputs,
                "model": model,
                "parent_run_id": parent_run_id,
            }
            
            _log(f"Started span: {name} ({kind}) - {span_id}")
            return span_id
        
        def _end_span(
            self,
            run_id: UUID,
            outputs: Optional[Dict[str, Any]] = None,
            error: Optional[str] = None,
            token_usage: Optional[Dict[str, int]] = None,
        ) -> None:
            run_key = self._get_run_id_str(run_id)
            span_info = self._span_stack.pop(run_key, None)
            
            if span_info is None:
                _log(f"Warning: No span found for run_id {run_id}")
                return
            
            end_time = int(time.time() * 1000)
            duration_ms = end_time - span_info["start_time"]
            
            attributes = {
                "fallom.sdk_version": "2",
                "fallom.integration": "langchain",
                "fallom.method": span_info["name"],
            }
            
            capture_content = should_capture_content()
            
            if capture_content:
                if span_info.get("inputs"):
                    attributes["fallom.raw.request"] = json.dumps(
                        span_info["inputs"], default=str
                    )
                if outputs:
                    attributes["fallom.raw.response"] = json.dumps(
                        outputs, default=str
                    )
            
            if token_usage:
                attributes["fallom.raw.usage"] = json.dumps(token_usage)
            
            attributes["fallom.raw.timings"] = json.dumps({
                "requestStart": 0,
                "requestEnd": duration_ms,
                "responseEnd": duration_ms,
                "totalDurationMs": duration_ms,
            })
            
            try:
                from fallom.prompts import get_prompt_context
                prompt_ctx = get_prompt_context()
            except ImportError:
                prompt_ctx = None
            
            trace_data = TraceData(
                config_key=self._ctx.config_key,
                session_id=self._ctx.session_id,
                customer_id=self._ctx.customer_id,
                metadata=self._ctx.metadata,
                tags=self._ctx.tags,
                trace_id=self._trace_id,
                span_id=span_info["span_id"],
                parent_span_id=self._get_parent_span_id(span_info.get("parent_run_id")),
                name=span_info["name"],
                kind=span_info["kind"],
                model=span_info.get("model"),
                start_time=timestamp_to_iso(span_info["start_time"]),
                end_time=timestamp_to_iso(end_time),
                duration_ms=duration_ms,
                status="ERROR" if error else "OK",
                error_message=error,
                attributes=attributes,
                prompt_key=prompt_ctx.get("prompt_key") if prompt_ctx else None,
                prompt_version=prompt_ctx.get("prompt_version") if prompt_ctx else None,
                prompt_ab_test_key=prompt_ctx.get("ab_test_key") if prompt_ctx else None,
                prompt_variant_index=prompt_ctx.get("variant_index") if prompt_ctx else None,
            )
            
            if is_initialized():
                send_trace(trace_data)
                _log(f"Sent trace: {span_info['name']} ({duration_ms}ms)")
            else:
                _log("Warning: Fallom not initialized, trace not sent")
        
        # =====================================================================
        # LLM Callbacks
        # =====================================================================
        
        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            model = _extract_model_from_serialized(serialized)
            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name="llm.completion",
                kind="llm",
                inputs={"prompts": prompts},
                model=model,
            )
        
        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[Any]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            model = _extract_model_from_serialized(serialized)
            serialized_messages = [
                _serialize_messages(msg_list) for msg_list in messages
            ]
            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name="chat.completions.create",
                kind="llm",
                inputs={"messages": serialized_messages[0] if len(serialized_messages) == 1 else serialized_messages},
                model=model,
            )
        
        def on_llm_end(
            self,
            response: Any,  # LLMResult
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            outputs = {}
            if response.generations:
                first_gen = response.generations[0]
                if first_gen:
                    gen = first_gen[0]
                    if isinstance(gen, ChatGeneration) and hasattr(gen, "message"):
                        outputs["text"] = gen.message.content
                        if hasattr(gen.message, "tool_calls") and gen.message.tool_calls:
                            outputs["toolCalls"] = [
                                {
                                    "id": tc.get("id"),
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                }
                                for tc in gen.message.tool_calls
                            ]
                    elif isinstance(gen, Generation):
                        outputs["text"] = gen.text
            
            token_usage = None
            if response.llm_output:
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
                if usage:
                    token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                
                if response.llm_output.get("model_name"):
                    run_key = self._get_run_id_str(run_id)
                    if run_key in self._span_stack:
                        self._span_stack[run_key]["model"] = response.llm_output["model_name"]
            
            self._end_span(run_id=run_id, outputs=outputs, token_usage=token_usage)
        
        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(run_id=run_id, error=str(error))
        
        # =====================================================================
        # Chain Callbacks
        # =====================================================================
        
        def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            chain_name = serialized.get("id", ["unknown"])[-1]
            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=f"chain.{chain_name}",
                kind="chain",
                inputs=inputs,
            )
        
        def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(run_id=run_id, outputs=outputs)
        
        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(run_id=run_id, error=str(error))
        
        # =====================================================================
        # Tool Callbacks
        # =====================================================================
        
        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            tool_name = serialized.get("name", "unknown_tool")
            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=f"tool.{tool_name}",
                kind="tool",
                inputs={"input": input_str},
            )
        
        def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(
                run_id=run_id,
                outputs={"output": str(output) if output else None},
            )
        
        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(run_id=run_id, error=str(error))
        
        # =====================================================================
        # Agent Callbacks
        # =====================================================================
        
        def on_agent_action(
            self,
            action: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            _log(f"Agent action: {action}")
        
        def on_agent_finish(
            self,
            finish: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            _log(f"Agent finished: {finish}")
        
        # =====================================================================
        # Retriever Callbacks
        # =====================================================================
        
        def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            retriever_name = serialized.get("id", ["retriever"])[-1]
            self._start_span(
                run_id=run_id,
                parent_run_id=parent_run_id,
                name=f"retriever.{retriever_name}",
                kind="retriever",
                inputs={"query": query},
            )
        
        def on_retriever_end(
            self,
            documents: List[Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            doc_summaries = []
            for doc in documents[:5]:
                if hasattr(doc, "page_content"):
                    doc_summaries.append({
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": getattr(doc, "metadata", {}),
                    })
            
            self._end_span(
                run_id=run_id,
                outputs={
                    "document_count": len(documents),
                    "documents": doc_summaries,
                },
            )
        
        def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
        ) -> None:
            self._end_span(run_id=run_id, error=str(error))
    
    _handler_class = _FallomCallbackHandler
    return _handler_class


class FallomCallbackHandler:
    """
    LangChain callback handler that sends traces to Fallom.
    
    This is a factory that creates the actual handler class on first use,
    lazily importing LangChain only when needed.
    
    Example:
        import fallom
        from fallom.trace.wrappers.langchain import FallomCallbackHandler
        
        fallom.init(api_key="your-api-key")
        
        handler = FallomCallbackHandler(
            config_key="my-app",
            session_id="session-123",
        )
        
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    """
    
    def __new__(
        cls,
        config_key: str,
        session_id: str,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a new Fallom callback handler instance."""
        handler_cls = _get_handler_class()
        return handler_cls(
            config_key=config_key,
            session_id=session_id,
            customer_id=customer_id,
            metadata=metadata,
            tags=tags,
        )


def callback_handler_from_session(session: "FallomSession") -> "FallomCallbackHandler":
    """
    Create a FallomCallbackHandler from an existing FallomSession.
    
    Example:
        session = fallom.session(config_key="my-app", session_id="123")
        handler = callback_handler_from_session(session)
        
        llm = ChatOpenAI(callbacks=[handler])
    """
    ctx = session.get_context()
    return FallomCallbackHandler(
        config_key=ctx.config_key,
        session_id=ctx.session_id,
        customer_id=ctx.customer_id,
        metadata=ctx.metadata,
        tags=ctx.tags,
    )
