"""
vLLM handler for the LLM service.

Provides interface to vLLM via OpenAI-compatible API with streaming support.
This serves as an alternative to ollama_handler.py for vLLM backend.
"""

import json
import time
import uuid
import logging
from typing import Generator, AsyncGenerator, Dict, List, Optional, Any
from threading import Lock

import httpx
from openai import OpenAI, AsyncOpenAI

from app.utils.error_handler import LLMServiceError, ErrorCategory, ErrorSeverity
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VLLMHandler:
    """
    Handles communication with vLLM backend via OpenAI-compatible API.

    Provides both sync and async streaming generation with tool calling support.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: float = 600.0,
    ):
        """
        Initialize vLLM handler.

        Args:
            base_url: Base URL for vLLM API (e.g., "http://vllm:8000/v1")
            model: Model name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            api_key: API key (vLLM doesn't require one, use placeholder)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

        # Initialize OpenAI clients
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

        # Request tracking for cancellation
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()

        # Connection status
        self._connection_ok = False

        logger.info(
            f"VLLMHandler initialized: model={model}, base_url={base_url}"
        )

    def check_connection(self) -> bool:
        """
        Check if vLLM server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # vLLM health endpoint is at /health (not /v1/health)
            health_url = self.base_url.replace("/v1", "/health")
            with httpx.Client(timeout=5.0) as client:
                response = client.get(health_url)
                response.raise_for_status()
            
            logger.info(f"Successfully connected to vLLM at {self.base_url}")
            self._connection_ok = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vLLM at {self.base_url}: {e}")
            self._connection_ok = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models.

        Returns:
            Dictionary with model information
        """
        try:
            models = self.client.models.list()
            return {
                "models": [m.id for m in models.data],
                "current_model": self.model,
            }
        except Exception as e:
            raise LLMServiceError(
                f"Failed to get model info: {e}",
                category=ErrorCategory.CONNECTION,
                severity=ErrorSeverity.MEDIUM,
            )

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        request_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text using vLLM API with streaming (synchronous).

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            tools: Optional list of tool definitions
            request_id: Optional request ID for tracking

        Yields:
            str: Individual tokens as they are generated

        Raises:
            LLMServiceError: If generation fails
        """
        req_id = request_id or f"vllm-{uuid.uuid4()}"
        logger.info(f"Starting generation (Request ID: {req_id})")

        # Build parameters
        params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        if tools is not None:
            params["tools"] = tools

        logger.debug(f"[{req_id}] Request params: {params}")

        try:
            # Register request for tracking
            with self._requests_lock:
                self._active_requests[req_id] = {
                    "start_time": time.time(),
                    "cancelled": False,
                }

            # Create streaming response
            stream = self.client.chat.completions.create(**params)
            
            token_count = 0
            for chunk in stream:
                # Check for cancellation
                with self._requests_lock:
                    if req_id not in self._active_requests:
                        logger.info(f"[{req_id}] Generation cancelled")
                        break
                    if self._active_requests[req_id].get("cancelled"):
                        logger.info(f"[{req_id}] Generation cancelled by user")
                        break

                # Extract content from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    token_count += 1
                    yield content

                # Check for finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    logger.debug(
                        f"[{req_id}] Finished with reason: {chunk.choices[0].finish_reason}"
                    )
                    break

            logger.info(f"[{req_id}] Generation complete: {token_count} tokens")

        except Exception as e:
            logger.error(f"[{req_id}] Generation error: {e}", exc_info=True)
            raise LLMServiceError(
                f"vLLM generation error: {e}",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
            )
        finally:
            with self._requests_lock:
                self._active_requests.pop(req_id, None)

    async def generate_stream_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using vLLM API with streaming (async).

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            tools: Optional list of tool definitions
            request_id: Optional request ID for tracking

        Yields:
            str: Individual tokens as they are generated
        """
        req_id = request_id or f"vllm-async-{uuid.uuid4()}"
        logger.info(f"Starting async generation (Request ID: {req_id})")

        # Build parameters
        params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        if tools is not None:
            params["tools"] = tools

        try:
            # Register request for tracking
            with self._requests_lock:
                self._active_requests[req_id] = {
                    "start_time": time.time(),
                    "cancelled": False,
                }

            # Create async streaming response
            stream = await self.async_client.chat.completions.create(**params)
            
            token_count = 0
            async for chunk in stream:
                # Check for cancellation
                with self._requests_lock:
                    if req_id not in self._active_requests:
                        logger.info(f"[{req_id}] Generation cancelled")
                        break
                    if self._active_requests[req_id].get("cancelled"):
                        logger.info(f"[{req_id}] Generation cancelled by user")
                        break

                # Extract content from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    token_count += 1
                    yield content

                # Check for finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    logger.debug(
                        f"[{req_id}] Finished with reason: {chunk.choices[0].finish_reason}"
                    )
                    break

            logger.info(f"[{req_id}] Async generation complete: {token_count} tokens")

        except Exception as e:
            logger.error(f"[{req_id}] Async generation error: {e}", exc_info=True)
            raise LLMServiceError(
                f"vLLM async generation error: {e}",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
            )
        finally:
            with self._requests_lock:
                self._active_requests.pop(req_id, None)

    def cancel_generation(self, request_id: str) -> bool:
        """
        Cancel an active generation request.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if request was found and marked for cancellation
        """
        with self._requests_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]["cancelled"] = True
                logger.info(f"Marked request {request_id} for cancellation")
                return True
            logger.warning(f"Request {request_id} not found for cancellation")
            return False

    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active requests."""
        with self._requests_lock:
            return {
                req_id: {
                    "start_time": info["start_time"],
                    "duration_s": time.time() - info["start_time"],
                    "cancelled": info.get("cancelled", False),
                }
                for req_id, info in self._active_requests.items()
            }


class VLLMWithToolsHandler(VLLMHandler):
    """
    Extended vLLM handler with tool calling support.
    
    Handles streaming responses that may include tool calls.
    """

    def generate_stream_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_functions: Dict[str, callable],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate with automatic tool execution.
        
        Args:
            messages: Conversation messages
            tools: List of tool definitions
            tool_functions: Dict mapping tool names to callables
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            request_id: Request ID for tracking
            
        Yields:
            str: Text tokens (tool results are processed internally)
        """
        req_id = request_id or f"vllm-tools-{uuid.uuid4()}"
        
        params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        try:
            stream = self.client.chat.completions.create(**params)
            
            # Collect tool calls from stream
            tool_calls = {}
            current_id = None
            
            for chunk in stream:
                # Handle tool calls
                if chunk.choices[0].delta.tool_calls:
                    for tc_chunk in chunk.choices[0].delta.tool_calls:
                        if hasattr(tc_chunk, "id") and tc_chunk.id:
                            current_id = tc_chunk.id
                            if current_id not in tool_calls:
                                tool_calls[current_id] = {
                                    "name": None,
                                    "arguments": "",
                                }
                        
                        if hasattr(tc_chunk, "function") and current_id:
                            if tc_chunk.function.name:
                                tool_calls[current_id]["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_calls[current_id]["arguments"] += tc_chunk.function.arguments
                
                # Handle regular content
                elif chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            # Execute tool calls if any
            if tool_calls:
                for tc_id, tc_info in tool_calls.items():
                    func_name = tc_info["name"]
                    if func_name in tool_functions:
                        try:
                            args = json.loads(tc_info["arguments"])
                            result = tool_functions[func_name](**args)
                            logger.info(f"[{req_id}] Tool {func_name} returned: {result[:100]}...")
                            
                            # Add tool result to messages and continue
                            messages.append({
                                "role": "assistant",
                                "tool_calls": [{
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": func_name,
                                        "arguments": tc_info["arguments"],
                                    }
                                }]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": str(result),
                            })
                            
                            # Continue generation with tool result
                            for token in self.generate_stream(
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                request_id=f"{req_id}-cont",
                            ):
                                yield token
                                
                        except Exception as e:
                            logger.error(f"[{req_id}] Tool execution error: {e}")
                            yield f"\n[Error executing tool: {e}]"

        except Exception as e:
            logger.error(f"[{req_id}] Tool generation error: {e}", exc_info=True)
            raise
