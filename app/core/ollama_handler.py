"""
Ollama LLM handler for the LLM service.

Provides interface to Ollama API with streaming support, connection management,
and request tracking.
"""

import json
import time
import uuid
import logging
from typing import Generator, Dict, List, Optional, Any
from threading import Lock

import requests
from requests import Session

from app.utils.error_handler import LLMServiceError, ErrorCategory, ErrorSeverity
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaHandler:
    """
    Handles communication with Ollama LLM backend.

    Provides streaming generation, conversation management, and error handling.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        keep_alive: str = "5m",
        timeout: float = 600.0,
    ):
        """
        Initialize Ollama handler.

        Args:
            base_url: Base URL for Ollama API (e.g., "http://ollama:11434")
            model: Model name to use (e.g., "llama3.2:1b")
            keep_alive: Time to keep model loaded (e.g., "5m", "1h")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout

        # Create session for connection pooling
        self.session: Session = requests.Session()
        self.session.timeout = timeout

        # Request tracking
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()

        # Connection status
        self._connection_ok = False
        self._init_lock = Lock()

        logger.info(
            f"OllamaHandler initialized: model={model}, base_url={base_url}"
        )

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            check_endpoint = f"{self.base_url}/"
            response = self.session.get(check_endpoint, timeout=5.0)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
            self._connection_ok = True
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            self._connection_ok = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information

        Raises:
            LLMServiceError: If model info cannot be retrieved
        """
        try:
            endpoint = f"{self.base_url}/api/show"
            payload = {"name": self.model}
            response = self.session.post(endpoint, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
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
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text using Ollama API with streaming.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            request_id: Optional request ID for tracking

        Yields:
            str: Individual tokens as they are generated

        Raises:
            LLMServiceError: If generation fails
        """
        req_id = request_id or f"ollama-{uuid.uuid4()}"
        logger.info(f"Starting generation (Request ID: {req_id})")

        # Build Ollama API endpoint
        api_endpoint = f"{self.base_url}/api/chat"

        # Build options dictionary
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if stop is not None:
            options["stop"] = stop

        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        logger.debug(f"[{req_id}] Sending request to {api_endpoint}")
        logger.debug(f"[{req_id}] Payload: {json.dumps(payload, indent=2)}")

        # Make streaming request
        response = None
        try:
            response = self.session.post(
                api_endpoint,
                json=payload,
                stream=True,
                timeout=(10.0, self.timeout),  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()

            # Register request for tracking/cancellation
            self._register_request(req_id, "ollama", response)

            # Stream and yield tokens
            yield from self._yield_ollama_chunks(response, req_id)

            logger.info(f"[{req_id}] Generation completed successfully")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{req_id}] Connection error: {e}")
            raise LLMServiceError(
                f"Connection error during generation: {e}",
                category=ErrorCategory.CONNECTION,
                severity=ErrorSeverity.HIGH,
                recoverable=True,
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"[{req_id}] Timeout error: {e}")
            raise LLMServiceError(
                f"Timeout during generation: {e}",
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                retry_after=30.0,
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{req_id}] HTTP error: {e}")
            raise LLMServiceError(
                f"HTTP error during generation: {e}",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
            )
        except Exception as e:
            logger.error(f"[{req_id}] Unexpected error: {e}", exc_info=True)
            raise LLMServiceError(
                f"Unexpected error during generation: {e}",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
            )
        finally:
            # Clean up request tracking
            with self._requests_lock:
                if req_id in self._active_requests:
                    self._active_requests.pop(req_id)
                    logger.debug(f"[{req_id}] Removed from active requests")

            # Ensure response is closed
            if response is not None:
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[{req_id}] Error closing response: {e}")

    def _yield_ollama_chunks(
        self, response: requests.Response, request_id: str
    ) -> Generator[str, None, None]:
        """
        Parse and yield tokens from Ollama streaming response.

        Args:
            response: Streaming response from Ollama API
            request_id: Request ID for logging

        Yields:
            str: Content chunks from the stream

        Raises:
            LLMServiceError: If stream returns an error
        """
        token_count = 0
        buffer = ""
        processed_done = False

        try:
            for chunk_bytes in response.iter_content(chunk_size=None):
                # Check for cancellation
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        logger.info(f"[{request_id}] Generation cancelled")
                        break

                if not chunk_bytes:
                    continue

                # Decode and add to buffer
                buffer += chunk_bytes.decode("utf-8")

                # Process complete JSON objects separated by newlines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)

                        # Check for errors
                        if chunk.get("error"):
                            error_msg = chunk["error"]
                            logger.error(f"[{request_id}] Ollama error: {error_msg}")
                            raise LLMServiceError(
                                f"Ollama stream error: {error_msg}",
                                category=ErrorCategory.PROCESSING,
                                severity=ErrorSeverity.HIGH,
                            )

                        # Extract content
                        content = chunk.get("message", {}).get("content")
                        if content:
                            token_count += 1
                            yield content

                        # Check for done signal
                        if chunk.get("done"):
                            logger.debug(f"[{request_id}] Ollama signaled done")
                            processed_done = True
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"[{request_id}] Failed to decode JSON: {line[:100]}")
                        continue
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing chunk: {e}")
                        raise

                # Break outer loop if done
                if processed_done:
                    break

            logger.debug(f"[{request_id}] Yielded {token_count} tokens")

        except requests.exceptions.ChunkedEncodingError as e:
            # Check if cancelled
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests

            if is_cancelled:
                logger.warning(f"[{request_id}] Chunked encoding error (likely due to cancellation)")
            else:
                logger.error(f"[{request_id}] Chunked encoding error: {e}")
                raise LLMServiceError(
                    f"Stream encoding error: {e}",
                    category=ErrorCategory.CONNECTION,
                    severity=ErrorSeverity.HIGH,
                )
        except AttributeError as e:
            # Handle race condition with response.close()
            if "'NoneType' object has no attribute 'read'" in str(e):
                logger.warning(f"[{request_id}] Stream closed concurrently")
            else:
                raise

        finally:
            # Ensure response is closed
            if response:
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[{request_id}] Error closing response in finally: {e}")

    def _register_request(self, request_id: str, request_type: str, stream_obj: Any):
        """Register an active generation request."""
        with self._requests_lock:
            if request_id in self._active_requests:
                logger.warning(f"Request ID {request_id} already registered")
            self._active_requests[request_id] = {
                "type": request_type,
                "stream": stream_obj,
                "start_time": time.time(),
            }
            logger.debug(f"Registered request: {request_id}")

    def cancel_generation(self, request_id: Optional[str] = None) -> bool:
        """
        Cancel active generation(s).

        Args:
            request_id: Specific request to cancel, or None to cancel all

        Returns:
            True if any requests were cancelled
        """
        cancelled = False

        with self._requests_lock:
            if request_id is None:
                # Cancel all
                ids_to_cancel = list(self._active_requests.keys())
                if not ids_to_cancel:
                    return False

                logger.info(f"Cancelling all {len(ids_to_cancel)} active requests")
                for req_id in ids_to_cancel:
                    if self._cancel_single_request(req_id):
                        cancelled = True
            else:
                # Cancel specific request
                if request_id in self._active_requests:
                    logger.info(f"Cancelling request: {request_id}")
                    cancelled = self._cancel_single_request(request_id)
                else:
                    logger.warning(f"Cannot cancel non-existent request: {request_id}")

        return cancelled

    def _cancel_single_request(self, request_id: str) -> bool:
        """
        Cancel a single request (must be called with lock held).

        Args:
            request_id: Request to cancel

        Returns:
            True if cancelled
        """
        request_data = self._active_requests.pop(request_id, None)
        if not request_data:
            return False

        # Try to close the stream
        stream_obj = request_data.get("stream")
        if stream_obj and hasattr(stream_obj, "close"):
            try:
                stream_obj.close()
                logger.debug(f"Closed stream for request {request_id}")
            except Exception as e:
                logger.warning(f"Error closing stream for {request_id}: {e}")

        logger.info(f"Cancelled request {request_id}")
        return True

    def get_active_requests(self) -> List[str]:
        """Get list of active request IDs."""
        with self._requests_lock:
            return list(self._active_requests.keys())

    def cleanup_stale_requests(self, timeout_seconds: int = 300) -> int:
        """
        Clean up requests older than timeout.

        Args:
            timeout_seconds: Maximum age before cleanup

        Returns:
            Number of requests cleaned up
        """
        stale_ids = []
        now = time.time()

        with self._requests_lock:
            stale_ids = [
                req_id
                for req_id, req_data in self._active_requests.items()
                if (now - req_data.get("start_time", 0)) > timeout_seconds
            ]

        if stale_ids:
            logger.info(f"Cleaning up {len(stale_ids)} stale requests")
            cleaned = 0
            for req_id in stale_ids:
                if self.cancel_generation(req_id):
                    cleaned += 1
            return cleaned

        return 0

    def close(self):
        """Close handler and clean up resources."""
        logger.info("Closing OllamaHandler")

        # Cancel all active requests
        self.cancel_generation(None)

        # Close session
        if self.session:
            self.session.close()

        logger.info("OllamaHandler closed")
