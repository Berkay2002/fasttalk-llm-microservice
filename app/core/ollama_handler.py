"""
Ollama LLM handler for the LLM service.

Enhanced version with backend-orchestration integration including:
- Connection checking with ollama ps fallback
- Advanced streaming implementation
- Request tracking and cancellation
- Proper error handling
"""

import json
import time
import uuid
import logging
import subprocess
import sys
import os
from typing import Generator, Dict, List, Optional, Any
from threading import Lock

import requests
from requests import Session

from app.utils.error_handler import LLMServiceError, ErrorCategory, ErrorSeverity
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _run_ollama_ps() -> bool:
    """
    Attempts to run the 'ollama ps' command via subprocess.

    This is used as a potential fallback diagnostic/recovery step if the initial
    HTTP connection check to the Ollama server fails. It assumes the `ollama` CLI
    is installed and in the system PATH.

    Returns:
        True if the command executes successfully (exit code 0), False otherwise
        (command not found, execution error, timeout).
    """
    try:
        logger.info("🤖🩺 Attempting to run 'ollama ps' to check server status...")
        # Added timeout to prevent hanging indefinitely
        result = subprocess.run(
            ["ollama", "ps"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0
        )
        logger.info(f"🤖🩺 'ollama ps' executed successfully. Output:\n{result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("🤖💥 'ollama ps' command not found. Make sure Ollama is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"🤖💥 'ollama ps' command failed with exit code {e.returncode}:")
        if e.stderr:
            logger.error(f"   stderr: {e.stderr.strip()}")
        if e.stdout:  # Log stdout even on error, might contain info
            logger.error(f"   stdout: {e.stdout.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("🤖💥 'ollama ps' command timed out after 10 seconds.")
        return False
    except Exception as e:
        logger.error(f"🤖💥 An unexpected error occurred while running 'ollama ps': {e}")
        return False


def _check_ollama_connection(base_url: str, session: Optional[Session]) -> bool:
    """
    Performs a quick HTTP GET request to check connectivity with an Ollama server.

    Uses the provided requests Session and base URL to attempt a connection.
    Logs success or specific connection errors.

    Args:
        base_url: The base URL of the Ollama server (e.g., "http://127.0.0.1:11434").
        session: An active requests.Session object to use for the check.

    Returns:
        True if the connection check is successful (HTTP 2xx status), False otherwise.
    """
    if not session:
        logger.warning("🤖⚠️ Cannot check Ollama connection: requests session not provided.")
        return False
    try:
        base_check_url = base_url.rstrip('/')
        if not base_check_url.startswith(('http://', 'https://')):
             base_check_url = 'http://' + base_check_url
        check_endpoint = f"{base_check_url}/"
        logger.debug(f"🤖🔌 Checking Ollama connection via GET to {check_endpoint}...")
        # Use a shorter timeout for the check
        response = session.get(check_endpoint, timeout=5.0)
        response.raise_for_status()
        logger.info(f"🤖🔌 Successfully connected to Ollama server via HTTP at: {base_url}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"🤖💥 Failed to connect to Ollama server at {base_url}: {e}")
        return False


class OllamaHandler:
    """
    Enhanced Ollama LLM handler with backend-orchestration integration.

    Provides streaming generation, conversation management, connection checking
    with ollama ps fallback, and comprehensive error handling.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        keep_alive: str = "5m",
        timeout: float = 600.0,
        no_think: bool = False,
    ):
        """
        Initialize enhanced Ollama handler.

        Args:
            base_url: Base URL for Ollama API (e.g., "http://ollama:11434")
            model: Model name to use (e.g., "llama3.2:1b")
            keep_alive: Time to keep model loaded (e.g., "5m", "1h")
            timeout: Request timeout in seconds
            no_think: Flag to modify prompts (experimental)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.no_think = no_think

        # Normalize base URL
        url = self.base_url
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        url = url.replace('/api/chat', '').replace('/api/generate', '').rstrip('/')
        self.base_url = url

        # Create session for connection pooling
        self.session: Session = requests.Session()
        self.session.timeout = timeout

        # Request tracking
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()

        # Connection and initialization status
        self._connection_ok = False
        self._client_initialized = False
        self._init_lock = Lock()

        logger.info(
            f"Enhanced OllamaHandler initialized: model={model}, base_url={self.base_url}"
        )

    def _lazy_initialize_connection(self) -> bool:
        """
        Initialize connection to Ollama server with fallback logic.

        Performs initial connection check and attempts 'ollama ps' fallback if needed.
        This mirrors the backend-orchestration initialization logic.

        Returns:
            True if connection established successfully, False otherwise
        """
        if self._client_initialized:
            return self._connection_ok

        with self._init_lock:
            if self._client_initialized:  # Double check
                return self._connection_ok

            logger.debug(f"🤖🔄 Lazy initializing/checking connection for Ollama at {self.base_url}")
            self._connection_ok = False  # Reset flag

            try:
                # Initial direct check
                initial_check_ok = _check_ollama_connection(self.base_url, self.session)
                if initial_check_ok:
                    self._connection_ok = True
                    logger.info("🤖✅ Ollama connection established successfully")
                else:
                    # Attempt ollama ps fallback
                    logger.warning(f"🤖🔌 Initial Ollama connection check failed for {self.base_url}. Attempting 'ollama ps' fallback.")
                    if _run_ollama_ps():
                        # ollama ps ran, wait a bit and try connecting again
                        logger.info("🤖⏳ 'ollama ps' succeeded, waiting 3 seconds before re-checking connection...")
                        time.sleep(3)
                        second_check_ok = _check_ollama_connection(self.base_url, self.session)
                        if second_check_ok:
                            logger.info("🤖🔌✅ Ollama connection successful after running 'ollama ps'.")
                            self._connection_ok = True
                        else:
                            logger.error(f"🤖💥 Ollama connection check still failed after running 'ollama ps'.")
                    else:
                        # ollama ps command failed or was not found
                        logger.error(f"🤖💥 'ollama ps' command failed or not found. Cannot verify/start server. Initialization failed for {self.base_url}.")

                if self._connection_ok:
                    logger.info(f"🤖✅ Connection initialized successfully for Ollama backend")
                else:
                    logger.error(f"🤖💥 Initialization failed for Ollama backend")

            except Exception as e:
                logger.exception(f"🤖💥 Critical failure during Ollama connection initialization: {e}")
                self._connection_ok = False
            finally:
                # Mark as initialized regardless of success/failure
                self._client_initialized = True

            return self._connection_ok

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        return self._lazy_initialize_connection()

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
        system_prompt: Optional[str] = None,
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
            system_prompt: Optional system prompt to prepend

        Yields:
            str: Individual tokens as they are generated

        Raises:
            LLMServiceError: If generation fails
        """
        req_id = request_id or f"ollama-{uuid.uuid4()}"
        logger.info(f"Starting generation (Request ID: {req_id})")

        # Ensure connection is initialized with fallback logic
        if not self._lazy_initialize_connection():
            logger.error(f"[{req_id}] Generation failed: Could not initialize Ollama connection")
            raise LLMServiceError(
                "Failed to initialize Ollama connection",
                category=ErrorCategory.CONNECTION,
                severity=ErrorSeverity.HIGH,
                recoverable=True,
                retry_after=10.0,
            )

        # Prepare messages with system prompt
        final_messages = []
        if system_prompt:
            # Add system prompt as the first message if not already present
            if not messages or messages[0].get("role") != "system":
                final_messages.append({"role": "system", "content": system_prompt})
            else:
                # Replace existing system prompt
                final_messages.append({"role": "system", "content": system_prompt})

        # Add the rest of the messages
        final_messages.extend(messages)

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

        # Build payload matching backend-orchestration format
        payload = {
            "model": self.model,
            "messages": final_messages,
            "stream": True,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        logger.info(f"🤖💬 [{req_id}] Sending Ollama request to {api_endpoint}")
        logger.debug(f"🤖💬 [{req_id}] Payload: {json.dumps(payload, indent=2)}")

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

            # Stream and yield tokens using enhanced implementation
            yield from self._yield_ollama_chunks(response, req_id)

            logger.info(f"🤖✅ [{req_id}] Generation completed successfully")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"🤖💥 [{req_id}] Connection error: {e}")
            # Reset connection flag to force re-check on retry
            self._client_initialized = False
            logger.debug(f"🤖🔄 [{req_id}] Resetting client initialized flag to force re-check on retry.")
            raise LLMServiceError(
                f"Connection error during generation: {e}",
                category=ErrorCategory.CONNECTION,
                severity=ErrorSeverity.HIGH,
                recoverable=True,
                retry_after=5.0,
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"🤖💥 [{req_id}] Timeout error: {e}")
            raise LLMServiceError(
                f"Timeout during generation: {e}",
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                retry_after=30.0,
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"🤖💥 [{req_id}] HTTP error: {e}")
            raise LLMServiceError(
                f"HTTP error during generation: {e}",
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
            )
        except Exception as e:
            logger.error(f"🤖💥 [{req_id}] Unexpected error: {e}", exc_info=True)
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
                    logger.debug(f"🤖🗑️ [{req_id}] Removed from active requests")

            # Ensure response is closed
            if response is not None:
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"🤖⚠️ [{req_id}] Error closing response: {e}")

    def _yield_ollama_chunks(
        self, response: requests.Response, request_id: str
    ) -> Generator[str, None, None]:
        """
        Enhanced parser for Ollama streaming response matching backend-orchestration.

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
            # Enhanced iteration with error handling matching backend
            try:
                for chunk_bytes in response.iter_content(chunk_size=None):  # None = read whatever is available
                    # Check for cancellation *before* processing chunk
                    with self._requests_lock:
                        if request_id not in self._active_requests:
                            logger.info(f"🤖🗑️ Ollama stream {request_id} cancelled or finished externally during iteration (pre-chunk check).")
                            break  # Exit the loop cleanly

                    if not chunk_bytes:
                        continue  # Skip empty chunks

                    buffer += chunk_bytes.decode('utf-8')

                    # Process complete JSON objects separated by newlines in the buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue  # Skip empty lines

                        try:
                            chunk = json.loads(line)
                            if chunk.get('error'):
                                logger.error(f"🤖💥 Ollama stream returned error for {request_id}: {chunk['error']}")
                                raise LLMServiceError(
                                    f"Ollama stream error: {chunk['error']}",
                                    category=ErrorCategory.PROCESSING,
                                    severity=ErrorSeverity.HIGH,
                                )

                            # Extract content using Ollama's message format
                            content = chunk.get('message', {}).get('content')
                            if content:
                                token_count += 1
                                yield content

                            # Check for done signal
                            if chunk.get('done'):
                                logger.debug(f"🤖✅ [{request_id}] Ollama signaled done")
                                processed_done = True
                                break

                        except json.JSONDecodeError:
                            logger.warning(f"🤖⚠️ [{request_id}] Failed to decode JSON: {line[:100]}")
                            continue
                        except Exception as e:
                            logger.error(f"🤖💥 [{request_id}] Error processing chunk: {e}")
                            raise

                    # Break outer loop if done
                    if processed_done:
                        break

            except AttributeError as e:
                # Handle race condition with response.close()
                if "'NoneType' object has no attribute 'read'" in str(e):
                    logger.warning(f"🤖⚠️ [{request_id}] Stream closed concurrently (likely due to cancellation)")
                    # Check if this was due to cancellation
                    is_cancelled = False
                    with self._requests_lock:
                        is_cancelled = request_id not in self._active_requests
                    if not is_cancelled:
                        # If not cancelled, this might be a real error
                        logger.error(f"🤖💥 [{request_id}] Unexpected stream closure: {e}")
                else:
                    raise

            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} tokens")

        except requests.exceptions.ChunkedEncodingError as e:
            # Check if cancelled
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests

            if is_cancelled:
                logger.warning(f"🤖🗑️ [{request_id}] Chunked encoding error (likely due to cancellation)")
            else:
                logger.error(f"🤖💥 [{request_id}] Chunked encoding error: {e}")
                raise LLMServiceError(
                    f"Stream encoding error: {e}",
                    category=ErrorCategory.CONNECTION,
                    severity=ErrorSeverity.HIGH,
                    recoverable=True,
                )

        finally:
            # Ensure response is closed
            if response:
                try:
                    logger.debug(f"🤖🗑️ [{request_id}] Closing Ollama response in _yield_ollama_chunks finally.")
                    response.close()
                except Exception as close_err:
                    logger.warning(f"🤖⚠️ [{request_id}] Error closing Ollama response in finally: {close_err}", exc_info=False)

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
