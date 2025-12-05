"""
WebSocket server lifecycle management for LLM service.

Handles server initialization, startup, shutdown, and signal handling.
Supports both vLLM (with PydanticAI) and Ollama backends.
"""

import signal
import sys
import logging
from typing import Optional

import uvicorn

from app.utils.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketLauncher:
    """
    Manages WebSocket server lifecycle.

    Handles graceful startup and shutdown.
    Supports both vLLM and Ollama backends.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize launcher.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or Config()
        self.server = None
        self.should_stop = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"WebSocketLauncher initialized (provider: {self.config.llm_provider})")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.should_stop = True
        sys.exit(0)

    def _create_server(self):
        """Create the appropriate WebSocket server based on provider."""
        if self.config.llm_provider == "vllm":
            # Use the new vLLM-enabled server
            from app.core.websocket_server_vllm import WebSocketLLMServer
            return WebSocketLLMServer(self.config)
        else:
            # Use the legacy Ollama server
            from app.core.websocket_server import WebSocketLLMServer
            return WebSocketLLMServer(self.config)

    def _verify_connection(self) -> bool:
        """Verify connection to the LLM backend."""
        if self.config.llm_provider == "vllm":
            return self._verify_vllm_connection()
        else:
            return self._verify_ollama_connection()

    def _verify_vllm_connection(self) -> bool:
        """Verify vLLM backend connection."""
        # Check if using PydanticAI agent or direct handler
        if self.server.use_pydantic_ai and self.server.voice_agent:
            if not self.server.voice_agent.check_connection():
                logger.error("Failed to connect to vLLM via PydanticAI agent")
                return False
            logger.info("vLLM connection verified (via PydanticAI agent)")
        elif self.server.vllm_handler:
            if not self.server.vllm_handler.check_connection():
                logger.error("Failed to connect to vLLM")
                return False
            logger.info("vLLM connection verified (direct handler)")
        else:
            logger.error("No vLLM handler available")
            return False
        return True

    def _verify_ollama_connection(self) -> bool:
        """Verify Ollama backend connection."""
        if not self.server.ollama_handler.check_connection():
            logger.error("Failed to connect to Ollama. Please ensure Ollama is running.")
            return False
        logger.info("Ollama connection verified")
        return True

    def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting LLM WebSocket server (provider: {self.config.llm_provider})...")

        # Create server instance
        self.server = self._create_server()

        # Verify backend connection
        if not self._verify_connection():
            sys.exit(1)

        # Log configuration
        logger.info(f"Server starting on {self.config.host}:{self.config.port}")
        
        if self.config.llm_provider == "vllm":
            logger.info(f"Model: {self.config.vllm_model}")
            logger.info(f"PydanticAI enabled: {self.config.enable_pydantic_ai}")
            logger.info(f"Web search enabled: {self.config.enable_web_search}")
            logger.info(f"Tools enabled: {self.config.enable_tools}")
        else:
            logger.info(f"Model: {self.config.model_name}")
            
        logger.info(f"Max connections: {self.config.max_connections}")

        # Run server
        try:
            uvicorn.run(
                self.server.app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower(),
                access_log=True,
            )
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            sys.exit(1)

    def stop(self):
        """Stop the WebSocket server."""
        logger.info("Stopping LLM WebSocket server...")

        if self.server:
            # Cleanup based on provider
            if self.config.llm_provider == "vllm":
                # vLLM handler cleanup if needed
                pass
            else:
                # Ollama handler cleanup
                if hasattr(self.server, "ollama_handler"):
                    self.server.ollama_handler.close()

        logger.info("LLM WebSocket server stopped")
