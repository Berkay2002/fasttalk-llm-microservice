"""
WebSocket server lifecycle management for LLM service.

Handles server initialization, startup, shutdown, and signal handling.
"""

import signal
import sys
import logging
from typing import Optional

import uvicorn

from app.utils.config import Config
from app.utils.logger import get_logger
from app.core.websocket_server import WebSocketLLMServer

logger = get_logger(__name__)


class WebSocketLauncher:
    """
    Manages WebSocket server lifecycle.

    Handles graceful startup and shutdown.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize launcher.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or Config()
        self.server: Optional[WebSocketLLMServer] = None
        self.should_stop = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("WebSocketLauncher initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.should_stop = True
        sys.exit(0)

    def start(self):
        """Start the WebSocket server."""
        logger.info("Starting LLM WebSocket server...")

        # Create server instance
        self.server = WebSocketLLMServer(self.config)

        # Check Ollama connection
        if not self.server.ollama_handler.check_connection():
            logger.error("Failed to connect to Ollama. Please ensure Ollama is running.")
            sys.exit(1)

        logger.info("Ollama connection verified")

        # Log configuration
        logger.info(f"Server starting on {self.config.host}:{self.config.port}")
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
            # Cleanup
            if hasattr(self.server, "ollama_handler"):
                self.server.ollama_handler.close()

        logger.info("LLM WebSocket server stopped")
