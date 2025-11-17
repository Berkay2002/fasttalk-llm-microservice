"""
Main entry point for LLM service.

Provides CLI interface for different operating modes.
"""

import sys
import argparse
import logging

from app.utils.config import Config
from app.utils.logger import get_logger
from app.core.websocket_launcher import WebSocketLauncher
from app.monitoring.service_monitor import MonitoringServer

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FastTalk LLM Service")
    parser.add_argument(
        "mode",
        choices=["websocket", "config", "test"],
        help="Operating mode",
    )
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--host", type=str, help="Server host")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("--log-level", type=str, help="Logging level")
    parser.add_argument("--show", action="store_true", help="Show configuration")

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override with CLI arguments
    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host
    if args.model:
        config.model_name = args.model
    if args.log_level:
        config.log_level = args.log_level

    # Handle modes
    if args.mode == "config":
        if args.show:
            print("\n" + "=" * 60)
            print("LLM Service Configuration")
            print("=" * 60)
            for key, value in config.to_dict().items():
                print(f"{key:30s}: {value}")
            print("=" * 60)
        return

    elif args.mode == "test":
        print("Test mode - checking Ollama connection...")
        from app.core.ollama_handler import OllamaHandler

        handler = OllamaHandler(
            base_url=config.ollama_base_url,
            model=config.model_name,
        )

        if handler.check_connection():
            print("✓ Ollama connection successful")
            print(f"✓ Model: {config.model_name}")
        else:
            print("✗ Ollama connection failed")
            sys.exit(1)

        return

    elif args.mode == "websocket":
        logger.info("Starting LLM service in WebSocket mode")

        # Start monitoring server
        monitoring = MonitoringServer(
            host=config.monitoring_host,
            port=config.monitoring_port,
        )
        monitoring.start()

        # Start WebSocket server
        launcher = WebSocketLauncher(config)
        launcher.start()


if __name__ == "__main__":
    main()
