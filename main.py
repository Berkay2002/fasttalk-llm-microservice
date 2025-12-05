"""
Main entry point for LLM service.

Provides CLI interface for different operating modes.
Supports both vLLM (with PydanticAI) and Ollama backends.
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
    parser.add_argument("--provider", type=str, choices=["vllm", "ollama", "openai"], 
                       help="LLM provider (vllm, ollama, openai)")
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
        if config.llm_provider == "vllm":
            config.vllm_model = args.model
        else:
            config.model_name = args.model
    if args.provider:
        config.llm_provider = args.provider
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
        _run_connection_test(config)
        return

    elif args.mode == "websocket":
        logger.info(f"Starting LLM service in WebSocket mode (provider: {config.llm_provider})")

        # Start monitoring server
        monitoring = MonitoringServer(
            host=config.monitoring_host,
            port=config.monitoring_port,
        )
        monitoring.start()

        # Start WebSocket server
        launcher = WebSocketLauncher(config)
        launcher.start()


def _run_connection_test(config: Config):
    """Run connection test based on configured provider."""
    print(f"\nTest mode - checking {config.llm_provider.upper()} connection...")
    print("=" * 50)
    
    if config.llm_provider == "vllm":
        _test_vllm_connection(config)
    else:
        _test_ollama_connection(config)


def _test_vllm_connection(config: Config):
    """Test vLLM connection and optionally PydanticAI agent."""
    print(f"Provider: vLLM")
    print(f"Base URL: {config.vllm_base_url}")
    print(f"Model: {config.vllm_model}")
    print("-" * 50)
    
    # Test direct vLLM connection
    try:
        from app.core.vllm_handler import VLLMHandler
        
        handler = VLLMHandler(
            base_url=config.vllm_base_url,
            model=config.vllm_model,
            api_key=config.vllm_api_key,
        )
        
        if handler.check_connection():
            print("✓ vLLM connection successful")
            print(f"✓ Model: {config.vllm_model}")
        else:
            print("✗ vLLM connection failed")
            sys.exit(1)
            
    except ImportError as e:
        print(f"✗ vLLM handler import failed: {e}")
        print("  Install with: pip install openai httpx")
        sys.exit(1)
    except Exception as e:
        print(f"✗ vLLM connection error: {e}")
        sys.exit(1)
    
    # Test PydanticAI agent if enabled
    if config.enable_pydantic_ai:
        print("-" * 50)
        print("Testing PydanticAI agent...")
        try:
            from app.agents.voice_agent import VoiceAgent, AgentConfig
            
            agent_config = AgentConfig(
                vllm_base_url=config.vllm_base_url,
                vllm_model=config.vllm_model,
                vllm_api_key=config.vllm_api_key,
                enable_web_search=config.enable_web_search,
                enable_tools=config.enable_tools,
            )
            
            agent = VoiceAgent(config=agent_config)
            
            if agent.check_connection():
                print("✓ PydanticAI agent initialized")
                print(f"✓ Web search enabled: {config.enable_web_search}")
                print(f"✓ Tools enabled: {config.enable_tools}")
            else:
                print("✗ PydanticAI agent connection failed")
                
        except ImportError as e:
            print(f"✗ PydanticAI import failed: {e}")
            print("  Install with: pip install pydantic-ai duckduckgo-search")
        except Exception as e:
            print(f"✗ PydanticAI agent error: {e}")
    
    print("=" * 50)
    print("vLLM test completed successfully!")


def _test_ollama_connection(config: Config):
    """Test Ollama connection (legacy)."""
    print(f"Provider: Ollama (legacy)")
    print(f"Base URL: {config.ollama_base_url}")
    print(f"Model: {config.model_name}")
    print("-" * 50)
    
    try:
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
            
    except Exception as e:
        print(f"✗ Ollama connection error: {e}")
        sys.exit(1)
    
    print("=" * 50)
    print("Ollama test completed successfully!")


if __name__ == "__main__":
    main()
