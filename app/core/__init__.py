"""
Core application components for LLM service.

Contains:
- WebSocket server and launcher
- Ollama LLM handler
- Conversation management
- Text processing utilities
"""

from app.core.ollama_handler import OllamaHandler
from app.core.conversation_manager import ConversationManager

__all__ = ["OllamaHandler", "ConversationManager"]
