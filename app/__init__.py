"""
FastTalk LLM Microservice
==========================

A production-ready LLM microservice providing WebSocket-based streaming inference
with conversation management, built on Ollama.

Architecture:
- WebSocket server for real-time token streaming
- Ollama backend for LLM inference
- Conversation state management
- Health monitoring and metrics
- GPU-optimized containerization

Author: FastTalk Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FastTalk Team"

from app.utils.config import Config
from app.utils.logger import StructuredLogger

__all__ = ["Config", "StructuredLogger"]
