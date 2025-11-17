"""
FastAPI WebSocket server for LLM service.

Provides real-time token streaming via WebSocket connections.
"""

import json
import uuid
import asyncio
import time
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.config import Config
from app.utils.logger import get_logger
from app.utils.connection_manager import ConnectionManager, ConnectionState
from app.utils.error_handler import ErrorHandler, LLMServiceError
from app.core.ollama_handler import OllamaHandler
from app.core.conversation_manager import ConversationManager

logger = get_logger(__name__)


class WebSocketLLMServer:
    """
    FastAPI WebSocket server for LLM streaming.

    Handles WebSocket connections, message routing, and LLM generation.
    """

    def __init__(self, config: Config):
        """
        Initialize WebSocket server.

        Args:
            config: Configuration instance
        """
        self.config = config

        # Initialize FastAPI app
        self.app = FastAPI(
            title="FastTalk LLM Service",
            description="WebSocket-based LLM streaming service",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self.connection_manager = ConnectionManager(max_connections=config.max_connections)
        self.conversation_manager = ConversationManager(max_history_length=config.max_history_length)
        self.error_handler = ErrorHandler()

        # Initialize Ollama handler
        self.ollama_handler = OllamaHandler(
            base_url=config.ollama_base_url,
            model=config.model_name,
            keep_alive=config.ollama_keep_alive,
            timeout=config.ollama_timeout,
        )

        # Register routes
        self._register_routes()

        logger.info("WebSocketLLMServer initialized")

    def _register_routes(self):
        """Register HTTP and WebSocket routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "FastTalk LLM Service",
                "status": "ready",
                "model": self.config.model_name,
                "version": "1.0.0",
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            try:
                # Check Ollama connection
                ollama_ok = self.ollama_handler.check_connection()

                health_status = {
                    "status": "healthy" if ollama_ok else "degraded",
                    "model": self.config.model_name,
                    "ollama_connection": ollama_ok,
                    "active_connections": self.connection_manager.get_active_count(),
                    "active_sessions": self.conversation_manager.get_session_count(),
                }

                return JSONResponse(
                    content=health_status,
                    status_code=200 if ollama_ok else 503,
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    content={"status": "unhealthy", "error": str(e)},
                    status_code=503,
                )

        @self.app.get("/stats")
        async def stats():
            """Statistics endpoint."""
            return {
                "connections": self.connection_manager.get_statistics(),
                "conversations": self.conversation_manager.get_statistics(),
                "errors": self.error_handler.get_error_stats(),
            }

        @self.app.websocket("/ws/llm")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for LLM streaming."""
            await self.handle_websocket(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        """
        Handle WebSocket connection lifecycle.

        Args:
            websocket: FastAPI WebSocket instance
        """
        session_id = str(uuid.uuid4())
        logger.info(f"New WebSocket connection: {session_id}")

        # Accept connection
        await websocket.accept()

        # Try to add connection
        conn_info = self.connection_manager.add_connection(session_id, websocket)
        if not conn_info:
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": "max_connections",
                    "message": "Maximum connections reached",
                    "severity": "high",
                },
            })
            await websocket.close()
            return

        try:
            # Send session started message
            await websocket.send_json({
                "type": "session_started",
                "session_id": session_id,
            })

            # Message loop
            while True:
                # Receive message
                data = await websocket.receive_text()
                self.connection_manager.record_message_received(session_id)

                try:
                    message = json.loads(data)
                    await self._handle_message(session_id, message, websocket)
                except json.JSONDecodeError:
                    logger.error(f"[{session_id}] Invalid JSON received")
                    await websocket.send_json({
                        "type": "error",
                        "error": {
                            "code": "invalid_json",
                            "message": "Invalid JSON format",
                        },
                    })
                except Exception as e:
                    logger.error(f"[{session_id}] Error handling message: {e}", exc_info=True)
                    self.connection_manager.record_error(session_id)
                    error_info = self.error_handler.handle_error(e, {"session_id": session_id})
                    await websocket.send_json({
                        "type": "error",
                        "error": {
                            "code": error_info.category.value,
                            "message": error_info.message,
                            "severity": error_info.severity.value,
                            "recoverable": error_info.recoverable,
                        },
                    })

        except WebSocketDisconnect:
            logger.info(f"[{session_id}] WebSocket disconnected")
        except Exception as e:
            logger.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
        finally:
            # Cleanup
            self.connection_manager.remove_connection(session_id)
            self.conversation_manager.end_session(session_id)
            logger.info(f"[{session_id}] Cleanup complete")

    async def _handle_message(self, session_id: str, message: Dict[str, Any], websocket: WebSocket):
        """
        Handle incoming WebSocket message.

        Args:
            session_id: Session identifier
            message: Parsed JSON message
            websocket: WebSocket instance
        """
        msg_type = message.get("type")

        if msg_type == "start_session":
            await self._handle_start_session(session_id, message, websocket)
        elif msg_type == "user_message":
            await self._handle_user_message(session_id, message, websocket)
        elif msg_type == "cancel":
            await self._handle_cancel(session_id, websocket)
        elif msg_type == "end_session":
            await self._handle_end_session(session_id, websocket)
        else:
            logger.warning(f"[{session_id}] Unknown message type: {msg_type}")
            await websocket.send_json({
                "type": "error",
                "error": {"code": "unknown_message_type", "message": f"Unknown message type: {msg_type}"},
            })

    async def _handle_start_session(
        self, session_id: str, message: Dict[str, Any], websocket: WebSocket
    ):
        """Handle session start message."""
        config = message.get("config", {})
        system_prompt = config.get("system_prompt")

        # Create conversation session
        self.conversation_manager.create_session(
            session_id=session_id,
            system_prompt=system_prompt,
        )

        logger.info(f"[{session_id}] Session started with config: {config}")

        await websocket.send_json({
            "type": "session_configured",
            "config": config,
        })
        self.connection_manager.record_message_sent(session_id)

    async def _handle_user_message(
        self, session_id: str, message: Dict[str, Any], websocket: WebSocket
    ):
        """Handle user message and generate response."""
        user_text = message.get("text", "")

        if not user_text:
            await websocket.send_json({
                "type": "error",
                "error": {"code": "empty_message", "message": "Empty user message"},
            })
            return

        logger.info(f"[{session_id}] User message: {user_text[:100]}...")

        # Add user message to conversation
        self.conversation_manager.add_user_message(session_id, user_text)

        # Update connection state
        self.connection_manager.update_connection_state(session_id, ConnectionState.PROCESSING)

        # Generate response
        start_time = time.time()
        token_count = 0
        full_response = ""

        try:
            # Get messages for API
            messages = self.conversation_manager.get_messages_for_generation(session_id)
            
            # Check if messages is None (session not found)
            if messages is None:
                await websocket.send_json({
                    "type": "error",
                    "error": {
                        "code": "session_not_found",
                        "message": "Conversation session not found",
                    },
                })
                return

            # Get generation config
            conn_info = self.connection_manager.get_connection(session_id)
            gen_config = conn_info.config if conn_info else {}

            # Generate with streaming
            generator = self.ollama_handler.generate_stream(
                messages=messages,
                temperature=gen_config.get("temperature", self.config.default_temperature),
                max_tokens=gen_config.get("max_tokens", self.config.default_max_tokens),
                top_p=gen_config.get("top_p", self.config.default_top_p),
                top_k=gen_config.get("top_k", self.config.default_top_k),
                request_id=session_id,
            )

            # Stream tokens
            for token in generator:
                await websocket.send_json({"type": "token", "data": token})
                self.connection_manager.record_message_sent(session_id)
                self.connection_manager.record_tokens_generated(session_id, 1)
                token_count += 1
                full_response += token

            # Generation complete
            end_time = time.time()
            duration = end_time - start_time
            tokens_per_second = token_count / duration if duration > 0 else 0.0

            # Add assistant message to conversation
            self.conversation_manager.add_assistant_message(
                session_id, full_response, tokens_generated=token_count
            )

            # Mark generation complete
            self.connection_manager.record_generation_complete(session_id)

            # Send completion message
            await websocket.send_json({
                "type": "response_complete",
                "stats": {
                    "tokens_generated": token_count,
                    "processing_time_ms": duration * 1000,
                    "tokens_per_second": tokens_per_second,
                },
            })
            self.connection_manager.record_message_sent(session_id)

            logger.info(
                f"[{session_id}] Generation complete: {token_count} tokens in {duration:.2f}s "
                f"({tokens_per_second:.1f} tok/s)"
            )

        except LLMServiceError as e:
            logger.error(f"[{session_id}] LLM service error: {e}")
            await websocket.send_json({
                "type": "error",
                "error": e.to_dict(),
            })
            self.connection_manager.record_error(session_id)
        except Exception as e:
            logger.error(f"[{session_id}] Generation error: {e}", exc_info=True)
            error_info = self.error_handler.handle_error(e, {"session_id": session_id})
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": error_info.category.value,
                    "message": error_info.message,
                    "severity": error_info.severity.value,
                },
            })
            self.connection_manager.record_error(session_id)
        finally:
            # Reset connection state
            self.connection_manager.update_connection_state(session_id, ConnectionState.ACTIVE)

    async def _handle_cancel(self, session_id: str, websocket: WebSocket):
        """Handle generation cancellation."""
        logger.info(f"[{session_id}] Cancellation requested")

        # Cancel generation
        cancelled = self.ollama_handler.cancel_generation(session_id)

        await websocket.send_json({
            "type": "cancelled",
            "success": cancelled,
        })
        self.connection_manager.record_message_sent(session_id)

    async def _handle_end_session(self, session_id: str, websocket: WebSocket):
        """Handle session end."""
        logger.info(f"[{session_id}] End session requested")

        # Get session stats
        conn_info = self.connection_manager.get_connection(session_id)
        stats = conn_info.to_dict() if conn_info else {}

        await websocket.send_json({
            "type": "session_ended",
            "stats": stats,
        })
        self.connection_manager.record_message_sent(session_id)

        # Close connection will be handled by finally block
