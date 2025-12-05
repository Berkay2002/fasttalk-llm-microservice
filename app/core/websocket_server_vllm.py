"""
FastAPI WebSocket server for LLM service with vLLM + PydanticAI support.

Provides real-time token streaming via WebSocket connections.
Supports both vLLM (with PydanticAI) and Ollama backends.
"""

import json
import uuid
import asyncio
import time
import logging
from typing import Optional, Dict, Any, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.config import Config
from app.utils.logger import get_logger
from app.utils.connection_manager import ConnectionManager, ConnectionState
from app.utils.error_handler import ErrorHandler, LLMServiceError
from app.core.conversation_manager import ConversationManager

logger = get_logger(__name__)


class WebSocketLLMServer:
    """
    FastAPI WebSocket server for LLM streaming.

    Handles WebSocket connections, message routing, and LLM generation.
    Supports both vLLM (with PydanticAI) and Ollama backends.
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
            description="WebSocket-based LLM streaming service with vLLM + PydanticAI support",
            version="2.0.0",
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

        # Initialize appropriate LLM handler based on provider
        self._init_llm_handler()

        # Register routes
        self._register_routes()

        logger.info(f"WebSocketLLMServer initialized with provider: {config.llm_provider}")

    def _init_llm_handler(self):
        """Initialize the appropriate LLM handler based on configuration."""
        if self.config.llm_provider == "vllm":
            self._init_vllm_handler()
        else:
            self._init_ollama_handler()

    def _init_vllm_handler(self):
        """Initialize vLLM handler with optional PydanticAI agent."""
        if self.config.enable_pydantic_ai:
            # Use PydanticAI agent for full agentic capabilities
            try:
                from app.agents.voice_agent import VoiceAgent, AgentConfig, ConversationContext
                
                agent_config = AgentConfig(
                    vllm_base_url=self.config.vllm_base_url,
                    vllm_model=self.config.vllm_model,
                    vllm_api_key=self.config.vllm_api_key,
                    temperature=self.config.default_temperature,
                    max_tokens=self.config.default_max_tokens,
                    top_p=self.config.default_top_p,
                    enable_web_search=self.config.enable_web_search,
                    enable_tools=self.config.enable_tools,
                    system_prompt=self.config.system_prompt,
                )
                
                self.voice_agent = VoiceAgent(config=agent_config)
                self.use_pydantic_ai = True
                self.vllm_handler = None
                logger.info("Initialized PydanticAI VoiceAgent for vLLM")
                
            except ImportError as e:
                logger.warning(f"PydanticAI not available, falling back to vLLM handler: {e}")
                self._init_vllm_handler_direct()
        else:
            self._init_vllm_handler_direct()

    def _init_vllm_handler_direct(self):
        """Initialize vLLM handler without PydanticAI."""
        from app.core.vllm_handler import VLLMHandler
        
        self.vllm_handler = VLLMHandler(
            base_url=self.config.vllm_base_url,
            model=self.config.vllm_model,
            api_key=self.config.vllm_api_key,
            timeout=self.config.vllm_timeout,
        )
        self.use_pydantic_ai = False
        self.voice_agent = None
        logger.info("Initialized direct vLLM handler")

    def _init_ollama_handler(self):
        """Initialize Ollama handler (legacy backend)."""
        from app.core.ollama_handler import OllamaHandler
        
        self.ollama_handler = OllamaHandler(
            base_url=self.config.ollama_base_url,
            model=self.config.model_name,
            keep_alive=self.config.ollama_keep_alive,
            timeout=self.config.ollama_timeout,
        )
        self.use_pydantic_ai = False
        self.voice_agent = None
        self.vllm_handler = None
        logger.info("Initialized Ollama handler (legacy)")

    def _register_routes(self):
        """Register HTTP and WebSocket routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "FastTalk LLM Service",
                "status": "ready",
                "version": "2.0.0",
                "provider": self.config.llm_provider,
                "model": self._get_current_model(),
                "pydantic_ai_enabled": self.use_pydantic_ai,
                "web_search_enabled": self.config.enable_web_search if self.config.llm_provider == "vllm" else False,
                "tools_enabled": self.config.enable_tools if self.config.llm_provider == "vllm" else False,
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            try:
                # Check connection based on provider
                connection_ok = self._check_backend_connection()

                health_status = {
                    "status": "healthy" if connection_ok else "degraded",
                    "provider": self.config.llm_provider,
                    "model": self._get_current_model(),
                    "backend_connection": connection_ok,
                    "pydantic_ai_enabled": self.use_pydantic_ai,
                    "active_connections": self.connection_manager.get_active_count(),
                    "active_sessions": self.conversation_manager.get_session_count(),
                }

                return JSONResponse(
                    content=health_status,
                    status_code=200 if connection_ok else 503,
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
                "provider": self.config.llm_provider,
                "pydantic_ai_enabled": self.use_pydantic_ai,
            }

        @self.app.get("/models")
        async def list_models():
            """List available models."""
            try:
                if self.config.llm_provider == "vllm":
                    if self.vllm_handler:
                        return self.vllm_handler.get_model_info()
                    elif self.voice_agent:
                        return self.voice_agent.get_model_info()
                else:
                    return self.ollama_handler.get_model_info()
            except Exception as e:
                return {"error": str(e)}

        @self.app.websocket("/ws/llm")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for LLM streaming."""
            await self.handle_websocket(websocket)

    def _get_current_model(self) -> str:
        """Get the current model name based on provider."""
        if self.config.llm_provider == "vllm":
            return self.config.vllm_model
        return self.config.model_name

    def _check_backend_connection(self) -> bool:
        """Check if the backend LLM is accessible."""
        try:
            if self.config.llm_provider == "vllm":
                if self.voice_agent:
                    return self.voice_agent.check_connection()
                elif self.vllm_handler:
                    return self.vllm_handler.check_connection()
            else:
                return self.ollama_handler.check_connection()
        except Exception as e:
            logger.error(f"Backend connection check failed: {e}")
            return False

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
                "provider": self.config.llm_provider,
                "model": self._get_current_model(),
                "pydantic_ai_enabled": self.use_pydantic_ai,
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
        elif msg_type == "update_config":
            await self._handle_update_config(session_id, message, websocket)
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
            "provider": self.config.llm_provider,
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

        # Generate response based on provider
        start_time = time.time()
        token_count = 0
        full_response = ""

        try:
            if self.config.llm_provider == "vllm" and self.use_pydantic_ai:
                # Use PydanticAI agent for generation
                token_count, full_response = await self._generate_with_pydantic_ai(
                    session_id, user_text, websocket
                )
            elif self.config.llm_provider == "vllm":
                # Use vLLM handler directly
                token_count, full_response = await self._generate_with_vllm(
                    session_id, websocket
                )
            else:
                # Use Ollama handler
                token_count, full_response = await self._generate_with_ollama(
                    session_id, websocket
                )

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
                    "provider": self.config.llm_provider,
                    "pydantic_ai_used": self.use_pydantic_ai,
                },
            })
            self.connection_manager.record_message_sent(session_id)

            logger.info(
                f"[{session_id}] Generation complete: {token_count} tokens in {duration:.2f}s "
                f"({tokens_per_second:.1f} tok/s) via {self.config.llm_provider}"
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

    async def _generate_with_pydantic_ai(
        self, session_id: str, user_text: str, websocket: WebSocket
    ) -> tuple[int, str]:
        """Generate response using PydanticAI agent."""
        from app.agents.voice_agent import ConversationContext
        from datetime import datetime
        
        # Get conversation history
        history = self.conversation_manager.get_messages_for_generation(session_id) or []
        
        # Create context for the agent
        context = ConversationContext(
            user_id=session_id,
            session_id=session_id,
            conversation_history=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in history[:-1]  # Exclude the current message
            ],
            created_at=datetime.now(),
        )
        
        # Get generation config
        conn_info = self.connection_manager.get_connection(session_id)
        gen_config = conn_info.config if conn_info else {}
        
        token_count = 0
        full_response = ""
        
        async for chunk in self.voice_agent.generate_stream(
            user_message=user_text,
            context=context,
            temperature=gen_config.get("temperature", self.config.default_temperature),
            max_tokens=gen_config.get("max_tokens", self.config.default_max_tokens),
        ):
            await websocket.send_json({"type": "token", "data": chunk})
            self.connection_manager.record_message_sent(session_id)
            self.connection_manager.record_tokens_generated(session_id, 1)
            token_count += 1
            full_response += chunk
        
        return token_count, full_response

    async def _generate_with_vllm(
        self, session_id: str, websocket: WebSocket
    ) -> tuple[int, str]:
        """Generate response using vLLM handler directly."""
        # Get messages for API
        messages = self.conversation_manager.get_messages_for_generation(session_id)
        
        if messages is None:
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": "session_not_found",
                    "message": "Conversation session not found",
                },
            })
            return 0, ""

        # Get generation config
        conn_info = self.connection_manager.get_connection(session_id)
        gen_config = conn_info.config if conn_info else {}

        token_count = 0
        full_response = ""

        # Use async generator
        async for token in self.vllm_handler.generate_stream_async(
            messages=messages,
            temperature=gen_config.get("temperature", self.config.default_temperature),
            max_tokens=gen_config.get("max_tokens", self.config.default_max_tokens),
            top_p=gen_config.get("top_p", self.config.default_top_p),
            request_id=session_id,
        ):
            await websocket.send_json({"type": "token", "data": token})
            self.connection_manager.record_message_sent(session_id)
            self.connection_manager.record_tokens_generated(session_id, 1)
            token_count += 1
            full_response += token

        return token_count, full_response

    async def _generate_with_ollama(
        self, session_id: str, websocket: WebSocket
    ) -> tuple[int, str]:
        """Generate response using Ollama handler (legacy)."""
        # Get messages for API
        messages = self.conversation_manager.get_messages_for_generation(session_id)
        
        if messages is None:
            await websocket.send_json({
                "type": "error",
                "error": {
                    "code": "session_not_found",
                    "message": "Conversation session not found",
                },
            })
            return 0, ""

        # Get generation config
        conn_info = self.connection_manager.get_connection(session_id)
        gen_config = conn_info.config if conn_info else {}

        token_count = 0
        full_response = ""

        # Generate with streaming (sync, but wrapped)
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

        return token_count, full_response

    async def _handle_cancel(self, session_id: str, websocket: WebSocket):
        """Handle generation cancellation."""
        logger.info(f"[{session_id}] Cancellation requested")

        # Cancel generation based on provider
        cancelled = False
        if self.config.llm_provider == "vllm":
            if self.vllm_handler:
                cancelled = self.vllm_handler.cancel_generation(session_id)
            # PydanticAI doesn't have built-in cancellation yet
        else:
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

    async def _handle_update_config(
        self, session_id: str, message: Dict[str, Any], websocket: WebSocket
    ):
        """Handle runtime configuration update."""
        new_config = message.get("config", {})
        
        logger.info(f"[{session_id}] Config update requested: {new_config}")
        
        # Update voice agent config if using PydanticAI
        if self.use_pydantic_ai and self.voice_agent:
            self.voice_agent.update_config(**new_config)
        
        await websocket.send_json({
            "type": "config_updated",
            "success": True,
            "config": new_config,
        })
        self.connection_manager.record_message_sent(session_id)
