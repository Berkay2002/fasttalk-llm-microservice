"""
PydanticAI Voice Agent for FastTalk.

Provides streaming agent with tool calling support via vLLM backend.
"""

import os
import logging
from typing import Optional, AsyncGenerator, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

logger = logging.getLogger(__name__)


# ============================================================================
# Context and Configuration Models
# ============================================================================


@dataclass
class ConversationContext:
    """
    Context for a voice conversation session.
    
    Passed to agent tools for session-aware operations.
    """
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages for context."""
        return self.conversation_history[-limit:]


class AgentConfig(BaseModel):
    """Configuration for the voice agent."""
    
    # vLLM Configuration (local GPU inference)
    # Default: AWQ 4-bit quantized Llama 3.1 8B (~4.5GB VRAM)
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    vllm_api_key: str = "not-needed"
    
    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    # Tool settings
    enable_web_search: bool = True
    enable_tools: bool = True
    duckduckgo_rate_limit: float = 1.0
    
    # System prompt
    system_prompt: str = """You are a helpful voice assistant for FastTalk. 
Keep your responses concise and conversational, suitable for speech synthesis.
When asked about current events or facts you're unsure about, use web search.
Avoid long lists or complex formatting - speak naturally as if in a conversation."""


# ============================================================================
# Voice Agent Implementation
# ============================================================================


class VoiceAgent:
    """
    PydanticAI-based voice agent with vLLM backend.
    
    Features:
    - Streaming text generation for low-latency TTS
    - Optional web search via DuckDuckGo
    - Custom tool support
    - Conversation context management
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the voice agent.
        
        Args:
            config: Agent configuration. If None, loads from environment.
        """
        self.config = config or self._load_config_from_env()
        self._agent: Optional[Agent] = None
        self._model: Optional[OpenAIChatModel] = None
        
        logger.info(
            f"VoiceAgent initialized: model={self.config.vllm_model}, "
            f"base_url={self.config.vllm_base_url}"
        )

    def _load_config_from_env(self) -> AgentConfig:
        """Load configuration from environment variables."""
        return AgentConfig(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1"),
            vllm_model=os.getenv("VLLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "not-needed"),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "2048")),
            top_p=float(os.getenv("DEFAULT_TOP_P", "0.9")),
            enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true",
            enable_tools=os.getenv("ENABLE_TOOLS", "true").lower() == "true",
            duckduckgo_rate_limit=float(os.getenv("DUCKDUCKGO_RATE_LIMIT", "1.0")),
            system_prompt=os.getenv("SYSTEM_PROMPT", AgentConfig().system_prompt),
        )

    def _get_model(self) -> OpenAIChatModel:
        """Get or create the OpenAI-compatible model for vLLM."""
        if self._model is None:
            # Create provider with custom base_url for vLLM
            provider = OpenAIProvider(
                base_url=self.config.vllm_base_url,
                api_key=self.config.vllm_api_key,
            )
            self._model = OpenAIChatModel(
                self.config.vllm_model,
                provider=provider,
            )
        return self._model

    def _get_agent(self) -> Agent:
        """Get or create the PydanticAI agent."""
        if self._agent is None:
            tools = []
            
            # Add web search tool if enabled
            if self.config.enable_web_search:
                try:
                    tools.append(duckduckgo_search_tool())
                    logger.info("Web search tool enabled (DuckDuckGo)")
                except Exception as e:
                    logger.warning(f"DuckDuckGo search tool not available: {e}")

            self._agent = Agent(
                model=self._get_model(),
                system_prompt=self.config.system_prompt,
                tools=tools if self.config.enable_tools else [],
                deps_type=ConversationContext,
            )
            
            # Register custom tools
            self._register_custom_tools()
            
        return self._agent

    def _register_custom_tools(self):
        """Register custom tools on the agent."""
        if not self.config.enable_tools:
            return

        agent = self._agent

        @agent.tool
        async def get_current_time(ctx: RunContext[ConversationContext]) -> str:
            """Get the current date and time."""
            now = datetime.now()
            return f"The current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')}."

        @agent.tool
        async def get_session_info(ctx: RunContext[ConversationContext]) -> str:
            """Get information about the current conversation session."""
            session = ctx.deps
            message_count = len(session.conversation_history)
            duration = (datetime.now() - session.created_at).seconds
            return (
                f"Session {session.session_id[:8]}... has been active for {duration} seconds "
                f"with {message_count} messages exchanged."
            )

        logger.info("Custom tools registered: get_current_time, get_session_info")

    async def generate_stream(
        self,
        user_message: str,
        context: ConversationContext,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to a user message.
        
        Args:
            user_message: The user's input message
            context: Conversation context with history
            temperature: Override temperature setting
            max_tokens: Override max tokens setting
            
        Yields:
            str: Individual text chunks as they're generated
        """
        agent = self._get_agent()
        
        # Build message history for context
        message_prompt = self._build_prompt_with_history(user_message, context)
        
        logger.info(f"[{context.session_id}] Generating response for: {user_message[:50]}...")
        
        try:
            async with agent.run_stream(
                message_prompt,
                deps=context,
                model_settings={
                    "temperature": temperature or self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "top_p": self.config.top_p,
                },
            ) as result:
                async for text in result.stream_text(delta=True):
                    yield text
                    
            logger.info(f"[{context.session_id}] Generation completed successfully")
            
        except Exception as e:
            logger.error(f"[{context.session_id}] Generation error: {e}", exc_info=True)
            raise

    def _build_prompt_with_history(
        self,
        user_message: str,
        context: ConversationContext,
    ) -> str:
        """
        Build a prompt that includes conversation history.
        
        Args:
            user_message: Current user message
            context: Conversation context
            
        Returns:
            Formatted prompt string
        """
        # Get recent messages for context
        recent = context.get_recent_messages(limit=10)
        
        if not recent:
            return user_message
        
        # Format history as context
        history_parts = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"]
            history_parts.append(f"{role}: {content}")
        
        history_str = "\n".join(history_parts)
        
        return f"""Previous conversation:
{history_str}

Current message from user: {user_message}"""

    async def generate(
        self,
        user_message: str,
        context: ConversationContext,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).
        
        Args:
            user_message: The user's input message
            context: Conversation context
            temperature: Override temperature setting
            max_tokens: Override max tokens setting
            
        Returns:
            Complete response text
        """
        chunks = []
        async for chunk in self.generate_stream(
            user_message, context, temperature, max_tokens
        ):
            chunks.append(chunk)
        return "".join(chunks)

    def update_config(self, **kwargs):
        """
        Update agent configuration.
        
        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key}={value}")
        
        # Reset agent to apply new config
        self._agent = None
        self._model = None

    def check_connection(self) -> bool:
        """
        Check if vLLM backend is accessible.
        
        Returns:
            True if connection successful
        """
        import httpx
        
        try:
            # Check vLLM health endpoint
            health_url = self.config.vllm_base_url.replace("/v1", "/health")
            with httpx.Client(timeout=5.0) as client:
                response = client.get(health_url)
                response.raise_for_status()
            logger.info(f"vLLM connection OK: {self.config.vllm_base_url}")
            return True
        except Exception as e:
            logger.error(f"vLLM connection failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.config.vllm_model,
            "base_url": self.config.vllm_base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "web_search_enabled": self.config.enable_web_search,
            "tools_enabled": self.config.enable_tools,
        }
