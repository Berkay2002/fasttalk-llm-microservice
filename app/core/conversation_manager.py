"""
Conversation management for LLM service.

Manages conversation history, context windows, and session state.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from app.core.text_processor import TextContext, calculate_text_similarity
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationState:
    """State of a conversation session."""

    session_id: str
    system_prompt: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history_length: int = 50
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Statistics
    total_turns: int = 0
    total_tokens_generated: int = 0

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
        self.total_turns += 1
        self.last_updated = time.time()

        # Trim history if needed
        if len(self.messages) > self.max_history_length:
            # Keep system prompt if it exists
            has_system = self.messages[0].get("role") == "system"
            start_idx = 1 if has_system else 0
            keep_count = self.max_history_length - (1 if has_system else 0)

            if has_system:
                self.messages = [self.messages[0]] + self.messages[-keep_count:]
            else:
                self.messages = self.messages[-keep_count:]

            logger.info(f"Trimmed conversation history to {len(self.messages)} messages")

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get formatted messages for API call."""
        return self.messages.copy()

    def clear_history(self, keep_system_prompt: bool = True):
        """Clear conversation history."""
        if keep_system_prompt and self.messages and self.messages[0].get("role") == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

        logger.info(f"Cleared conversation history (session: {self.session_id})")

    def get_age(self) -> float:
        """Get conversation age in seconds."""
        return time.time() - self.created_at

    def get_idle_time(self) -> float:
        """Get time since last update in seconds."""
        return time.time() - self.last_updated


class ConversationManager:
    """
    Manages multiple conversation sessions.

    Handles conversation history, context windows, and message formatting.
    """

    def __init__(self, max_history_length: int = 50):
        """
        Initialize conversation manager.

        Args:
            max_history_length: Maximum messages to keep per conversation
        """
        self.max_history_length = max_history_length
        self.conversations: Dict[str, ConversationState] = {}
        self.text_context_processor = TextContext()

        logger.info(f"ConversationManager initialized (max_history: {max_history_length})")

    def create_session(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
        max_history_length: Optional[int] = None,
    ) -> ConversationState:
        """
        Create a new conversation session.

        Args:
            session_id: Unique session identifier
            system_prompt: Optional system prompt
            max_history_length: Optional override for history length

        Returns:
            ConversationState instance
        """
        if session_id in self.conversations:
            logger.warning(f"Session {session_id} already exists, recreating")
            self.end_session(session_id)

        history_length = max_history_length or self.max_history_length
        conv_state = ConversationState(
            session_id=session_id,
            system_prompt=system_prompt,
            max_history_length=history_length,
        )

        # Add system prompt if provided
        if system_prompt:
            conv_state.add_message("system", system_prompt)

        self.conversations[session_id] = conv_state

        logger.info(f"Created conversation session: {session_id}")
        return conv_state

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Get conversation session."""
        return self.conversations.get(session_id)

    def add_user_message(self, session_id: str, content: str) -> bool:
        """
        Add user message to conversation.

        Args:
            session_id: Session identifier
            content: User message content

        Returns:
            True if successful, False if session not found
        """
        conv = self.conversations.get(session_id)
        if not conv:
            logger.error(f"Cannot add message: session {session_id} not found")
            return False

        conv.add_message("user", content)
        logger.debug(f"Added user message to session {session_id}")
        return True

    def add_assistant_message(
        self, session_id: str, content: str, tokens_generated: int = 0
    ) -> bool:
        """
        Add assistant message to conversation.

        Args:
            session_id: Session identifier
            content: Assistant message content
            tokens_generated: Number of tokens generated

        Returns:
            True if successful, False if session not found
        """
        conv = self.conversations.get(session_id)
        if not conv:
            logger.error(f"Cannot add message: session {session_id} not found")
            return False

        conv.add_message("assistant", content)
        conv.total_tokens_generated += tokens_generated
        logger.debug(f"Added assistant message to session {session_id}")
        return True

    def get_messages_for_generation(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get conversation messages formatted for LLM API.

        Args:
            session_id: Session identifier

        Returns:
            List of messages or None if session not found
        """
        conv = self.conversations.get(session_id)
        if not conv:
            logger.error(f"Cannot get messages: session {session_id} not found")
            return None

        return conv.get_messages_for_api()

    def clear_history(self, session_id: str, keep_system_prompt: bool = True) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
            keep_system_prompt: Whether to keep system prompt

        Returns:
            True if successful, False if session not found
        """
        conv = self.conversations.get(session_id)
        if not conv:
            logger.error(f"Cannot clear history: session {session_id} not found")
            return False

        conv.clear_history(keep_system_prompt)
        return True

    def end_session(self, session_id: str) -> bool:
        """
        End and remove a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was removed, False if not found
        """
        conv = self.conversations.pop(session_id, None)
        if conv:
            logger.info(
                f"Ended session {session_id} "
                f"(turns: {conv.total_turns}, tokens: {conv.total_tokens_generated}, "
                f"duration: {conv.get_age():.1f}s)"
            )
            return True

        logger.warning(f"Cannot end session: {session_id} not found")
        return False

    def cleanup_idle_sessions(self, idle_timeout: float = 3600.0) -> int:
        """
        Clean up sessions that have been idle too long.

        Args:
            idle_timeout: Idle timeout in seconds

        Returns:
            Number of sessions cleaned up
        """
        idle_sessions = [
            session_id
            for session_id, conv in self.conversations.items()
            if conv.get_idle_time() > idle_timeout
        ]

        for session_id in idle_sessions:
            self.end_session(session_id)

        if idle_sessions:
            logger.info(f"Cleaned up {len(idle_sessions)} idle sessions")

        return len(idle_sessions)

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.conversations)

    def get_all_session_ids(self) -> List[str]:
        """Get list of all active session IDs."""
        return list(self.conversations.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get conversation manager statistics.

        Returns:
            Dictionary with statistics
        """
        total_turns = sum(conv.total_turns for conv in self.conversations.values())
        total_tokens = sum(conv.total_tokens_generated for conv in self.conversations.values())

        return {
            "active_sessions": len(self.conversations),
            "total_turns": total_turns,
            "total_tokens_generated": total_tokens,
        }
