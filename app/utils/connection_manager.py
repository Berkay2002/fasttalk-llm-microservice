"""
WebSocket connection management for LLM service.

Tracks active connections, manages session state, and provides connection lifecycle management.
"""

import time
import logging
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    ACTIVE = "active"
    PROCESSING = "processing"
    DISCONNECTING = "disconnecting"
    CLOSED = "closed"


@dataclass
class ConnectionInfo:
    """
    Information about a WebSocket connection.

    Tracks session state, metrics, and configuration.
    """
    session_id: str
    websocket: WebSocket
    state: ConnectionState = ConnectionState.CONNECTING
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Session metrics
    messages_received: int = 0
    messages_sent: int = 0
    tokens_generated: int = 0
    generations_completed: int = 0
    errors_count: int = 0

    # Session configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Conversation context
    conversation_history: list = field(default_factory=list)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def get_duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time

    def get_idle_time(self) -> float:
        """Get time since last activity in seconds."""
        return time.time() - self.last_activity

    def to_dict(self) -> Dict[str, Any]:
        """Convert connection info to dictionary."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "duration_seconds": self.get_duration(),
            "idle_seconds": self.get_idle_time(),
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "tokens_generated": self.tokens_generated,
            "generations_completed": self.generations_completed,
            "errors_count": self.errors_count,
            "config": self.config,
        }


class ConnectionManager:
    """
    Manages WebSocket connections for the LLM service.

    Provides thread-safe connection tracking, lifecycle management, and statistics.
    """

    def __init__(self, max_connections: int = 50):
        """
        Initialize connection manager.

        Args:
            max_connections: Maximum number of concurrent connections allowed
        """
        self.max_connections = max_connections
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self._lock = Lock()

        # Global statistics
        self.total_connections = 0
        self.total_disconnections = 0
        self.total_messages_received = 0
        self.total_messages_sent = 0
        self.total_tokens_generated = 0
        self.total_generations_completed = 0

        logger.info(f"ConnectionManager initialized with max_connections={max_connections}")

    def add_connection(
        self,
        session_id: str,
        websocket: WebSocket,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConnectionInfo]:
        """
        Add a new WebSocket connection.

        Args:
            session_id: Unique session identifier
            websocket: FastAPI WebSocket instance
            config: Optional session configuration

        Returns:
            ConnectionInfo if successful, None if max connections reached
        """
        with self._lock:
            # Check if max connections reached
            if len(self.active_connections) >= self.max_connections:
                logger.warning(
                    f"Max connections ({self.max_connections}) reached. "
                    f"Rejecting connection {session_id}"
                )
                return None

            # Check if session already exists
            if session_id in self.active_connections:
                logger.warning(f"Session {session_id} already exists. Closing old connection.")
                # Clean up old connection
                self.active_connections.pop(session_id)

            # Create connection info
            conn_info = ConnectionInfo(
                session_id=session_id,
                websocket=websocket,
                state=ConnectionState.ACTIVE,
                config=config or {},
            )

            self.active_connections[session_id] = conn_info
            self.total_connections += 1

            logger.info(
                f"Connection added: {session_id} "
                f"(active: {len(self.active_connections)}/{self.max_connections})"
            )

            return conn_info

    def remove_connection(self, session_id: str) -> bool:
        """
        Remove a WebSocket connection.

        Args:
            session_id: Session identifier to remove

        Returns:
            True if connection was removed, False if not found
        """
        with self._lock:
            conn_info = self.active_connections.pop(session_id, None)

            if conn_info:
                # Update global statistics
                self.total_disconnections += 1
                self.total_messages_received += conn_info.messages_received
                self.total_messages_sent += conn_info.messages_sent
                self.total_tokens_generated += conn_info.tokens_generated
                self.total_generations_completed += conn_info.generations_completed

                logger.info(
                    f"Connection removed: {session_id} "
                    f"(duration: {conn_info.get_duration():.1f}s, "
                    f"msgs: {conn_info.messages_received}/{conn_info.messages_sent}, "
                    f"tokens: {conn_info.tokens_generated})"
                )
                return True

            logger.warning(f"Attempted to remove non-existent connection: {session_id}")
            return False

    def get_connection(self, session_id: str) -> Optional[ConnectionInfo]:
        """
        Get connection info for a session.

        Args:
            session_id: Session identifier

        Returns:
            ConnectionInfo if found, None otherwise
        """
        with self._lock:
            return self.active_connections.get(session_id)

    def update_connection_state(self, session_id: str, state: ConnectionState) -> bool:
        """
        Update connection state.

        Args:
            session_id: Session identifier
            state: New connection state

        Returns:
            True if successful, False if connection not found
        """
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                old_state = conn_info.state
                conn_info.state = state
                conn_info.update_activity()
                logger.debug(f"Connection {session_id} state: {old_state.value} -> {state.value}")
                return True

            logger.warning(f"Cannot update state for non-existent connection: {session_id}")
            return False

    def record_message_received(self, session_id: str):
        """Record a message received from client."""
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                conn_info.messages_received += 1
                conn_info.update_activity()

    def record_message_sent(self, session_id: str):
        """Record a message sent to client."""
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                conn_info.messages_sent += 1
                conn_info.update_activity()

    def record_tokens_generated(self, session_id: str, count: int):
        """Record tokens generated for a session."""
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                conn_info.tokens_generated += count
                conn_info.update_activity()

    def record_generation_complete(self, session_id: str):
        """Record a completed generation."""
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                conn_info.generations_completed += 1
                conn_info.update_activity()

    def record_error(self, session_id: str):
        """Record an error for a session."""
        with self._lock:
            conn_info = self.active_connections.get(session_id)
            if conn_info:
                conn_info.errors_count += 1
                conn_info.update_activity()

    def get_active_count(self) -> int:
        """Get number of active connections."""
        with self._lock:
            return len(self.active_connections)

    def get_session_list(self) -> list[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self.active_connections.keys())

    def cleanup_idle_connections(self, idle_timeout: float = 3600.0) -> int:
        """
        Clean up connections that have been idle too long.

        Args:
            idle_timeout: Idle timeout in seconds (default: 1 hour)

        Returns:
            Number of connections cleaned up
        """
        cleaned = 0
        with self._lock:
            idle_sessions = [
                session_id
                for session_id, conn_info in self.active_connections.items()
                if conn_info.get_idle_time() > idle_timeout
            ]

            for session_id in idle_sessions:
                logger.info(
                    f"Cleaning up idle connection: {session_id} "
                    f"(idle: {self.active_connections[session_id].get_idle_time():.1f}s)"
                )
                self.active_connections.pop(session_id)
                cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} idle connections")

        return cleaned

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        with self._lock:
            active_connections = len(self.active_connections)

            # Calculate average session duration for active connections
            if active_connections > 0:
                avg_duration = sum(
                    conn.get_duration() for conn in self.active_connections.values()
                ) / active_connections
            else:
                avg_duration = 0.0

            return {
                "active_connections": active_connections,
                "max_connections": self.max_connections,
                "utilization_percent": (active_connections / self.max_connections * 100)
                if self.max_connections > 0
                else 0.0,
                "total_connections": self.total_connections,
                "total_disconnections": self.total_disconnections,
                "total_messages_received": self.total_messages_received,
                "total_messages_sent": self.total_messages_sent,
                "total_tokens_generated": self.total_tokens_generated,
                "total_generations_completed": self.total_generations_completed,
                "average_session_duration_seconds": avg_duration,
            }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics including per-session info.

        Returns:
            Dictionary with detailed statistics
        """
        with self._lock:
            return {
                "summary": self.get_statistics(),
                "active_sessions": {
                    session_id: conn_info.to_dict()
                    for session_id, conn_info in self.active_connections.items()
                },
            }

    def reset_statistics(self):
        """Reset global statistics."""
        with self._lock:
            self.total_connections = len(self.active_connections)
            self.total_disconnections = 0
            self.total_messages_received = 0
            self.total_messages_sent = 0
            self.total_tokens_generated = 0
            self.total_generations_completed = 0
            logger.info("Connection statistics reset")
