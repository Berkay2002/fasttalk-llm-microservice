"""
Structured logging system for LLM service.

Provides JSON-formatted logging for production with human-readable console output.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Context variable for request tracking
request_context: ContextVar[Optional[str]] = ContextVar("request_context", default=None)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "llm-service",
            "component": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        req_id = request_context.get()
        if req_id:
            log_data["request_id"] = req_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter with colors.

    Provides colored output for different log levels for better readability.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Build formatted message
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:8s}{reset}"
        component = f"{record.name:30s}"
        message = record.getMessage()

        # Add request context if available
        req_id = request_context.get()
        if req_id:
            message = f"[{req_id[:8]}] {message}"

        formatted = f"{timestamp} | {level} | {component} | {message}"

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


class StructuredLogger:
    """
    Structured logger with request context tracking and specialized logging methods.

    Provides both JSON (file) and console (human-readable) output.
    """

    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if file logging enabled)
            enable_console: Enable console output
            enable_file: Enable file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()  # Clear any existing handlers

        # Console handler with color formatting
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            self.logger.addHandler(console_handler)

        # File handler with JSON formatting
        if enable_file and log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)

    def set_request_context(self, request_id: str):
        """Set request context for correlation."""
        request_context.set(request_id)

    def clear_request_context(self):
        """Clear request context."""
        request_context.set(None)

    def _log_with_extra(self, level: str, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Internal method to log with extra fields."""
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level),
            "",
            0,
            message,
            (),
            None,
        )
        if extra_fields:
            record.extra_fields = extra_fields
        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_extra("DEBUG", message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_extra("INFO", message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_extra("WARNING", message, kwargs if kwargs else None)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_extra("ERROR", message, kwargs if kwargs else None)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_extra("CRITICAL", message, kwargs if kwargs else None)

    def log_generation(
        self,
        prompt: str,
        tokens_generated: int,
        processing_time: float,
        tokens_per_second: float,
        model: str,
    ):
        """
        Log LLM generation details.

        Args:
            prompt: Input prompt (truncated if too long)
            tokens_generated: Number of tokens generated
            processing_time: Total processing time in seconds
            tokens_per_second: Generation speed (tokens/second)
            model: Model name used
        """
        # Truncate prompt if too long
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt

        self.info(
            f"Generation complete: {tokens_generated} tokens in {processing_time:.2f}s ({tokens_per_second:.1f} tok/s)",
            prompt_preview=prompt_preview,
            tokens_generated=tokens_generated,
            processing_time_seconds=processing_time,
            tokens_per_second=tokens_per_second,
            model=model,
        )

    def log_performance(self, component: str, operation: str, duration: float, **kwargs):
        """
        Log performance metrics.

        Args:
            component: Component name (e.g., "ollama_handler")
            operation: Operation name (e.g., "generate_stream")
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        self.info(
            f"Performance: {component}.{operation} took {duration:.4f}s",
            component=component,
            operation=operation,
            duration_seconds=duration,
            **kwargs,
        )

    def log_connection(self, session_id: str, action: str, **kwargs):
        """
        Log WebSocket connection events.

        Args:
            session_id: Session identifier
            action: Action type (connected, disconnected, error)
            **kwargs: Additional context
        """
        self.info(
            f"Connection {action}: {session_id}",
            session_id=session_id,
            action=action,
            **kwargs,
        )


def get_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name
        log_level: Log level (default: from config)
        log_file: Log file path (optional)

    Returns:
        StructuredLogger instance
    """
    if log_level is None:
        import os
        log_level = os.getenv("LOG_LEVEL", "INFO")

    return StructuredLogger(
        name=name,
        log_level=log_level,
        log_file=log_file,
        enable_console=True,
        enable_file=log_file is not None,
    )
