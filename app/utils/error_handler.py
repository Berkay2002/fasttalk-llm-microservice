"""
Comprehensive error handling for LLM service.

Provides error categorization, circuit breakers, retry logic, and error tracking.
"""

import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONNECTION = "connection"
    PROCESSING = "processing"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    GPU = "gpu"
    TIMEOUT = "timeout"
    VALIDATION = "validation"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    recoverable: bool
    timestamp: float = field(default_factory=time.time)
    retry_after: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.retry_after = retry_after

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    Monitors operation failures and opens circuit when threshold is exceeded.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting half-open state
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            LLMServiceError: If circuit is open
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time and (time.time() - self._last_failure_time) >= self.timeout:
                    logger.info(f"Circuit breaker '{self.name}': Transitioning to HALF_OPEN")
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    raise LLMServiceError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        category=ErrorCategory.RESOURCE,
                        severity=ErrorSeverity.HIGH,
                        recoverable=True,
                        retry_after=self.timeout,
                    )

            # Check if we're in HALF_OPEN and exceeded max calls
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise LLMServiceError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN (max calls reached)",
                        category=ErrorCategory.RESOURCE,
                        severity=ErrorSeverity.HIGH,
                        recoverable=True,
                        retry_after=10.0,
                    )
                self._half_open_calls += 1

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                logger.info(f"Circuit breaker '{self.name}': Closing (recovery successful)")
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                logger.warning(f"Circuit breaker '{self.name}': Reopening (recovery failed)")
                self._state = CircuitBreakerState.OPEN
                self._half_open_calls = 0
            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker '{self.name}': Opening (threshold {self.failure_threshold} reached)"
                    )
                    self._state = CircuitBreakerState.OPEN

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}': Manual reset")
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0


class RetryManager:
    """
    Manages retry logic with exponential backoff.
    """

    @staticmethod
    def retry_with_backoff(
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        retriable_exceptions: tuple = (Exception,),
    ) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            max_attempts: Maximum number of attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for exponential backoff
            retriable_exceptions: Tuple of exceptions to retry on

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        delay = base_delay

        for attempt in range(1, max_attempts + 1):
            try:
                return func()
            except retriable_exceptions as e:
                last_exception = e
                if attempt == max_attempts:
                    logger.error(f"All {max_attempts} retry attempts failed")
                    raise

                logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)

        # Should never reach here, but for type safety
        raise last_exception


class ErrorHandler:
    """
    Central error handler with tracking and circuit breakers.
    """

    def __init__(self, max_error_history: int = 1000):
        """
        Initialize error handler.

        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.max_error_history = max_error_history
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self._lock = Lock()

        # Circuit breakers for critical operations
        self.ollama_circuit_breaker = CircuitBreaker(
            name="ollama_connection",
            failure_threshold=3,
            timeout=300.0,
        )
        self.generation_circuit_breaker = CircuitBreaker(
            name="generation",
            failure_threshold=5,
            timeout=120.0,
        )

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """
        Handle and categorize an error.

        Args:
            error: Exception that occurred
            context: Additional context information

        Returns:
            ErrorInfo with categorized error details
        """
        # Categorize error
        if isinstance(error, LLMServiceError):
            category = error.category
            severity = error.severity
            recoverable = error.recoverable
            message = error.message
        else:
            category, severity, recoverable = self._categorize_error(error)
            message = str(error)

        # Create error info
        error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=message,
            recoverable=recoverable,
            context=context,
        )

        # Track error
        with self._lock:
            self.error_history.append(error_info)
            self.error_counts[category] += 1

        # Log error
        log_method = logger.error if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else logger.warning
        log_method(
            f"Error handled: [{category.value}] {message}",
            extra={
                "category": category.value,
                "severity": severity.value,
                "recoverable": recoverable,
                "context": context,
            },
        )

        return error_info

    def _categorize_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
        """
        Categorize an error based on its type and message.

        Args:
            error: Exception to categorize

        Returns:
            Tuple of (category, severity, recoverable)
        """
        error_str = str(error).lower()

        # Connection errors
        if "connection" in error_str or "refused" in error_str:
            return ErrorCategory.CONNECTION, ErrorSeverity.HIGH, True

        # Timeout errors
        if "timeout" in error_str:
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, True

        # GPU errors
        if "cuda" in error_str or "gpu" in error_str or "out of memory" in error_str:
            return ErrorCategory.GPU, ErrorSeverity.CRITICAL, True

        # Resource errors
        if "resource" in error_str or "capacity" in error_str:
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH, True

        # Validation errors
        if "invalid" in error_str or "validation" in error_str:
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW, False

        # Default to processing error
        return ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM, True

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error counts and recent errors
        """
        with self._lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "by_category": {cat.value: count for cat, count in self.error_counts.items()},
                "recent_errors": len(self.error_history),
                "circuit_breakers": {
                    "ollama": self.ollama_circuit_breaker.state.value,
                    "generation": self.generation_circuit_breaker.state.value,
                },
            }

    def reset(self):
        """Reset error tracking."""
        with self._lock:
            self.error_history.clear()
            self.error_counts = {cat: 0 for cat in ErrorCategory}
            self.ollama_circuit_breaker.reset()
            self.generation_circuit_breaker.reset()
            logger.info("Error handler reset")
