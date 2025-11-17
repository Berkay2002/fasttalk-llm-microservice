"""
Configuration management for LLM service.

Handles environment-based configuration with sensible defaults
and presets for different use cases.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _detect_compute_device() -> str:
    """
    Detect and validate the compute device to use.
    
    Priority:
    1. COMPUTE_DEVICE environment variable (cuda/cpu/mps)
    2. Auto-detection based on availability
    
    Returns:
        Device string: 'cuda', 'cpu', or 'mps'
    """
    device = os.getenv("COMPUTE_DEVICE", "").lower()
    
    if device in ("cuda", "cpu", "mps"):
        # Explicit device specified
        if device == "cuda":
            if os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES"):
                logger.info("Using CUDA GPU (NVIDIA)")
                return "cuda"
            else:
                logger.warning("CUDA requested but no GPU devices found, falling back to CPU")
                return "cpu"
        elif device == "mps":
            # Check if running on macOS with Apple Silicon
            if sys.platform == "darwin":
                logger.info("Using MPS (Apple Silicon GPU)")
                return "mps"
            else:
                logger.warning("MPS requested but not on macOS, falling back to CPU")
                return "cpu"
        else:
            logger.info("Using CPU")
            return "cpu"
    else:
        # Auto-detect
        if os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES"):
            logger.info("Auto-detected CUDA GPU")
            return "cuda"
        elif sys.platform == "darwin":
            logger.info("Auto-detected macOS, using MPS if available")
            return "mps"
        else:
            logger.info("Auto-detected CPU")
            return "cpu"


@dataclass
class Config:
    """
    Configuration settings for the LLM service.

    All settings can be overridden via environment variables.
    """

    # ============================================================================
    # Compute Device Configuration
    # ============================================================================
    compute_device: str = field(default_factory=_detect_compute_device)
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2:1b"))
    # Legacy device field for backward compatibility
    device: str = field(default_factory=_detect_compute_device)

    # ============================================================================
    # Ollama Configuration
    # ============================================================================
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "5m"))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "600.0")))

    # ============================================================================
    # Generation Configuration
    # ============================================================================
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.7")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "2048")))
    default_context_window: int = field(default_factory=lambda: int(os.getenv("DEFAULT_CONTEXT_WINDOW", "8192")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "0.9")))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "40")))

    # ============================================================================
    # Server Configuration
    # ============================================================================
    host: str = field(default_factory=lambda: os.getenv("LLM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("LLM_PORT", "8000")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CONNECTIONS", "50")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ============================================================================
    # Monitoring Configuration
    # ============================================================================
    monitoring_port: int = field(default_factory=lambda: int(os.getenv("LLM_MONITORING_PORT", "9092")))
    monitoring_host: str = field(default_factory=lambda: os.getenv("LLM_MONITORING_HOST", "0.0.0.0"))

    # ============================================================================
    # Performance Configuration
    # ============================================================================
    num_threads: int = field(default_factory=lambda: int(os.getenv("NUM_THREADS", "12")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "6")))

    # ============================================================================
    # Session Configuration
    # ============================================================================
    session_timeout: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT", "3600")))  # 1 hour
    max_history_length: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_LENGTH", "50")))

    # ============================================================================
    # Paths
    # ============================================================================
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "/app/models"))
    log_path: str = field(default_factory=lambda: os.getenv("LOG_PATH", "/app/logs"))

    def __post_init__(self):
        """Validate and log configuration after initialization."""
        self._validate()
        self._log_config()

    def _validate(self):
        """Validate configuration values."""
        # Validate temperature
        if not 0.0 <= self.default_temperature <= 2.0:
            logger.warning(f"Temperature {self.default_temperature} is outside recommended range [0.0, 2.0]")

        # Validate top_p
        if not 0.0 <= self.default_top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.default_top_p}")

        # Validate top_k
        if self.default_top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.default_top_k}")

        # Validate max_tokens
        if self.default_max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.default_max_tokens}")

        # Validate context_window
        if self.default_context_window < self.default_max_tokens:
            logger.warning(
                f"Context window ({self.default_context_window}) is smaller than max_tokens ({self.default_max_tokens})"
            )

        # Validate ports
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Port must be between 1024 and 65535, got {self.port}")
        if not 1024 <= self.monitoring_port <= 65535:
            raise ValueError(f"Monitoring port must be between 1024 and 65535, got {self.monitoring_port}")

        # Validate max_connections
        if self.max_connections < 1:
            raise ValueError(f"max_connections must be >= 1, got {self.max_connections}")

    def _log_config(self):
        """Log current configuration."""
        logger.info("=" * 60)
        logger.info("LLM Service Configuration")
        logger.info("=" * 60)
        logger.info(f"Compute Device: {self.compute_device}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Ollama URL: {self.ollama_base_url}")
        logger.info(f"Server: {self.host}:{self.port}")
        logger.info(f"Monitoring: {self.monitoring_host}:{self.monitoring_port}")
        logger.info(f"Max Connections: {self.max_connections}")
        logger.info(f"Temperature: {self.default_temperature}")
        logger.info(f"Max Tokens: {self.default_max_tokens}")
        logger.info(f"Context Window: {self.default_context_window}")
        logger.info(f"Num Threads: {self.num_threads}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "compute_device": self.compute_device,
            "model_name": self.model_name,
            "device": self.device,  # Legacy field
            "ollama_base_url": self.ollama_base_url,
            "ollama_keep_alive": self.ollama_keep_alive,
            "host": self.host,
            "port": self.port,
            "monitoring_port": self.monitoring_port,
            "max_connections": self.max_connections,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_context_window": self.default_context_window,
            "default_top_p": self.default_top_p,
            "default_top_k": self.default_top_k,
            "num_threads": self.num_threads,
            "num_workers": self.num_workers,
            "log_level": self.log_level,
        }

    @classmethod
    def from_preset(cls, preset: str) -> "Config":
        """
        Create configuration from a preset.

        Args:
            preset: One of 'fast', 'balanced', 'quality'

        Returns:
            Config instance with preset values
        """
        presets = {
            "fast": {
                "default_temperature": 0.5,
                "default_max_tokens": 1024,
                "default_context_window": 4096,
                "default_top_p": 0.9,
                "default_top_k": 40,
            },
            "balanced": {
                "default_temperature": 0.7,
                "default_max_tokens": 2048,
                "default_context_window": 8192,
                "default_top_p": 0.9,
                "default_top_k": 40,
            },
            "quality": {
                "default_temperature": 0.8,
                "default_max_tokens": 4096,
                "default_context_window": 16384,
                "default_top_p": 0.95,
                "default_top_k": 50,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

        logger.info(f"Loading preset: {preset}")
        config = cls()

        # Update with preset values
        for key, value in presets[preset].items():
            setattr(config, key, value)

        return config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    return Config()
