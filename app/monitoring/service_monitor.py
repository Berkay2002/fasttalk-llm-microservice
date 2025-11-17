"""
Monitoring and health check service for LLM service.

Provides Flask-based health check endpoints and metrics.
"""

import time
import psutil
import logging
from typing import Dict, Any
from threading import Thread

from flask import Flask, jsonify

logger = logging.getLogger(__name__)


class ServiceMonitor:
    """Tracks service metrics and health."""

    def __init__(self):
        """Initialize service monitor."""
        self.start_time = time.time()
        self.request_count = 0
        self.generation_count = 0
        self.error_count = 0
        self.total_tokens_generated = 0
        self.total_processing_time = 0.0

    def record_request(self):
        """Record a request."""
        self.request_count += 1

    def record_generation(self, tokens: int, processing_time: float):
        """Record a generation."""
        self.generation_count += 1
        self.total_tokens_generated += tokens
        self.total_processing_time += processing_time

    def record_error(self):
        """Record an error."""
        self.error_count += 1

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_processing_time = (
            self.total_processing_time / self.generation_count if self.generation_count > 0 else 0.0
        )

        return {
            "uptime_seconds": self.get_uptime(),
            "requests": self.request_count,
            "generations": self.generation_count,
            "errors": self.error_count,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_processing_time_seconds": avg_processing_time,
        }


class MonitoringServer:
    """Flask-based monitoring server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9092):
        """
        Initialize monitoring server.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.monitor = ServiceMonitor()
        self.app = Flask(__name__)

        # Register routes
        self._register_routes()

        logger.info(f"MonitoringServer initialized on {host}:{port}")

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            """Full health check."""
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            health_data = {
                "status": "healthy",
                "uptime_seconds": self.monitor.get_uptime(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024 ** 3),
                },
                "metrics": self.monitor.get_metrics(),
            }

            # Add warnings
            if cpu_percent > 90:
                health_data["warnings"] = health_data.get("warnings", [])
                health_data["warnings"].append("High CPU usage")
            if memory.percent > 90:
                health_data["warnings"] = health_data.get("warnings", [])
                health_data["warnings"].append("High memory usage")

            return jsonify(health_data)

        @self.app.route("/health/ready", methods=["GET"])
        def ready():
            """Readiness probe."""
            return jsonify({"status": "ready"})

        @self.app.route("/health/live", methods=["GET"])
        def live():
            """Liveness probe."""
            return jsonify({"status": "live"})

        @self.app.route("/metrics", methods=["GET"])
        def metrics():
            """Service metrics."""
            return jsonify(self.monitor.get_metrics())

        @self.app.route("/info", methods=["GET"])
        def info():
            """Service information."""
            return jsonify({
                "service": "llm-service",
                "version": "1.0.0",
                "uptime_seconds": self.monitor.get_uptime(),
            })

    def start(self):
        """Start monitoring server in background thread."""
        thread = Thread(target=self._run_server, daemon=True)
        thread.start()
        logger.info(f"Monitoring server started on {self.host}:{self.port}")

    def _run_server(self):
        """Run Flask server."""
        self.app.run(host=self.host, port=self.port, debug=False)
