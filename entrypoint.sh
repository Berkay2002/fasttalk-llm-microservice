#!/bin/bash
# Entrypoint script for LLM service container

set -e

echo "===================================================================="
echo "FastTalk LLM Service - Container Starting"
echo "===================================================================="

# Set correct permissions for cache and log directories
echo "Setting up permissions..."
chown -R appuser:appuser /app/logs || true
chmod -R 755 /app/logs || true

# Print configuration
echo "Configuration:"
echo "  Model: ${LLM_MODEL:-llama3.2:1b}"
echo "  Ollama URL: ${OLLAMA_BASE_URL:-http://ollama:11434}"
echo "  Port: ${LLM_PORT:-8000}"
echo "  Monitoring Port: ${LLM_MONITORING_PORT:-9092}"
echo "===================================================================="

# Start the service
echo "Starting LLM service..."
exec python main.py websocket
