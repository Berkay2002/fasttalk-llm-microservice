#!/bin/bash
# Quick start script for LLM service (Linux/Mac)

set -e

echo "===================================================================="
echo "FastTalk LLM Service - Quick Start"
echo "===================================================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
    exit 1
fi

# Load environment
source .env

echo "Configuration:"
echo "  Model: ${LLM_MODEL:-llama3.2:1b}"
echo "  Ollama URL: ${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "  Port: ${LLM_PORT:-8000}"
echo "===================================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start service
echo "Starting LLM service..."
python main.py websocket

deactivate
