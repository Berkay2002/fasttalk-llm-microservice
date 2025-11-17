#!/bin/bash
# Quick start script for LLM service on Apple Silicon (Mac)

set -e

echo "===================================================================="
echo "FastTalk LLM Service - Apple Silicon (MPS) Mode Quick Start"
echo "===================================================================="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS with Apple Silicon only"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "Warning: Not running on Apple Silicon (arm64), detected: $ARCH"
    echo "Falling back to CPU mode..."
    export COMPUTE_DEVICE=cpu
else
    echo "Detected Apple Silicon (arm64)"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Set MPS as default for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        sed -i.bak 's/COMPUTE_DEVICE=cuda/COMPUTE_DEVICE=mps/' .env
    else
        sed -i.bak 's/COMPUTE_DEVICE=cuda/COMPUTE_DEVICE=cpu/' .env
    fi
    rm .env.bak
    echo "Please edit .env with your configuration"
    echo "Note: COMPUTE_DEVICE is set to '${COMPUTE_DEVICE:-mps}'"
    exit 1
fi

# Load environment
source .env

echo "Configuration:"
echo "  Compute Device: ${COMPUTE_DEVICE:-mps}"
echo "  Model: ${LLM_MODEL:-llama3.2:1b}"
echo "  Ollama URL: ${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "  Port: ${LLM_PORT:-8000}"
echo "  CPU Threads: ${NUM_THREADS:-8}"
echo "===================================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies (PyTorch with MPS support)
echo "Installing Apple Silicon optimized dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set Apple Silicon-specific environment variables
export COMPUTE_DEVICE=${COMPUTE_DEVICE:-mps}
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=${NUM_THREADS:-8}

# Start service
echo "Starting LLM service in Apple Silicon (MPS) mode..."
python main.py websocket

deactivate
