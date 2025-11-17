#!/bin/bash
# Quick start script for LLM service on CPU (Linux/Mac)

set -e

echo "===================================================================="
echo "FastTalk LLM Service - CPU Mode Quick Start"
echo "===================================================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Set CPU as default
    sed -i.bak 's/COMPUTE_DEVICE=cuda/COMPUTE_DEVICE=cpu/' .env
    rm .env.bak
    echo "Please edit .env with your configuration"
    echo "Note: COMPUTE_DEVICE is set to 'cpu'"
    exit 1
fi

# Load environment
source .env

echo "Configuration:"
echo "  Compute Device: ${COMPUTE_DEVICE:-cpu}"
echo "  Model: ${LLM_MODEL:-llama3.2:1b}"
echo "  Ollama URL: ${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "  Port: ${LLM_PORT:-8000}"
echo "  CPU Threads: ${NUM_THREADS:-12}"
echo "===================================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install CPU version of PyTorch
echo "Installing CPU-optimized dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Set CPU-specific environment variables
export COMPUTE_DEVICE=cpu
export OMP_NUM_THREADS=${NUM_THREADS:-12}
export MKL_NUM_THREADS=${NUM_THREADS:-12}
export OPENBLAS_NUM_THREADS=${NUM_THREADS:-12}

# Start service
echo "Starting LLM service in CPU mode..."
python main.py websocket

deactivate
