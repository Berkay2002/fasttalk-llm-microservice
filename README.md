# FastTalk LLM Microservice

A production-ready Large Language Model microservice providing WebSocket-based streaming inference with conversation management, built on Ollama.

## Features

- **Real-time Streaming**: Token-by-token streaming via WebSocket
- **Conversation Management**: Automatic conversation history and context management
- **Multiple Sessions**: Support for concurrent sessions with isolated state
- **Health Monitoring**: Comprehensive health checks and metrics
- **Multi-Platform Support**: CUDA GPU, CPU, and Apple Silicon (MPS) acceleration
- **Error Handling**: Circuit breakers, retry logic, and comprehensive error tracking
- **Production Ready**: Docker containerization, structured logging, and monitoring

## Compute Platform Support

The service supports three compute platforms:

1. **CUDA GPU** (NVIDIA GPUs) - Best performance for production workloads
2. **CPU** - Universal compatibility, good for development and low-traffic deployments
3. **Apple Silicon (MPS)** - Optimized for M1/M2/M3 Macs with GPU acceleration

## Quick Start

### Option 1: Docker Compose (Recommended)

#### CUDA GPU (NVIDIA)

```bash
# From the llm-service directory
cd microservices/llm-service

# Copy and configure environment variables
cp .env.example .env
# Edit .env and set: COMPUTE_DEVICE=cuda

# Start the service
docker-compose -f docker-compose.gpu.yml up -d

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f llm-service
```

#### CPU

```bash
# From the llm-service directory
cd microservices/llm-service

# Copy and configure environment variables
cp .env.example .env
# Edit .env and set: COMPUTE_DEVICE=cpu

# Start the service
docker-compose -f docker-compose.cpu.yml up -d

# Check logs
docker-compose -f docker-compose.cpu.yml logs -f llm-service
```

#### Apple Silicon (MPS)

```bash
# From the llm-service directory (macOS only)
cd microservices/llm-service

# Copy and configure environment variables
cp .env.example .env
# Edit .env and set: COMPUTE_DEVICE=mps

# Start the service
docker-compose -f docker-compose.apple.yml up -d

# Check logs
docker-compose -f docker-compose.apple.yml logs -f llm-service
```

### Option 2: Standalone Deployment

#### Using Run Scripts

**CUDA GPU (NVIDIA):**
```bash
./run-gpu.sh          # Linux/Mac
run-gpu.bat           # Windows
```

**CPU:**
```bash
./run-cpu.sh          # Linux/Mac
run-cpu.bat           # Windows
```

**Apple Silicon (macOS):**
```bash
./run-apple.sh        # Mac only
```

#### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For CPU, install CPU-optimized PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Configure environment
export COMPUTE_DEVICE=cpu  # or cuda or mps
export LLM_MODEL=llama3.2:1b
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_PORT=8000

# Start server
python main.py websocket
```

## Configuration

### Environment Variables

Key configuration options (see `.env.example` for full list):

```bash
# Compute Device
COMPUTE_DEVICE=cuda            # Options: cuda, cpu, mps

# GPU Configuration (CUDA only)
GPU_DEVICE_ID=0
CUDA_VISIBLE_DEVICES=0

# Model
LLM_MODEL=llama3.2:1b          # Ollama model name

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_KEEP_ALIVE=5m

# Server
LLM_PORT=8000                  # WebSocket port
LLM_MONITORING_PORT=9092       # Monitoring port
LLM_MAX_CONNECTIONS=50

# Generation
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048
DEFAULT_CONTEXT_WINDOW=8192

# Performance (CPU/MPS)
NUM_THREADS=12                 # CPU threads (12 for CPU, 8 for Apple Silicon)
NUM_WORKERS=6                  # Worker processes
```

### Compute Device Selection

The `COMPUTE_DEVICE` environment variable controls which compute platform to use:

- **`cuda`**: Use NVIDIA GPU with CUDA acceleration
  - Requires NVIDIA GPU and drivers
  - Best performance for production
  - Uses `docker-compose.gpu.yml` and `Dockerfile.gpu`

- **`cpu`**: Use CPU only
  - Universal compatibility
  - Good for development and low-traffic scenarios
  - Uses `docker-compose.cpu.yml` and `Dockerfile.cpu`

- **`mps`**: Use Apple Silicon GPU (Metal Performance Shaders)
  - For M1/M2/M3 Macs
  - Better performance than CPU on Apple Silicon
  - Uses `docker-compose.apple.yml` and `Dockerfile.cpu`

If not specified, the service will auto-detect based on available hardware.

## WebSocket API

### Connection

Connect to: `ws://localhost:8002/ws/llm`

### Message Protocol

#### Client → Server

**Start Session:**
```json
{
  "type": "start_session",
  "config": {
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

**Send Message:**
```json
{
  "type": "user_message",
  "text": "What is the capital of France?"
}
```

**Cancel Generation:**
```json
{
  "type": "cancel"
}
```

**End Session:**
```json
{
  "type": "end_session"
}
```

#### Server → Client

**Session Started:**
```json
{
  "type": "session_started",
  "session_id": "uuid-here"
}
```

**Token Stream:**
```json
{
  "type": "token",
  "data": "Paris"
}
```

**Response Complete:**
```json
{
  "type": "response_complete",
  "stats": {
    "tokens_generated": 150,
    "processing_time_ms": 2300,
    "tokens_per_second": 65.2
  }
}
```

**Error:**
```json
{
  "type": "error",
  "error": {
    "code": "generation_failed",
    "message": "Model unavailable",
    "severity": "high",
    "recoverable": true
  }
}
```

## HTTP Endpoints

### Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
{
  "status": "healthy",
  "model": "llama3.2:1b",
  "ollama_connection": true,
  "active_connections": 3,
  "active_sessions": 3
}
```

### Statistics

```bash
curl http://localhost:8002/stats
```

Response:
```json
{
  "connections": {
    "active_connections": 3,
    "total_connections": 127,
    "total_tokens_generated": 45678
  },
  "conversations": {
    "active_sessions": 3,
    "total_turns": 89,
    "total_tokens_generated": 45678
  },
  "errors": {
    "total_errors": 2,
    "by_category": {...}
  }
}
```

### Monitoring

```bash
curl http://localhost:9092/health
curl http://localhost:9092/metrics
```

## Architecture

```
┌────────────────────────────────────────────────┐
│              LLM Microservice                  │
│                                                │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   WebSocket  │────────▶│  Conversation   │  │
│  │    Server    │         │    Manager      │  │
│  │  (FastAPI)   │         └─────────────────┘  │
│  └──────┬───────┘                              │
│         │                                      │
│         ▼                                      │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   Ollama     │────────▶│     Ollama      │  │
│  │   Handler    │         │     Server      │  │
│  └──────────────┘         └─────────────────┘  │
│                                                │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │  Monitoring  │         │ Error Handler + │  │
│  │   Service    │         │ Circuit Breaker │  │
│  └──────────────┘         └─────────────────┘  │
└────────────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Test Ollama connection
python main.py test

# Show configuration
python main.py config --show
```

### Local Development

```bash
# Start with auto-reload
python main.py websocket --log-level DEBUG

# Custom port
python main.py websocket --port 8080

# Custom model
python main.py websocket --model llama3:8b
```

## Integration with Backend

The LLM service can be integrated with the backend-orchestration service:

```yaml
# backend-orchestration/docker-compose.yml
services:
  app:
    environment:
      - USE_REMOTE_LLM=true
      - LLM_WEBSOCKET_URL=ws://llm-service:8000/ws/llm
    depends_on:
      - llm-service
```

## Deployment Comparison

### File Structure by Platform

| File | CUDA GPU | CPU | Apple Silicon |
|------|----------|-----|---------------|
| Dockerfile | `Dockerfile.gpu` | `Dockerfile.cpu` | `Dockerfile.cpu` |
| Docker Compose | `docker-compose.gpu.yml` | `docker-compose.cpu.yml` | `docker-compose.apple.yml` |
| Run Script (Unix) | `run-gpu.sh` | `run-cpu.sh` | `run-apple.sh` |
| Run Script (Windows) | `run-gpu.bat` | `run-cpu.bat` | N/A |

### Choosing the Right Platform

**Use CUDA GPU when:**
- You have NVIDIA GPU hardware
- Running in production with high traffic
- Need maximum performance and throughput
- Working with large models (70B+)

**Use CPU when:**
- No GPU available
- Development and testing
- Low traffic scenarios
- Running small models (1B-3B)
- Universal deployment across different hardware

**Use Apple Silicon (MPS) when:**
- Running on M1/M2/M3 Mac
- Development on macOS
- Better performance than CPU on Mac
- Testing before production deployment

### Performance Comparison

| Platform | llama3.2:1b | llama3:8b | llama3:70b | Best Use Case |
|----------|-------------|-----------|------------|---------------|
| CUDA GPU (RTX 3090) | ~150 tok/s | ~80 tok/s | ~20 tok/s | Production |
| Apple M2 Max (MPS) | ~40 tok/s | ~15 tok/s | ~3 tok/s | Development |
| CPU (12-core i7) | ~10 tok/s | ~3 tok/s | N/A | Testing |

*Performance varies based on hardware specifications*

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.{gpu|cpu|apple}.yml logs llm-service

# Verify Ollama connection
curl http://localhost:11434/

# Test configuration
python main.py test
```

### Platform-Specific Issues

#### CUDA GPU
```bash
# Verify GPU is detected
nvidia-smi

# Check CUDA drivers
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# If GPU not detected, ensure:
# - NVIDIA drivers are installed
# - nvidia-docker2 is installed
# - Docker daemon has GPU support enabled
```

#### CPU
```bash
# Optimize CPU performance
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Check CPU usage
htop

# If performance is poor:
# - Reduce NUM_THREADS in .env
# - Use smaller models (llama3.2:1b)
# - Reduce concurrent connections
```

#### Apple Silicon (MPS)
```bash
# Verify architecture
uname -m  # Should show arm64

# Check if MPS is being used (in Python)
python -c "import torch; print(torch.backends.mps.is_available())"

# If MPS fails:
# - Ensure PYTORCH_ENABLE_MPS_FALLBACK=1 is set
# - Update to latest PyTorch version
# - Fall back to CPU: COMPUTE_DEVICE=cpu
```

### High Latency

- Use smaller model (e.g., `llama3.2:1b` instead of `llama3:70b`)
- Reduce `max_tokens` for faster responses
- Check GPU availability: `nvidia-smi`
- Monitor metrics: `curl http://localhost:9092/metrics`

### Connection Errors

```bash
# Verify Ollama is running
docker ps | grep ollama

# Check network
docker network inspect fasttalk-network

# Test WebSocket connection
wscat -c ws://localhost:8002/ws/llm
```

## Performance

### Expected Latency (RTX 3090)

| Model | Tokens/Second | First Token | Notes |
|-------|---------------|-------------|-------|
| llama3.2:1b | ~100-150 | ~100ms | Fast, good for chat |
| llama3:8b | ~50-80 | ~200ms | Balanced |
| llama3:70b | ~10-20 | ~1s | High quality |

### Resource Usage

- **Memory**: 32GB RAM recommended
- **GPU**: 8GB VRAM minimum (depends on model)
- **CPU**: 8+ cores for concurrent requests

## License

Part of the FastTalk project.

## Support

For issues and questions:
- Check logs: `docker-compose logs llm-service`
- Monitor health: `curl http://localhost:8002/health`
- Review configuration: `python main.py config --show`
