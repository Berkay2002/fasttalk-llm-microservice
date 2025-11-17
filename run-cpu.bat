@echo off
REM Quick start script for LLM service on CPU (Windows)

echo ====================================================================
echo FastTalk LLM Service - CPU Mode Quick Start
echo ====================================================================

REM Check if .env exists
if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
    powershell -Command "(gc .env) -replace 'COMPUTE_DEVICE=cuda', 'COMPUTE_DEVICE=cpu' | Out-File -encoding ASCII .env"
    echo Please edit .env with your configuration
    echo Note: COMPUTE_DEVICE is set to 'cpu'
    exit /b 1
)

REM Load environment (manual for Windows)
echo Please ensure your .env file has COMPUTE_DEVICE=cpu

echo Configuration:
echo   Compute Device: cpu
echo   Model: llama3.2:1b (default)
echo   Port: 8000 (default)
echo ====================================================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install CPU version of PyTorch
echo Installing CPU-optimized dependencies...
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

REM Set CPU-specific environment variables
set COMPUTE_DEVICE=cpu
set OMP_NUM_THREADS=12
set MKL_NUM_THREADS=12

REM Start service
echo Starting LLM service in CPU mode...
python main.py websocket

deactivate
