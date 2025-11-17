@echo off
REM Quick start script for LLM service (Windows)

echo ====================================================================
echo FastTalk LLM Service - Quick Start
echo ====================================================================

REM Check if .env exists
if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo Please edit .env with your configuration
    exit /b 1
)

REM Load environment (note: this doesn't actually load vars in batch)
echo Loading configuration from .env...

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start service
echo Starting LLM service...
python main.py websocket

deactivate
