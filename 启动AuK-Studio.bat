@echo off
chcp 65001 >nul
title AuK Studio - Quick Start

cd /d "%~dp0"

echo ==========================================
echo          AuK Studio Quick Start
echo ==========================================
echo.
echo Project: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Please create the environment first:
    echo     uv venv --python 3.10
    echo     uv pip install -e .
    echo     uv pip install -r requirements-web.txt
    echo.
    pause
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found in PATH.
    echo Please install uv first.
    echo.
    pause
    exit /b 1
)

echo [1/2] Starting AuK Studio...
echo.
echo NOTE: Models are NOT loaded at startup.
echo       They will be loaded when you click Generate
echo       or manually load them in Model & Settings.
echo.

start "" http://127.0.0.1:8080/

echo [2/2] Starting FastAPI server...
echo.
uv run python run.py

echo.
echo ==========================================
echo AuK Studio has stopped.
echo ==========================================
echo.
pause
