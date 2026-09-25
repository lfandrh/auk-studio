@echo off
setlocal
title AuK Studio
cd /d "%~dp0"

REM Keep this file pure ASCII. cmd.exe reads it with the ANSI codepage, so a UTF-8 BOM
REM would corrupt line 1 ("@echo off") before any chcp could help.
REM
REM run.py owns the decisions: if the port already holds an AuK instance it ends that one and
REM starts a fresh server (so a second double click can never leave you on stale code, and can
REM never stack a second invisible model process), it never touches processes that are not
REM provably AuK, and it opens the browser only once the port accepts connections.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create the environment first:
    echo     uv venv --python 3.10
    echo     uv pip install -e .
    echo     uv pip install -r requirements-web.txt
    echo.
    pause
    exit /b 1
)

REM A .venv folder can exist and still be unusable: dependencies installed on another machine,
REM or the uv-managed interpreter it points at since deleted. Probing the imports the server
REM needs costs ~0.3s and turns an instant window flash into a readable instruction.
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, pydantic, auk" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] .venv exists but cannot import the web server packages.
    echo Repair the environment from this folder:
    echo     uv pip install -e .
    echo     uv pip install -r requirements-web.txt
    echo.
    echo Run the line below without the nul redirect to see the real error:
    echo     ".venv\Scripts\python.exe" -c "import fastapi, uvicorn, pydantic, auk"
    echo.
    pause
    exit /b 1
)

echo ==========================================
echo          AuK Studio
echo ==========================================
echo.
echo Project: %CD%
echo Models are NOT loaded at startup; they load on first use
echo or when you click "Load model" in Model and Settings.
echo.
echo To stop cleanly press Ctrl+C, or run the stop script in this folder.
echo.

".venv\Scripts\python.exe" run.py %*
set "RC=%ERRORLEVEL%"

echo.
echo ==========================================
echo AuK Studio has stopped (exit code %RC%).
echo ==========================================
echo.
pause
exit /b %RC%
