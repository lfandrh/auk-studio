@echo off
setlocal
title AuK Studio - Stop
cd /d "%~dp0"

REM Kills whatever is listening on the studio port plus its whole process tree (the model
REM worker is a child of the server), then any multiprocessing-spawn python left parentless.
REM Pure ASCII on purpose -- see the note in the start script.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run.py --stop

echo.
pause
