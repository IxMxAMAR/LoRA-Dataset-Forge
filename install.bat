@echo off
REM One-shot Windows installer: creates a local .venv and installs dependencies.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python not found on PATH. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [info] Creating virtual environment at .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Failed to create venv.
        pause
        exit /b 1
    )
)

echo [info] Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Dependency install failed.
    pause
    exit /b 1
)

echo.
echo [ok] Ready. Launch with: run.bat
pause
