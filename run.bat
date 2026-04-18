@echo off
REM LoRA-Dataset-Forge launcher (Windows)
REM Prefers a local .venv if present; otherwise uses system python.
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    where python >nul 2>nul && set "PY=python"
)

if "%PY%"=="" (
    echo [error] Python not found. Install Python 3.10+ or run install.bat to create a local venv.
    pause
    exit /b 1
)

REM Verify dependencies; offer install if missing.
REM Also guard against the wrong SDK — the legacy `google-generativeai` package
REM installs a `google.genai` namespace too but has no `Client`. Require the
REM new `google-genai` SDK by checking for `genai.Client`.
"%PY%" -c "import google.genai as g, PIL; assert hasattr(g, 'Client'), 'wrong SDK'" 2>nul
if errorlevel 1 (
    echo [info] Missing or wrong dependencies. Run install.bat, or install manually:
    echo        %PY% -m pip install -r requirements.txt
    echo        (Requires `google-genai`, not the legacy `google-generativeai`.)
    pause
    exit /b 1
)

"%PY%" forge.py
if errorlevel 1 pause
