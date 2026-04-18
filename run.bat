@echo off
REM LoRA-Dataset-Forge launcher (Windows)
REM Prefers a local .venv if present; otherwise uses system python.
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if "%PY%"=="" (
    echo.
    echo [error] Python not found on PATH and no local .venv/venv folder exists.
    echo.
    echo Fix one of these two ways:
    echo   1. Run install.bat to create a local .venv with the correct dependencies
    echo   2. Install Python 3.10+ from https://python.org and re-run this script
    echo.
    pause
    exit /b 1
)

echo Using Python: %PY%
echo.

REM Verify dependencies. Print the real error if the check fails — hiding it
REM with 2^>nul was a bug in a prior release: it left users with a silent
REM "press any key to continue" and no diagnostic.
"%PY%" -c "import sys; print('Python', sys.version); import google.genai as g; print('google-genai OK:', hasattr(g, 'Client')); import PIL; print('Pillow', PIL.__version__)"
if errorlevel 1 (
    echo.
    echo [error] Dependency check failed — see the Python error above.
    echo.
    echo Fix one of these:
    echo   * Run install.bat to create a local .venv with the right packages
    echo   * Install manually: "%PY%" -m pip install -r requirements.txt
    echo.
    echo Note: this tool requires the `google-genai` package.
    echo       The legacy `google-generativeai` package has the same namespace
    echo       but is a different SDK and will NOT work. If you have the wrong
    echo       one installed, run:  "%PY%" -m pip uninstall google-generativeai
    echo.
    pause
    exit /b 1
)

echo.
echo Launching LoRA-Dataset-Forge...
echo.
"%PY%" forge.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [warn] forge.py exited with code %RC% ^(non-zero usually means an uncaught exception^).
    echo        The error, if any, should be visible above this line.
    pause
)
endlocal
