#!/usr/bin/env bash
# One-shot installer for macOS / Linux.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[error] python3 not found. Install Python 3.10+."
    exit 1
fi

# Require 3.10+
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    python3 -c "import sys; print('[error] Python 3.10+ required, found:', sys.version)"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[info] Creating virtual environment at .venv ..."
    python3 -m venv .venv
fi

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

echo
echo "[ok] Ready. Launch with: ./run.sh"
