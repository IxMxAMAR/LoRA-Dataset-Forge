#!/usr/bin/env bash
# LoRA-Dataset-Forge launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "[error] Python not found. Install Python 3.10+ or run install.sh."
    exit 1
fi

"$PY" -c "import google.genai as g, PIL; assert hasattr(g, 'Client'), 'wrong SDK'" 2>/dev/null || {
    echo "[info] Missing or wrong dependencies. Run install.sh, or:"
    echo "       $PY -m pip install -r requirements.txt"
    echo "       (Requires google-genai, not the legacy google-generativeai.)"
    exit 1
}

"$PY" forge.py
