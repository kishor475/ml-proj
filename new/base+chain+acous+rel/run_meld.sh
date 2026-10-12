#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/../../.venv/bin/python"

if [ -f "$VENV_PY" ]; then
    PYTHON="$VENV_PY"
else
    PYTHON="python3"
fi

"$PYTHON" "$SCRIPT_DIR/run_local_ours.py" --dataset meld "$@"
