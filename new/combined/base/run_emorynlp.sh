#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../../../.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../../.venv/bin/python"
else
    PYTHON="python3"
fi

# Run combined mixed training by default (pass --steer to run steer instead)
if [[ "$*" == *"--steer"* ]]; then
    ARGS=("${@/--steer/}")
    "$PYTHON" "$SCRIPT_DIR/run_local_steer.py" --dataset emorynlp "${ARGS[@]}"
else
    "$PYTHON" "$SCRIPT_DIR/run_local_combined.py" --dataset emorynlp "$@"
fi
