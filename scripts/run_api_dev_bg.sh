#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
RUNNER_SCRIPT="$ROOT_DIR/scripts/run_api_dev.py"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment interpreter: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Missing API runner script: $RUNNER_SCRIPT" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$RUNNER_SCRIPT"