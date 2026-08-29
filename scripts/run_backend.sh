#!/bin/bash
# Linux: install deps (first run only) then start the NIDS backend + UI.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/backend/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$REPO/backend/.venv"
fi

"$PY" -m pip install -r "$REPO/backend/requirements.txt"
"$PY" "$REPO/scripts/patch_cicflowmeter.py" "$PY"

cd "$REPO"
"$PY" -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8765
