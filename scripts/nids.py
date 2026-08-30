#!/usr/bin/env python3
"""Thin shim so the CLI runs as `python scripts/nids.py ...` without an
install. The real logic is in `backend/cli/`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
