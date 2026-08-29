from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.json"

PCAPS_DIR = REPO_ROOT / "pcaps"
FEATURES_DIR = REPO_ROOT / "features"
RESULTS_DIR = REPO_ROOT / "results"
UPLOADS_DIR = REPO_ROOT / "uploads" / "sessions"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}
