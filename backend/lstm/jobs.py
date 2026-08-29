from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path

from .config import ARTIFACT_ROOT, STATUS_PATH
from .training import forecast_latest, train_forecaster

_lock = threading.Lock()
_process: multiprocessing.Process | None = None


def _default_status() -> dict:
    return {
        "stage": "idle",
        "rows_processed": 0,
        "cache_state": "unknown",
        "epoch": 0,
        "loss": None,
        "validation_loss": None,
        "validation_macro_f1": None,
        "error": None,
    }


def read_status() -> dict:
    if not STATUS_PATH.is_file():
        return _default_status()
    try:
        return {**_default_status(), **json.loads(STATUS_PATH.read_text())}
    except (OSError, json.JSONDecodeError):
        return {**_default_status(), "stage": "error", "error": "Training status file is unreadable."}


def write_status(**updates) -> dict:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    current = read_status()
    current.update(updates)
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2))
    os.replace(temporary, STATUS_PATH)
    return current


def _worker(force_rebuild: bool) -> None:
    write_status(**_default_status(), stage="starting", error=None)
    try:
        train_forecaster(force_rebuild=force_rebuild, status=write_status)
    except Exception as error:
        write_status(stage="error", error=f"{type(error).__name__}: {error}")


def start_training(force_rebuild: bool = False) -> dict:
    global _process
    with _lock:
        if _process is not None and _process.is_alive():
            raise RuntimeError("LSTM training is already running.")
        context = multiprocessing.get_context("spawn")
        _process = context.Process(target=_worker, args=(bool(force_rebuild),), daemon=True)
        _process.start()
        return write_status(**_default_status(), stage="starting", pid=_process.pid, error=None)


__all__ = ["forecast_latest", "read_status", "start_training", "write_status"]
