"""Training-job control for the world model — mirrors backend/lstm/jobs.py
(atomic status file + spawned worker process)."""
from __future__ import annotations

import json
import multiprocessing
import os
import threading

from .config import ARTIFACT_ROOT, STATUS_PATH
from .engine import forecast  # re-exported for the API layer
from .training import train_world_model

_lock = threading.Lock()
_process: "multiprocessing.Process | None" = None


def _default_status() -> dict:
    return {"stage": "idle", "evaluation_status": None, "error": None}


def read_status() -> dict:
    if not STATUS_PATH.is_file():
        return _default_status()
    try:
        return {**_default_status(), **json.loads(STATUS_PATH.read_text())}
    except (OSError, json.JSONDecodeError):
        return {**_default_status(), "stage": "error", "error": "status file unreadable"}


def write_status(**updates) -> dict:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    current = read_status()
    current.update(updates)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2))
    os.replace(tmp, STATUS_PATH)
    return current


def _worker(force_rebuild: bool, allow_ungated: bool = False) -> None:
    write_status(**_default_status(), stage="starting")
    try:
        train_world_model(force_rebuild=force_rebuild, status=write_status, allow_ungated=allow_ungated)
    except Exception as error:  # noqa: BLE001 - surfaced via status file
        write_status(stage="error", error=f"{type(error).__name__}: {error}")


def start_training(force_rebuild: bool = False, allow_ungated: bool = False) -> dict:
    global _process
    with _lock:
        if _process is not None and _process.is_alive():
            raise RuntimeError("World-model training is already running.")
        ctx = multiprocessing.get_context("spawn")
        _process = ctx.Process(target=_worker, args=(bool(force_rebuild), bool(allow_ungated)), daemon=True)
        _process.start()
        return write_status(**_default_status(), stage="starting", pid=_process.pid)


__all__ = ["forecast", "read_status", "start_training", "write_status"]
