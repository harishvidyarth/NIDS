"""forecast-from-prepared-data, multi-interface capture, /api/benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.capture import capture as cap
from backend.lstm.training import _load_recent_windows
from backend.temporal.schema import STATE_FEATURE_NAMES


# ---------- A. forecast reads the user's prepared temporal dataset ----------

def _write_states(dir_path, n_windows):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_windows):
        row = {name: float(rng.random()) for name in STATE_FEATURE_NAMES}
        row["window_id"] = i
        row["dominant_state"] = "BENIGN"
        rows.append(row)
    (dir_path / "temporal_states.csv").write_text(pd.DataFrame(rows).to_csv(index=False))


def test_load_recent_windows_from_prepared_dir(tmp_path):
    _write_states(tmp_path, 8)
    recent = _load_recent_windows(tmp_path)
    assert len(recent) == 5
    assert list(recent["window_id"]) == [3, 4, 5, 6, 7]
    assert set(STATE_FEATURE_NAMES).issubset(recent.columns)


def test_load_recent_windows_too_few(tmp_path):
    _write_states(tmp_path, 3)
    with pytest.raises(RuntimeError, match="contiguous 10-second windows"):
        _load_recent_windows(tmp_path)


def test_load_recent_windows_missing_cache_is_runtimeerror(monkeypatch):
    import backend.lstm.config as lcfg

    monkeypatch.setattr(lcfg, "CACHE_ROOT", lcfg.CACHE_ROOT / "definitely-absent")
    with pytest.raises(RuntimeError):
        _load_recent_windows(None)


# ---------- B. multi-interface capture ----------

class _FakePopen:
    def __init__(self, cmd, *a, **kw):
        self.cmd = cmd
        self.returncode = None
        self.stderr = None
        self.stdout = None

    def poll(self):
        return None


def test_start_capture_multi_interface_builds_repeated_i(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(cap, "IS_WINDOWS", False)
    monkeypatch.setattr(cap.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(cap.time, "sleep", lambda *_: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dumpcap")

    session = cap.start_capture(["lo0", "en0"], tmp_path, duration_seconds=None,
                                packet_target=None, buffer_mb=None)
    assert session.process.cmd.count("-i") == 2
    assert "lo0" in session.process.cmd and "en0" in session.process.cmd
    assert session.interfaces == ["lo0", "en0"]
    assert "2 interfaces" in session.interface


def test_start_capture_all_expands(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(cap, "IS_WINDOWS", False)
    monkeypatch.setattr(cap.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(cap.time, "sleep", lambda *_: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dumpcap")
    monkeypatch.setattr(cap, "list_interfaces", lambda: [
        {"device": "en0", "description": "Wi-Fi - en0 (Up, Running, Connected)"},
        {"device": "en1", "description": "en1 (Up, Running, Disconnected)"},
        {"device": "lo0", "description": "Loopback - lo0 (Up, Running, Loopback)"},
    ])
    session = cap.start_capture("all", tmp_path, duration_seconds=None,
                                packet_target=None, buffer_mb=None)
    assert session.interfaces == ["en0", "lo0"]  # en1 skipped (Disconnected)


def test_multi_interface_needs_dumpcap(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(cap, "IS_WINDOWS", False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(cap.CaptureError, match="dumpcap"):
        cap.start_capture(["lo0", "en0"], tmp_path)


# ---------- C. /api/benchmark ----------

def test_benchmark_endpoint():
    from backend.api.main import benchmark

    body = benchmark()
    v = body["one_step"]["validation"]
    assert isinstance(v["lstm"]["macro_f1"], float)
    assert isinstance(v["logistic_regression"]["macro_f1"], float)
    assert v["lstm"]["attack_false_positive_rate"] is not None
    assert body["source"].endswith("lstm_evaluation_report.json")
