"""
Tests for backend/temporal/validate.py — the post-hoc temporal dataset
validator (task: "Temporal Dataset Validation"). Reuses the synthetic
flow-CSV builder from test_temporal.py (real cicflowmeter schema, not
invented column names) rather than duplicating it.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_temporal import make_flow_csv  # noqa: E402
from backend.temporal.temporal_dataset import prepare_temporal_dataset  # noqa: E402
from backend.temporal.validate import validate_temporal_dataset  # noqa: E402


def _prepare(tmp_path, df, name="flows.csv", window_size=10, sequence_length=5):
    csv_path = tmp_path / name
    df.to_csv(csv_path, index=False)
    temporal_dir = tmp_path / "temporal_out"
    prepare_temporal_dataset(csv_path, temporal_dir, window_size, sequence_length)
    return csv_path, temporal_dir


# ---------- 1. Valid dataset -> PASS ----------

def test_valid_dataset_passes(tmp_path):
    df = make_flow_csv(120, step_seconds=1)  # continuous, single class, spans ~12 windows
    csv_path, temporal_dir = _prepare(tmp_path, df)
    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["overall_status"] == "PASS", report["checks"]
    assert report["rows_checked"] == 120
    for key in ("current_state", "timestamps", "windows", "features", "transitions",
                "sequences", "chronological_split", "leakage", "missing_data", "duplicates"):
        assert key in report["checks"]


# ---------- 2. Invalid state -> FAIL ----------

def test_invalid_current_state_label_fails(tmp_path):
    # One minority UNKNOWN label that never dominates a window, so
    # prepare_temporal_dataset()'s own build-time gate still succeeds —
    # isolating this test to the validator's current_state check.
    states = ["BENIGN"] * 59 + ["UNKNOWN"]
    df = make_flow_csv(60, step_seconds=2, states=states)
    csv_path, temporal_dir = _prepare(tmp_path, df)
    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["checks"]["current_state"] == "FAIL"
    assert "UNKNOWN" in report["details"]["current_state"]["details"]["unknown_labels"]
    assert report["overall_status"] == "FAIL"


# ---------- 3. Timestamp disorder -> FAIL ----------

def test_timestamp_disorder_fails(tmp_path):
    df = make_flow_csv(60, step_seconds=2)
    csv_path, temporal_dir = _prepare(tmp_path, df)

    disordered = df.copy()
    disordered.loc[[10, 11], "timestamp"] = disordered.loc[[11, 10], "timestamp"].to_numpy()
    disordered_csv = tmp_path / "disordered.csv"
    disordered.to_csv(disordered_csv, index=False)

    report = validate_temporal_dataset(disordered_csv, temporal_dir)
    assert report["checks"]["timestamps"] == "FAIL"
    assert report["details"]["timestamps"]["details"]["order_status"] == "FAIL"
    assert report["details"]["timestamps"]["details"]["out_of_order_rows"] > 0
    assert report["overall_status"] == "FAIL"


# ---------- 4. Invalid (overlapping) window -> FAIL ----------

def test_overlapping_window_fails(tmp_path):
    df = make_flow_csv(60, step_seconds=2)
    csv_path, temporal_dir = _prepare(tmp_path, df)

    states_path = temporal_dir / "temporal_states.csv"
    states_df = pd.read_csv(states_path)
    assert len(states_df) >= 2
    # Force window 1 to start at the same time as window 0 -> overlap.
    states_df.loc[1, "window_start"] = states_df.loc[0, "window_start"]
    states_df.to_csv(states_path, index=False)

    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["checks"]["windows"] == "FAIL"
    assert report["details"]["windows"]["details"]["overlapping_windows"] > 0


# ---------- 5. Feature NaN -> WARNING (or FAIL per policy) ----------

def test_feature_nan_triggers_warning(tmp_path):
    df = make_flow_csv(60, step_seconds=2)
    csv_path, temporal_dir = _prepare(tmp_path, df)

    states_path = temporal_dir / "temporal_states.csv"
    states_df = pd.read_csv(states_path)
    states_df.loc[0, "mean_iat"] = None
    states_df.to_csv(states_path, index=False)

    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["checks"]["features"] in ("WARNING", "FAIL")
    assert report["details"]["features"]["details"]["nan_values"] >= 1


# ---------- 6. Chronological split violation -> FAIL ----------

def test_chronological_split_violation_fails(tmp_path):
    df = make_flow_csv(400, step_seconds=2)  # enough windows for train+val+test to each have sequences
    csv_path, temporal_dir = _prepare(tmp_path, df)

    train_path = temporal_dir / "temporal_train.npz"
    train = dict(np.load(train_path, allow_pickle=True))
    assert len(train["target_window_id"]) > 0
    # Push every train target window far into the "future" so it violates
    # max(train) < min(validation).
    train["target_window_id"] = train["target_window_id"] + 100000
    np.savez(train_path, **train)

    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["checks"]["chronological_split"] == "FAIL"
    assert report["details"]["chronological_split"]["details"]["order_valid"] is False


# ---------- 7. Cross-split sequence leakage -> FAIL ----------

def test_cross_split_sequence_leakage_fails(tmp_path):
    df = make_flow_csv(400, step_seconds=2)
    csv_path, temporal_dir = _prepare(tmp_path, df)

    train_path = temporal_dir / "temporal_train.npz"
    test_path = temporal_dir / "temporal_test.npz"
    train = dict(np.load(train_path, allow_pickle=True))
    test = dict(np.load(test_path, allow_pickle=True))
    assert len(train["X"]) > 0

    # Inject an exact copy of train's first sequence into test -> exact
    # cross-split duplicate (the clearest possible leakage case).
    for key in ("X", "X_scaled", "input_window_ids"):
        test[key] = np.concatenate([test[key], train[key][:1]], axis=0)
    for key in ("y_state_vector", "y_state_vector_scaled"):
        test[key] = np.concatenate([test[key], train[key][:1]], axis=0)
    test["y_dominant_state"] = np.concatenate([test["y_dominant_state"], train["y_dominant_state"][:1]], axis=0)
    test["y_attack_present"] = np.concatenate([test["y_attack_present"], train["y_attack_present"][:1]], axis=0)
    test["target_window_id"] = np.concatenate([test["target_window_id"], train["target_window_id"][:1]], axis=0)
    np.savez(test_path, **test)

    report = validate_temporal_dataset(csv_path, temporal_dir)
    assert report["checks"]["leakage"] == "FAIL"
    assert report["details"]["leakage"]["details"]["exact_duplicate_sequences"]["train_test"] >= 1


# ---------- 8. Large dataset (synthetic, performance only — never used as real app output) ----------

def test_large_dataset_50k_rows_no_crash():
    import tempfile
    df = make_flow_csv(50_000, step_seconds=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_path = tmp_path / "large.csv"
        df.to_csv(csv_path, index=False)
        temporal_dir = tmp_path / "out"

        t0 = time.time()
        prepare_temporal_dataset(csv_path, temporal_dir, window_size_seconds=10, sequence_length=5)
        prep_seconds = time.time() - t0

        t1 = time.time()
        report = validate_temporal_dataset(csv_path, temporal_dir)
        validate_seconds = time.time() - t1

        assert report["rows_checked"] == 50_000
        assert report["details"]["windows"]["details"]["total_windows"] > 100
        # Generous bounds — this is a correctness/no-crash/no-memory-blowup
        # check, not a strict perf benchmark.
        assert prep_seconds < 120
        assert validate_seconds < 120
