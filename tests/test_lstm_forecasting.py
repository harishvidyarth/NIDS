from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.lstm.config import FORECAST_CLASSES, REPO_ROOT, SEQUENCE_LENGTH, repository_path, repository_relative
from backend.lstm.dataset import (
    _aggregate_proxy_chunk,
    build_sequences,
    cache_identity,
    cache_key,
    concat_sequence_sets,
    score_ann_chunk,
)
from backend.lstm.evaluation import evaluate_predictions
from backend.lstm.training import _fit_scaler, build_model, rolling_origin_windows
from backend.prediction.features import TRAINING_FEATURES
from backend.temporal.schema import STATE_FEATURE_NAMES
from backend.temporal.state_builder import build_temporal_states


def make_windows(count=30, session="session", start=0):
    rows = []
    for index in range(count):
        row = {name: float(index + feature_index) for feature_index, name in enumerate(STATE_FEATURE_NAMES)}
        row.update({
            "session_id": session,
            "window_id": start + index,
            "dominant_state": FORECAST_CLASSES[index % len(FORECAST_CLASSES)],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def make_flow_frame(count=20):
    values = {name: np.arange(1, count + 1, dtype=float) for name in TRAINING_FEATURES}
    return pd.DataFrame(values)


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values)


class FakeAnn:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, values, batch_size=None, verbose=None):
        self.batch_sizes.append((len(values), batch_size))
        probabilities = np.zeros((len(values), 4), dtype=float)
        probabilities[:, 0] = 0.7
        probabilities[:, 1] = 0.3
        return probabilities


def test_chunked_ann_scoring_preserves_rows_and_batches():
    frame = make_flow_frame(7)
    frame.loc[2, TRAINING_FEATURES[0]] = np.inf
    model = FakeAnn()
    labels, confidence, valid = score_ann_chunk(frame, model=model, scaler=IdentityScaler())
    assert labels.tolist() == ["BENIGN", "BENIGN", "INVALID_FEATURES", "BENIGN", "BENIGN", "BENIGN", "BENIGN"]
    assert valid.sum() == 6
    assert np.isnan(confidence[2])
    assert model.batch_sizes == [(6, 4096)]


def test_invalid_states_are_excluded_and_empty_windows_dropped():
    frame = make_flow_frame(20)
    labels = np.asarray(["INVALID_FEATURES"] * 10 + ["DDoS"] * 10, dtype=object)
    windows = _aggregate_proxy_chunk(frame, labels, first_row=0)
    assert windows["window_id"].tolist() == [1]
    assert windows.iloc[0]["dominant_state"] == "DDoS"
    assert windows.iloc[0]["scoreable_flow_count"] == 10


def test_proxy_aggregation_matches_chunked_aggregation():
    frame = make_flow_frame(20)
    labels = np.asarray(["BENIGN"] * 10 + ["PortScan"] * 10, dtype=object)
    whole = _aggregate_proxy_chunk(frame, labels, 0)
    chunked = pd.concat([
        _aggregate_proxy_chunk(frame.iloc[:10].copy(), labels[:10], 0),
        _aggregate_proxy_chunk(frame.iloc[10:].copy(), labels[10:], 10),
    ], ignore_index=True)
    pd.testing.assert_frame_equal(whole, chunked)


def test_cache_key_invalidates_on_source_or_artifact_change():
    source = {"name": "a.csv", "sha256": "one", "size_bytes": 1}
    artifacts = {"ann_model_sha256": "m", "ann_scaler_sha256": "s"}
    original = cache_key(cache_identity(source, artifacts))
    assert cache_key(cache_identity({**source, "sha256": "two"}, artifacts)) != original
    assert cache_key(cache_identity(source, {**artifacts, "ann_model_sha256": "other"})) == original
    assert cache_identity(source, artifacts)["target_provenance"] == "cicids2017_ground_truth_label"


def test_repository_artifact_pointers_are_portable_and_legacy_compatible():
    relative = Path("artifacts/lstm_forecaster/example/model.keras")
    assert repository_path(relative) == REPO_ROOT / relative
    assert repository_path(REPO_ROOT / relative) == REPO_ROOT / relative
    assert repository_relative(REPO_ROOT / relative) == relative.as_posix()


def test_sequences_do_not_cross_sessions_or_missing_windows():
    first = make_windows(8, "one", 0)
    second = make_windows(8, "two", 0)
    separated = concat_sequence_sets([build_sequences(first), build_sequences(second)])
    assert len(separated["X"]) == 6
    gapped = pd.concat([make_windows(6, "gap", 0), make_windows(6, "gap", 10)], ignore_index=True)
    assert len(build_sequences(gapped)["X"]) == 2


def test_sequence_alignment_and_next_target_are_exact():
    windows = make_windows(9)
    sequences = build_sequences(windows)
    assert sequences["X"].shape == (4, 5, 28)
    assert sequences["input_window_ids"][0].tolist() == [0, 1, 2, 3, 4]
    assert sequences["target_window_id"][0] == 5
    assert sequences["y"][0] == windows.loc[5, "dominant_state"]


def test_scaler_is_fit_only_on_training_windows():
    train = make_windows(10)
    holdout = make_windows(5)
    holdout[STATE_FEATURE_NAMES] = 10_000
    scaler = _fit_scaler([train])
    assert scaler.data_max_.max() < 10_000


def test_three_rolling_origin_folds_expand_without_overlap():
    folds = rolling_origin_windows(make_windows(100))
    assert len(folds) == 3
    assert [len(train) for train, _ in folds] == sorted(len(train) for train, _ in folds)
    for train, validation in folds:
        assert train["window_id"].max() < validation["window_id"].min()


def test_label_mapping_and_limited_metrics_are_deterministic():
    assert FORECAST_CLASSES == ("BENIGN", "DDoS", "DoS", "PortScan")
    metrics = evaluate_predictions(["BENIGN", "BENIGN"], ["BENIGN", "BENIGN"], [[1, 0, 0, 0], [1, 0, 0, 0]])
    assert metrics["evaluation_status"] == "benign_only_holdout"
    assert metrics["roc_auc"]["DDoS"] == "N/A — class absent from evaluation set"


def test_existing_state_builder_ignores_invalid_dominance_and_uses_zero_unavailable_counts():
    frame = make_flow_frame(3)
    frame["Current_State"] = ["INVALID_FEATURES", "INVALID_FEATURES", "DDoS"]
    frame["window_id"] = 0
    frame["window_start"] = pd.Timestamp("2026-01-01")
    frame["window_end"] = pd.Timestamp("2026-01-01 00:00:10")
    states, warnings = build_temporal_states(frame, "Current_State", 10)
    assert states.loc[0, "dominant_state"] == "DDoS"
    assert states.loc[0, "unique_src_ip_count"] == 0
    assert "unavailable" in warnings["src_ip"]


def test_model_save_load_and_batch_inference(tmp_path):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    model = build_model()
    sample = np.zeros((3, 5, 28), dtype=np.float32)
    before = model.predict(sample, batch_size=2, verbose=0)
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    after = loaded.predict(sample, batch_size=2, verbose=0)
    assert set(before) == {"dominant_state", "attack_alert"}
    assert before["dominant_state"].shape == (3, 4)
    assert before["attack_alert"].shape == (3, 4)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], rtol=1e-6, atol=1e-6)


def test_job_status_round_trip_and_api_contract(tmp_path, monkeypatch):
    from backend.lstm import jobs

    monkeypatch.setattr(jobs, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "STATUS_PATH", tmp_path / "status.json")
    jobs.write_status(stage="training", epoch=2, rows_processed=50)
    assert jobs.read_status()["epoch"] == 2
    source = Path("backend/api/main.py").read_text()
    assert '@app.post("/api/lstm/train")' in source
    assert '@app.get("/api/lstm/status")' in source
    assert '@app.post("/api/lstm/forecast")' in source
    assert '@app.get("/api/lstm/report")' in source
    assert 'extra="forbid"' in source


def test_ui_renders_only_bounded_aggregates():
    html = Path("frontend/index.html").read_text()
    script = Path("frontend/app.js").read_text()
    assert "ROW-ORDER PROXY" in html
    assert "lstm-probabilities" in html
    assert "lstmReport.counts" in script
    assert "flow-level training rows" not in script
