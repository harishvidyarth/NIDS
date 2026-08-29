from __future__ import annotations

import numpy as np
import pandas as pd

from backend.lstm.config import FORECAST_CLASSES
from backend.lstm_multistep.config import HORIZONS, SEQUENCE_LENGTH
from backend.lstm_multistep.dataset import build_multistep_sequences, split_session_windows
from backend.lstm_multistep.evaluation import evaluate_horizons, onset_metrics, select_early_warning_threshold
from backend.lstm_multistep.training import build_model
from backend.temporal.schema import STATE_FEATURE_NAMES


def _windows(session="Monday-WorkingHours.pcap_ISCX", count=18, start=0):
    rows = []
    labels = ["BENIGN"] * 8 + ["PortScan", "DDoS", "DoS"] * 4
    for index in range(count):
        rows.append({
            "session_id": session,
            "window_id": start + index,
            "dominant_state": labels[index],
            **{name: float(index + feature) for feature, name in enumerate(STATE_FEATURE_NAMES)},
        })
    return pd.DataFrame(rows)


def test_direct_sequence_shape_alignment_and_identity():
    sequences = build_multistep_sequences(_windows())
    assert sequences["X"].shape == (8, 5, 28)
    assert sequences["y"].shape == (8, 6)
    assert np.array_equal(sequences["history_window_ids"][0], np.arange(5))
    assert np.array_equal(sequences["target_window_ids"][0], np.arange(5, 11))
    assert sequences["sample_id"][0].endswith(":0:10")


def test_sequences_never_bridge_gap_or_session():
    first = _windows(count=11)
    second = _windows(session="Tuesday-WorkingHours.pcap_ISCX", count=11, start=50)
    sequences = build_multistep_sequences(pd.concat([first, second], ignore_index=True))
    assert len(sequences["X"]) == 2
    assert set(sequences["session_id"]) == {first.session_id.iloc[0], second.session_id.iloc[0]}


def test_direct_model_output_probabilities_and_batching(tmp_path):
    model = build_model()
    batch = np.random.default_rng(42).normal(size=(7, SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES))).astype(np.float32)
    probabilities = model.predict(batch, verbose=0)
    assert probabilities.shape == (7, HORIZONS, len(FORECAST_CLASSES))
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    path = tmp_path / "model.keras"
    model.save(path)
    import tensorflow as tf
    restored = tf.keras.models.load_model(path)
    assert np.allclose(probabilities, restored.predict(batch, verbose=0), atol=1e-6)


def test_threshold_selection_is_validation_only_and_tie_prefers_lower():
    y = np.full((2, HORIZONS), "BENIGN", dtype=object)
    y[0, 0] = "PortScan"
    probabilities = np.zeros((2, HORIZONS, 4), dtype=float)
    probabilities[:, :, 0] = 0.95
    probabilities[:, :, 1] = 0.05
    probabilities[0, 0] = [0.10, 0.90, 0.0, 0.0]
    selected = select_early_warning_threshold(y, probabilities)
    assert selected["selected_threshold"] == 0.5


def test_horizon_metrics_absent_classes_and_onset_support():
    y = np.full((3, HORIZONS), "BENIGN", dtype=object)
    y[0, 2:] = "PortScan"
    probabilities = np.zeros((3, HORIZONS, 4), dtype=float)
    probabilities[:, :, 0] = 0.9
    probabilities[:, :, 3] = 0.1
    probabilities[0, 1:, 0] = 0.2
    probabilities[0, 1:, 3] = 0.8
    reports = evaluate_horizons(y, probabilities)
    assert len(reports) == HORIZONS
    assert reports[0]["roc_auc"]["DDoS"].startswith("N/A")
    onset = onset_metrics(y, probabilities, np.full((3, SEQUENCE_LENGTH), "BENIGN"), 0.5)
    assert onset["supported_onsets"] == 1
    assert onset["early_detections"] == 1


def test_official_split_keeps_friday_ddos_test_only():
    names = (
        "Monday-WorkingHours.pcap_ISCX", "Tuesday-WorkingHours.pcap_ISCX",
        "Wednesday-workingHours.pcap_ISCX", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX", "Friday-WorkingHours-Morning.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",
    )
    split = split_session_windows([_windows(name, 11) for name in names])
    assert len(split["train"]) == 5 and len(split["validation"]) == 2 and len(split["test"]) == 1
    assert "DDos" in split["test"][0].session_id.iloc[0]
