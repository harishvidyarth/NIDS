from __future__ import annotations

import numpy as np

from backend.lstm.evaluation import evaluate_predictions


def test_rigorous_metrics_include_forecast_and_probability_quality():
    probabilities = np.asarray([
        [0.80, 0.05, 0.10, 0.05],
        [0.20, 0.05, 0.65, 0.10],
        [0.10, 0.05, 0.15, 0.70],
        [0.55, 0.05, 0.30, 0.10],
    ])
    metrics = evaluate_predictions(
        ["BENIGN", "DoS", "PortScan", "DoS"],
        ["BENIGN", "DoS", "PortScan", "BENIGN"],
        probabilities,
    )
    assert metrics["balanced_accuracy"] == 0.8333333333333334
    assert metrics["attack_forecasting"]["recall"] == 2 / 3
    assert metrics["attack_forecasting"]["false_negative_rate"] == 1 / 3
    assert metrics["probability_quality"]["multiclass_log_loss"] > 0
    assert metrics["probability_quality"]["multiclass_brier_score"] > 0
    assert metrics["roc_auc"]["DDoS"] == "N/A — class absent from evaluation set"


def test_transition_metrics_preserve_actual_state_transition():
    probabilities = np.asarray([
        [0.10, 0.10, 0.70, 0.10],
        [0.80, 0.05, 0.10, 0.05],
    ])
    metrics = evaluate_predictions(
        ["DoS", "BENIGN"],
        ["DoS", "BENIGN"],
        probabilities,
        current_states=["PortScan", "DoS"],
    )
    transitions = {item["actual_transition"]: item for item in metrics["transitions"]}
    assert transitions["PortScan -> DoS"]["count"] == 1
    assert transitions["PortScan -> DoS"]["mean_true_state_probability"] == 0.7
    assert transitions["PortScan -> DoS"]["low_support"] is True

