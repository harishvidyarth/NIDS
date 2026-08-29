from __future__ import annotations

import numpy as np

from ..lstm.config import FORECAST_CLASSES
from ..lstm.evaluation import ABSENT_CLASS, evaluate_predictions
from .config import EARLY_WARNING_THRESHOLDS, HORIZONS


def evaluate_horizons(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    reports = []
    for horizon in range(HORIZONS):
        predictions = np.asarray(FORECAST_CLASSES)[np.argmax(probabilities[:, horizon, :], axis=1)]
        report = evaluate_predictions(y_true[:, horizon], predictions, probabilities[:, horizon, :])
        report["horizon"] = horizon + 1
        report["class_distribution"] = report["class_support"]
        report["low_support_classes"] = [label for label, count in report["class_support"].items() if count < 30]
        reports.append(report)
    return reports


def select_early_warning_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    actual = (y_true != "BENIGN").reshape(-1)
    attack_probability = (1.0 - probabilities[:, :, 0]).reshape(-1)
    candidates = []
    for threshold in EARLY_WARNING_THRESHOLDS:
        predicted = attack_probability >= threshold
        tp = int(np.sum(actual & predicted)); tn = int(np.sum(~actual & ~predicted))
        fp = int(np.sum(~actual & predicted)); fn = int(np.sum(actual & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        candidates.append({"threshold": threshold, "attack_f1": f1, "false_positive_rate": fpr})
    eligible = [item for item in candidates if item["false_positive_rate"] <= 0.05]
    selected = max(eligible, key=lambda item: (item["attack_f1"], -item["threshold"])) if eligible else None
    return {"selected_threshold": selected["threshold"] if selected else None, "candidates": candidates,
            "selection_rule": "maximum validation attack F1 subject to FPR <= 5%; ties prefer lower threshold"}


def onset_metrics(y_true: np.ndarray, probabilities: np.ndarray, input_labels: np.ndarray, threshold: float | None) -> dict:
    if threshold is None:
        return {"status": ABSENT_CLASS, "reason": "No validation threshold satisfied the FPR constraint."}
    benign_history = np.all(input_labels == "BENIGN", axis=1)
    actual_onset = np.full(len(y_true), -1, dtype=int)
    predicted_onset = np.full(len(y_true), -1, dtype=int)
    for row in range(len(y_true)):
        actual_hits = np.flatnonzero(y_true[row] != "BENIGN")
        predicted_hits = np.flatnonzero((1.0 - probabilities[row, :, 0]) >= threshold)
        if len(actual_hits): actual_onset[row] = int(actual_hits[0] + 1)
        if len(predicted_hits): predicted_onset[row] = int(predicted_hits[0] + 1)
    supported = benign_history & (actual_onset > 0)
    false_warning = benign_history & (actual_onset < 0) & (predicted_onset > 0)
    detected = supported & (predicted_onset > 0)
    errors = predicted_onset[detected] - actual_onset[detected]
    return {
        "status": "AVAILABLE" if supported.any() else ABSENT_CLASS,
        "supported_onsets": int(supported.sum()),
        "detected": int(detected.sum()),
        "misses": int((supported & (predicted_onset < 0)).sum()),
        "false_warnings": int(false_warning.sum()),
        "early_detections": int(np.sum(errors < 0)),
        "on_time_detections": int(np.sum(errors == 0)),
        "late_detections": int(np.sum(errors > 0)),
        "mean_horizon_error": float(errors.mean()) if len(errors) else ABSENT_CLASS,
        "mean_lead_windows": float((-errors[errors < 0]).mean()) if np.any(errors < 0) else ABSENT_CLASS,
    }


def degradation_table(reports: list[dict]) -> list[dict]:
    h1 = reports[0]
    return [{
        "horizon": item["horizon"],
        "accuracy": item["accuracy"],
        "macro_f1": item["macro_f1"],
        "attack_f1": item["attack_forecasting"]["f1"],
        "accuracy_delta_from_h1": item["accuracy"] - h1["accuracy"],
        "macro_f1_delta_from_h1": item["macro_f1"] - h1["macro_f1"],
    } for item in reports]
