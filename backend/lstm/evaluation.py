from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import FORECAST_CLASSES

ABSENT_CLASS = "N/A — class absent from evaluation set"


def _safe_ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else ABSENT_CLASS


def _attack_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_attack = y_true != "BENIGN"
    predicted_attack = y_pred != "BENIGN"
    true_positive = int(np.sum(true_attack & predicted_attack))
    true_negative = int(np.sum(~true_attack & ~predicted_attack))
    false_positive = int(np.sum(~true_attack & predicted_attack))
    false_negative = int(np.sum(true_attack & ~predicted_attack))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    if not isinstance(precision, float) or not isinstance(recall, float):
        f1 = ABSENT_CLASS
    elif precision + recall:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_ratio(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_ratio(false_negative, false_negative + true_positive),
        "confusion": {"tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative},
    }


def _probability_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    confidences = probabilities.max(axis=1)
    correct = y_true == y_pred
    mapping = {label: index for index, label in enumerate(labels)}
    encoded = np.asarray([mapping[label] for label in y_true], dtype=int)
    one_hot = np.eye(len(labels), dtype=float)[encoded]
    bins = []
    expected_calibration_error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (confidences <= upper if index == 9 else confidences < upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            observed_accuracy = float(correct[mask].mean())
            expected_calibration_error += count / len(y_true) * abs(mean_confidence - observed_accuracy)
        else:
            mean_confidence = observed_accuracy = None
        bins.append({
            "lower": float(lower), "upper": float(upper), "count": count,
            "mean_confidence": mean_confidence, "observed_accuracy": observed_accuracy,
        })
    return {
        "average_confidence": float(confidences.mean()),
        "confidence_correct": float(confidences[correct].mean()) if correct.any() else ABSENT_CLASS,
        "confidence_incorrect": float(confidences[~correct].mean()) if (~correct).any() else ABSENT_CLASS,
        "multiclass_log_loss": float(log_loss(encoded, probabilities, labels=list(range(len(labels))))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bin": float(expected_calibration_error),
        "calibration_bins": bins,
    }


def _transition_metrics(y_true, y_pred, probabilities, current_states, labels) -> list[dict]:
    if current_states is None:
        return []
    current_states = np.asarray(current_states, dtype=str)
    mapping = {label: index for index, label in enumerate(labels)}
    result = []
    for current in labels:
        for target in labels:
            mask = (current_states == current) & (y_true == target)
            count = int(mask.sum())
            if not count:
                continue
            correct = int(np.sum(y_pred[mask] == target))
            result.append({
                "actual_transition": f"{current} -> {target}",
                "count": count,
                "correct_next_state_predictions": correct,
                "accuracy": float(correct / count),
                "mean_true_state_probability": float(probabilities[mask, mapping[target]].mean()) if probabilities is not None else None,
                "low_support": count < 30,
            })
    return sorted(result, key=lambda item: (-item["count"], item["actual_transition"]))


def evaluate_predictions(y_true, y_pred, probabilities=None, current_states=None) -> dict:
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    labels = list(FORECAST_CLASSES)
    support = {label: int(np.sum(y_true == label)) for label in labels}
    precision, recall, f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present_recall = recall[class_support > 0]
    report = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": float(np.mean(present_recall)) if len(present_recall) else None,
        "macro_precision": float(np.mean(precision)) if len(y_true) else None,
        "macro_recall": float(np.mean(recall)) if len(y_true) else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else None,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)) if len(y_true) else None,
        "class_support": support,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(class_support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "roc_auc": {label: ABSENT_CLASS for label in labels},
        "pr_auc": {label: ABSENT_CLASS for label in labels},
    }
    if probabilities is not None and len(y_true):
        probabilities = np.asarray(probabilities, dtype=float)
        binary = label_binarize(y_true, classes=labels)
        for index, label in enumerate(labels):
            positives = support[label]
            negatives = len(y_true) - positives
            if positives == 0 or negatives == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report["roc_auc"][label] = float(roc_auc_score(binary[:, index], probabilities[:, index]))
                report["pr_auc"][label] = float(average_precision_score(binary[:, index], probabilities[:, index]))
        report["multiclass_roc_auc"] = (
            float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
            if all(support[label] > 0 for label in labels) else ABSENT_CLASS
        )
        report["multiclass_pr_auc"] = (
            float(average_precision_score(binary, probabilities, average="macro"))
            if all(support[label] > 0 for label in labels) else ABSENT_CLASS
        )
        report["probability_quality"] = _probability_metrics(y_true, y_pred, probabilities, labels)
    else:
        report["multiclass_roc_auc"] = report["multiclass_pr_auc"] = None
        report["probability_quality"] = None
    report["attack_forecasting"] = _attack_metrics(y_true, y_pred)
    report["transitions"] = _transition_metrics(y_true, y_pred, probabilities, current_states, labels)
    attack_classes_supported = sum(support[label] > 0 for label in labels[1:])
    report["evaluation_status"] = (
        "sufficient_attack_support" if attack_classes_supported == len(labels) - 1
        else "limited_attack_support" if attack_classes_supported
        else "benign_only_holdout"
    )
    report["limitations"] = []
    if report["evaluation_status"] != "sufficient_attack_support":
        report["limitations"].append(
            "Evaluation does not support every attack class; unsupported AUC values are marked N/A and results are not complete attack-performance evidence."
        )
    return report
