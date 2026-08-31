"""Comparable, deliberately simple baselines for the world model."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..lstm.evaluation import evaluate_predictions
from .config import FORECAST_CLASSES


def train_logistic_baseline(X_train_seq, y_class_train, seed=42):
    """Fit a reproducible linear baseline over flattened scaled sequences."""
    X = np.asarray(X_train_seq, dtype=np.float32)
    return LogisticRegression(max_iter=300, random_state=seed).fit(X.reshape(len(X), -1), y_class_train)


def persistence_baseline(X_seq):
    """Return a predictor using split metadata's last observed class.

    ``X_seq`` may be a label vector or a mapping containing ``input_labels``;
    feature vectors are intentionally not decoded to labels.
    """
    if isinstance(X_seq, dict):
        return np.asarray(X_seq["input_labels"])[:, -1]
    return np.asarray(X_seq)


def evaluate_baseline(clf_or_fn, X_test_seq, y_class_test):
    y = np.asarray(y_class_test)
    labels = np.asarray(FORECAST_CLASSES)
    if hasattr(clf_or_fn, "predict"):
        X = np.asarray(X_test_seq, dtype=np.float32)
        pred_ids = clf_or_fn.predict(X.reshape(len(X), -1))
        probs = clf_or_fn.predict_proba(X.reshape(len(X), -1))
        pred = labels[pred_ids]
    else:
        pred = np.asarray(clf_or_fn(X_test_seq) if callable(clf_or_fn) else clf_or_fn)
        probs = None
    truth = labels[y] if np.issubdtype(y.dtype, np.integer) else y.astype(str)
    report = evaluate_predictions(truth, pred, probs)
    attack = report["attack_forecasting"]
    return {"macro_f1": report["macro_f1"], "precision": report["macro_precision"],
            "recall": report["macro_recall"], "false_positive_rate": attack["false_positive_rate"],
            "per_class_f1": {k: v["f1"] for k, v in report["per_class"].items()}, "detail": report}
