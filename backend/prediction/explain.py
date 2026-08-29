"""
Per-prediction feature attribution for the ANN (models/ISAA_ANN.h5).

The SIH "World Models" brief requires interpretable decision support:
"which specific flags, ports, or flow patterns are contributing most …
via attention weights or SHAP values. Black-box outputs without
interpretability are not acceptable."

This module supplies that with **gradient x input** saliency — a single
`tf.GradientTape` pass of the predicted-class score with respect to the
scaled 77-feature input, multiplied element-wise by the input. It is:
  - fully offline, no network, no extra dependency (pure TensorFlow),
  - deterministic (no sampling, unlike SHAP KernelExplainer),
  - cheap (one backward pass for the whole batch),
  - defined in the model's own scaled feature space, then reported
    against the human `TRAINING_FEATURES` names.

SHAP is deliberately not used: DeepExplainer is fragile on TF 2.18 /
Keras 3 and KernelExplainer is slow and stochastic. Gradient x input is
the "feature attribution or equivalent" the brief allows.
"""
from __future__ import annotations

import numpy as np

from .features import TRAINING_FEATURES


def attribute(model, scaled_rows: np.ndarray, predicted_index: np.ndarray) -> np.ndarray:
    """
    Signed contribution of each of the 77 scaled features to the score of
    the predicted class, for every row.

    Args:
        model: the loaded Keras ANN.
        scaled_rows: (n, 77) float array — already MinMax-scaled, exactly
            what was fed to ``model.predict``.
        predicted_index: (n,) int array — argmax class per row.

    Returns:
        (n, 77) float array of ``gradient * input`` attributions. Positive
        means the feature's current value pushed the model toward the
        predicted class; negative means it pushed away.
    """
    import tensorflow as tf

    x = tf.constant(np.asarray(scaled_rows, dtype=np.float32))
    idx = tf.constant(np.asarray(predicted_index, dtype=np.int32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = model(x, training=False)
        # score of the predicted class for each row
        row_ids = tf.range(tf.shape(probs)[0])
        gather_idx = tf.stack([row_ids, idx], axis=1)
        class_score = tf.gather_nd(probs, gather_idx)
    grads = tape.gradient(class_score, x)
    if grads is None:  # pragma: no cover - only if the graph is detached
        return np.zeros_like(np.asarray(scaled_rows, dtype=np.float64))
    return (grads.numpy() * x.numpy()).astype(np.float64)


def top_features_for_row(attribution_row: np.ndarray, k: int = 8) -> list[dict]:
    """Top-k features for one row by absolute contribution, most first."""
    order = np.argsort(np.abs(attribution_row))[::-1][:k]
    return [
        {
            "feature": TRAINING_FEATURES[i].strip(),
            "contribution": round(float(attribution_row[i]), 6),
        }
        for i in order
    ]


def driving_features(attribution: np.ndarray, k: int = 10) -> list[dict]:
    """
    Aggregate attribution across all scored rows: mean |contribution| per
    feature, ranked. This is the "top contributing traffic features" for
    the capture as a whole.
    """
    if attribution.size == 0:
        return []
    mean_abs = np.abs(attribution).mean(axis=0)
    mean_signed = attribution.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        {
            "feature": TRAINING_FEATURES[i].strip(),
            "mean_abs_contribution": round(float(mean_abs[i]), 6),
            "mean_signed_contribution": round(float(mean_signed[i]), 6),
        }
        for i in order
    ]
