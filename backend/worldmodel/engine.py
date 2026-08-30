"""K-step infiltration forecast — the runtime side of the world model.

Loads the latest trained artifact, reads the last SEQUENCE_LENGTH prepared
temporal windows (the user's own capture), autoregressively rolls the
world model forward K steps, and returns, per step: infiltration
probability, predicted state, seconds-ahead, risk level, kill-chain stage
— plus the top contributing state features (gradient x input on the LSTM
input, same technique as backend/prediction/explain.py) and the top
contributing input windows (attention).
"""
from __future__ import annotations

import json

import joblib
import numpy as np

from ..lstm.config import repository_path
from ..lstm.training import _load_recent_windows
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import (
    DEFAULT_K,
    FORECAST_CLASSES,
    LATEST_PATH,
    MAX_K,
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
)
from .killchain import PROGRESS_STAGES, presentation_state, risk_level, state_to_stage
from .model import rollout

_BENIGN_IDX = FORECAST_CLASSES.index("BENIGN")


class WorldModelUnavailable(RuntimeError):
    """Raised (-> HTTP 409) when there is no trained artifact or not enough
    prepared windows to forecast from."""


def _rung(stage: str) -> int:
    return PROGRESS_STAGES.index(stage) if stage in PROGRESS_STAGES else -1


def _load_artifact():
    if not LATEST_PATH.is_file():
        raise WorldModelUnavailable(
            "No trained world-model. POST /api/worldmodel/train after preparing "
            "a temporal dataset (needs the CICIDS2017 CSVs / NIDS_CICIDS2017_DIR)."
        )
    latest = json.loads(LATEST_PATH.read_text())
    artifact_dir = repository_path(latest["artifact_dir"])
    import tensorflow as tf

    try:
        # safe_mode=False: build_world_model() uses a Lambda layer (a Python
        # lambda over tf.reduce_sum), which Keras 3 refuses to deserialize
        # under the default safe_mode. This is our own just-trained artifact.
        model = tf.keras.models.load_model(
            artifact_dir / "model.keras", compile=False, safe_mode=False
        )
        scaler = joblib.load(artifact_dir / "scaler.bin")
    except Exception as error:  # a corrupt/incompatible artifact -> clean 409, not a 500
        raise WorldModelUnavailable(
            f"Trained world-model artifact could not be loaded ({error}). "
            "Re-run POST /api/worldmodel/train."
        ) from error
    report = {}
    report_path = artifact_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    return model, scaler, latest, report


def _feature_attribution(model, seq_scaled: np.ndarray) -> list[dict]:
    """gradient x input of the infiltration score (1 - P(BENIGN)) w.r.t.
    the scaled [1, seq_len, 28] input, averaged |contribution| per feature
    across the input windows -> top-8 STATE_FEATURE_NAMES."""
    import tensorflow as tf

    x = tf.constant(np.asarray(seq_scaled, dtype=np.float32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        class_probs, _ = model(x, training=False)
        infil = 1.0 - class_probs[:, _BENIGN_IDX]
    grads = tape.gradient(infil, x)
    if grads is None:  # pragma: no cover
        return []
    contrib = (grads.numpy() * x.numpy())[0]            # [seq_len, 28]
    mean_abs = np.abs(contrib).mean(axis=0)             # [28]
    mean_signed = contrib.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:8]
    return [
        {
            "feature": STATE_FEATURE_NAMES[i],
            "mean_abs_contribution": float(mean_abs[i]),
            "mean_signed_contribution": float(mean_signed[i]),
        }
        for i in order
    ]


def forecast(windows_source=None, k: int | None = None) -> dict:
    k = DEFAULT_K if not k else max(1, min(int(k), MAX_K))
    model, scaler, latest, report = _load_artifact()

    try:
        recent = _load_recent_windows(windows_source)
    except RuntimeError as exc:
        raise WorldModelUnavailable(str(exc)) from exc

    window_ids = recent["window_id"].to_numpy(dtype=np.int64)
    current_state = str(recent["dominant_state"].iloc[-1])
    X = recent[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X)[None, :, :]          # [1, seq_len, 28]

    stage_probs, _next_states, attn_hist = rollout(model, X_scaled, k)

    steps = []
    most_advanced = state_to_stage(current_state)
    for i in range(k):
        probs = stage_probs[i]
        idx = int(np.argmax(probs))
        state = FORECAST_CLASSES[idx]
        infil = float(1.0 - probs[_BENIGN_IDX])
        stage = state_to_stage(state)
        if _rung(stage) > _rung(most_advanced):
            most_advanced = stage
        steps.append({
            "step": i + 1,
            "offset_seconds": (i + 1) * WINDOW_SECONDS,
            "infiltration_probability": round(infil, 4),
            "predicted_state": state,
            "probabilities": {c: float(probs[j]) for j, c in enumerate(FORECAST_CLASSES)},
            "risk_level": risk_level(infil),
            "mitre_stage": stage,
        })

    max_infil = max(s["infiltration_probability"] for s in steps)
    earliest_alarm = next(
        (s["step"] for s in steps if s["infiltration_probability"] >= 0.60), None
    )

    # top contributing input windows (attention on the first rollout step)
    attn0 = attn_hist[0]
    win_order = np.argsort(attn0)[::-1]
    top_windows = [
        {
            "window_id": int(window_ids[j]),
            "position": int(j) - SEQUENCE_LENGTH,          # -5..-1 (most recent = -1)
            "attention": float(attn0[j]),
        }
        for j in win_order[:3]
    ]

    return {
        "model_version": latest.get("model_version"),
        "k": k,
        "window_seconds": WINDOW_SECONDS,
        "current_state": current_state,
        "current_window": int(window_ids[-1]),
        "current_mitre_stage": state_to_stage(current_state),
        "infiltration_probability": max_infil,
        "maximum_infiltration_probability": max_infil,
        "earliest_alarm_step": earliest_alarm,
        "early_warning_threshold": 0.60,
        "predicted_mitre_stage": presentation_state(
            most_advanced, risk_level(max_infil)
        ),
        "k_steps": steps,
        "top_features": _feature_attribution(model, X_scaled),
        "top_windows": top_windows,
        "evaluation_status": report.get("evaluation_status", "UNVERIFIED"),
        "limitations": [
            "row_order_proxy — windows are 10s slices in capture order, not "
            "wall-clock-validated seconds ahead.",
            "NIDS classes cover BENIGN/DDoS/DoS/PortScan only; kill-chain "
            "stages beyond Reconnaissance/Impact are shown but never predicted.",
        ],
    }
