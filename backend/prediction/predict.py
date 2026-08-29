"""
Traffic-state prediction using the repository's existing trained ANN
(models/ISAA_ANN.h5) and its MinMaxScaler (models/minmax.bin).

This fixes the one real bug found in notebooks/NIDS (Prediction).ipynb:
that notebook casts the raw CICFlowMeter columns straight to float32 and
calls model.predict() WITHOUT ever loading/applying minmax.bin, even
though the model was trained exclusively on MinMax-scaled input. Its own
committed output shows the same 4 softmax values repeated for every row,
i.e. it never worked. Here the scaler is loaded and applied, and columns
are aligned by name (see features.py) instead of trusted by position.

No new model is introduced — same architecture, same weights, same
4-class output (BENIGN / DDoS / DoS / PortScan), matching
pd.get_dummies alphabetical order used at training time.

In addition to the JSON summary, predict_csv() now persists the same
per-flow prediction back into the feature CSV itself as a trailing
`Current_State` column (see README/task: "Add Current_State Label to
Extracted Feature CSV"). This column is the same argmax-decoded class
computed for the JSON output — never recomputed separately — and is
never fed back into the scaler/model as a feature.
"""
from __future__ import annotations

import logging
import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import TRAINING_FEATURES, match_columns, is_id_or_label_column

CLASS_NAMES = ["BENIGN", "DDoS", "DoS", "PortScan"]
CURRENT_STATE_COLUMN = "Current_State"
# Sentinel for a row the ANN genuinely could not score (inf/NaN in one of
# its 77 features after cleaning — a real, well-documented artifact of
# CICFlowMeter/CICIDS2017 output for zero-duration flows, e.g. "Flow
# Bytes/s" = Infinity). Never one of CLASS_NAMES, so nothing downstream
# can mistake it for a real traffic classification, and it is never
# fabricated — it is the model's own "could not score this" state.
INVALID_FEATURES_LABEL = "INVALID_FEATURES"

logger = logging.getLogger("nids.prediction")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "ISAA_ANN.h5"
SCALER_PATH = MODELS_DIR / "minmax.bin"


class PredictionError(Exception):
    pass


_model = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    if _model is None:
        if not MODEL_PATH.exists():
            raise PredictionError(f"Model not found at {MODEL_PATH}")
        try:
            from tensorflow.keras.models import load_model
        except ImportError as error:
            raise PredictionError(
                "TensorFlow is required for ANN inference. Install backend/requirements.txt in a compatible environment."
            ) from error
        _model = load_model(str(MODEL_PATH))
    if _scaler is None:
        if not SCALER_PATH.exists():
            raise PredictionError(f"Scaler not found at {SCALER_PATH}")
        _scaler = joblib.load(str(SCALER_PATH))
    return _model, _scaler


def predict_csv(csv_path: Path) -> dict:
    """
    Run inference on a CICFlowMeter feature CSV. Returns only values the
    model actually produces: per-flow predicted class + confidence, and
    an aggregate summary. Raises PredictionError with a clear message on
    any failure (empty CSV, missing/unmatchable columns, model/scaler
    load failure, or every row being unscoreable) rather than fabricating
    a result.

    Rows whose features are inf/NaN after cleaning (a real, common
    CICFlowMeter/CICIDS2017 artifact — e.g. a zero-duration flow makes
    "Flow Bytes/s" literally Infinity) are labelled INVALID_FEATURES
    rather than a real class, and are excluded from class_counts/
    flows_analyzed but still written back to the CSV (never silently
    dropped, never fabricated as BENIGN/DDoS/DoS/PortScan). This keeps
    real-world datasets — including the public CICIDS2017 CSVs, which
    contain a small fraction of such rows — usable end-to-end instead of
    the whole file being rejected for a handful of unscoreable rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise PredictionError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise PredictionError(f"Feature CSV is empty: {csv_path}")

    # Re-running prediction on an already-labelled CSV: drop any stale
    # Current_State column up front so it never leaks into id/feature
    # detection below, and so there is exactly one such column at the end.
    if CURRENT_STATE_COLUMN in df.columns:
        logger.info(f"[Prediction] Found existing {CURRENT_STATE_COLUMN} column, will regenerate it")
        df = df.drop(columns=[CURRENT_STATE_COLUMN])

    # Some CICFlowMeter output repeats the header row when flows were
    # captured across multiple internal batches; drop any such rows.
    if "Protocol" in df.columns:
        df = df[df["Protocol"] != "Protocol"]
    elif " Protocol" in df.columns:
        df = df[df[" Protocol"] != " Protocol"]
    df = df.reset_index(drop=True)

    if df.empty:
        raise PredictionError(
            f"Feature CSV had rows but none were valid data rows: {csv_path}"
        )

    logger.info(f"[Prediction] Loaded {len(df)} flows from {csv_path.name}")

    id_cols = [c for c in df.columns if is_id_or_label_column(c)]
    flow_meta = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

    try:
        col_map = match_columns(list(df.columns))
    except ValueError as e:
        raise PredictionError(str(e))

    ordered = df[[col_map[f] for f in TRAINING_FEATURES]].copy()
    ordered.columns = TRAINING_FEATURES

    try:
        ordered = ordered.replace([np.inf, -np.inf], np.nan)
        ordered = ordered.astype(np.float64)
    except (ValueError, TypeError) as e:
        raise PredictionError(f"Non-numeric values in feature columns: {e}")

    valid_mask = (~ordered.isna().any(axis=1)).to_numpy()
    n_total = len(df)
    n_dropped = int((~valid_mask).sum())

    if not valid_mask.any():
        raise PredictionError(
            "All rows contained invalid/missing feature values after cleaning."
        )

    model, scaler = _load_artifacts()

    valid_ordered = ordered.to_numpy(dtype=np.float64)[valid_mask]
    scaled = scaler.transform(valid_ordered).astype(np.float32)

    t0 = time.time()
    valid_probabilities = model.predict(scaled, verbose=0)
    inference_seconds = round(time.time() - t0, 3)
    logger.info("[Prediction] ANN inference completed")

    valid_idx = np.argmax(valid_probabilities, axis=1)
    valid_labels = [CLASS_NAMES[i] for i in valid_idx]
    valid_confidences = valid_probabilities[np.arange(len(valid_probabilities)), valid_idx]

    # Full-length, in original row order: real predictions for scoreable
    # rows, an explicit non-fabricated sentinel for the rest. This keeps
    # CSV and JSON in perfect 1:1 correspondence with every row of the
    # input CSV — never a length mismatch, never a silently-dropped row,
    # and never a real class name invented for a row the model couldn't
    # actually score.
    predicted_labels = np.full(n_total, INVALID_FEATURES_LABEL, dtype=object)
    confidences = np.full(n_total, np.nan)
    valid_positions = np.flatnonzero(valid_mask)
    predicted_labels[valid_positions] = valid_labels
    confidences[valid_positions] = valid_confidences

    flows = []
    for i in range(n_total):
        conf = confidences[i]
        row = {
            "predicted_state": str(predicted_labels[i]),
            "confidence": round(float(conf), 4) if not np.isnan(conf) else None,
        }
        for col in id_cols:
            value = flow_meta.iloc[i][col]
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            else:
                value = str(value)
            row[col.strip()] = value
        flows.append(row)

    counts = {c: int((predicted_labels == c).sum()) for c in CLASS_NAMES}
    n_scored = int(valid_mask.sum())
    overall_state = "MALICIOUS" if counts["BENIGN"] < n_scored else "BENIGN"

    distribution_str = ", ".join(f"{c}={counts[c]}" for c in CLASS_NAMES)
    if n_dropped:
        distribution_str += f", {INVALID_FEATURES_LABEL}={n_dropped}"
    logger.info(f"[Prediction] Predicted states: {distribution_str}")
    if n_dropped:
        logger.info(
            f"[Prediction] {n_dropped} of {n_total} row(s) had invalid/missing "
            f"feature values (inf/NaN) after cleaning and were labelled "
            f"'{INVALID_FEATURES_LABEL}' rather than scored — a real, "
            f"documented CICFlowMeter/CICIDS2017 artifact (e.g. zero-duration "
            f"flows producing Infinity for Flow Bytes/s), not a pipeline error."
        )

    # Persist the SAME predictions used above onto the CSV as the final
    # Current_State column — never recomputed, so CSV and JSON can never
    # disagree. predicted_labels has exactly one entry per row of df, in
    # the same order, so this is a safe positional assignment.
    out_df = df.copy()
    out_df[CURRENT_STATE_COLUMN] = predicted_labels
    out_df.to_csv(csv_path, index=False)
    logger.info(f"[Prediction] Added {CURRENT_STATE_COLUMN} column")
    logger.info(f"[Prediction] Updated CSV: {csv_path}")

    return {
        "model": "ISAA_ANN (Keras Sequential MLP, 4-class softmax)",
        "flows_analyzed": n_scored,
        "flows_dropped_invalid": n_dropped,
        "invalid_features_label": INVALID_FEATURES_LABEL if n_dropped else None,
        "class_counts": counts,
        "overall_traffic_state": overall_state,
        "inference_seconds": inference_seconds,
        "feature_csv_updated": True,
        "current_state_column": CURRENT_STATE_COLUMN,
        "output_csv": str(csv_path),
        "flows": flows,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m backend.prediction.predict <csv_path> <output_json_path>")
        sys.exit(1)
    result = predict_csv(Path(sys.argv[1]))
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
