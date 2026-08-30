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
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from ..config import load_config
from .features import TRAINING_FEATURES, match_columns, is_id_or_label_column
from .explain import attribute, top_features_for_row, driving_features
from .signatures import flow_signatures

CLASS_NAMES = ["BENIGN", "DDoS", "DoS", "PortScan"]
CURRENT_STATE_COLUMN = "Current_State"
# Sentinel for a row the ANN genuinely could not score (inf/NaN in one of
# its 77 features after cleaning — a real, well-documented artifact of
# CICFlowMeter/CICIDS2017 output for zero-duration flows, e.g. "Flow
# Bytes/s" = Infinity). Never one of CLASS_NAMES, so nothing downstream
# can mistake it for a real traffic classification, and it is never
# fabricated — it is the model's own "could not score this" state.
INVALID_FEATURES_LABEL = "INVALID_FEATURES"


# Confidence gate for the headline verdict. A small fraction of ANN-only
# flagged flows is not a confirmed attack. Below the gate the hero reads
# "SUSPICIOUS — <class>?" (amber, review) instead of "ATTACK — <class>"
# (red). The gate is purely percentage-based: the attack class's OWN share
# of scored flows must reach `min_ratio` for its tier — a minority DoS is
# not promoted to ATTACK by unrelated DDoS flows also present in the
# capture, and there is no absolute flow-count condition. The deterministic
# signature layer firing for the same class always clears the gate.
# Values come from config/config.json -> "verdict_gate" (read once at
# import; restart the backend after editing). Defaults below are the
# fallback when the key is absent.
_VERDICT_GATE = load_config().get("verdict_gate", {})
_GATE_DDOS = _VERDICT_GATE.get("ddos", {})
_GATE_OTHER = _VERDICT_GATE.get("other", {})
MIN_ATTACK_RATIO_DDOS = _GATE_DDOS.get("min_ratio", 0.20)
MIN_ATTACK_RATIO_OTHER = _GATE_OTHER.get("min_ratio", 0.20)


def summarize_states(
    counts: dict, n_scored: int, *, signature_attack_class: str | None = None
) -> dict:
    """Headline state = the class the most scored flows fall into (ties
    resolve to the earlier CLASS_NAMES entry, i.e. BENIGN first) — the
    same "dominant by flow count" rule the temporal state builder uses.
    The old logic flipped the whole capture to "MALICIOUS" on a single
    non-BENIGN flow, which reads as "everything here is an attack" even at
    1% attack flows. The real signal is the ratio, returned alongside.

    `verdict` / `confidence` apply the gate above: BENIGN (none) → SUSPICIOUS
    (low) → ATTACK (high). The gate is judged purely on the attack class's
    OWN share of scored flows (`attack_class_ratio`) vs `min_ratio`, not the
    combined non-BENIGN ratio and not any absolute flow count — so a
    minority DoS is not promoted to ATTACK by unrelated DDoS flows in the
    same capture. `signature_attack_class` is the deterministic signature
    layer's verdict for this capture, if any — it clears the gate for its
    class."""
    dominant_state = max(
        CLASS_NAMES, key=lambda c: (counts.get(c, 0), -CLASS_NAMES.index(c))
    )
    attack_flow_count = int(n_scored - counts.get("BENIGN", 0))
    ratio = round(attack_flow_count / n_scored, 4) if n_scored else 0.0

    # `attack_class` = the single most common *non-BENIGN* class among
    # scored flows (ties -> the more severe, i.e. earlier non-BENIGN entry).
    attack_classes = [c for c in CLASS_NAMES if c != "BENIGN"]
    attack_class_counts = {c: counts.get(c, 0) for c in attack_classes}
    attack_class = None
    if attack_flow_count > 0:
        attack_class = max(
            attack_classes,
            key=lambda c: (attack_class_counts[c], -attack_classes.index(c)),
        )

    # Per-class share of scored flows — the figure the gate is judged on.
    cls_count = attack_class_counts[attack_class] if attack_class else 0
    cls_ratio = round(cls_count / n_scored, 4) if (attack_class and n_scored) else 0.0

    if attack_class is None:
        verdict, confidence = "BENIGN", "none"
    else:
        min_ratio = (
            MIN_ATTACK_RATIO_DDOS if attack_class == "DDoS" else MIN_ATTACK_RATIO_OTHER
        )
        confirmed = (
            signature_attack_class == attack_class
            or cls_ratio >= min_ratio
        )
        if confirmed:
            verdict, confidence = f"ATTACK — {attack_class}", "high"
        else:
            verdict, confidence = f"SUSPICIOUS — {attack_class}?", "low"

    return {
        "dominant_state": dominant_state,
        "attack_flow_count": attack_flow_count,
        "malicious_flow_ratio": ratio,
        "attack_class_ratio": cls_ratio,
        "attack_present": attack_flow_count > 0,
        "attack_class": attack_class,
        "attack_class_counts": attack_class_counts,
        "verdict": verdict,
        "confidence": confidence,
    }


# Curated per-flow feature VALUES surfaced on the Main Stage "FEATURE
# VALUES" panel (the FEATURE SPACE accordion only ever showed names). Keys
# are the CICFlowMeter column names in this build's feature CSV.
_FLOW_VALUE_COLUMNS = [
    "flow_duration", "flow_byts_s", "flow_pkts_s",
    "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt",
    "down_up_ratio", "pkt_len_mean", "pkt_len_max",
    "init_fwd_win_byts", "init_bwd_win_byts",
]
# TCP flag bitmask, high->low bit: URG ACK PSH RST SYN FIN.
_FLAG_BITS = [
    ("urg_flag_cnt", 0x20, "U"), ("ack_flag_cnt", 0x10, "A"),
    ("psh_flag_cnt", 0x08, "P"), ("rst_flag_cnt", 0x04, "R"),
    ("syn_flag_cnt", 0x02, "S"), ("fin_flag_cnt", 0x01, "F"),
]


def _flow_feature_values(df: pd.DataFrame) -> list[dict]:
    """Per-row dict of the curated flow-level feature VALUES, plus a
    derived TCP-flag bitmask (hex + letters) and IAT variance."""
    cols = {c.strip().lower(): c for c in df.columns}
    out: list[dict] = []
    for i in range(len(df)):
        rec: dict = {}
        for name in _FLOW_VALUE_COLUMNS:
            real = cols.get(name)
            if real is None:
                continue
            val = df.iloc[i][real]
            try:
                num = float(val)
            except (TypeError, ValueError):
                rec[name] = str(val)
                continue
            rec[name] = round(num, 4)
        if "flow_iat_std" in rec and isinstance(rec["flow_iat_std"], float):
            rec["flow_iat_var"] = round(rec["flow_iat_std"] ** 2, 4)
        bits, letters = 0, ""
        for name, bit, ch in _FLAG_BITS:
            real = cols.get(name)
            try:
                on = real is not None and float(df.iloc[i][real]) > 0
            except (TypeError, ValueError):
                on = False
            bits |= bit if on else 0
            letters += ch if on else "."
        rec["tcp_flag_bitmask"] = f"0x{bits:02X}"
        rec["tcp_flags"] = letters
        out.append(rec)
    return out


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
        # compile=False: inference only, no optimizer state needed (also
        # silences Keras' "Skipping variable loading for optimizer 'adam'"
        # UserWarning). ISAA_ANN.h5 is a legacy Sequential built with
        # `input_dim=` on its first Dense layer; Keras 3 warns about that
        # style on load. The weights load correctly and the architecture
        # is frozen (no retraining in scope), so that warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _model = load_model(str(MODEL_PATH), compile=False)
    if _scaler is None:
        if not SCALER_PATH.exists():
            raise PredictionError(f"Scaler not found at {SCALER_PATH}")
        # minmax.bin was pickled with scikit-learn 1.0.2; this project runs
        # 1.5.2. MinMaxScaler's fitted attributes (data_min_/data_max_/
        # scale_/min_) are plain numpy arrays and load unchanged across
        # this gap — verified by comparing transform() output. The
        # InconsistentVersionWarning is therefore noise here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
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
    # `ordered` was explicitly re-columned to TRAINING_FEATURES above, so
    # column *order* is already correct; feeding the scaler a nameless
    # array is intentional. Silence sklearn's "X does not have valid
    # feature names" UserWarning that fires because minmax.bin was fitted
    # on a named DataFrame.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaled = scaler.transform(valid_ordered).astype(np.float32)

    t0 = time.time()
    valid_probabilities = model.predict(scaled, verbose=0)
    inference_seconds = round(time.time() - t0, 3)
    logger.info("[Prediction] ANN inference completed")

    valid_idx = np.argmax(valid_probabilities, axis=1)
    valid_labels = [CLASS_NAMES[i] for i in valid_idx]
    valid_confidences = valid_probabilities[np.arange(len(valid_probabilities)), valid_idx]

    # Feature attribution (SIH brief: interpretable decision support —
    # "which flags/ports/flow patterns are contributing most"). Gradient x
    # input over the scaled features w.r.t. the predicted class. Never
    # allowed to break a prediction: on any failure it is simply omitted.
    try:
        valid_attribution = attribute(model, scaled, valid_idx)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(f"[Prediction] Feature attribution unavailable: {exc}")
        valid_attribution = np.zeros((len(scaled), len(TRAINING_FEATURES)))

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

    attribution = np.full((n_total, len(TRAINING_FEATURES)), np.nan)
    attribution[valid_positions] = valid_attribution

    # Deterministic cross-flow signature layer — catches a volumetric DDoS
    # the per-flow ANN dilutes to BENIGN, and gives INVALID_FEATURES rows a
    # verdict. It never overwrites an ANN label; `effective_state` below is
    # what the UI treats as the row's class, and both are always reported.
    try:
        sig = flow_signatures(df)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(f"[Prediction] Signature layer unavailable: {exc}")
        sig = {"states": ["BENIGN"] * n_total, "hits": [], "attack_class": None,
               "counts": {"BENIGN": n_total}, "port_scan": None}
    sig_states = sig["states"]

    try:
        feature_values = _flow_feature_values(df)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(f"[Prediction] Feature-value panel unavailable: {exc}")
        feature_values = [{} for _ in range(n_total)]

    flows = []
    for i in range(n_total):
        conf = confidences[i]
        ann_state = str(predicted_labels[i])
        sig_state = sig_states[i] if i < len(sig_states) else "BENIGN"
        # ANN wins when it made a real call; the signature only speaks up
        # for rows the ANN left BENIGN or could not score at all.
        effective_state = ann_state
        if sig_state != "BENIGN" and ann_state in ("BENIGN", INVALID_FEATURES_LABEL):
            effective_state = sig_state
        row = {
            "predicted_state": ann_state,
            "signature_state": sig_state,
            "effective_state": effective_state,
            "confidence": round(float(conf), 4) if not np.isnan(conf) else None,
        }
        if not np.isnan(attribution[i]).all():
            row["top_features"] = top_features_for_row(attribution[i])
        if i < len(feature_values):
            row["features"] = feature_values[i]
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
    sig_attack_class = sig.get("attack_class")
    state_summary = summarize_states(
        counts, n_scored, signature_attack_class=sig_attack_class
    )
    dominant_state = state_summary["dominant_state"]
    overall_state = dominant_state

    # Combined verdict: the ANN verdict, escalated if the deterministic
    # signature layer independently flagged an attack the ANN missed.
    effective_states = [f["effective_state"] for f in flows]
    effective_counts = {
        c: int(sum(1 for s in effective_states if s == c)) for c in CLASS_NAMES
    }
    effective_summary = summarize_states(
        effective_counts, sum(effective_counts.values()),
        signature_attack_class=sig_attack_class,
    )
    sig_hits = sig.get("hits", [])
    if sig.get("attack_class") and sig["attack_class"] != state_summary["attack_class"]:
        logger.info(
            f"[Prediction] Signature layer flagged {sig['attack_class']} "
            f"(ANN verdict: {state_summary['verdict']}) — {len(sig_hits)} rule hit(s)"
        )

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
    out_df["Signature_State"] = sig["states"][:n_total]
    out_df["Effective_State"] = effective_states
    out_df.to_csv(csv_path, index=False)
    logger.info(f"[Prediction] Added {CURRENT_STATE_COLUMN} column")
    logger.info(f"[Prediction] Updated CSV: {csv_path}")

    explanation_jobs = []
    try:
        from .explanation_runtime import submit_explanation
        attack_scores = 1.0 - valid_probabilities[:, 0]
        selected = [int(np.argmax(attack_scores))]
        benign = np.flatnonzero(np.asarray(valid_labels) == "BENIGN")
        if len(benign):
            selected.append(int(benign[len(benign) // 2]))
        for sample_index in dict.fromkeys(selected):
            class_name = CLASS_NAMES[int(valid_idx[sample_index])]
            explanation_jobs.append(submit_explanation("ann", scaled[sample_index:sample_index + 1], class_name))
    except Exception as exc:
        logger.info(f"[Prediction] Asynchronous SHAP unavailable: {exc}")

    return {
        "model": "ISAA_ANN (Keras Sequential MLP, 4-class softmax)",
        "flows_analyzed": n_scored,
        "flows_dropped_invalid": n_dropped,
        "invalid_features_label": INVALID_FEATURES_LABEL if n_dropped else None,
        "class_counts": counts,
        "overall_traffic_state": overall_state,
        "dominant_state": dominant_state,
        "attack_alert": effective_summary["attack_class"] or "NONE",
        "attack_ratio": effective_summary["malicious_flow_ratio"],
        # Headline the UI shows: "ATTACK — DDoS" whenever any non-BENIGN
        # flow is present, so a DDoS in a mostly-benign capture is never
        # silently swallowed by the dominant-by-count rule.
        "verdict": state_summary["verdict"],
        # "none" | "low" (SUSPICIOUS — few ANN-only flows) | "high"
        # (ATTACK — cleared the flow-count/ratio gate or the signature
        # layer confirmed it).
        "confidence": state_summary["confidence"],
        "attack_class": state_summary["attack_class"],
        "attack_class_counts": state_summary["attack_class_counts"],
        # The attack class's own share of scored flows — the figure the
        # confidence gate is judged on (not the combined non-BENIGN ratio).
        "attack_class_ratio": state_summary["attack_class_ratio"],
        "attack_flow_count": state_summary["attack_flow_count"],
        "malicious_flow_ratio": state_summary["malicious_flow_ratio"],
        "attack_present": state_summary["attack_present"],
        # Deterministic signature/heuristic layer (cross-flow) run alongside
        # the ANN — independent verdict, never overwrites an ANN label.
        "signature_verdict": sig.get("attack_class"),
        "signature_hits": sig_hits,
        "signature_class_counts": sig.get("counts", {}),
        "port_scan_signature": sig.get("port_scan"),
        "distribution_evidence": sig.get("distribution_evidence", []),
        "signature_evidence": {
            "rate_method": sig.get("rate_method", "unavailable"),
            "rolling_window_seconds": 1,
            "capture_wide_flow_rate_sum_used": False,
        },
        # ANN verdict escalated by the signature layer where they disagree.
        "effective_verdict": effective_summary["verdict"],
        "effective_confidence": effective_summary["confidence"],
        "effective_attack_class": effective_summary["attack_class"],
        "effective_class_counts": effective_counts,
        "inference_seconds": inference_seconds,
        "feature_csv_updated": True,
        "current_state_column": CURRENT_STATE_COLUMN,
        "output_csv": str(csv_path),
        "driving_features": driving_features(valid_attribution),
        "attribution_method": "gradient_x_input_fallback",
        "attribution_is_shap": False,
        "explanation_jobs": explanation_jobs,
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
