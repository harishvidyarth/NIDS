"""
Ground-truth evaluation metrics for the per-flow ANN classifier
(models/ISAA_ANN.h5) — precision / recall / F1 / support per class,
macro & weighted F1, attack-vs-benign precision / recall / F1 / **false
positive rate**, and a 4x4 confusion matrix.

The LSTM forecaster already reports these (backend/lstm/evaluation.py,
GET /api/benchmark); the ANN never had them. This module fills that gap.

Label source, in priority order (the endpoint adds a third, unlabelled
"proxy" fallback):
  1. NIDS_CICIDS2017_DIR  — a stratified held-out slice of the official
     CICIDS2017 CSVs, real per-flow `Label`.
  2. an explicit CSV path that carries a `Label` / `label` column.
Both are genuine ground truth (`is_ground_truth = True`).
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .predict import CLASS_NAMES
from ..lstm.dataset import score_ann_chunk
from ..lstm.evaluation import _attack_metrics

logger = logging.getLogger("nids.prediction")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
CACHE_PATH = MODELS_DIR / "ann_metrics.json"

# CICIDS2017 raw `Label` value -> one of CLASS_NAMES, or None to drop the
# row (the ANN was trained on a pre-filtered 4-class dataset, so brute
# force / web attack / infiltration / bot rows are out of scope and are
# excluded from the evaluation rather than force-fit).
_LABEL_MAP = {
    "benign": "BENIGN",
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    "portscan": "PortScan",
}

_MAX_EVAL_ROWS = 80_000  # cap so the endpoint stays interactive


def _canon_label(raw: str) -> Optional[str]:
    key = str(raw).strip().lower().replace("\x96", "-").replace("–", "-")
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    if key.startswith("dos "):
        return "DoS"
    if "ddos" in key:
        return "DDoS"
    if key == "portscan":
        return "PortScan"
    return None


def _cicids_dir() -> Optional[Path]:
    configured = os.environ.get("NIDS_CICIDS2017_DIR")
    if configured:
        p = Path(configured).expanduser()
        if p.is_dir():
            return p
    fallback = Path(__file__).resolve().parent.parent.parent / "data" / "cicids2017"
    return fallback if fallback.is_dir() else None


def _iter_label_csv_paths() -> list[Path]:
    d = _cicids_dir()
    if not d:
        return []
    return sorted(Path(p) for p in glob.glob(str(d / "*.csv")))


def _collect_labelled(paths: list[Path], per_class_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Score the ANN over labelled CSV rows, returning aligned
    (y_true, y_pred) for rows the ANN could actually score, roughly
    balanced across the classes present."""
    kept_true: list[str] = []
    kept_pred: list[str] = []
    per_class_seen: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for path in paths:
        try:
            reader = pd.read_csv(path, chunksize=20_000, low_memory=False)
        except Exception as exc:  # pragma: no cover
            logger.info(f"[Metrics] skipping {path.name}: {exc}")
            continue
        for chunk in reader:
            label_col = next(
                (c for c in chunk.columns if str(c).strip().lower() == "label"), None
            )
            if label_col is None:
                break
            canon = chunk[label_col].map(_canon_label)
            mask = canon.notna()
            if not mask.any():
                continue
            sub = chunk[mask].reset_index(drop=True)
            sub_true = canon[mask].to_numpy()
            take = np.array(
                [per_class_seen.get(t, 0) < per_class_cap for t in sub_true]
            )
            if not take.any():
                continue
            sub = sub[take].reset_index(drop=True)
            sub_true = sub_true[take]
            try:
                pred, _, valid = score_ann_chunk(sub)
            except Exception as exc:  # pragma: no cover
                logger.info(f"[Metrics] score failure on {path.name}: {exc}")
                continue
            for t, p, v in zip(sub_true, pred, valid):
                if not v:
                    continue
                kept_true.append(str(t))
                kept_pred.append(str(p))
                per_class_seen[t] = per_class_seen.get(t, 0) + 1
            if sum(per_class_seen.values()) >= _MAX_EVAL_ROWS:
                return np.asarray(kept_true), np.asarray(kept_pred)
    return np.asarray(kept_true), np.asarray(kept_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class + macro/weighted + attack-vs-benign + confusion, over the
    fixed CLASS_NAMES label set (absent classes report support 0)."""
    labels = list(CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    attack = _attack_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {
        "n": int(len(y_true)),
        "labels": labels,
        "classes_present": sorted({str(t) for t in y_true}),
        "per_class": per_class,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "attack": attack,
        "confusion_matrix": cm,
    }


def evaluate_ann_metrics(csv_path: Optional[str] = None, use_cache: bool = True) -> Optional[dict]:
    """Ground-truth ANN metrics, or None if no labelled source is
    available (the endpoint then falls back to the proxy view)."""
    if csv_path:
        p = Path(csv_path)
        if not p.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(p, low_memory=False)
        label_col = next((c for c in df.columns if str(c).strip().lower() == "label"), None)
        if label_col is None:
            raise ValueError("CSV has no 'Label' column — cannot compute ground-truth metrics.")
        canon = df[label_col].map(_canon_label)
        mask = canon.notna()
        if not mask.any():
            raise ValueError("No rows with a recognised BENIGN/DDoS/DoS/PortScan label.")
        sub = df[mask].reset_index(drop=True)
        pred, _, valid = score_ann_chunk(sub)
        y_true = canon[mask].to_numpy()[valid]
        y_pred = np.asarray(pred)[valid]
        out = compute_metrics(y_true, y_pred)
        out.update(source=f"labelled_csv:{p.name}", is_ground_truth=True,
                   generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return out

    paths = _iter_label_csv_paths()
    if not paths:
        return None

    if use_cache and CACHE_PATH.is_file():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            newest_src = max(p.stat().st_mtime for p in paths)
            if cached.get("_source_mtime", 0) >= newest_src:
                return cached
        except Exception:  # pragma: no cover
            pass

    y_true, y_pred = _collect_labelled(paths, per_class_cap=_MAX_EVAL_ROWS // 4)
    if len(y_true) == 0:
        return None
    out = compute_metrics(y_true, y_pred)
    out.update(
        source="cicids2017",
        is_ground_truth=True,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_files=[p.name for p in paths],
        _source_mtime=max(p.stat().st_mtime for p in paths),
    )
    absent = [c for c in CLASS_NAMES if out["per_class"][c]["support"] == 0]
    if absent:
        out["note"] = (
            "No labelled rows for " + ", ".join(absent)
            + " in the available CICIDS2017 files (Friday PortScan/DDoS CSVs "
            "are not published in the phase-3 release) — those rows/columns "
            "report support 0."
        )
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(out, indent=2))
    except Exception:  # pragma: no cover
        pass
    return out


def proxy_agreement_metrics(flows: list[dict]) -> dict:
    """Fallback when no labels exist: ANN `predicted_state` vs the
    deterministic signature layer's `effective_state` on the current
    capture. NOT ground truth — an agreement check only."""
    y_ann = np.asarray([str(f.get("predicted_state", "")) for f in flows])
    y_sig = np.asarray([str(f.get("effective_state", f.get("predicted_state", ""))) for f in flows])
    keep = np.isin(y_ann, CLASS_NAMES) & np.isin(y_sig, CLASS_NAMES)
    y_ann, y_sig = y_ann[keep], y_sig[keep]
    if len(y_ann) == 0:
        return {
            "source": "proxy_ann_vs_signature", "is_ground_truth": False,
            "n": 0, "note": "No scored flows to compare.",
        }
    out = compute_metrics(y_sig, y_ann)  # "truth" = signature layer
    out.update(
        source="proxy_ann_vs_signature",
        is_ground_truth=False,
        agreement_rate=float(np.mean(y_ann == y_sig)),
        note=(
            "PROXY — not validated against ground truth. 'Truth' here is the "
            "deterministic signature layer, not real labels. Set "
            "NIDS_CICIDS2017_DIR or pass a labelled CSV for genuine metrics."
        ),
    )
    return out
