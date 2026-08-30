#!/usr/bin/env python3
"""
Build models/ann_shap_background.npy — the reference sample
`shap.GradientExplainer` needs to explain an ANN flow prediction
(backend/prediction/explanation_runtime.py:submit_explanation("ann", ...)).
Without it every Feature Explanation panel falls back to a non-SHAP
gradient x input attribution.

The canonical producer is `backend/prediction/flow_challenger.py`, which
stratifies the background from the labelled CICIDS2017 CSVs
(`NIDS_CICIDS2017_DIR`). When those CSVs aren't available this script
builds an equivalent file from the project's own captured feature CSVs
(`features/*.csv`), stratified by the ANN's own predicted class so every
class present is represented. The background only has to be a
representative sample of the scaled input distribution — labels here just
spread the sample across classes — so this is a valid stand-in. Prefer
regenerating from CICIDS2017 when you have it.

Output: (<=100, 77) float32, already MinMax-scaled (models/minmax.bin),
matching what the ANN and GradientExplainer consume.

Usage:
    python scripts/build_shap_background.py [--size 100] [--glob "features/*.csv"]
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.prediction.features import TRAINING_FEATURES, match_columns  # noqa: E402
from backend.prediction.predict import CLASS_NAMES, _load_artifacts  # noqa: E402
from backend.prediction.shap_service import stratified_background  # noqa: E402

OUT_PATH = REPO_ROOT / "models" / "ann_shap_background.npy"


def _ordered_features(frame: pd.DataFrame) -> np.ndarray:
    if "Protocol" in frame.columns:
        frame = frame[frame["Protocol"] != "Protocol"]
    col_map = match_columns(list(frame.columns))
    ordered = frame[[col_map[name] for name in TRAINING_FEATURES]].copy()
    ordered.columns = TRAINING_FEATURES
    ordered = ordered.replace([np.inf, -np.inf, "Infinity", "-Infinity"], np.nan)
    ordered = ordered.apply(pd.to_numeric, errors="coerce")
    values = ordered.to_numpy(dtype=np.float64)
    return values[np.isfinite(values).all(axis=1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100, help="background sample size (default 100)")
    parser.add_argument("--glob", default="features/*.csv", help="feature CSV glob (repo-relative)")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(str(REPO_ROOT / args.glob)))
    if not csv_paths:
        print(f"No feature CSVs matched {args.glob!r} under {REPO_ROOT}", file=sys.stderr)
        return 1

    frames = []
    for path in csv_paths:
        try:
            rows = _ordered_features(pd.read_csv(path, low_memory=False))
        except Exception as error:  # a malformed capture CSV shouldn't abort the whole build
            print(f"  skip {Path(path).name}: {error}")
            continue
        if len(rows):
            frames.append(rows)
            print(f"  {Path(path).name}: {len(rows)} usable flows")

    if not frames:
        print("No usable flow rows across any CSV.", file=sys.stderr)
        return 1

    raw = np.concatenate(frames, axis=0)
    model, scaler = _load_artifacts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaled = scaler.transform(raw).astype(np.float32)
        predicted = np.array([CLASS_NAMES[i] for i in np.argmax(model.predict(scaled, verbose=0), axis=1)])

    background = stratified_background(scaled, predicted, size=args.size, seed=42).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, background)

    classes, counts = np.unique(predicted, return_counts=True)
    print()
    print(f"pool: {len(raw)} flows  ->  predicted class mix: " +
          ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}  shape={background.shape}  dtype={background.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
