from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score

from ..config import REPO_ROOT
from ..lstm.config import FORECAST_CLASSES, SEED, repository_relative
from ..lstm.dataset import cicids_ground_truth_state
from .features import TRAINING_FEATURES, match_columns
from .predict import SCALER_PATH
from .shap_service import stratified_background

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "flow_challenger"


def _clean_features(frame: pd.DataFrame) -> np.ndarray:
    columns = match_columns(list(frame.columns))
    values = frame[[columns[name] for name in TRAINING_FEATURES]].copy()
    values.columns = TRAINING_FEATURES
    values = values.replace([np.inf, -np.inf, "Infinity", "-Infinity"], np.nan)
    values = values.apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=np.float64)


def fit_challenger(values: np.ndarray, labels: np.ndarray) -> tuple[HistGradientBoostingClassifier, dict, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    valid = np.isfinite(values).all(axis=1) & np.isin(labels, FORECAST_CLASSES)
    values, labels = values[valid], labels[valid]
    train_indices, validation_indices = [], []
    for label in FORECAST_CLASSES:
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise RuntimeError(f"Flow challenger class {label} has insufficient support.")
        boundary = max(1, int(len(indices) * 0.8))
        train_indices.extend(indices[:boundary]); validation_indices.extend(indices[boundary:])
    train_indices = np.asarray(sorted(train_indices)); validation_indices = np.asarray(sorted(validation_indices))
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=180, max_leaf_nodes=31,
        class_weight="balanced", random_state=SEED,
    ).fit(values[train_indices], labels[train_indices])
    probabilities = model.predict_proba(values[validation_indices])
    predictions = model.predict(values[validation_indices])
    attack_true = labels[validation_indices] != "BENIGN"
    benign_index = list(model.classes_).index("BENIGN")
    attack_probability = 1.0 - probabilities[:, benign_index]
    tn, fp, _, _ = confusion_matrix(attack_true, predictions != "BENIGN", labels=[False, True]).ravel()
    metrics = {
        "macro_f1": float(f1_score(labels[validation_indices], predictions, labels=FORECAST_CLASSES, average="macro", zero_division=0)),
        "ddos_recall": float(recall_score(labels[validation_indices] == "DDoS", predictions == "DDoS", zero_division=0)),
        "attack_pr_auc": float(average_precision_score(attack_true, attack_probability)),
        "benign_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "class_support": {label: int(np.sum(labels[validation_indices] == label)) for label in FORECAST_CLASSES},
    }
    background = stratified_background(values[train_indices], labels[train_indices])
    return model, metrics, background


def train_flow_challenger(paths: list[Path], max_rows_per_class: int = 25_000) -> dict:
    samples: dict[str, list[np.ndarray]] = {label: [] for label in FORECAST_CLASSES}
    counts = {label: 0 for label in FORECAST_CLASSES}
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
            label_column = next((column for column in chunk.columns if str(column).strip().lower() == "label"), None)
            if label_column is None:
                raise RuntimeError(f"Ground-truth Label column is missing from {path.name}.")
            mapped = chunk[label_column].map(cicids_ground_truth_state)
            values = _clean_features(chunk)
            for label in FORECAST_CLASSES:
                remaining = max_rows_per_class - counts[label]
                if remaining <= 0:
                    continue
                selected = values[(mapped == label).to_numpy()][:remaining]
                if len(selected):
                    samples[label].append(selected); counts[label] += len(selected)
        if all(counts[label] >= max_rows_per_class for label in FORECAST_CLASSES):
            break
    values = np.concatenate([np.concatenate(samples[label]) for label in FORECAST_CLASSES])
    labels = np.concatenate([np.repeat(label, counts[label]) for label in FORECAST_CLASSES])
    model, metrics, background = fit_challenger(values, labels)
    version = "candidate-" + hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()[:12]
    artifact_dir = ARTIFACT_ROOT / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.bin")
    np.save(artifact_dir / "shap_background.npy", background)
    ann_scaler = joblib.load(SCALER_PATH)
    np.save(REPO_ROOT / "models" / "ann_shap_background.npy", ann_scaler.transform(background).astype(np.float32))
    (artifact_dir / "feature_names.json").write_text(json.dumps(TRAINING_FEATURES, indent=2))
    report = {"model": "HistGradientBoostingClassifier", "status": "challenger_not_automatically_promoted", "metrics": metrics, "training_class_support": counts}
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "latest.json").write_text(json.dumps({"artifact_dir": repository_relative(artifact_dir)}, indent=2))
    return report
