from __future__ import annotations

import json
import os
import random
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from ..temporal.schema import STATE_FEATURE_NAMES
from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    FORECAST_CLASSES,
    HOLDOUT_RATIO,
    LATEST_PATH,
    MAX_EPOCHS,
    ROLLING_FOLDS,
    SCHEMA_VERSION,
    SEED,
    SEQUENCE_LENGTH,
    WINDOW_SIZE_SECONDS,
    repository_path,
    repository_relative,
    source_paths,
)
from .dataset import (
    analyze_sources,
    artifact_fingerprints,
    build_sequences,
    concat_sequence_sets,
    dataset_fingerprints,
    prepare_session_windows,
)
from .evaluation import evaluate_predictions

TRAINING_PROTOCOL_VERSION = "strict-final-train-scaler/v1"


def _set_determinism() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for LSTM training. Install backend/requirements.txt in a Python 3.11 environment."
        ) from error
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf


def build_model(input_shape=(SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES))):
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    values = tf.keras.layers.LSTM(64)(inputs)
    values = tf.keras.layers.Dropout(0.3)(values)
    values = tf.keras.layers.Dense(32, activation="relu")(values)
    outputs = tf.keras.layers.Dense(len(FORECAST_CLASSES), activation="softmax")(values)
    model = tf.keras.Model(inputs, outputs, name="next_window_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def rolling_origin_windows(windows: pd.DataFrame, folds: int = ROLLING_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(windows)
    if n < (SEQUENCE_LENGTH + 1) * (folds + 1):
        return []
    edges = np.linspace(0.40, 1.0, folds + 1)
    results = []
    for index in range(folds):
        train_end = max(SEQUENCE_LENGTH + 1, int(n * edges[index]))
        validation_end = max(train_end + SEQUENCE_LENGTH + 1, int(n * edges[index + 1]))
        validation_end = min(validation_end, n)
        train = windows.iloc[:train_end].reset_index(drop=True)
        validation = windows.iloc[train_end:validation_end].reset_index(drop=True)
        if len(validation) > SEQUENCE_LENGTH:
            results.append((train, validation))
    return results


def split_sessions(session_windows: list[pd.DataFrame]):
    train_parts, holdout_parts = [], []
    for windows in session_windows:
        boundary = max(SEQUENCE_LENGTH + 1, int(len(windows) * (1.0 - HOLDOUT_RATIO)))
        boundary = min(boundary, len(windows))
        train_parts.append(windows.iloc[:boundary].reset_index(drop=True))
        holdout_parts.append(windows.iloc[boundary:].reset_index(drop=True))
    return train_parts, holdout_parts


def _fit_scaler(windows: list[pd.DataFrame]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate([frame[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32) for frame in windows], axis=0))
    return scaler


def _scale_sequences(sequences: dict, scaler: MinMaxScaler) -> np.ndarray:
    shape = sequences["X"].shape
    if not shape[0]:
        return sequences["X"].copy()
    return scaler.transform(sequences["X"].reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def _encode(labels) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(FORECAST_CLASSES)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _class_weights(labels, capped: bool) -> dict[int, float] | None:
    if not capped:
        return None
    encoded = _encode(labels)
    present = np.unique(encoded)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=encoded)
    return {int(label): float(min(weight, 5.0)) for label, weight in zip(present, weights)}


def _fit_keras(train, validation, scaler, weighted, output_path: Path, status) -> tuple[object, dict]:
    tf = _tensorflow()
    model = build_model()
    X_train = _scale_sequences(train, scaler)
    X_validation = _scale_sequences(validation, scaler)
    y_train = _encode(train["y"])
    y_validation = _encode(validation["y"])

    class StatusCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            probabilities = self.model.predict(X_validation, batch_size=BATCH_SIZE, verbose=0)
            predictions = np.asarray(FORECAST_CLASSES)[np.argmax(probabilities, axis=1)]
            metrics = evaluate_predictions(validation["y"], predictions, probabilities)
            status(
                stage="training",
                epoch=int(epoch + 1),
                loss=float(logs.get("loss", 0.0)),
                validation_loss=float(logs.get("val_loss", 0.0)),
                validation_macro_f1=metrics["macro_f1"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(output_path), monitor="val_loss", save_best_only=True),
        StatusCallback(),
    ]
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=_class_weights(train["y"], weighted),
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    probabilities = model.predict(X_validation, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.asarray(FORECAST_CLASSES)[np.argmax(probabilities, axis=1)]
    metrics = evaluate_predictions(validation["y"], predictions, probabilities)
    metrics["epochs"] = len(history.history["loss"])
    metrics["training_seconds"] = round(time.perf_counter() - started, 3)
    return model, metrics


def _persistence(sequences: dict) -> tuple[np.ndarray, np.ndarray]:
    predictions = sequences["input_labels"][:, -1]
    probabilities = np.zeros((len(predictions), len(FORECAST_CLASSES)), dtype=float)
    mapping = {label: index for index, label in enumerate(FORECAST_CLASSES)}
    for row, label in enumerate(predictions):
        probabilities[row, mapping[label]] = 1.0
    return predictions, probabilities


def _logistic(train, evaluation, scaler):
    X_train = _scale_sequences(train, scaler).reshape(len(train["X"]), -1)
    X_evaluation = _scale_sequences(evaluation, scaler).reshape(len(evaluation["X"]), -1)
    model = LogisticRegression(max_iter=300, random_state=SEED, class_weight=None)
    model.fit(X_train, train["y"])
    probabilities_partial = model.predict_proba(X_evaluation)
    probabilities = np.zeros((len(evaluation["X"]), len(FORECAST_CLASSES)), dtype=float)
    for index, label in enumerate(model.classes_):
        probabilities[:, list(FORECAST_CLASSES).index(label)] = probabilities_partial[:, index]
    return model, model.predict(X_evaluation), probabilities


def _session_support(parts: list[pd.DataFrame]) -> dict:
    result = {}
    for frame in parts:
        session = str(frame["session_id"].iloc[0]) if len(frame) else "empty"
        result[session] = {label: int((frame["dominant_state"] == label).sum()) for label in FORECAST_CLASSES}
    return result


def train_forecaster(force_rebuild: bool = False, status=lambda **kwargs: None) -> dict:
    _set_determinism()
    started_total = time.perf_counter()
    paths = source_paths()
    fingerprints = dataset_fingerprints(paths)
    artifacts = artifact_fingerprints()
    version_seed = json.dumps({
        "sources": fingerprints,
        "artifacts": artifacts,
        "schema_version": SCHEMA_VERSION,
        "window_size_seconds": WINDOW_SIZE_SECONDS,
        "sequence_length": SEQUENCE_LENGTH,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
    }, sort_keys=True).encode()
    version = "v1-" + __import__("hashlib").sha256(version_seed).hexdigest()[:12]
    artifact_dir = ARTIFACT_ROOT / version
    analysis_dir = artifact_dir / "analysis"
    status(
        stage="analyzing", rows_processed=0, cache_state="checking", model_version=version,
        report_path=None, evaluation_status=None, error=None,
    )
    analysis_started = time.perf_counter()
    analysis = analyze_sources(paths, analysis_dir, progress=status)
    analysis_seconds = time.perf_counter() - analysis_started

    session_windows, cache_metadata = [], []
    rows_seen = 0
    preprocessing_started = time.perf_counter()
    for path, fingerprint in zip(paths, fingerprints):
        fingerprint = dict(fingerprint)
        fingerprint["rows"] = next(item["rows"] for item in analysis["sources"] if item["source"] == path.name)
        windows, metadata = prepare_session_windows(
            path, fingerprint, artifacts, force_rebuild=force_rebuild, progress=status
        )
        if "session_id" not in windows:
            windows.insert(0, "session_id", path.stem)
        session_windows.append(windows)
        cache_metadata.append(metadata)
        rows_seen += fingerprint["rows"]
        status(stage="preparing_sequences", rows_processed=rows_seen)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    sequence_started = time.perf_counter()
    train_windows, holdout_windows = split_sessions(session_windows)
    train_sequences = concat_sequence_sets([build_sequences(frame) for frame in train_windows])
    holdout_sequences = concat_sequence_sets([build_sequences(frame) for frame in holdout_windows])
    if not len(train_sequences["X"]) or not len(holdout_sequences["X"]):
        raise RuntimeError("Prepared sessions do not contain enough contiguous train/holdout sequences.")
    sequence_seconds = time.perf_counter() - sequence_started

    selection = []
    candidate_scores = {"unweighted": [], "capped_class_weighted": []}
    fold_pairs = [rolling_origin_windows(frame) for frame in train_windows]
    for fold_index in range(ROLLING_FOLDS):
        fold_train_windows = [pairs[fold_index][0] for pairs in fold_pairs if len(pairs) > fold_index]
        fold_validation_windows = [pairs[fold_index][1] for pairs in fold_pairs if len(pairs) > fold_index]
        fold_train = concat_sequence_sets([build_sequences(frame) for frame in fold_train_windows])
        fold_validation = concat_sequence_sets([build_sequences(frame) for frame in fold_validation_windows])
        fold_scaler = _fit_scaler(fold_train_windows)
        fold_record = {"fold": fold_index + 1, "train_samples": len(fold_train["X"]), "validation_samples": len(fold_validation["X"]), "candidates": {}}
        for weighted, name in ((False, "unweighted"), (True, "capped_class_weighted")):
            checkpoint = artifact_dir / "selection" / f"fold_{fold_index + 1}_{name}.keras"
            _, metrics = _fit_keras(fold_train, fold_validation, fold_scaler, weighted, checkpoint, status)
            candidate_scores[name].append(metrics["macro_f1"])
            fold_record["candidates"][name] = metrics
        selection.append(fold_record)

    mean_scores = {name: float(np.mean(scores)) for name, scores in candidate_scores.items()}
    selected = max(mean_scores, key=lambda name: (mean_scores[name], name == "unweighted"))
    weighted = selected == "capped_class_weighted"
    final_train_windows, final_validation_windows = [], []
    for frame in train_windows:
        boundary = max(SEQUENCE_LENGTH + 1, int(len(frame) * 0.90))
        final_train_windows.append(frame.iloc[:boundary].reset_index(drop=True))
        final_validation_windows.append(frame.iloc[boundary:].reset_index(drop=True))
    final_train = concat_sequence_sets([build_sequences(frame) for frame in final_train_windows])
    final_validation = concat_sequence_sets([build_sequences(frame) for frame in final_validation_windows])
    scaler = _fit_scaler(final_train_windows)
    model_path = artifact_dir / "model.keras"
    model, final_validation_metrics = _fit_keras(final_train, final_validation, scaler, weighted, model_path, status)

    holdout_scaled = _scale_sequences(holdout_sequences, scaler)
    started_inference = time.perf_counter()
    probabilities = model.predict(holdout_scaled, batch_size=BATCH_SIZE, verbose=0)
    inference_seconds = time.perf_counter() - started_inference
    predictions = np.asarray(FORECAST_CLASSES)[np.argmax(probabilities, axis=1)]
    lstm_metrics = evaluate_predictions(holdout_sequences["y"], predictions, probabilities)

    baseline_started = time.perf_counter()
    persistence_pred, persistence_prob = _persistence(holdout_sequences)
    persistence_metrics = evaluate_predictions(holdout_sequences["y"], persistence_pred, persistence_prob)
    logistic_model, logistic_pred, logistic_prob = _logistic(final_train, holdout_sequences, scaler)
    logistic_metrics = evaluate_predictions(holdout_sequences["y"], logistic_pred, logistic_prob)
    baseline_seconds = time.perf_counter() - baseline_started

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifact_dir / "scaler.bin")
    joblib.dump(logistic_model, artifact_dir / "baseline_logistic.bin")
    (artifact_dir / "label_map.json").write_text(json.dumps({label: index for index, label in enumerate(FORECAST_CLASSES)}, indent=2))
    (artifact_dir / "feature_names.json").write_text(json.dumps(STATE_FEATURE_NAMES, indent=2))
    metrics = {
        "selection": selection,
        "mean_rolling_macro_f1": mean_scores,
        "selected_training": selected,
        "final_validation": final_validation_metrics,
        "holdout": {
            "lstm": lstm_metrics,
            "persistence": persistence_metrics,
            "logistic_regression": logistic_metrics,
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rolling_training_seconds = sum(
        candidate["training_seconds"]
        for fold in selection for candidate in fold["candidates"].values()
    )
    training_seconds = rolling_training_seconds + final_validation_metrics["training_seconds"]
    training_sequence_epochs = sum(
        fold["train_samples"] * candidate["epochs"]
        for fold in selection for candidate in fold["candidates"].values()
    ) + len(final_train["X"]) * final_validation_metrics["epochs"]
    report = {
        "model_version": version,
        "architecture": "LSTM(64) -> Dropout(0.3) -> Dense(32, ReLU) -> Softmax(4)",
        "seed": SEED,
        "chronology": "row_order_proxy",
        "limitation": "One source row is treated as one proxy second; this is not measured wall-clock forecasting.",
        "window_size_seconds": WINDOW_SIZE_SECONDS,
        "sequence_length": SEQUENCE_LENGTH,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "scaler_fit_scope": "final_training_windows_only",
        "dataset_fingerprints": fingerprints,
        "ann_artifact_fingerprints": artifacts,
        "analysis": analysis,
        "cache": cache_metadata,
        "counts": {
            "rows": analysis["rows"],
            "windows": int(sum(len(frame) for frame in session_windows)),
            "train_windows": int(sum(len(frame) for frame in train_windows)),
            "holdout_windows": int(sum(len(frame) for frame in holdout_windows)),
            "train_sequences": int(len(train_sequences["X"])),
            "final_train_sequences": int(len(final_train["X"])),
            "validation_sequences": int(len(final_validation["X"])),
            "holdout_sequences": int(len(holdout_sequences["X"])),
        },
        "class_support_windows": {"train": _session_support(train_windows), "holdout": _session_support(holdout_windows)},
        "metrics": metrics,
        "evaluation_status": lstm_metrics["evaluation_status"],
        "benchmark": {
            "total_seconds": round(time.perf_counter() - started_total, 3),
            "analysis_seconds": round(analysis_seconds, 3),
            "analysis_rows_per_second": round(analysis["rows"] / max(analysis_seconds, 1e-9), 3),
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "preprocessing_rows_per_second": round(analysis["rows"] / max(preprocessing_seconds, 1e-9), 3),
            "sequence_build_seconds": round(sequence_seconds, 3),
            "sequences_per_second": round((len(train_sequences["X"]) + len(holdout_sequences["X"])) / max(sequence_seconds, 1e-9), 3),
            "training_seconds": round(training_seconds, 3),
            "training_sequence_epochs_per_second": round(training_sequence_epochs / max(training_seconds, 1e-9), 3),
            "baseline_seconds": round(baseline_seconds, 3),
            "holdout_inference_seconds": round(inference_seconds, 3),
            "holdout_sequences_per_second": round(len(holdout_sequences["X"]) / max(inference_seconds, 1e-9), 3),
            "peak_ram_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024), 3),
        },
        "artifact_dir": repository_relative(artifact_dir),
    }
    report_path = artifact_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    latest = {
        "model_version": version,
        "artifact_dir": repository_relative(artifact_dir),
        "report_path": repository_relative(report_path),
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(latest, indent=2))
    status(
        stage="completed",
        rows_processed=analysis["rows"],
        report_path=repository_relative(report_path),
        evaluation_status=report["evaluation_status"],
    )
    return report


def _load_recent_windows(windows_source) -> "pd.DataFrame":
    """Last SEQUENCE_LENGTH contiguous 10-second windows to forecast from.

    windows_source is a `data/temporal/<session>/` directory produced by
    `Temporal Dataset -> Prepare` — i.e. the user's own capture. If it is
    None, fall back to the frozen training-set window cache
    (`report.cache[-1]`), which only exists on a machine that rebuilt it
    from the CICIDS2017 CSVs; a missing cache raises RuntimeError (409),
    never FileNotFoundError (500)."""
    from pathlib import Path

    if windows_source is not None:
        states_csv = Path(windows_source) / "temporal_states.csv"
        if not states_csv.is_file():
            raise RuntimeError(
                f"Prepared temporal dataset has no temporal_states.csv at {states_csv}."
            )
        windows = pd.read_csv(states_csv)
    else:
        artifact_dir = repository_path(json.loads(LATEST_PATH.read_text())["artifact_dir"])
        report = json.loads((artifact_dir / "report.json").read_text())
        from .config import CACHE_ROOT

        cache_key = report["cache"][-1]["cache_key"]
        npz = CACHE_ROOT / cache_key / "windows.npz"
        if not npz.is_file():
            raise RuntimeError(
                "One-step forecast needs a prepared temporal dataset "
                "(Temporal Dataset -> Prepare) or the CICIDS2017 training "
                "window cache, which is not on this machine."
            )
        data = np.load(npz, allow_pickle=False)
        windows = pd.DataFrame({name: data[name] for name in data.files})

    if "window_id" not in windows.columns or "dominant_state" not in windows.columns:
        raise RuntimeError("Window source is missing window_id / dominant_state columns.")
    block = windows.sort_values("window_id").reset_index(drop=True)
    if len(block) < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Need {SEQUENCE_LENGTH} contiguous 10-second windows to forecast, "
            f"have {len(block)}. Capture ~{SEQUENCE_LENGTH * 10 + 20}s+ of "
            f"continuous traffic and re-run Prepare Temporal Dataset."
        )
    recent = block.iloc[-SEQUENCE_LENGTH:]
    if not np.all(np.diff(recent["window_id"].to_numpy()) == 1):
        raise RuntimeError(
            f"The last {SEQUENCE_LENGTH} windows are not contiguous "
            "(gaps from invalid-state exclusion). Capture longer / steadier traffic."
        )
    return recent


def forecast_latest(windows_source=None) -> dict:
    if not LATEST_PATH.is_file():
        raise RuntimeError("No completed LSTM forecasting artifact is available.")
    latest = json.loads(LATEST_PATH.read_text())
    artifact_dir = repository_path(latest["artifact_dir"])
    tf = _tensorflow()
    model = tf.keras.models.load_model(artifact_dir / "model.keras", compile=False)
    scaler = joblib.load(artifact_dir / "scaler.bin")
    report = json.loads((artifact_dir / "report.json").read_text())
    recent = _load_recent_windows(windows_source)
    X = recent[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X)[None, :, :]
    probabilities = model.predict(X_scaled, verbose=0)[0]
    index = int(np.argmax(probabilities))
    probability_map = {label: float(probabilities[i]) for i, label in enumerate(FORECAST_CLASSES)}
    from ..mitre.mapper import MitreAttackMapper
    mitre_mapping = MitreAttackMapper().map_forecast(
        str(recent["dominant_state"].iloc[-1]),
        probability_map,
        recent.iloc[-1][STATE_FEATURE_NAMES].to_dict(),
    )
    return {
        "current_state": str(recent["dominant_state"].iloc[-1]),
        "current_window": int(recent["window_id"].iloc[-1]),
        "predicted_state": FORECAST_CLASSES[index],
        "confidence": float(probabilities[index]),
        "probabilities": probability_map,
        "forecast_probability": float(probabilities[index]),
        "mitre_mapping": mitre_mapping,
        "model_version": latest["model_version"],
        "evaluation_status": report["evaluation_status"],
        "limitation": "row_order_proxy",
    }
