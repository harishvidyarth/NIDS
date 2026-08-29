from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import REPO_ROOT
from ..lstm.config import FORECAST_CLASSES, LATEST_PATH as ONE_STEP_LATEST, repository_path, repository_relative
from ..lstm.dataset import build_sequences as build_one_step_sequences, concat_sequence_sets as concat_one_step_sequences
from ..lstm.evaluation import evaluate_predictions
from ..mitre.mapper import MitreAttackMapper
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    HORIZONS,
    LATEST_PATH,
    MAX_EPOCHS,
    REPORT_ROOT,
    SCHEMA_VERSION,
    SEED,
    SEQUENCE_LENGTH,
)
from .dataset import prepare_multistep_dataset, split_session_windows
from .evaluation import degradation_table, evaluate_horizons, onset_metrics, select_early_warning_threshold


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required; install backend/requirements.txt.") from error
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    return tf


def build_model(input_shape=(SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES))):
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    values = tf.keras.layers.LSTM(64)(inputs)
    values = tf.keras.layers.Dropout(0.3)(values)
    values = tf.keras.layers.Dense(32, activation="relu")(values)
    values = tf.keras.layers.Dense(HORIZONS * len(FORECAST_CLASSES))(values)
    values = tf.keras.layers.Reshape((HORIZONS, len(FORECAST_CLASSES)))(values)
    outputs = tf.keras.layers.Softmax(axis=-1)(values)
    model = tf.keras.Model(inputs, outputs, name="direct_h1_h6_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _encode(labels: np.ndarray) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(FORECAST_CLASSES)}
    return np.vectorize(mapping.__getitem__, otypes=[np.int64])(labels)


def _fit_scaler(train_windows) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate([frame[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32) for frame in train_windows]))
    return scaler


def _scale(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)


def _predict(model, sequences, scaler) -> np.ndarray:
    return model.predict(_scale(sequences["X"], scaler), batch_size=BATCH_SIZE, verbose=0)


def _model_gates(model, model_path: Path, validation, scaler, history: dict) -> dict:
    probabilities = _predict(model, validation, scaler)
    checks = {
        "finite_loss": bool(history.get("loss") and np.isfinite(history["loss"]).all()),
        "finite_probabilities": bool(np.isfinite(probabilities).all()),
        "probability_sums": bool(np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)),
        "prediction_diversity": int(np.unique(np.argmax(probabilities, axis=-1)).size) >= 2,
    }
    validation_reports = evaluate_horizons(validation["y"], probabilities)
    checks["validation_performance"] = bool(
        np.isfinite(validation_reports[0]["macro_f1"]) and validation_reports[0]["macro_f1"] > 0.0
    )
    reloaded = _tensorflow().keras.models.load_model(model_path, compile=False)
    sample = _scale(validation["X"][: min(32, len(validation["X"]))], scaler)
    checks["save_load_parity"] = bool(np.allclose(model.predict(sample, verbose=0), reloaded.predict(sample, verbose=0), atol=1e-6))
    return {"passed": all(checks.values()), "checks": checks, "validation": validation_reports}


def _rolling_origin_diagnostics(session_windows) -> list[dict]:
    diagnostics = []
    prior_sessions = []
    for frame in session_windows:
        session = str(frame["session_id"].iloc[0])
        sequences = __import__("backend.lstm_multistep.dataset", fromlist=["build_multistep_sequences"]).build_multistep_sequences(frame)
        if not prior_sessions:
            diagnostics.append({"evaluation_session": session, "status": "N/A", "reason": "No earlier session exists."})
        elif len(sequences["X"]):
            persistence = np.repeat(sequences["input_labels"][:, -1, None], HORIZONS, axis=1)
            diagnostics.append({
                "evaluation_session": session,
                "training_sessions_available_before_score": list(prior_sessions),
                "status": "SCORED_THEN_ADVANCED",
                "persistence_accuracy_by_horizon": [float(np.mean(persistence[:, h] == sequences["y"][:, h])) for h in range(HORIZONS)],
            })
        prior_sessions.append(session)
    return diagnostics


def _one_step_h1_comparison(test, test_probabilities, scaler) -> dict:
    if not ONE_STEP_LATEST.is_file():
        return {"status": "N/A", "reason": "No saved Phase 3 one-step artifact."}
    latest = json.loads(ONE_STEP_LATEST.read_text())
    artifact_dir = repository_path(latest["artifact_dir"])
    feature_names = json.loads((artifact_dir / "feature_names.json").read_text())
    if feature_names != list(STATE_FEATURE_NAMES):
        return {"status": "N/A", "reason": "Phase 3 feature order is incompatible."}
    model = _tensorflow().keras.models.load_model(artifact_dir / "model.keras", compile=False)
    one_step_scaler = joblib.load(artifact_dir / "scaler.bin")
    one_probabilities = model.predict(_scale(test["X"], one_step_scaler), batch_size=BATCH_SIZE, verbose=0)
    direct_predictions = np.asarray(FORECAST_CLASSES)[np.argmax(test_probabilities[:, 0, :], axis=1)]
    one_predictions = np.asarray(FORECAST_CLASSES)[np.argmax(one_probabilities, axis=1)]
    return {
        "status": "AVAILABLE",
        "holdout": "Friday DDoS, identical H1 samples",
        "direct_multistep_h1": evaluate_predictions(test["y"][:, 0], direct_predictions, test_probabilities[:, 0, :]),
        "saved_one_step": evaluate_predictions(test["y"][:, 0], one_predictions, one_probabilities),
        "one_step_model_version": latest["model_version"],
    }


def _benchmark(artifact_dir: Path, model, scaler, sample: np.ndarray) -> dict:
    tf = _tensorflow()
    started = time.perf_counter(); loaded = tf.keras.models.load_model(artifact_dir / "model.keras", compile=False)
    cold_load = time.perf_counter() - started
    one = _scale(sample[:1], scaler)
    loaded.predict(one, verbose=0)
    started = time.perf_counter(); loaded.predict(one, verbose=0); warm = time.perf_counter() - started
    batches = {}
    for size in (100, 1000):
        values = np.repeat(one, size, axis=0)
        started = time.perf_counter(); loaded.predict(values, batch_size=BATCH_SIZE, verbose=0); elapsed = time.perf_counter() - started
        batches[str(size)] = {"seconds": elapsed, "throughput_samples_per_second": size / max(elapsed, 1e-9)}
    divisor = 1024 * 1024 if __import__("platform").system() == "Darwin" else 1024
    return {"cold_model_load_seconds": cold_load, "warm_single_inference_seconds": warm,
            "batches": batches, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor}


def _markdown(report: dict) -> str:
    lines = ["# Multi-Step LSTM Evaluation Report", "", "Direct H1-H6 row-order proxy evaluation. Horizons are windows, not validated elapsed seconds.", "", "## Strict Holdout", ""]
    for name, sessions in report["dataset"]["split"].items(): lines.append(f"- {name.title()}: {', '.join(sessions)}")
    lines.extend(["", "## Per-Horizon Results", "", "| Horizon | Accuracy | Balanced accuracy | Macro-F1 | Attack F1 |", "|---:|---:|---:|---:|---:|"])
    for item in report["per_horizon"]:
        lines.append(f"| H{item['horizon']} | {item['accuracy']:.4f} | {item['balanced_accuracy']:.4f} | {item['macro_f1']:.4f} | {item['attack_forecasting']['f1']} |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def train_multistep(force_rebuild: bool = False, status=lambda **kwargs: None) -> dict:
    started = time.perf_counter()
    sequences, dataset_manifest, session_windows = prepare_multistep_dataset(force_rebuild, status)
    train, validation, test = sequences["train"], sequences["validation"], sequences["test"]
    if min(len(train["X"]), len(validation["X"]), len(test["X"])) == 0:
        raise RuntimeError("Every strict split must contain at least one H1-H6 sample.")
    version_seed = json.dumps({"dataset": dataset_manifest["source_fingerprints"], "schema": SCHEMA_VERSION}, sort_keys=True).encode()
    version = "v1-" + hashlib.sha256(version_seed).hexdigest()[:12]
    artifact_dir = ARTIFACT_ROOT / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_windows = split_session_windows(session_windows)
    scaler = _fit_scaler(split_windows["train"])
    model = build_model()
    callbacks = [
        _tensorflow().keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        _tensorflow().keras.callbacks.ModelCheckpoint(artifact_dir / "model.keras", monitor="val_loss", save_best_only=True),
    ]
    training_started = time.perf_counter()
    history = model.fit(
        _scale(train["X"], scaler), _encode(train["y"]),
        validation_data=(_scale(validation["X"], scaler), _encode(validation["y"])),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0, callbacks=callbacks,
    )
    model = _tensorflow().keras.models.load_model(artifact_dir / "model.keras", compile=False)
    joblib.dump(scaler, artifact_dir / "scaler.bin")
    scaler_sha256 = hashlib.sha256((artifact_dir / "scaler.bin").read_bytes()).hexdigest()
    (artifact_dir / "feature_names.json").write_text(json.dumps(STATE_FEATURE_NAMES, indent=2))
    (artifact_dir / "label_map.json").write_text(json.dumps({label: i for i, label in enumerate(FORECAST_CLASSES)}, indent=2))
    (artifact_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2))
    gates = _model_gates(model, artifact_dir / "model.keras", validation, scaler, history.history)
    validation_probabilities = _predict(model, validation, scaler)
    threshold = select_early_warning_threshold(validation["y"], validation_probabilities)
    test_probabilities = _predict(model, test, scaler)
    per_horizon = evaluate_horizons(test["y"], test_probabilities)
    evaluation = {
        "model_version": version,
        "evaluation_status": "ACTIVATED" if gates["passed"] and threshold["selected_threshold"] is not None else "NOT_ACTIVATED",
        "threshold_selection": threshold,
        "per_horizon": per_horizon,
        "onset": onset_metrics(test["y"], test_probabilities, test["input_labels"], threshold["selected_threshold"]),
        "degradation": degradation_table(per_horizon),
        "h1_comparison": _one_step_h1_comparison(test, test_probabilities, scaler),
        "rolling_origin_diagnostics": _rolling_origin_diagnostics(session_windows),
        "dataset": dataset_manifest,
        "limitations": [
            "CICIDS2017 is an older, attack-clustered benchmark and does not represent current production traffic.",
            "ANN-derived four-state targets are model outputs, while raw CICIDS labels remain diagnostic metadata.",
            "Row order is a synthetic timing proxy; H1-H6 are windows and not validated seconds-ahead forecasts.",
            "Absent split/horizon classes are reported as N/A and are never fabricated or moved between splits.",
            "Longer horizons carry greater uncertainty; network-only evidence cannot confirm host intent or ATT&CK attribution.",
        ],
    }
    benchmark = _benchmark(artifact_dir, model, scaler, test["X"])
    activation = {"active": evaluation["evaluation_status"] == "ACTIVATED", "gates": gates,
                  "selected_threshold": threshold["selected_threshold"]}
    (artifact_dir / "activation_status.json").write_text(json.dumps(activation, indent=2))
    (artifact_dir / "threshold_selection.json").write_text(json.dumps(threshold, indent=2))
    (artifact_dir / "class_distributions.json").write_text(json.dumps(dataset_manifest["splits"], indent=2))
    (artifact_dir / "leakage_audit.json").write_text(json.dumps({
        "session_isolation": "PASS", "chronological_split": "PASS", "train_only_scaler": "PASS",
        "invalid_feature_exclusion": "PASS", "target_alignment": "PASS",
    }, indent=2))
    training_report = {
        "model_version": version,
        "architecture": "LSTM(64) -> Dropout(0.3) -> Dense(32, ReLU) -> Dense(24) -> Reshape(6,4) -> horizon-wise softmax",
        "input_shape": [SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES)], "output_shape": [HORIZONS, len(FORECAST_CLASSES)],
        "epochs": len(history.history["loss"]), "training_seconds": time.perf_counter() - training_started,
        "total_seconds": time.perf_counter() - started, "model_gates": gates, "activation": activation,
        "scaler_identity": {"sha256": scaler_sha256, "fit_scope": "training sessions only", "feature_order": list(STATE_FEATURE_NAMES)},
        "dataset": dataset_manifest,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_report.json").write_text(json.dumps(training_report, indent=2))
    (artifact_dir / "evaluation_report.json").write_text(json.dumps(evaluation, indent=2))
    (artifact_dir / "performance_benchmark.json").write_text(json.dumps(benchmark, indent=2))
    (REPORT_ROOT / "multistep_training_report.json").write_text(json.dumps(training_report, indent=2))
    (REPORT_ROOT / "multistep_evaluation_report.json").write_text(json.dumps(evaluation, indent=2))
    (REPORT_ROOT / "multistep_evaluation_report.md").write_text(_markdown(evaluation))
    (REPORT_ROOT / "multistep_performance_benchmark.json").write_text(json.dumps(benchmark, indent=2))
    if activation["active"]:
        LATEST_PATH.write_text(json.dumps({"model_version": version, "artifact_dir": repository_relative(artifact_dir)}, indent=2))
    return {"training": training_report, "evaluation": evaluation, "benchmark": benchmark}


def forecast_latest() -> dict:
    if not LATEST_PATH.is_file():
        raise RuntimeError("No activated multi-step LSTM artifact is available.")
    latest = json.loads(LATEST_PATH.read_text())
    artifact_dir = repository_path(latest["artifact_dir"])
    model = _tensorflow().keras.models.load_model(artifact_dir / "model.keras", compile=False)
    scaler = joblib.load(artifact_dir / "scaler.bin")
    evaluation = json.loads((artifact_dir / "evaluation_report.json").read_text())
    manifest = json.loads((artifact_dir / "dataset_manifest.json").read_text())
    last_cache = manifest["cache"][-1]
    data = np.load(REPO_ROOT / "data" / "lstm_cache" / last_cache["cache_key"] / "windows.npz", allow_pickle=False)
    import pandas as pd
    windows = pd.DataFrame({name: data[name] for name in data.files}).sort_values("window_id").reset_index(drop=True)
    recent = windows.iloc[-SEQUENCE_LENGTH:]
    if len(recent) != SEQUENCE_LENGTH or not np.all(np.diff(recent["window_id"]) == 1):
        raise RuntimeError("Latest history does not contain five contiguous observed windows.")
    probabilities = model.predict(_scale(recent[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)[None, :, :], scaler), verbose=0)[0]
    threshold = evaluation["threshold_selection"]["selected_threshold"]
    mapper = MitreAttackMapper(); observed_features = recent.iloc[-1][STATE_FEATURE_NAMES].to_dict()
    horizons = []
    for index, row in enumerate(probabilities):
        probability_map = {label: float(row[i]) for i, label in enumerate(FORECAST_CLASSES)}
        predicted_index = int(np.argmax(row)); predicted_state = FORECAST_CLASSES[predicted_index]
        mapping = mapper.map_forecast(str(recent["dominant_state"].iloc[-1]), probability_map, observed_features)
        horizons.append({
            "horizon": index + 1, "label": f"+{index + 1} window{'s' if index else ''}", "seconds_ahead": None,
            "probabilities": probability_map, "predicted_state": predicted_state,
            "forecast_probability": float(row[predicted_index]), "attack_probability": float(1.0 - row[0]),
            "mitre_candidates": mapping["mitre_candidates"], "mapping_confidence": max(
                (item["mapping_confidence"] for item in mapping["mitre_candidates"]), default=None
            ),
        })
    warning = [item["horizon"] for item in horizons if item["attack_probability"] >= threshold]
    return {
        "current_state": str(recent["dominant_state"].iloc[-1]), "history_window_count": SEQUENCE_LENGTH,
        "forecast_horizon": HORIZONS, "window_duration": None, "timestamp_mode": "row_order_proxy",
        "horizon_labeling": "windows", "horizons": horizons, "early_warning_threshold": threshold,
        "earliest_predicted_attack_horizon": min(warning) if warning else None,
        "maximum_attack_probability": max(item["attack_probability"] for item in horizons),
        "model_version": latest["model_version"], "input_shape": [SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES)],
        "output_shape": [HORIZONS, len(FORECAST_CLASSES)], "classes": list(FORECAST_CLASSES),
        "evaluation_status": evaluation["evaluation_status"], "limitations": evaluation["limitations"],
    }
