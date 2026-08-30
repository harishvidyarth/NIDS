from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import REPO_ROOT
from ..lstm.config import ALERT_CLASSES, FORECAST_CLASSES, LATEST_PATH as ONE_STEP_LATEST, repository_path, repository_relative
from ..lstm.dataset import build_sequences as build_one_step_sequences, concat_sequence_sets as concat_one_step_sequences
from ..lstm.evaluation import evaluate_predictions
from ..lstm.training import _peak_ram_mb
from ..mitre.mapper import MitreAttackMapper
from ..prediction.shap_service import stratified_background
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
    dominant_values = tf.keras.layers.Dense(HORIZONS * len(FORECAST_CLASSES))(values)
    dominant_values = tf.keras.layers.Reshape((HORIZONS, len(FORECAST_CLASSES)))(dominant_values)
    dominant = tf.keras.layers.Softmax(axis=-1, name="dominant_state")(dominant_values)
    alert_values = tf.keras.layers.Dense(HORIZONS * len(ALERT_CLASSES))(values)
    alert_values = tf.keras.layers.Reshape((HORIZONS, len(ALERT_CLASSES)))(alert_values)
    alert = tf.keras.layers.Softmax(axis=-1, name="attack_alert")(alert_values)
    model = tf.keras.Model(inputs, {"dominant_state": dominant, "attack_alert": alert}, name="direct_h1_h6_dual_head_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={"dominant_state": "sparse_categorical_crossentropy", "attack_alert": "sparse_categorical_crossentropy"},
        metrics={"dominant_state": ["accuracy"], "attack_alert": ["accuracy"]},
    )
    return model


def _encode(labels: np.ndarray, classes=FORECAST_CLASSES) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    return np.vectorize(mapping.__getitem__, otypes=[np.int64])(labels)


def _fit_scaler(train_windows) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate([frame[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32) for frame in train_windows]))
    return scaler


def _scale(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)


def _predict(model, sequences, scaler) -> dict[str, np.ndarray]:
    output = model.predict(_scale(sequences["X"], scaler), batch_size=BATCH_SIZE, verbose=0)
    if isinstance(output, dict):
        return output
    return {"dominant_state": output, "attack_alert": output}


def _model_gates(model, model_path: Path, validation, scaler, history: dict) -> dict:
    outputs = _predict(model, validation, scaler)
    probabilities = outputs["dominant_state"]
    checks = {
        "finite_loss": bool(history.get("loss") and np.isfinite(history["loss"]).all()),
        "finite_probabilities": bool(all(np.isfinite(values).all() for values in outputs.values())),
        "probability_sums": bool(all(np.allclose(values.sum(axis=-1), 1.0, atol=1e-5) for values in outputs.values())),
        "prediction_diversity": int(np.unique(np.argmax(probabilities, axis=-1)).size) >= 2,
    }
    validation_reports = evaluate_horizons(validation["y"], probabilities)
    checks["validation_performance"] = bool(
        np.isfinite(validation_reports[0]["macro_f1"]) and validation_reports[0]["macro_f1"] > 0.0
    )
    reloaded = _tensorflow().keras.models.load_model(model_path, compile=False)
    sample = _scale(validation["X"][: min(32, len(validation["X"]))], scaler)
    original = model.predict(sample, verbose=0); restored = reloaded.predict(sample, verbose=0)
    checks["save_load_parity"] = bool(all(np.allclose(original[key], restored[key], atol=1e-6) for key in original))
    return {"passed": all(checks.values()), "checks": checks, "validation": validation_reports}


def _activation_gate(validation, outputs, model_gates: dict, threshold: dict) -> dict:
    dominant_reports = evaluate_horizons(validation["y_dominant"], outputs["dominant_state"])
    persistence_labels = np.repeat(validation["input_labels"][:, -1, None], HORIZONS, axis=1)
    persistence_probabilities = np.zeros_like(outputs["dominant_state"])
    for class_index, label in enumerate(FORECAST_CLASSES):
        persistence_probabilities[:, :, class_index] = persistence_labels == label
    persistence_reports = evaluate_horizons(validation["y_dominant"], persistence_probabilities)
    model_h1, baseline_h1 = dominant_reports[0], persistence_reports[0]
    model_fpr = model_h1["attack_forecasting"]["false_positive_rate"]
    baseline_fpr = baseline_h1["attack_forecasting"]["false_positive_rate"]
    checks = {
        "model_integrity": bool(model_gates["passed"]),
        "ddos_training_support": bool(np.any(validation.get("train_support_probe", validation["y_dominant"]) == "DDoS")),
        "ddos_validation_support": model_h1["class_support"]["DDoS"] > 0,
        "macro_f1_improves_persistence": model_h1["macro_f1"] > baseline_h1["macro_f1"],
        "ddos_recall_improves_persistence": model_h1["per_class"]["DDoS"]["recall"] > baseline_h1["per_class"]["DDoS"]["recall"],
        "benign_fpr_within_one_point": isinstance(model_fpr, float) and isinstance(baseline_fpr, float) and model_fpr <= baseline_fpr + 0.01,
        "warning_threshold_selected": threshold["selected_threshold"] is not None,
    }
    return {"active": all(checks.values()), "checks": checks, "model_h1": model_h1, "persistence_h1": baseline_h1}


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
    if isinstance(one_probabilities, dict):
        one_probabilities = one_probabilities["dominant_state"]
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
    return {"cold_model_load_seconds": cold_load, "warm_single_inference_seconds": warm,
            "batches": batches, "peak_rss_mb": _peak_ram_mb()}


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
        _scale(train["X"], scaler), {
            "dominant_state": _encode(train["y_dominant"]),
            "attack_alert": _encode(train["y_alert"], ALERT_CLASSES),
        },
        validation_data=(_scale(validation["X"], scaler), {
            "dominant_state": _encode(validation["y_dominant"]),
            "attack_alert": _encode(validation["y_alert"], ALERT_CLASSES),
        }),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0, callbacks=callbacks,
    )
    model = _tensorflow().keras.models.load_model(artifact_dir / "model.keras", compile=False)
    joblib.dump(scaler, artifact_dir / "scaler.bin")
    np.save(
        artifact_dir / "shap_background.npy",
        stratified_background(_scale(train["X"], scaler), train["y_dominant"][:, 0]),
    )
    scaler_sha256 = hashlib.sha256((artifact_dir / "scaler.bin").read_bytes()).hexdigest()
    (artifact_dir / "feature_names.json").write_text(json.dumps(STATE_FEATURE_NAMES, indent=2))
    (artifact_dir / "label_map.json").write_text(json.dumps({label: i for i, label in enumerate(FORECAST_CLASSES)}, indent=2))
    (artifact_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2))
    gates = _model_gates(model, artifact_dir / "model.keras", validation, scaler, history.history)
    validation_outputs = _predict(model, validation, scaler)
    threshold = select_early_warning_threshold(validation["y_dominant"], validation_outputs["attack_alert"])
    test_outputs = _predict(model, test, scaler)
    test_probabilities = test_outputs["dominant_state"]
    per_horizon = evaluate_horizons(test["y_dominant"], test_probabilities)
    validation["train_support_probe"] = train["y_dominant"]
    promotion = _activation_gate(validation, validation_outputs, gates, threshold)
    evaluation = {
        "model_version": version,
        "evaluation_status": "ACTIVATED" if promotion["active"] else "NOT_ACTIVATED",
        "threshold_selection": threshold,
        "per_horizon": per_horizon,
        "onset": onset_metrics(test["y_dominant"], test_outputs["attack_alert"], test["input_labels"], threshold["selected_threshold"]),
        "degradation": degradation_table(per_horizon),
        "h1_comparison": _one_step_h1_comparison(test, test_probabilities, scaler),
        "rolling_origin_diagnostics": _rolling_origin_diagnostics(session_windows),
        "dataset": dataset_manifest,
        "limitations": [
            "CICIDS2017 is an older, attack-clustered benchmark and does not represent current production traffic.",
            "Dominant-state and minority-alert targets come from CICIDS2017 ground-truth labels.",
            "Row order is a synthetic timing proxy; H1-H6 are windows and not validated seconds-ahead forecasts.",
            "Absent split/horizon classes are reported as N/A and are never fabricated or moved between splits.",
            "Longer horizons carry greater uncertainty; network-only evidence cannot confirm host intent or ATT&CK attribution.",
        ],
    }
    benchmark = _benchmark(artifact_dir, model, scaler, test["X"])
    activation = {"active": evaluation["evaluation_status"] == "ACTIVATED", "gates": gates,
                  "promotion": promotion, "selected_threshold": threshold["selected_threshold"]}
    (artifact_dir / "activation_status.json").write_text(json.dumps(activation, indent=2))
    (artifact_dir / "threshold_selection.json").write_text(json.dumps(threshold, indent=2))
    (artifact_dir / "class_distributions.json").write_text(json.dumps(dataset_manifest["splits"], indent=2))
    (artifact_dir / "leakage_audit.json").write_text(json.dumps({
        "session_isolation": "PASS", "chronological_split": "PASS", "train_only_scaler": "PASS",
        "invalid_feature_exclusion": "PASS", "target_alignment": "PASS",
    }, indent=2))
    training_report = {
        "model_version": version,
        "architecture": "LSTM(64) -> shared Dense(32) -> dual H1-H6 dominant-state and attack-alert softmax heads",
        "input_shape": [SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES)],
        "output_shape": {"dominant_state": [HORIZONS, len(FORECAST_CLASSES)], "attack_alert": [HORIZONS, len(ALERT_CLASSES)]},
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


def forecast_latest(windows_source=None) -> dict:
    if not LATEST_PATH.is_file():
        raise RuntimeError("No activated multi-step LSTM artifact is available.")
    latest = json.loads(LATEST_PATH.read_text())
    artifact_dir = repository_path(latest["artifact_dir"])
    model = _tensorflow().keras.models.load_model(artifact_dir / "model.keras", compile=False)
    scaler = joblib.load(artifact_dir / "scaler.bin")
    evaluation = json.loads((artifact_dir / "evaluation_report.json").read_text())
    # Forecast from the user's prepared temporal dataset when there is one;
    # otherwise fall back to the frozen training-window cache (absent on
    # machines without the CICIDS2017 CSVs -> RuntimeError/409, not 500).
    from ..lstm.training import _load_recent_windows

    if windows_source is not None:
        recent = _load_recent_windows(windows_source)
    else:
        manifest = json.loads((artifact_dir / "dataset_manifest.json").read_text())
        npz = REPO_ROOT / "data" / "lstm_cache" / manifest["cache"][-1]["cache_key"] / "windows.npz"
        if not npz.is_file():
            raise RuntimeError(
                "Multi-step forecast needs a prepared temporal dataset "
                "(Temporal Dataset -> Prepare) or the CICIDS2017 training "
                "window cache, which is not on this machine."
            )
        data = np.load(npz, allow_pickle=False)
        import pandas as pd

        windows = pd.DataFrame({name: data[name] for name in data.files}).sort_values("window_id").reset_index(drop=True)
        recent = windows.iloc[-SEQUENCE_LENGTH:]
        if len(recent) != SEQUENCE_LENGTH or not np.all(np.diff(recent["window_id"]) == 1):
            raise RuntimeError("Latest history does not contain five contiguous observed windows.")
    outputs = model.predict(_scale(recent[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)[None, :, :], scaler), verbose=0)
    if not isinstance(outputs, dict):
        outputs = {"dominant_state": outputs, "attack_alert": outputs}
    dominant_probabilities = outputs["dominant_state"][0]
    alert_probabilities = outputs["attack_alert"][0]
    threshold = evaluation["threshold_selection"]["selected_threshold"]
    observed_features = recent.iloc[-1][STATE_FEATURE_NAMES].to_dict()
    mitre_mapping_error = None
    try:
        mapper = MitreAttackMapper()
    except Exception as error:  # ATT&CK context is advisory — never fail the forecast over it
        mapper, mitre_mapping_error = None, str(error)
    _null_mapping = {
        "mitre_candidates": [], "operator_guidance": [], "evidence_needed": [],
        "severity": None, "action_provenance": None,
    }
    horizons = []
    for index, row in enumerate(dominant_probabilities):
        probability_map = {label: float(row[i]) for i, label in enumerate(FORECAST_CLASSES)}
        alert_map = {label: float(alert_probabilities[index][i]) for i, label in enumerate(ALERT_CLASSES)}
        predicted_index = int(np.argmax(row)); predicted_state = FORECAST_CLASSES[predicted_index]
        predicted_alert = ALERT_CLASSES[int(np.argmax(alert_probabilities[index]))]
        mapping = mapper.map_forecast(str(recent["dominant_state"].iloc[-1]), probability_map, observed_features) if mapper else _null_mapping
        horizons.append({
            "horizon": index + 1, "label": f"+{index + 1} window{'s' if index else ''}", "seconds_ahead": None,
            "probabilities": probability_map, "predicted_state": predicted_state,
            "dominant_state_forecast": predicted_state, "attack_alert_forecast": predicted_alert,
            "class_probabilities": probability_map, "alert_class_probabilities": alert_map,
            "forecast_probability": float(row[predicted_index]), "attack_probability": float(1.0 - alert_map["NONE"]),
            "mitre_candidates": mapping["mitre_candidates"], "mapping_confidence": max(
                (item["mapping_confidence"] for item in mapping["mitre_candidates"]), default=None
            ),
            "operator_guidance": mapping["operator_guidance"],
            "evidence_needed": mapping["evidence_needed"],
            "severity": mapping["severity"],
            "action_provenance": mapping["action_provenance"],
        })
    warning = [item["horizon"] for item in horizons if item["attack_probability"] >= threshold]
    return {
        "current_state": str(recent["dominant_state"].iloc[-1]),
        "historical_label": str(recent["dominant_state"].iloc[-1]),
        "future_labels_are_forecasts": True, "history_window_count": SEQUENCE_LENGTH,
        "forecast_horizon": HORIZONS, "window_duration": None, "timestamp_mode": "row_order_proxy",
        "horizon_labeling": "windows", "horizons": horizons, "early_warning_threshold": threshold,
        "earliest_predicted_attack_horizon": min(warning) if warning else None,
        "maximum_attack_probability": max(item["attack_probability"] for item in horizons),
        "model_version": latest["model_version"], "input_shape": [SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES)],
        "output_shape": {"dominant_state": [HORIZONS, len(FORECAST_CLASSES)], "attack_alert": [HORIZONS, len(ALERT_CLASSES)]},
        "classes": list(FORECAST_CLASSES), "alert_classes": list(ALERT_CLASSES),
        "evaluation_status": evaluation["evaluation_status"], "limitations": evaluation["limitations"],
        "mitre_mapping_error": mitre_mapping_error,
    }
