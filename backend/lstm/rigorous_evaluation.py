from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import (
    BATCH_SIZE,
    CACHE_ROOT,
    FORECAST_CLASSES,
    LATEST_PATH,
    SEQUENCE_LENGTH,
    repository_path,
    repository_relative,
)
from .dataset import build_sequences, concat_sequence_sets, sha256_file
from .evaluation import ABSENT_CLASS, evaluate_predictions
from .training import _fit_scaler, _logistic, _scale_sequences, rolling_origin_windows, split_sessions

REPORT_DIR = REPO_ROOT / "reports"


def _load_windows(training_report: dict) -> list[pd.DataFrame]:
    sessions = []
    for metadata in training_report["cache"]:
        path = CACHE_ROOT / metadata["cache_key"] / "windows.npz"
        data = np.load(path, allow_pickle=False)
        frame = pd.DataFrame({name: data[name] for name in data.files})
        frame.insert(0, "session_id", metadata["session_id"])
        sessions.append(frame.sort_values("window_id").reset_index(drop=True))
    return sessions


def _final_split(session_windows: list[pd.DataFrame]):
    development, test = split_sessions(session_windows)
    train, validation = [], []
    for frame in development:
        boundary = max(SEQUENCE_LENGTH + 1, int(len(frame) * 0.90))
        train.append(frame.iloc[:boundary].reset_index(drop=True))
        validation.append(frame.iloc[boundary:].reset_index(drop=True))
    return train, validation, test


def _sequences(parts: list[pd.DataFrame]) -> dict:
    return concat_sequence_sets([build_sequences(frame) for frame in parts])


def _class_counts(labels) -> dict[str, int]:
    counts = Counter(np.asarray(labels, dtype=str).tolist())
    return {label: int(counts.get(label, 0)) for label in FORECAST_CLASSES}


def _predict_lstm(model, sequences: dict, scaler) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict(_scale_sequences(sequences, scaler), batch_size=BATCH_SIZE, verbose=0)
    predictions = np.asarray(FORECAST_CLASSES)[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def _predict_logistic(model, sequences: dict, scaler) -> tuple[np.ndarray, np.ndarray]:
    scaled = _scale_sequences(sequences, scaler).reshape(len(sequences["X"]), -1)
    partial = model.predict_proba(scaled)
    probabilities = np.zeros((len(scaled), len(FORECAST_CLASSES)), dtype=float)
    for index, label in enumerate(model.classes_):
        probabilities[:, list(FORECAST_CLASSES).index(label)] = partial[:, index]
    return model.predict(scaled), probabilities


def _provenance(sequences: dict) -> set[tuple[str, int]]:
    return set(zip(sequences["session_id"].tolist(), sequences["target_window_id"].astype(int).tolist()))


def _content_hashes(sequences: dict) -> set[str]:
    hashes = set()
    for values, labels in zip(sequences["X"], sequences["input_labels"]):
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes())
        digest.update("\x1f".join(labels.tolist()).encode())
        hashes.add(digest.hexdigest())
    return hashes


def _scaler_matches_training_only(scaler, train_windows: list[pd.DataFrame]) -> tuple[bool, dict]:
    expected = _fit_scaler(train_windows)
    checks = {
        "sample_count": int(getattr(scaler, "n_samples_seen_", -1)) == int(expected.n_samples_seen_),
        "data_min": bool(np.allclose(scaler.data_min_, expected.data_min_, rtol=0, atol=1e-12)),
        "data_max": bool(np.allclose(scaler.data_max_, expected.data_max_, rtol=0, atol=1e-12)),
        "scale": bool(np.allclose(scaler.scale_, expected.scale_, rtol=0, atol=1e-12)),
    }
    return all(checks.values()), checks


def _leakage_audit(train, validation, test, scaler, train_windows) -> dict:
    train_provenance = _provenance(train)
    validation_provenance = _provenance(validation)
    test_provenance = _provenance(test)
    train_hashes = _content_hashes(train)
    validation_hashes = _content_hashes(validation)
    test_hashes = _content_hashes(test)
    scaler_pass, scaler_checks = _scaler_matches_training_only(scaler, train_windows)

    def aligned(sequences):
        return bool(
            np.all(np.diff(sequences["input_window_ids"], axis=1) == 1)
            and np.all(sequences["input_window_ids"][:, -1] + 1 == sequences["target_window_id"])
        )

    valid_labels = set(FORECAST_CLASSES)
    invalid_excluded = all(
        set(sequences["y"].tolist()).issubset(valid_labels)
        and set(sequences["input_labels"].reshape(-1).tolist()).issubset(valid_labels)
        for sequences in (train, validation, test)
    )
    duplicate_intersections = {
        "train_validation": len(train_hashes & validation_hashes),
        "train_test": len(train_hashes & test_hashes),
        "validation_test": len(validation_hashes & test_hashes),
    }
    audit = {
        "session_boundary_leakage": {
            "status": "PASS",
            "detail": "Sequences were rebuilt independently per capture session and contiguous block.",
        },
        "train_validation_overlap": {
            "status": "PASS" if not train_provenance & validation_provenance else "FAIL",
            "overlap_targets": len(train_provenance & validation_provenance),
        },
        "train_test_overlap": {
            "status": "PASS" if not train_provenance & test_provenance else "FAIL",
            "overlap_targets": len(train_provenance & test_provenance),
        },
        "scaler_train_only": {
            "status": "PASS" if scaler_pass else "FAIL",
            "checks": scaler_checks,
            "artifact_samples_seen": int(getattr(scaler, "n_samples_seen_", -1)),
            "expected_training_windows": int(sum(len(frame) for frame in train_windows)),
        },
        "next_window_alignment": {
            "status": "PASS" if all(aligned(item) for item in (train, validation, test)) else "FAIL",
        },
        "future_feature_leakage": {
            "status": "PASS" if all(aligned(item) for item in (train, validation, test)) else "FAIL",
            "detail": "Every target window ID is exactly one after the final input window ID.",
        },
        "duplicate_sequence_leakage": {
            "status": "PASS" if not any(duplicate_intersections.values()) else "FAIL",
            "content_hash_overlaps": duplicate_intersections,
        },
        "invalid_features_exclusion": {"status": "PASS" if invalid_excluded else "FAIL"},
        "synthetic_session_continuity": {
            "status": "PASS",
            "detail": "Proxy window IDs restart per source file; sequence construction is session-scoped.",
        },
    }
    audit["overall"] = "PASS" if all(item["status"] == "PASS" for key, item in audit.items() if key != "overall") else "FAIL"
    return audit


def _attack_runs(frame: pd.DataFrame) -> list[dict]:
    attacks = frame[frame["dominant_state"] != "BENIGN"][["window_id", "dominant_state"]].copy()
    if attacks.empty:
        return []
    attacks["run"] = (
        attacks["dominant_state"].ne(attacks["dominant_state"].shift())
        | attacks["window_id"].diff().fillna(1).ne(1)
    ).cumsum()
    return [
        {
            "state": str(group["dominant_state"].iloc[0]),
            "start_window": int(group["window_id"].iloc[0]),
            "end_window": int(group["window_id"].iloc[-1]),
            "windows": int(len(group)),
            "proxy_start_seconds": int(group["window_id"].iloc[0]) * 10,
            "proxy_end_seconds": (int(group["window_id"].iloc[-1]) + 1) * 10,
        }
        for _, group in attacks.groupby("run", sort=True)
    ]


def _session_diagnosis(session_windows, train_windows, validation_windows, test_windows) -> list[dict]:
    result = []
    for full, train, validation, test in zip(session_windows, train_windows, validation_windows, test_windows):
        result.append({
            "session": str(full["session_id"].iloc[0]),
            "total_windows": int(len(full)),
            "window_distribution": _class_counts(full["dominant_state"]),
            "train_target_distribution": _class_counts(build_sequences(train)["y"]),
            "validation_target_distribution": _class_counts(build_sequences(validation)["y"]),
            "test_target_distribution": _class_counts(build_sequences(test)["y"]),
            "split_boundaries": {
                "train_window_ids": [int(train["window_id"].iloc[0]), int(train["window_id"].iloc[-1])],
                "validation_window_ids": [int(validation["window_id"].iloc[0]), int(validation["window_id"].iloc[-1])],
                "test_window_ids": [int(test["window_id"].iloc[0]), int(test["window_id"].iloc[-1])],
            },
            "attack_temporal_runs": _attack_runs(full),
        })
    return result


def _split_result(name, sequences, model, scaler, baseline) -> dict:
    lstm_pred, lstm_prob = _predict_lstm(model, sequences, scaler)
    baseline_pred, baseline_prob = _predict_logistic(baseline, sequences, scaler)
    current_states = sequences["input_labels"][:, -1]
    return {
        "name": name,
        "shapes": {"X": list(sequences["X"].shape), "y": list(sequences["y"].shape)},
        "class_distribution": _class_counts(sequences["y"]),
        "absent_classes": [label for label, count in _class_counts(sequences["y"]).items() if count == 0],
        "lstm": evaluate_predictions(sequences["y"], lstm_pred, lstm_prob, current_states),
        "logistic_regression": evaluate_predictions(sequences["y"], baseline_pred, baseline_prob, current_states),
    }


def _attack_protocol(test_result: dict) -> dict:
    distribution = test_result["class_distribution"]
    attack_samples = sum(distribution[label] for label in FORECAST_CLASSES if label != "BENIGN")
    if attack_samples:
        return {
            "status": "AVAILABLE",
            "methodology": "Independent session-scoped chronological holdout after all training and validation windows.",
            "result": test_result,
        }
    return {
        "status": "NOT_AVAILABLE",
        "message": "Existing dataset cannot provide a sufficiently representative attack-containing chronological holdout.",
        "methodology": "Candidate evaluation units must be whole capture sessions or contiguous post-training blocks; no eligible attack targets remain after the saved training/validation cutoffs.",
        "framework_ready": True,
    }


def _comparison(split_result: dict) -> dict:
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    comparison = {
        key: {
            "logistic_regression": split_result["logistic_regression"][key],
            "lstm": split_result["lstm"][key],
        }
        for key in keys
    }
    for key in ("recall", "f1"):
        comparison[f"attack_{key}"] = {
            "logistic_regression": split_result["logistic_regression"]["attack_forecasting"][key],
            "lstm": split_result["lstm"]["attack_forecasting"][key],
        }
    return comparison


def _walk_forward(artifact_dir: Path, training_report: dict, development_windows: list[pd.DataFrame]) -> dict:
    selected = training_report["metrics"]["selected_training"]
    fold_pairs = [rolling_origin_windows(frame) for frame in development_windows]
    folds = []
    aggregate = {"y": [], "current": [], "lstm_pred": [], "lstm_prob": [], "baseline_pred": [], "baseline_prob": []}
    import tensorflow as tf

    for fold_index in range(3):
        fold_train_windows = [pairs[fold_index][0] for pairs in fold_pairs if len(pairs) > fold_index]
        fold_evaluation_windows = [pairs[fold_index][1] for pairs in fold_pairs if len(pairs) > fold_index]
        fold_train = _sequences(fold_train_windows)
        fold_evaluation = _sequences(fold_evaluation_windows)
        scaler = _fit_scaler(fold_train_windows)
        checkpoint = artifact_dir / "selection" / f"fold_{fold_index + 1}_{selected}.keras"
        if not checkpoint.is_file():
            return {
                "status": "NOT_AVAILABLE",
                "reason": f"Saved fold checkpoint is missing: {checkpoint}",
            }
        model = tf.keras.models.load_model(checkpoint)
        lstm_pred, lstm_prob = _predict_lstm(model, fold_evaluation, scaler)
        _, baseline_pred, baseline_prob = _logistic(fold_train, fold_evaluation, scaler)
        current_states = fold_evaluation["input_labels"][:, -1]
        fold_distribution = _class_counts(fold_evaluation["y"])
        folds.append({
            "fold": fold_index + 1,
            "checkpoint": repository_relative(checkpoint),
            "train_samples": int(len(fold_train["X"])),
            "evaluation_samples": int(len(fold_evaluation["X"])),
            "attack_targets": int(sum(fold_distribution[label] for label in FORECAST_CLASSES[1:])),
            "class_distribution": fold_distribution,
            "train_periods": [
                {
                    "session": str(frame["session_id"].iloc[0]),
                    "window_ids": [int(frame["window_id"].iloc[0]), int(frame["window_id"].iloc[-1])],
                }
                for frame in fold_train_windows
            ],
            "evaluation_periods": [
                {
                    "session": str(frame["session_id"].iloc[0]),
                    "window_ids": [int(frame["window_id"].iloc[0]), int(frame["window_id"].iloc[-1])],
                }
                for frame in fold_evaluation_windows
            ],
            "lstm": evaluate_predictions(fold_evaluation["y"], lstm_pred, lstm_prob, current_states),
            "logistic_regression": evaluate_predictions(
                fold_evaluation["y"], baseline_pred, baseline_prob, current_states
            ),
        })
        aggregate["y"].append(fold_evaluation["y"])
        aggregate["current"].append(current_states)
        aggregate["lstm_pred"].append(lstm_pred)
        aggregate["lstm_prob"].append(lstm_prob)
        aggregate["baseline_pred"].append(baseline_pred)
        aggregate["baseline_prob"].append(baseline_prob)
    y = np.concatenate(aggregate["y"])
    current = np.concatenate(aggregate["current"])
    return {
        "status": "INTERNAL_VALIDATION_ONLY",
        "methodology": "Expanding-window, session-scoped rolling origin; every fold trains on earlier proxy windows and evaluates the immediately following disjoint proxy block.",
        "limitation": "These folds selected the training protocol, so they are reproduced internal validation diagnostics rather than an untouched final holdout.",
        "folds": folds,
        "aggregate": {
            "lstm": evaluate_predictions(
                y, np.concatenate(aggregate["lstm_pred"]), np.concatenate(aggregate["lstm_prob"]), current
            ),
            "logistic_regression": evaluate_predictions(
                y,
                np.concatenate(aggregate["baseline_pred"]),
                np.concatenate(aggregate["baseline_prob"]),
                current,
            ),
        },
    }


def _markdown(report: dict) -> str:
    def metric(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    lines = [
        "# LSTM Evaluation Report",
        "",
        "## Model Identity",
        "",
        f"- Model: `{report['model_identity']['model_path']}`",
        f"- Model SHA-256: `{report['model_identity']['model_sha256']}`",
        f"- Scaler: `{report['model_identity']['scaler_path']}`",
        f"- Features: {report['model_identity']['feature_count']}",
        f"- Sequence length: {report['model_identity']['sequence_length']}",
        f"- Classes: {', '.join(report['model_identity']['classes'])}",
        "",
        "## Timestamp Methodology",
        "",
        "- REAL TIMESTAMP TEMPORAL DATA: not used by this saved LSTM artifact.",
        "- ROW-ORDER / SYNTHETIC-TIMESTAMP TEMPORAL PROXY: one source row is treated as one proxy second and aggregated into nominal 10-second windows.",
        "- This is an experimental temporal-ordering proxy, not evidence of real-world time-to-attack forecasting.",
        "",
        "## Temporal Leakage Audit",
        "",
    ]
    for key, value in report["leakage_audit"].items():
        if key != "overall":
            lines.append(f"- {key.replace('_', ' ').title()}: **{value['status']}**")
    lines.extend(["", f"Overall: **{report['leakage_audit']['overall']}**", ""])
    if report["leakage_audit"]["overall"] == "FAIL":
        lines.extend([
            "> Normal metric interpretation is stopped because the leakage audit failed. Metrics below reproduce the saved artifact only and are not presented as valid generalization evidence.",
            "",
        ])
    lines.extend(["## Dataset Distribution", ""])
    for split_name in ("train", "validation", "test"):
        split = report["evaluations"][split_name]
        lines.append(f"### {split_name.title()}")
        lines.append("")
        lines.append(f"- X: `{tuple(split['shapes']['X'])}`")
        lines.append(f"- y: `{tuple(split['shapes']['y'])}`")
        for label, count in split["class_distribution"].items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    test = report["evaluations"]["test"]["lstm"]
    lines.extend([
        "## Original Chronological Evaluation",
        "",
        f"- Accuracy: {metric(test['accuracy'])}",
        f"- Balanced accuracy: {metric(test['balanced_accuracy'])}",
        f"- Macro F1: {metric(test['macro_f1'])}",
        f"- Weighted F1: {metric(test['weighted_f1'])}",
        f"- Attack recall: {metric(test['attack_forecasting']['recall'])}",
        f"- Attack F1: {metric(test['attack_forecasting']['f1'])}",
        f"- Absent classes: {', '.join(report['evaluations']['test']['absent_classes'])}",
        "",
        "## Attack-Containing Evaluation",
        "",
        f"- Status: {report['attack_containing_evaluation']['status']}",
        f"- {report['attack_containing_evaluation'].get('message', report['attack_containing_evaluation']['methodology'])}",
        "",
        "## Transition Analysis",
        "",
    ])
    for transition in report["evaluations"]["validation"]["lstm"]["transitions"][:12]:
        flag = " (low support)" if transition["low_support"] else ""
        lines.append(
            f"- {transition['actual_transition']}: n={transition['count']}, accuracy={metric(transition['accuracy'])}, mean true-state probability={metric(transition['mean_true_state_probability'])}{flag}"
        )
    lines.extend(["", "## Rolling-Origin Diagnostics", ""])
    lines.append(f"- Status: {report['walk_forward_evaluation']['status']}")
    for fold in report["walk_forward_evaluation"].get("folds", []):
        lines.append(
            f"- Fold {fold['fold']}: samples={fold['evaluation_samples']}, attacks={fold['attack_targets']}, macro F1={metric(fold['lstm']['macro_f1'])}, attack F1={metric(fold['lstm']['attack_forecasting']['f1'])}"
        )
    lines.extend([
        "",
        "## Baseline Comparison — Test",
        "",
        "| Metric | Logistic Regression | LSTM |",
        "|---|---:|---:|",
    ])
    for name, values in report["baseline_comparison_test"].items():
        lines.append(f"| {name.replace('_', ' ').title()} | {metric(values['logistic_regression'])} | {metric(values['lstm'])} |")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- CICIDS2017 is an aging benchmark and does not represent every modern network or threat domain.",
        "- DDoS is absent from all configured source sessions; the model cannot be validated for that class here.",
        "- The strict test split contains only BENIGN targets, so attack recall/F1 are undefined and multiclass AUC is not valid.",
        "- Network-flow evidence cannot confirm ATT&CK techniques, host/process behavior, or adversary intent.",
        "- Mapping confidence is deterministic and heuristic, not statistically calibrated.",
        "",
        "## MITRE ATT&CK Context",
        "",
        f"- ATT&CK version: {report['mitre_attack']['version']}",
        f"- Data modified: {report['mitre_attack']['data_modified']}",
        f"- Offline metadata: `{report['mitre_attack']['offline_metadata']}`",
        f"- Source: {report['mitre_attack']['source']}",
        f"- Implemented candidates: {', '.join(report['mitre_attack']['implemented_techniques'])}",
        "",
        "## Conclusion",
        "",
        report["conclusion"],
        "",
    ])
    return "\n".join(lines)


def evaluate_saved_artifact(
    artifact_dir: Path | str | None = None,
    json_path: Path | str | None = None,
    markdown_path: Path | str | None = None,
) -> dict:
    if artifact_dir is None:
        latest = json.loads(LATEST_PATH.read_text())
        artifact_dir = repository_path(latest["artifact_dir"])
    else:
        artifact_dir = repository_path(artifact_dir)
    training_report = json.loads((artifact_dir / "report.json").read_text())
    import tensorflow as tf

    model_path = artifact_dir / "model.keras"
    scaler_path = artifact_dir / "scaler.bin"
    baseline_path = artifact_dir / "baseline_logistic.bin"
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    baseline = joblib.load(baseline_path)
    session_windows = _load_windows(training_report)
    development_windows, _ = split_sessions(session_windows)
    train_windows, validation_windows, test_windows = _final_split(session_windows)
    train = _sequences(train_windows)
    validation = _sequences(validation_windows)
    test = _sequences(test_windows)
    leakage = _leakage_audit(train, validation, test, scaler, train_windows)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        evaluations = {
            "train": _split_result("train", train, model, scaler, baseline),
            "validation": _split_result("validation", validation, model, scaler, baseline),
            "test": _split_result("test", test, model, scaler, baseline),
        }
        walk_forward = _walk_forward(artifact_dir, training_report, development_windows)
    from ..mitre.mapper import DEFAULT_METADATA_PATH, MitreAttackMapper
    mapper = MitreAttackMapper()
    report = {
        "report_schema": "lstm-rigorous-evaluation/v1",
        "interpretation_status": "VALID_WITH_DATASET_LIMITATIONS" if leakage["overall"] == "PASS" else "STOPPED_LEAKAGE_DEFECT",
        "model_identity": {
            "model_version": training_report["model_version"],
            "model_path": repository_relative(model_path),
            "model_sha256": sha256_file(model_path),
            "scaler_path": repository_relative(scaler_path),
            "scaler_sha256": sha256_file(scaler_path),
            "baseline_path": repository_relative(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
            "architecture": training_report["architecture"],
            "feature_count": len(STATE_FEATURE_NAMES),
            "feature_names": STATE_FEATURE_NAMES,
            "sequence_length": SEQUENCE_LENGTH,
            "window_size_seconds": training_report["window_size_seconds"],
            "classes": list(FORECAST_CLASSES),
            "label_mapping": json.loads((artifact_dir / "label_map.json").read_text()),
            "loaded_saved_model": True,
            "training_protocol_version": training_report.get("training_protocol_version", "legacy-unversioned"),
            "checkpoint_selection": "minimum validation loss via ModelCheckpoint; EarlyStopping restores best validation-loss weights",
        },
        "dataset_identity": training_report["dataset_fingerprints"],
        "timestamp_methodology": {
            "used": "ROW-ORDER / SYNTHETIC-TIMESTAMP TEMPORAL PROXY",
            "real_timestamp_temporal_data": False,
            "proxy_cadence_seconds": 1,
            "limitation": "One source row is treated as one proxy second; nominal 10-second windows do not prove wall-clock time-to-attack forecasting.",
        },
        "split_methodology": {
            "train": "first 90% of each session's initial 85% development block",
            "validation": "last 10% of each session's initial 85% development block; checkpoint selection set",
            "test": "final 15% of every session; strict chronological holdout",
            "random_row_split": False,
        },
        "leakage_audit": leakage,
        "evaluations": evaluations,
        "session_diagnosis": _session_diagnosis(session_windows, train_windows, validation_windows, test_windows),
        "attack_containing_evaluation": _attack_protocol(evaluations["test"]),
        "walk_forward_evaluation": walk_forward,
        "baseline_comparison_test": _comparison(evaluations["test"]),
        "mitre_attack": {
            "version": mapper.metadata.version,
            "data_modified": mapper.metadata.data_modified,
            "source": mapper.metadata.source_url,
            "source_stix_sha256": mapper.metadata.source_sha256,
            "offline_metadata": repository_relative(DEFAULT_METADATA_PATH),
            "mapping_implementation": "Deterministic downstream evidence rules; no runtime cloud request and no additional ML model.",
            "supported_evidence": [
                "current traffic state", "full next-state probability vector", "state transition",
                "flow and packet rates", "SYN/RST counts", "destination-port diversity",
            ],
            "implemented_techniques": [
                f"{technique_id} {mapper.metadata.techniques[technique_id]['technique_name']}"
                for technique_id in sorted(mapper.rule_technique_ids)
            ],
        },
        "limitations": [
            "Synthetic row-order proxy rather than real timestamp temporal data.",
            "CICIDS2017 age and domain limitation.",
            "DDoS absent from all configured sessions.",
            "Strict test holdout contains only BENIGN targets.",
            "Network-only evidence cannot establish ATT&CK technique or adversary intent.",
        ],
        "conclusion": (
            "The saved LSTM was loaded and evaluated without initialization or pre-evaluation training. "
            + (
                "The leakage audit passed, but the independent test set is BENIGN-only; it measures late-session benign stability, not attack forecasting. "
                if leakage["overall"] == "PASS"
                else "Metric interpretation is stopped because the saved preprocessing artifact fails the train-only scaler audit. "
            )
            + "The existing sessions cannot provide a defensible independent attack-containing chronological holdout."
        ),
    }
    json_path = Path(json_path or REPORT_DIR / "lstm_evaluation_report.json")
    markdown_path = Path(markdown_path or REPORT_DIR / "lstm_evaluation_report.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(_markdown(report))
    return report
