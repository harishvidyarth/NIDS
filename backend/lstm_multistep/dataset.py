from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..lstm.config import ALERT_CLASSES, FORECAST_CLASSES, SEQUENCE_LENGTH
from ..lstm.dataset import (
    artifact_fingerprints,
    contiguous_blocks,
    dataset_fingerprints,
    prepare_session_windows,
)
from ..prediction.predict import INVALID_FEATURES_LABEL
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import DATASET_MANIFEST, HORIZONS, SOURCE_FILENAMES, TEST_SESSIONS, TRAIN_SESSIONS, VALIDATION_SESSIONS, source_paths

EMBARGO_WINDOWS = HORIZONS


def build_multistep_sequences(windows: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH, horizons: int = HORIZONS) -> dict:
    values = {key: [] for key in (
        "X", "y", "y_dominant", "y_alert", "input_labels", "input_alert_labels", "history_window_ids", "target_window_ids",
        "session_id", "sample_id",
    )}
    for block in contiguous_blocks(windows):
        if len(block) < sequence_length + horizons:
            continue
        features = block[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = block["dominant_state"].to_numpy(dtype=str)
        alerts = block.get("alert_state", block["dominant_state"].where(block["dominant_state"] != "BENIGN", "NONE")).to_numpy(dtype=str)
        window_ids = block["window_id"].to_numpy(dtype=np.int64)
        session = str(block["session_id"].iloc[0])
        for start in range(len(block) - sequence_length - horizons + 1):
            target_start = start + sequence_length
            history_labels = labels[start:target_start]
            targets = labels[target_start:target_start + horizons]
            alert_targets = alerts[target_start:target_start + horizons]
            if INVALID_FEATURES_LABEL in history_labels or INVALID_FEATURES_LABEL in targets:
                continue
            if not set(history_labels).issubset(FORECAST_CLASSES) or not set(targets).issubset(FORECAST_CLASSES):
                continue
            history_ids = window_ids[start:target_start]
            target_ids = window_ids[target_start:target_start + horizons]
            if not np.all(np.diff(np.concatenate([history_ids, target_ids])) == 1):
                continue
            values["X"].append(features[start:target_start])
            values["y"].append(targets)
            values["y_dominant"].append(targets)
            values["y_alert"].append(alert_targets)
            values["input_labels"].append(history_labels)
            values["input_alert_labels"].append(alerts[start:target_start])
            values["history_window_ids"].append(history_ids)
            values["target_window_ids"].append(target_ids)
            values["session_id"].append(session)
            values["sample_id"].append(f"{session}:{int(history_ids[0])}:{int(target_ids[-1])}")
    feature_count = len(STATE_FEATURE_NAMES)
    return {
        "X": np.asarray(values["X"], dtype=np.float32).reshape(-1, sequence_length, feature_count),
        "y": np.asarray(values["y"], dtype=str).reshape(-1, horizons),
        "y_dominant": np.asarray(values["y_dominant"], dtype=str).reshape(-1, horizons),
        "y_alert": np.asarray(values["y_alert"], dtype=str).reshape(-1, horizons),
        "input_labels": np.asarray(values["input_labels"], dtype=str).reshape(-1, sequence_length),
        "input_alert_labels": np.asarray(values["input_alert_labels"], dtype=str).reshape(-1, sequence_length),
        "history_window_ids": np.asarray(values["history_window_ids"], dtype=np.int64).reshape(-1, sequence_length),
        "target_window_ids": np.asarray(values["target_window_ids"], dtype=np.int64).reshape(-1, horizons),
        "session_id": np.asarray(values["session_id"], dtype=str),
        "sample_id": np.asarray(values["sample_id"], dtype=str),
    }


def concat_sequence_sets(items: list[dict]) -> dict:
    if not items:
        empty = pd.DataFrame(columns=["window_id", "dominant_state", "session_id", *STATE_FEATURE_NAMES])
        return build_multistep_sequences(empty)
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in items[0]}


def split_session_windows(session_windows: list[pd.DataFrame]) -> dict[str, list[pd.DataFrame]]:
    by_name = {f"{str(frame['session_id'].iloc[0])}.csv": frame for frame in session_windows}
    ddos = by_name[TEST_SESSIONS[0]].sort_values("window_id").reset_index(drop=True)
    train_end = int(len(ddos) * 0.60)
    validation_start = train_end + EMBARGO_WINDOWS
    validation_end = int(len(ddos) * 0.80)
    test_start = validation_end + EMBARGO_WINDOWS
    minimum = SEQUENCE_LENGTH + HORIZONS
    partitions = {
        "train": ddos.iloc[:train_end].copy(),
        "validation": ddos.iloc[validation_start:validation_end].copy(),
        "test": ddos.iloc[test_start:].copy(),
    }
    if any(len(frame) < minimum for frame in partitions.values()):
        raise RuntimeError("Friday DDoS is too short for chronological train/validation/test partitions with embargo gaps.")
    return {
        "train": [by_name[name] for name in TRAIN_SESSIONS] + [partitions["train"]],
        "validation": [by_name[name] for name in VALIDATION_SESSIONS] + [partitions["validation"]],
        "test": [partitions["test"]],
    }


def class_distribution(labels: np.ndarray) -> dict[str, list[int]]:
    return {
        label: [int(np.sum(labels[:, horizon] == label)) for horizon in range(HORIZONS)]
        for label in FORECAST_CLASSES
    }


def validate_class_support(train: dict, validation: dict) -> None:
    train_labels = np.asarray(train["y_dominant"]).reshape(-1)
    missing = [label for label in FORECAST_CLASSES if not np.any(train_labels == label)]
    if missing:
        raise RuntimeError(f"Required class(es) have zero training support: {', '.join(missing)}")
    validation_labels = np.asarray(validation["y_dominant"]).reshape(-1)
    if not np.any(validation_labels == "DDoS"):
        raise RuntimeError("DDoS has zero validation support; activation is forbidden.")


def prepare_multistep_dataset(force_rebuild: bool = False, progress=lambda **kwargs: None) -> tuple[dict, dict, list[pd.DataFrame]]:
    paths = source_paths()
    fingerprints = dataset_fingerprints(paths)
    ann_fingerprints = {}
    session_windows = []
    cache = []
    raw_label_diagnostics = {}
    for path, fingerprint in zip(paths, fingerprints):
        windows, metadata = prepare_session_windows(
            path, fingerprint, ann_fingerprints, force_rebuild=force_rebuild, progress=progress
        )
        session_windows.append(windows)
        cache.append(metadata)
        label_column = next((column for column in pd.read_csv(path, nrows=0).columns if str(column).strip().lower() == "label"), None)
        counts = Counter()
        if label_column:
            for chunk in pd.read_csv(path, usecols=[label_column], chunksize=100_000, low_memory=False):
                counts.update(chunk[label_column].astype(str).str.strip().value_counts().to_dict())
        raw_label_diagnostics[path.name] = dict(sorted(counts.items()))
    split_windows = split_session_windows(session_windows)
    sequences = {
        name: concat_sequence_sets([build_multistep_sequences(frame) for frame in frames])
        for name, frames in split_windows.items()
    }
    validate_class_support(sequences["train"], sequences["validation"])
    identities = {
        name: {
            "samples": int(len(item["X"])),
            "sessions": sorted(set(item["session_id"].tolist())),
            "sample_ids_sha256": __import__("hashlib").sha256("\n".join(item["sample_id"]).encode()).hexdigest(),
            "dominant_class_distribution_by_horizon": class_distribution(item["y_dominant"]),
            "alert_class_distribution_by_horizon": {
                label: [int(np.sum(item["y_alert"][:, horizon] == label)) for horizon in range(HORIZONS)]
                for label in ALERT_CLASSES
            },
        }
        for name, item in sequences.items()
    }
    manifest = {
        "sources_in_official_order": list(SOURCE_FILENAMES),
        "source_fingerprints": fingerprints,
        "raw_cicids_label_diagnostics": raw_label_diagnostics,
        "target_policy": "Targets come only from the CICIDS2017 ground-truth Label column; ANN predictions are not training targets.",
        "target_provenance": "cicids2017_ground_truth_label",
        "split": {
            "train": list(TRAIN_SESSIONS),
            "validation": list(VALIDATION_SESSIONS),
            "test": list(TEST_SESSIONS),
            "ddos_partition": "chronological 60/20/20 with six-window embargo gaps",
        },
        "shapes": {"history": [SEQUENCE_LENGTH, len(STATE_FEATURE_NAMES)], "targets": [HORIZONS]},
        "feature_order": list(STATE_FEATURE_NAMES),
        "label_order": list(FORECAST_CLASSES),
        "timestamp_policy": "row_order_proxy; one source row per proxy unit; horizons are windows, not seconds",
        "invalid_features_policy": "exclude any non-contiguous sample or history/target label outside the four ANN states",
        "sample_identity": "session_id:first_history_window:last_target_window",
        "cache": cache,
        "splits": identities,
    }
    DATASET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DATASET_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return sequences, manifest, session_windows
