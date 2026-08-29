from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..lstm.config import FORECAST_CLASSES, SEQUENCE_LENGTH
from ..lstm.dataset import (
    artifact_fingerprints,
    contiguous_blocks,
    dataset_fingerprints,
    prepare_session_windows,
)
from ..prediction.predict import INVALID_FEATURES_LABEL
from ..temporal.schema import STATE_FEATURE_NAMES
from .config import DATASET_MANIFEST, HORIZONS, SOURCE_FILENAMES, TEST_SESSIONS, TRAIN_SESSIONS, VALIDATION_SESSIONS, source_paths


def build_multistep_sequences(windows: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH, horizons: int = HORIZONS) -> dict:
    values = {key: [] for key in (
        "X", "y", "input_labels", "history_window_ids", "target_window_ids",
        "session_id", "sample_id",
    )}
    for block in contiguous_blocks(windows):
        if len(block) < sequence_length + horizons:
            continue
        features = block[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = block["dominant_state"].to_numpy(dtype=str)
        window_ids = block["window_id"].to_numpy(dtype=np.int64)
        session = str(block["session_id"].iloc[0])
        for start in range(len(block) - sequence_length - horizons + 1):
            target_start = start + sequence_length
            history_labels = labels[start:target_start]
            targets = labels[target_start:target_start + horizons]
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
            values["input_labels"].append(history_labels)
            values["history_window_ids"].append(history_ids)
            values["target_window_ids"].append(target_ids)
            values["session_id"].append(session)
            values["sample_id"].append(f"{session}:{int(history_ids[0])}:{int(target_ids[-1])}")
    feature_count = len(STATE_FEATURE_NAMES)
    return {
        "X": np.asarray(values["X"], dtype=np.float32).reshape(-1, sequence_length, feature_count),
        "y": np.asarray(values["y"], dtype=str).reshape(-1, horizons),
        "input_labels": np.asarray(values["input_labels"], dtype=str).reshape(-1, sequence_length),
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
    return {
        "train": [by_name[name] for name in TRAIN_SESSIONS],
        "validation": [by_name[name] for name in VALIDATION_SESSIONS],
        "test": [by_name[name] for name in TEST_SESSIONS],
    }


def class_distribution(labels: np.ndarray) -> dict[str, list[int]]:
    return {
        label: [int(np.sum(labels[:, horizon] == label)) for horizon in range(HORIZONS)]
        for label in FORECAST_CLASSES
    }


def prepare_multistep_dataset(force_rebuild: bool = False, progress=lambda **kwargs: None) -> tuple[dict, dict, list[pd.DataFrame]]:
    paths = source_paths()
    fingerprints = dataset_fingerprints(paths)
    ann_fingerprints = artifact_fingerprints()
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
    identities = {
        name: {
            "samples": int(len(item["X"])),
            "sessions": sorted(set(item["session_id"].tolist())),
            "sample_ids_sha256": __import__("hashlib").sha256("\n".join(item["sample_id"]).encode()).hexdigest(),
            "class_distribution_by_horizon": class_distribution(item["y"]),
        }
        for name, item in sequences.items()
    }
    manifest = {
        "sources_in_official_order": list(SOURCE_FILENAMES),
        "source_fingerprints": fingerprints,
        "ann_artifact_fingerprints": ann_fingerprints,
        "raw_cicids_label_diagnostics": raw_label_diagnostics,
        "target_policy": "Targets are the existing ANN's four-class output; raw CICIDS labels are diagnostics only.",
        "split": {
            "train": list(TRAIN_SESSIONS),
            "validation": list(VALIDATION_SESSIONS),
            "test": list(TEST_SESSIONS),
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
