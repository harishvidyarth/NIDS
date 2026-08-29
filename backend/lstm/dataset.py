from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..prediction.features import TRAINING_FEATURES, match_columns
from ..prediction.predict import CLASS_NAMES, INVALID_FEATURES_LABEL, _load_artifacts
from ..temporal.schema import (
    AGGREGATIONS,
    DST_PORT_TRAINING_FEATURE,
    STATE_FEATURE_NAMES,
    find_id_columns,
)
from .config import (
    CACHE_ROOT,
    CHUNK_SIZE,
    FORECAST_CLASSES,
    PROXY_CADENCE_SECONDS,
    SCHEMA_VERSION,
    SEQUENCE_LENGTH,
    WINDOW_SIZE_SECONDS,
)

Progress = Callable[..., None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_fingerprints() -> dict[str, str]:
    from ..prediction.predict import MODEL_PATH, SCALER_PATH

    return {
        "ann_model_sha256": sha256_file(MODEL_PATH),
        "ann_scaler_sha256": sha256_file(SCALER_PATH),
    }


def dataset_fingerprints(paths: Iterable[Path]) -> list[dict]:
    return [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def cache_identity(source: dict, artifact_hashes: dict) -> dict:
    return {
        "source": source,
        "ann_artifacts": artifact_hashes,
        "schema_version": SCHEMA_VERSION,
        "window_size_seconds": WINDOW_SIZE_SECONDS,
        "sequence_length": SEQUENCE_LENGTH,
        "proxy_cadence_seconds": PROXY_CADENCE_SECONDS,
        "state_feature_names": STATE_FEATURE_NAMES,
    }


def cache_key(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_number(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def analyze_sources(
    paths: list[Path], output_dir: Path, chunk_size: int = CHUNK_SIZE, progress: Progress | None = None
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    combined_labels = Counter()
    combined_rows = combined_invalid = combined_duplicates = 0
    schema_rows = []

    for path in paths:
        rows = invalid_values = 0
        labels = Counter()
        minimums: dict[str, float] = {}
        maximums: dict[str, float] = {}
        non_null_counts: Counter = Counter()
        unique_samples: dict[str, set] = {}
        row_hashes = []
        columns = None
        started = time.perf_counter()

        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            if columns is None:
                columns = [str(column) for column in chunk.columns]
                unique_samples = {column: set() for column in columns}
            rows += len(chunk)
            normalized = chunk.replace([np.inf, -np.inf, "Infinity", "-Infinity"], np.nan)
            invalid_values += int(normalized.isna().sum().sum())
            label_col = next((c for c in chunk.columns if str(c).strip().lower() == "label"), None)
            if label_col is not None:
                labels.update(chunk[label_col].astype(str).str.strip().value_counts().to_dict())
            row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64))

            for column in chunk.columns:
                values = pd.to_numeric(normalized[column], errors="coerce")
                valid = values.dropna()
                non_null_counts[str(column)] += int(chunk[column].notna().sum())
                if not valid.empty:
                    current_min = float(valid.min())
                    current_max = float(valid.max())
                    minimums[str(column)] = min(minimums.get(str(column), current_min), current_min)
                    maximums[str(column)] = max(maximums.get(str(column), current_max), current_max)
                if len(unique_samples[str(column)]) <= 2:
                    unique_samples[str(column)].update(chunk[column].dropna().astype(str).head(3).tolist())

            if progress:
                progress(stage="analyzing", rows_processed=combined_rows + rows)

        hashes = np.concatenate(row_hashes) if row_hashes else np.empty(0, dtype=np.uint64)
        duplicates = int(len(hashes) - len(np.unique(hashes)))
        constants = [column for column, values in unique_samples.items() if len(values) <= 1]
        report = {
            "source": path.name,
            "rows": rows,
            "columns": len(columns or []),
            "schema": columns or [],
            "labels": dict(sorted(labels.items())),
            "invalid_values": invalid_values,
            "duplicate_rows": duplicates,
            "constant_columns": constants,
            "numeric_ranges": {
                column: {"min": _json_number(minimums.get(column)), "max": _json_number(maximums.get(column))}
                for column in (columns or []) if column in minimums
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        reports.append(report)
        (output_dir / f"{path.stem}_analysis.json").write_text(json.dumps(report, indent=2))
        for column in columns or []:
            schema_rows.append({
                "source": path.name,
                "column": column,
                "non_null_values": int(non_null_counts[column]),
                "is_constant": column in constants,
                "numeric_min": _json_number(minimums.get(column)),
                "numeric_max": _json_number(maximums.get(column)),
            })
        combined_rows += rows
        combined_invalid += invalid_values
        combined_duplicates += duplicates
        combined_labels.update(labels)

    pd.DataFrame(schema_rows).to_csv(output_dir / "column_statistics.csv", index=False)
    combined = {
        "sources": reports,
        "source_count": len(paths),
        "rows": combined_rows,
        "labels": dict(sorted(combined_labels.items())),
        "invalid_values": combined_invalid,
        "duplicate_rows_within_sessions": combined_duplicates,
        "session_policy": "independent_capture_sessions",
        "chronology": "row_order_proxy",
        "proxy_cadence_seconds": PROXY_CADENCE_SECONDS,
    }
    (output_dir / "combined_analysis.json").write_text(json.dumps(combined, indent=2))
    return combined


def score_ann_chunk(frame: pd.DataFrame, model=None, scaler=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model is None or scaler is None:
        model, scaler = _load_artifacts()
    col_map = match_columns(list(frame.columns))
    ordered = frame[[col_map[name] for name in TRAINING_FEATURES]].copy()
    ordered.columns = TRAINING_FEATURES
    ordered = ordered.replace([np.inf, -np.inf, "Infinity", "-Infinity"], np.nan)
    ordered = ordered.apply(pd.to_numeric, errors="coerce")
    valid = (~ordered.isna().any(axis=1)).to_numpy()
    labels = np.full(len(frame), INVALID_FEATURES_LABEL, dtype=object)
    confidences = np.full(len(frame), np.nan, dtype=np.float32)
    if valid.any():
        values = scaler.transform(ordered.loc[valid].to_numpy(dtype=np.float64)).astype(np.float32)
        probabilities = model.predict(values, batch_size=4096, verbose=0)
        indices = np.argmax(probabilities, axis=1)
        labels[valid] = np.asarray(CLASS_NAMES, dtype=object)[indices]
        confidences[valid] = probabilities[np.arange(len(indices)), indices]
    return labels, confidences, valid


def _aggregate_proxy_chunk(frame: pd.DataFrame, labels: np.ndarray, first_row: int) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    col_map = match_columns(list(frame.columns))
    id_cols = find_id_columns(list(frame.columns))
    window_ids = (np.arange(first_row, first_row + len(frame)) // WINDOW_SIZE_SECONDS).astype(np.int64)
    working = pd.DataFrame({"window_id": window_ids})
    working["state"] = labels

    for output_name, training_feature, _ in AGGREGATIONS:
        working[output_name] = pd.to_numeric(frame[col_map[training_feature]], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dst_port = col_map[DST_PORT_TRAINING_FEATURE]
    working["dst_port"] = frame[dst_port]
    for role in ("src_ip", "dst_ip", "src_port"):
        if role in id_cols:
            working[role] = frame[id_cols[role]]

    grouped = working.groupby("window_id", sort=True)
    output = pd.DataFrame(index=grouped.size().index)
    output["flow_count"] = grouped.size().astype(np.float64)
    for role in ("src_ip", "dst_ip", "src_port"):
        output[f"unique_{role}_count"] = grouped[role].nunique() if role in working else 0.0
    output["unique_dst_port_count"] = grouped["dst_port"].nunique()

    for output_name, _, function in AGGREGATIONS:
        series_group = grouped[output_name]
        if function == "sum": values = series_group.sum(min_count=1)
        elif function == "mean": values = series_group.mean()
        elif function == "min": values = series_group.min()
        elif function == "max": values = series_group.max()
        else: values = series_group.var()
        output[output_name] = values.fillna(0.0)

    output["total_packets"] = output["total_fwd_packets"] + output["total_bwd_packets"]
    output["total_bytes"] = output["total_fwd_bytes"] + output["total_bwd_bytes"]
    output["packets_per_second"] = output["total_packets"] / WINDOW_SIZE_SECONDS
    output["bytes_per_second"] = output["total_bytes"] / WINDOW_SIZE_SECONDS
    output["flows_per_second"] = output["flow_count"] / WINDOW_SIZE_SECONDS

    scoreable = working[working["state"].isin(FORECAST_CLASSES)]
    counts = pd.crosstab(scoreable["window_id"], scoreable["state"]).reindex(
        index=output.index, columns=FORECAST_CLASSES, fill_value=0
    )
    output["scoreable_flow_count"] = counts.sum(axis=1)
    for state in FORECAST_CLASSES:
        output[f"{state.lower()}_flow_count"] = counts[state].astype(np.int64)
    output = output[output["scoreable_flow_count"] > 0].copy()
    counts = counts.loc[output.index]
    output["dominant_state"] = np.asarray(FORECAST_CLASSES, dtype=object)[
        np.argmax(counts.to_numpy(), axis=1)
    ]
    output["attack_present"] = (counts[list(FORECAST_CLASSES[1:])].sum(axis=1) > 0).astype(np.int64)
    output["proxy_start_row"] = output.index.to_numpy(dtype=np.int64) * WINDOW_SIZE_SECONDS
    output["proxy_end_row"] = output["proxy_start_row"] + WINDOW_SIZE_SECONDS
    return output.reset_index()


def prepare_session_windows(
    path: Path,
    source_fingerprint: dict,
    artifact_hashes: dict,
    cache_root: Path = CACHE_ROOT,
    chunk_size: int = CHUNK_SIZE,
    force_rebuild: bool = False,
    progress: Progress | None = None,
) -> tuple[pd.DataFrame, dict]:
    identity = cache_identity(source_fingerprint, artifact_hashes)
    key = cache_key(identity)
    session_dir = Path(cache_root) / key
    windows_path = session_dir / "windows.npz"
    metadata_path = session_dir / "metadata.json"
    if not force_rebuild and windows_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["source_path"] = path.name
        data = np.load(windows_path, allow_pickle=False)
        windows = pd.DataFrame({name: data[name] for name in data.files})
        windows.insert(0, "session_id", metadata["session_id"])
        if progress:
            progress(stage="cache", cache_state="hit", rows_processed=source_fingerprint.get("rows", 0))
        return windows, metadata

    model, scaler = _load_artifacts()
    frames = []
    rows_processed = invalid_rows = 0
    started = time.perf_counter()
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        labels, _, valid = score_ann_chunk(chunk, model, scaler)
        frames.append(_aggregate_proxy_chunk(chunk, labels, rows_processed))
        rows_processed += len(chunk)
        invalid_rows += int((~valid).sum())
        if progress:
            progress(stage="ann_scoring", rows_processed=rows_processed, cache_state="miss")

    windows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    windows.insert(0, "session_id", path.stem)
    session_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        column: (windows[column].to_numpy(dtype=str) if column == "dominant_state" else windows[column].to_numpy())
        for column in windows.columns if column != "session_id"
    }
    np.savez_compressed(windows_path, **arrays)
    metadata = {
        "cache_key": key,
        "identity": identity,
        "session_id": path.stem,
        "source_path": path.name,
        "rows": rows_processed,
        "invalid_rows": invalid_rows,
        "windows": len(windows),
        "dropped_empty_scoreable_windows": (rows_processed + WINDOW_SIZE_SECONDS - 1) // WINDOW_SIZE_SECONDS - len(windows),
        "feature_availability": {
            "unique_src_ip_count": "measured" if "src_ip" in find_id_columns(list(pd.read_csv(path, nrows=0).columns)) else "unavailable_zero",
            "unique_dst_ip_count": "measured" if "dst_ip" in find_id_columns(list(pd.read_csv(path, nrows=0).columns)) else "unavailable_zero",
            "unique_src_port_count": "measured" if "src_port" in find_id_columns(list(pd.read_csv(path, nrows=0).columns)) else "unavailable_zero",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return windows, metadata


def contiguous_blocks(windows: pd.DataFrame) -> list[pd.DataFrame]:
    if windows.empty:
        return []
    ordered = windows.sort_values("window_id").reset_index(drop=True)
    groups = ordered["window_id"].diff().fillna(1).ne(1).cumsum()
    return [block.reset_index(drop=True) for _, block in ordered.groupby(groups, sort=True)]


def build_sequences(windows: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH) -> dict:
    arrays = {"X": [], "y": [], "input_labels": [], "input_window_ids": [], "target_window_id": [], "session_id": []}
    for block in contiguous_blocks(windows):
        if len(block) <= sequence_length:
            continue
        features = block[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = block["dominant_state"].to_numpy(dtype=str)
        ids = block["window_id"].to_numpy(dtype=np.int64)
        session_id = str(block["session_id"].iloc[0]) if "session_id" in block else "unknown"
        for index in range(len(block) - sequence_length):
            target = index + sequence_length
            arrays["X"].append(features[index:target])
            arrays["y"].append(labels[target])
            arrays["input_labels"].append(labels[index:target])
            arrays["input_window_ids"].append(ids[index:target])
            arrays["target_window_id"].append(ids[target])
            arrays["session_id"].append(session_id)
    feature_count = len(STATE_FEATURE_NAMES)
    return {
        "X": np.asarray(arrays["X"], dtype=np.float32).reshape(-1, sequence_length, feature_count),
        "y": np.asarray(arrays["y"], dtype=str),
        "input_labels": np.asarray(arrays["input_labels"], dtype=str).reshape(-1, sequence_length),
        "input_window_ids": np.asarray(arrays["input_window_ids"], dtype=np.int64).reshape(-1, sequence_length),
        "target_window_id": np.asarray(arrays["target_window_id"], dtype=np.int64),
        "session_id": np.asarray(arrays["session_id"], dtype=str),
    }


def concat_sequence_sets(sequence_sets: list[dict]) -> dict:
    if not sequence_sets:
        return build_sequences(pd.DataFrame(columns=["window_id", "dominant_state", *STATE_FEATURE_NAMES]))
    return {key: np.concatenate([item[key] for item in sequence_sets], axis=0) for key in sequence_sets[0]}
