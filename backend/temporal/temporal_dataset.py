"""
Orchestrates the full temporal dataset / state-transition pipeline:

  Feature CSV -> timestamp parsing -> chronological sort -> time windows
    -> network-state vectors -> state labels -> state transitions
    -> sliding sequences -> chronological train/val/test split
    -> temporal scaler (fit on train only)

This module does NOT train any forecasting model — it only prepares the
dataset a future phase will train on. See config.py for the tunables
(WINDOW_SIZE_SECONDS, SEQUENCE_LENGTH) — nothing below hard-codes them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_config
from .config import DEFAULT_SEQUENCE_LENGTH, DEFAULT_WINDOW_SIZE_SECONDS, STATE_CLASSES
from .schema import STATE_FEATURE_NAMES, STATE_FEATURE_NAMES_V2, XDR_FEATURE_NAMES, find_id_columns
from .sequence_builder import build_state_transitions, raise_if_insufficient_windows
from .splitting import build_split_sequences, chronological_split, fit_scaler_on_train
from .state_builder import build_temporal_states
from .validation import run_all_checks
from .windowing import TemporalError, assign_windows, parse_timestamps, sort_chronologically


def _empty_sequence_set(sequence_length: int) -> dict:
    n_feat = len(STATE_FEATURE_NAMES)
    return {
        "X": np.empty((0, sequence_length, n_feat)),
        "X_scaled": np.empty((0, sequence_length, n_feat)),
        "y_state_vector": np.empty((0, n_feat)),
        "y_state_vector_scaled": np.empty((0, n_feat)),
        "y_dominant_state": np.empty((0,), dtype=object),
        "y_attack_present": np.empty((0,), dtype=np.int64),
        "input_window_ids": np.empty((0, sequence_length), dtype=np.int64),
        "target_window_id": np.empty((0,), dtype=np.int64),
    }


def _insufficient_data_report(
    df: pd.DataFrame, sorted_df: pd.DataFrame, states_df: pd.DataFrame,
    sequence_length: int, window_size_seconds: int,
) -> str:
    required = sequence_length + 1
    available = len(states_df)
    first_ts = sorted_df["timestamp_parsed"].min()
    last_ts = sorted_df["timestamp_parsed"].max()
    duration = (last_ts - first_ts).total_seconds()
    n_flows = len(df)
    n_invalid_features = int((df["Current_State"] == "INVALID_FEATURES").sum()) if "Current_State" in df.columns else 0

    return (
        "Insufficient temporal data.\n\n"
        f"Required windows: {required}\n"
        f"Available windows: {available}\n\n"
        f"Capture start: {first_ts}\n"
        f"Capture end: {last_ts}\n"
        f"Capture duration: {duration:.2f} sec\n\n"
        f"Flow records: {n_flows}\n"
        f"Invalid features: {n_invalid_features}\n"
        f"{window_size_seconds}-second windows: {available}\n"
        f"Sequence length: {sequence_length}\n\n"
        "Recommendation:\n"
        "Capture at least 90-120 seconds of continuous traffic. Packet "
        "count alone does not guarantee enough temporal windows — a burst "
        "of packets arriving within a few seconds still only spans a "
        f"few {window_size_seconds}-second windows regardless of how many "
        "packets it contains; what matters is the real time span between "
        "the first and last packet."
    )


def prepare_temporal_dataset(
    input_csv: Path,
    output_dir: Path,
    window_size_seconds: int = DEFAULT_WINDOW_SIZE_SECONDS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    session_id: str | None = None,
    ingest_enrichment: dict | None = None,
    enrich_windows: bool | None = None,
) -> dict:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    if not input_csv.exists():
        raise TemporalError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        raise TemporalError(f"Input CSV is empty: {input_csv}")

    if "Current_State" not in df.columns:
        raise TemporalError(
            "Input CSV has no Current_State column — run prediction "
            "(backend/prediction/predict.py) on it first."
        )

    id_cols = find_id_columns(list(df.columns))
    ts_col = id_cols.get("timestamp")
    check1 = ("1_timestamp_column_exists", ts_col is not None)
    if ts_col is None:
        raise TemporalError("No timestamp column found in the input CSV.")

    ts_result = parse_timestamps(df[ts_col])
    if ts_result.n_valid == 0:
        raise TemporalError(f"None of {ts_result.n_total} timestamp values could be parsed.")

    df = df.copy()
    df["timestamp_parsed"] = ts_result.parsed
    valid_df = df[ts_result.valid_mask].copy()
    n_dropped_invalid_ts = ts_result.n_invalid

    sorted_df = sort_chronologically(valid_df, "timestamp_parsed")
    windowed_df = assign_windows(sorted_df, "timestamp_parsed", window_size_seconds)

    states_df, agg_warnings = build_temporal_states(windowed_df, "Current_State", window_size_seconds)

    if enrich_windows is None:
        enrich_windows = bool(load_config().get("xdr", {}).get("enrich_windows", False))
    if enrich_windows:
        if ingest_enrichment is None:
            try:
                from ..ingest import get_ingest_store
                ingest_enrichment = get_ingest_store().enrichment(session_id or input_csv.stem)
            except ImportError:
                ingest_enrichment = None
        enrichment = ingest_enrichment or {}
        values = {
            "dns_entropy": enrichment.get("dns_query_entropy_mean", 0.0),
            "unique_sni": enrichment.get("unique_sni_count", 0.0),
            "beacon_score": enrichment.get("beacon_score_max", 0.0),
            "byte_asymmetry": enrichment.get("byte_asymmetry_max", 0.0),
            "ja3_novelty": enrichment.get("ja3_novelty", 0.0),
            "http_error_rate": enrichment.get("http_error_rate", 0.0),
        }
        for feature in XDR_FEATURE_NAMES:
            states_df[feature] = float(values[feature] or 0.0)

    # Hard gate — checked against the TOTAL dataset before any split. On
    # failure this is re-raised with full diagnostics (capture start/end/
    # duration, flow/invalid-feature counts) rather than just the bare
    # "Required N, Available M" — the point is to make it obvious the
    # problem is temporal DURATION, not packet count, row count, or a
    # validation bug, and to say exactly how much more real traffic is
    # needed. Nothing here fabricates a window/packet/timestamp to get
    # past this — it only changes what the error message reports.
    try:
        raise_if_insufficient_windows(len(states_df), sequence_length)
    except TemporalError:
        raise TemporalError(_insufficient_data_report(
            df, sorted_df, states_df, sequence_length, window_size_seconds
        ))

    checks = run_all_checks({
        "columns": list(df.columns), "ts_col": ts_col,
        "n_total": ts_result.n_total, "n_valid": ts_result.n_valid, "n_invalid": ts_result.n_invalid,
        "sorted_df": windowed_df, "windowed_df": windowed_df,
        "window_size_seconds": window_size_seconds,
        "states_df": states_df, "feature_names": STATE_FEATURE_NAMES,
    })
    failed_checks = {k: v for k, v in checks.items() if not v[0]}
    if failed_checks:
        raise TemporalError(f"Temporal validation failed: {failed_checks}")

    transitions_df = build_state_transitions(states_df)

    train_df, val_df, test_df, split_metadata = chronological_split(states_df)
    scaler = fit_scaler_on_train(train_df)

    split_warnings = {}
    splits = {}
    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        try:
            splits[name] = build_split_sequences(split_df, sequence_length, scaler, name)
        except TemporalError as e:
            split_warnings[name] = str(e)
            splits[name] = _empty_sequence_set(sequence_length)

    if len(splits["train"]["X"]) == 0:
        raise TemporalError(
            "Training split has zero usable sequences "
            f"({len(train_df)} windows, need {sequence_length + 1}+). "
            "Capture/upload a longer traffic sample or reduce --sequence-length."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    states_out = output_dir / "temporal_states.csv"
    states_df.to_csv(states_out, index=False)

    transitions_out = output_dir / "state_transitions.csv"
    transitions_df.to_csv(transitions_out, index=False)

    # Full-dataset (unsplit) sequences, for inspection/completeness.
    all_seq = None
    try:
        from .sequence_builder import build_sequences
        all_seq = build_sequences(states_df, sequence_length)
        np.savez(
            output_dir / "temporal_sequences.npz",
            X=all_seq["X"], y_state_vector=all_seq["y_state_vector"],
            y_dominant_state=all_seq["y_dominant_state"], y_attack_present=all_seq["y_attack_present"],
            input_window_ids=all_seq["input_window_ids"], target_window_id=all_seq["target_window_id"],
        )
    except TemporalError:
        pass  # already guaranteed >= sequence_length+1 above; defensive only

    for name in ("train", "validation", "test"):
        s = splits[name]
        np.savez(
            output_dir / f"temporal_{name}.npz",
            X=s["X"], X_scaled=s["X_scaled"],
            y_state_vector=s["y_state_vector"], y_state_vector_scaled=s["y_state_vector_scaled"],
            y_dominant_state=s["y_dominant_state"], y_attack_present=s["y_attack_present"],
            input_window_ids=s["input_window_ids"], target_window_id=s["target_window_id"],
        )

    feature_names_out = output_dir / "state_feature_names.json"
    output_features = STATE_FEATURE_NAMES_V2 if enrich_windows else STATE_FEATURE_NAMES
    feature_names_out.write_text(json.dumps({"features": output_features}, indent=2))

    seq_metadata = {
        "window_size_seconds": window_size_seconds,
        "sequence_length": sequence_length,
        "num_sequences": int(len(all_seq["X"])) if all_seq is not None else 0,
        "num_state_features": len(output_features),
        "state_classes": STATE_CLASSES,
    }
    (output_dir / "temporal_sequences_metadata.json").write_text(json.dumps(seq_metadata, indent=2))

    split_metadata["split_sequence_counts"] = {
        name: int(len(splits[name]["X"])) for name in ("train", "validation", "test")
    }
    split_metadata["split_warnings"] = split_warnings
    (output_dir / "temporal_split_metadata.json").write_text(json.dumps(split_metadata, indent=2, default=str))

    models_dir = REPO_ROOT / "models"
    scaler_path = models_dir / "temporal_scaler.bin" if models_dir.exists() else output_dir / "temporal_scaler.bin"
    joblib.dump(scaler, scaler_path)

    summary = {
        "input_csv": str(input_csv),
        "input_rows": int(len(df)),
        "timestamp_column": ts_col,
        "timestamps_valid": ts_result.n_valid,
        "timestamps_invalid": n_dropped_invalid_ts,
        "timestamp_format_used": ts_result.format_used,
        "window_size_seconds": window_size_seconds,
        "total_windows": int(len(states_df)),
        "state_features": len(output_features),
        "forecast_state_features": len(STATE_FEATURE_NAMES),
        "xdr_enrichment_enabled": bool(enrich_windows),
        "transitions": int(len(transitions_df)),
        "sequence_length": sequence_length,
        "total_sequences": int(len(all_seq["X"])) if all_seq is not None else 0,
        "train_windows": len(train_df), "validation_windows": len(val_df), "test_windows": len(test_df),
        "train_sequences": int(len(splits["train"]["X"])),
        "validation_sequences": int(len(splits["validation"]["X"])),
        "test_sequences": int(len(splits["test"]["X"])),
        "state_distribution": {
            c: int(states_df[f"{c.lower()}_flow_count"].sum()) for c in STATE_CLASSES
        },
        "aggregation_warnings": agg_warnings,
        "split_warnings": split_warnings,
        "validation_checks": {k: {"passed": v[0], "message": v[1]} for k, v in checks.items()},
        "output_dir": str(output_dir),
        "temporal_scaler_path": str(scaler_path),
        "states_df": states_df,
        "transitions_df": transitions_df,
        "sequences": all_seq,
    }
    return summary


def _print_summary(summary: dict):
    print("Temporal Dataset Preparation")
    print("─" * 45)
    print(f"Input rows:                 {summary['input_rows']}")
    print(f"Time window:                {summary['window_size_seconds']} seconds")
    print(f"Total windows:              {summary['total_windows']}")
    print(f"State features:             {summary['state_features']}")
    print(f"Transitions:                {summary['transitions']}")
    print(f"Sequences:                  {summary['total_sequences']}")
    print()
    print(f"Sequence length:            {summary['sequence_length']}")
    print()
    print(f"Train windows:              {summary['train_windows']}")
    print(f"Validation windows:         {summary['validation_windows']}")
    print(f"Test windows:               {summary['test_windows']}")
    print()
    print(f"Train sequences:            {summary['train_sequences']}")
    print(f"Validation sequences:       {summary['validation_sequences']}")
    print(f"Test sequences:             {summary['test_sequences']}")
    print()
    print("State distribution:")
    print()
    for c, v in summary["state_distribution"].items():
        print(f"{c}:{' ' * (28 - len(c))}{v}")


def main():
    parser = argparse.ArgumentParser(description="Build the temporal state-transition dataset.")
    parser.add_argument("--input", required=True, help="Path to a features/<session>_prediction.csv (must have Current_State)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE_SECONDS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    args = parser.parse_args()

    try:
        summary = prepare_temporal_dataset(
            Path(args.input), Path(args.output), args.window_size, args.sequence_length
        )
    except TemporalError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    _print_summary(summary)


if __name__ == "__main__":
    main()
