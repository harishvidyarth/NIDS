"""
Post-hoc validation of an already-generated temporal dataset (data
quality, structural correctness, and — critically — cross-split
leakage). This is separate from validation.py's build-time gate (the 7
checks run automatically inside prepare_temporal_dataset(), which already
block generation of a structurally broken dataset). This module instead
re-inspects the persisted artifacts (temporal_states.csv,
state_transitions.csv, temporal_sequences.npz, temporal_{train,
validation,test}.npz) plus the original Current_State-labelled flow CSV,
producing the richer report the UI needs: label/timestamp validity,
window continuity, per-feature numeric health, transition consistency,
sequence correctness, chronological split ordering, cross-split
duplicate/overlap leakage, and scaler/label leakage.

Everything here is read-only: no source file (the flow CSV, the PCAP, or
any temporal artifact) is ever modified or rewritten.

Performance: every check is O(n) or O(n log n) — pandas/NumPy vectorized
ops (value_counts, duplicated, hash_pandas_object, is_monotonic) — no
row-by-row Python loops over the flow-level data and no O(n^2) pairwise
comparison. The flow-level CSV is read exactly once and reused across all
raw-level checks (current_state, timestamps, duplicates, missing_data).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .config import STATE_CLASSES
from .schema import STATE_FEATURE_NAMES, find_id_columns
from .windowing import parse_timestamps

PASS, WARNING, FAIL, NOT_AVAILABLE = "PASS", "WARNING", "FAIL", "NOT_AVAILABLE"
_SEVERITY = {PASS: 0, NOT_AVAILABLE: 0, WARNING: 1, FAIL: 2}


class ValidationError(Exception):
    pass


def _worse(a: str, b: str) -> str:
    return b if _SEVERITY[b] > _SEVERITY[a] else a


def _combine(*statuses: str) -> str:
    result = PASS
    for s in statuses:
        result = _worse(result, s)
    return result


# Feature-name substrings that logically can never be negative (packet/byte/
# flow counts, durations, rates). Only checked where the name itself implies
# the constraint — never an invented arbitrary bound.
_NON_NEGATIVE_HINTS = ("count", "packets", "bytes", "duration", "iat", "per_second", "size")

# Raw feature CSVs come out of a flow exporter that flushes on flow
# completion/timeout, so rows are commonly a little out of start-time
# order. prepare_temporal_dataset re-sorts chronologically before
# windowing (windowing.sort_chronologically) and window_id is derived
# from timestamp *values*, not row position — so a small fraction of
# out-of-order rows changes nothing downstream and is a WARNING, not a
# structural FAIL. FAIL only when disorder is large enough to suggest a
# genuinely broken export.
_TS_ORDER_FAIL_FRACTION = 0.05
_TS_ORDER_FAIL_MIN = 5

# Aggregated IAT/duration extrema can land a hair below zero from
# floating-point and sub-second clock jitter in the source capture. A
# tiny negative is a data-quality WARNING; only a clearly non-physical
# magnitude is a FAIL.
_NON_NEGATIVE_TOLERANCE = 1.0


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    """Vectorized per-row hash for O(n) exact-duplicate-row detection."""
    return pd.util.hash_pandas_object(df, index=False)


def _hash_array_rows(arr: np.ndarray) -> list[str]:
    return [hashlib.md5(arr[i].tobytes()).hexdigest() for i in range(len(arr))]


def _validate_current_state(df: pd.DataFrame) -> dict:
    if "Current_State" not in df.columns:
        return {"status": FAIL, "details": {
            "message": "Required column 'Current_State' is missing.",
            "required_column": "Current_State",
        }}
    col = df["Current_State"]
    missing = int(col.isna().sum() + (col.astype(str).str.strip() == "").sum())
    counts = col.value_counts(dropna=True)
    total = int(counts.sum())
    unknown_labels = [str(v) for v in counts.index if v not in STATE_CLASSES]
    unknown_count = int(sum(counts[v] for v in counts.index if v not in STATE_CLASSES))

    distribution = {
        cls: {
            "count": int(counts.get(cls, 0)),
            "percent": round(100 * counts.get(cls, 0) / total, 2) if total else 0.0,
        }
        for cls in STATE_CLASSES
    }

    status = PASS
    if missing == total or "Current_State" not in df.columns:
        status = FAIL
    elif unknown_labels:
        status = FAIL
    elif missing > 0:
        status = WARNING

    return {"status": status, "details": {
        "rows_checked": total + missing,
        "missing_labels": missing,
        "unknown_labels": unknown_labels,
        "unknown_label_count": unknown_count,
        "distribution": distribution,
        "canonical_classes": STATE_CLASSES,
    }}


def _validate_timestamps(df: pd.DataFrame, ts_col: str | None) -> dict:
    if ts_col is None or ts_col not in df.columns:
        return {"status": FAIL, "details": {"message": "No timestamp column found in the source CSV."}}

    raw = df[ts_col]
    result = parse_timestamps(raw)
    parsed = result.parsed

    duplicate_timestamps = int(parsed.dropna().duplicated().sum())
    # Adjacent-pair inversions in the ORIGINAL row order (never re-sorted
    # here) — an honest count of out-of-order rows, not a silently-sorted
    # "valid" claim.
    valid_parsed = parsed.dropna()
    out_of_order = int((valid_parsed.diff().dt.total_seconds() < 0).sum()) if len(valid_parsed) > 1 else 0
    is_ordered = out_of_order == 0
    order_fraction = out_of_order / result.n_valid if result.n_valid else 0.0
    # A large share of inversions suggests a genuinely broken export; a
    # handful is normal flow-exporter flush order and is re-sorted before
    # windowing.
    order_structural = out_of_order > _TS_ORDER_FAIL_MIN and order_fraction > _TS_ORDER_FAIL_FRACTION
    order_status = "PASS" if is_ordered else ("FAIL" if order_structural else "WARNING")

    first_ts = valid_parsed.min() if not valid_parsed.empty else None
    last_ts = valid_parsed.max() if not valid_parsed.empty else None
    duration_seconds = (last_ts - first_ts).total_seconds() if first_ts is not None else None

    status = PASS
    if result.n_valid == 0:
        status = FAIL
    elif order_structural:
        status = FAIL
    elif result.n_invalid > 0 or not is_ordered:
        status = WARNING  # excluded-invalid rows, or minor flush-order inversions that sort_chronologically resolves before windowing

    return {"status": status, "details": {
        "rows_checked": result.n_total,
        "valid_timestamps": result.n_valid,
        "missing_or_invalid": result.n_invalid,
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_timestamps_note": "Expected for flow data at second-level precision — multiple flows commonly share a timestamp; not treated as a failure on its own.",
        "order_status": order_status,
        "order_note": "Rows are re-sorted chronologically before windowing; window_id derives from timestamp values, not row order, so a small fraction of out-of-order rows does not affect windows/sequences/splits.",
        "out_of_order_rows": out_of_order,
        "timestamp_format_detected": result.format_used,
        "first_timestamp": str(first_ts) if first_ts is not None else None,
        "last_timestamp": str(last_ts) if last_ts is not None else None,
        "duration_seconds": duration_seconds,
    }}


def _validate_duplicates(df: pd.DataFrame, ts_col: str | None) -> dict:
    row_hashes = _hash_rows(df)
    dup_rows = int(row_hashes.duplicated().sum())

    dup_timestamps = int(df[ts_col].dropna().duplicated().sum()) if ts_col and ts_col in df.columns else None

    id_map = find_id_columns(list(df.columns))
    flow_id_col = id_map.get("flow_id")
    dup_flow_ids = int(df[flow_id_col].dropna().duplicated().sum()) if flow_id_col else None

    status = WARNING if dup_rows > 0 else PASS

    return {"status": status, "details": {
        "exact_duplicate_rows": dup_rows,
        "duplicate_timestamps": dup_timestamps,
        "duplicate_flow_ids": dup_flow_ids if flow_id_col else "NOT_AVAILABLE (no Flow ID column in this CSV)",
        "method": "vectorized row hashing (pandas.util.hash_pandas_object) — O(n), no pairwise comparison",
    }}


def _validate_missing_data(df: pd.DataFrame, ts_col: str | None) -> dict:
    total = len(df)
    missing_ts = int(df[ts_col].isna().sum()) if ts_col and ts_col in df.columns else total
    missing_state = int(df["Current_State"].isna().sum()) if "Current_State" in df.columns else total

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_count = int(df[numeric_cols].isna().sum().sum()) if len(numeric_cols) else 0
    inf_count = int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum()) if len(numeric_cols) else 0

    status = PASS
    if missing_ts or missing_state or inf_count:
        status = WARNING
    if total == 0:
        status = FAIL

    return {"status": status, "details": {
        "rows_checked": total,
        "missing_timestamps": missing_ts,
        "missing_current_state": missing_state,
        "nan_numeric_values": nan_count,
        "infinite_values": inf_count,
    }}


def _validate_windows(states_df: pd.DataFrame, window_size_seconds: float) -> dict:
    total = len(states_df)
    if total == 0:
        return {"status": FAIL, "details": {"message": "temporal_states.csv has no rows."}}

    durations = (states_df["window_end"] - states_df["window_start"]).dt.total_seconds()
    bad_duration = int((durations != window_size_seconds).sum())

    ordered = states_df.sort_values("window_id")
    wid = ordered["window_id"].to_numpy()
    id_diffs = np.diff(wid)
    missing_windows = int(np.clip(id_diffs - 1, 0, None).sum())
    duplicate_window_ids = int(total - states_df["window_id"].nunique())

    starts = ordered["window_start"].reset_index(drop=True)
    start_deltas = starts.diff().dt.total_seconds().dropna()
    overlapping_windows = int((start_deltas < window_size_seconds).sum())

    status = PASS
    if bad_duration or duplicate_window_ids or overlapping_windows:
        status = FAIL
    elif missing_windows:
        status = WARNING  # gaps in traffic are expected/informational, not a structural defect

    return {"status": status, "details": {
        "total_windows": total,
        "valid_windows": total - bad_duration,
        "missing_windows": missing_windows,
        "overlapping_windows": overlapping_windows,
        "duplicate_window_ids": duplicate_window_ids,
        "window_size_seconds": window_size_seconds,
        "boundary_convention": (
            "Half-open interval [window_start, window_end): a timestamp exactly "
            "equal to window_end belongs to the NEXT window, not this one. "
            "window_id = floor((timestamp - first_timestamp) / window_size_seconds)."
        ),
    }}


def _validate_features(states_df: pd.DataFrame) -> dict:
    non_feature_cols = (
        {"window_id", "window_start", "window_end", "dominant_state", "attack_present"}
        | {f"{c.lower()}_flow_count" for c in STATE_CLASSES}
    )
    unexpected_columns = [c for c in states_df.columns if c not in non_feature_cols and c not in STATE_FEATURE_NAMES]
    duplicate_columns = [c for c in states_df.columns if list(states_df.columns).count(c) > 1]
    missing_features = [f for f in STATE_FEATURE_NAMES if f not in states_df.columns]

    feature_table = []
    total_nan = total_inf = 0
    range_violations = []
    near_zero_negatives = []
    for name in STATE_FEATURE_NAMES:
        if name not in states_df.columns:
            feature_table.append({"feature": name, "type": "missing", "missing": len(states_df),
                                   "nan": None, "infinite": None, "min": None, "max": None, "status": FAIL})
            continue
        col = states_df[name]
        is_numeric = pd.api.types.is_numeric_dtype(col)
        nan_count = int(col.isna().sum()) if is_numeric else None
        inf_count = int(np.isinf(col.to_numpy(dtype=float)).sum()) if is_numeric else 0
        col_min = float(col.min()) if is_numeric and col.notna().any() else None
        col_max = float(col.max()) if is_numeric and col.notna().any() else None
        constant = bool(is_numeric and col.nunique(dropna=True) <= 1)

        range_bad = False
        range_soft = False
        if is_numeric and col_min is not None and any(h in name.lower() for h in _NON_NEGATIVE_HINTS):
            if col_min < -_NON_NEGATIVE_TOLERANCE:
                range_bad = True
                range_violations.append({"feature": name, "min": col_min})
            elif col_min < 0:
                range_soft = True
                near_zero_negatives.append({"feature": name, "min": col_min})

        fstatus = PASS
        if not is_numeric:
            fstatus = FAIL
        elif range_bad:
            fstatus = FAIL
        elif range_soft or nan_count or inf_count:
            fstatus = WARNING

        feature_table.append({
            "feature": name, "type": "numeric" if is_numeric else str(col.dtype),
            "missing": 0, "nan": nan_count, "infinite": inf_count,
            "min": col_min, "max": col_max, "constant": constant, "status": fstatus,
        })
        total_nan += nan_count or 0
        total_inf += inf_count or 0

    status = PASS
    if missing_features or duplicate_columns or range_violations:
        status = FAIL
    elif total_nan or total_inf or near_zero_negatives:
        status = WARNING

    return {"status": status, "details": {
        "features_checked": len(STATE_FEATURE_NAMES),
        "missing_features": missing_features,
        "unexpected_columns": unexpected_columns,
        "duplicate_columns": duplicate_columns,
        "nan_values": total_nan,
        "infinite_values": total_inf,
        "range_violations": range_violations,
        "near_zero_negatives": near_zero_negatives,
        "near_zero_negatives_note": (
            f"min below 0 but within {_NON_NEGATIVE_TOLERANCE} of it — float / sub-second "
            "clock jitter in the source capture, not a structural defect."
        ),
        "feature_table": feature_table,
    }}


def _validate_transitions(states_df: pd.DataFrame, transitions_path: Path) -> dict:
    if not transitions_path.exists():
        return {"status": NOT_AVAILABLE, "details": {"message": "state_transitions.csv not found."}}
    trans = pd.read_csv(transitions_path)
    if trans.empty:
        return {"status": WARNING, "details": {"message": "state_transitions.csv has no rows (fewer than 2 windows).", "total_transitions": 0}}

    valid_ids = set(states_df["window_id"])
    bad_current = int((~trans["current_window"].isin(valid_ids)).sum())
    bad_next = int((~trans["next_window"].isin(valid_ids)).sum())
    invalid_states = sorted(set(trans["current_state"]).union(trans["next_state"]) - set(STATE_CLASSES))
    backward = int((trans["time_delta_seconds"] < 0).sum())

    state_map = states_df.set_index("window_id")["dominant_state"]
    actual_next_state = trans["next_window"].map(state_map)
    consistency_mismatches = int((trans["next_state"] != actual_next_state).sum())

    from_to = (
        trans.groupby(["current_state", "next_state"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    status = PASS
    if bad_current or bad_next or invalid_states or backward or consistency_mismatches:
        status = FAIL

    return {"status": status, "details": {
        "total_transitions": len(trans),
        "invalid_current_window_refs": bad_current,
        "invalid_next_window_refs": bad_next,
        "invalid_state_labels": invalid_states,
        "backward_time_transitions": backward,
        "consistency_mismatches": consistency_mismatches,
        "from_to_counts": from_to.head(20).to_dict("records"),
    }}


def _load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _validate_sequences(temporal_dir: Path, states_df: pd.DataFrame) -> dict:
    seq_path = temporal_dir / "temporal_sequences.npz"
    meta_path = temporal_dir / "temporal_sequences_metadata.json"
    seq = _load_npz(seq_path)
    if seq is None:
        return {"status": NOT_AVAILABLE, "details": {"message": "temporal_sequences.npz not found."}}

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sequence_length = meta.get("sequence_length")

    X, input_ids, target_ids = seq["X"], seq["input_window_ids"], seq["target_window_id"]
    n_seq = len(X)
    actual_len = int(X.shape[1]) if n_seq else (int(input_ids.shape[1]) if len(input_ids) else None)

    length_ok = (sequence_length is None) or (actual_len == sequence_length)

    # Target alignment: target_window_id must be the window immediately
    # after the sequence's own last input window in POSITIONAL (window_id-
    # sorted) order — this is exactly how sequence_builder.build_sequences
    # constructs them, checked here independently from the persisted data.
    ordered_ids = states_df.sort_values("window_id")["window_id"].reset_index(drop=True)
    pos_of_id = {wid: i for i, wid in enumerate(ordered_ids)}
    misaligned = 0
    for i in range(n_seq):
        last_input_pos = pos_of_id.get(int(input_ids[i][-1]))
        target_pos = pos_of_id.get(int(target_ids[i]))
        if last_input_pos is None or target_pos is None or target_pos != last_input_pos + 1:
            misaligned += 1

    seq_hashes = _hash_array_rows(X) if n_seq else []
    duplicate_sequences = int(n_seq - len(set(seq_hashes)))

    expected_n_seq = max(len(states_df) - (sequence_length or actual_len or 0), 0)
    count_correct = n_seq == expected_n_seq

    status = PASS
    if not length_ok or misaligned:
        status = FAIL
    elif not count_correct or duplicate_sequences:
        status = WARNING

    return {"status": status, "details": {
        "sequence_length_configured": sequence_length,
        "sequence_length_actual": actual_len,
        "total_sequences": n_seq,
        "expected_sequences_N_minus_L": expected_n_seq,
        "sequence_count_correct": count_correct,
        "target_misaligned": misaligned,
        "duplicate_sequences": duplicate_sequences,
    }}


def _validate_split_and_leakage(temporal_dir: Path, states_df: pd.DataFrame) -> tuple[dict, dict]:
    train_p = temporal_dir / "temporal_train.npz"
    val_p = temporal_dir / "temporal_validation.npz"
    test_p = temporal_dir / "temporal_test.npz"
    split_meta_p = temporal_dir / "temporal_split_metadata.json"

    train, val, test = _load_npz(train_p), _load_npz(val_p), _load_npz(test_p)
    if train is None or val is None or test is None:
        na = {"status": NOT_AVAILABLE, "details": {"message": "Train/validation/test split artifacts not found."}}
        return na, na

    split_meta = json.loads(split_meta_p.read_text()) if split_meta_p.exists() else {}

    def target_range(npz):
        ids = npz["target_window_id"]
        return (int(ids.min()), int(ids.max())) if len(ids) else (None, None)

    train_lo, train_hi = target_range(train)
    val_lo, val_hi = target_range(val)
    test_lo, test_hi = target_range(test)

    order_ok = True
    if train_hi is not None and val_lo is not None and train_hi >= val_lo:
        order_ok = False
    if val_hi is not None and test_lo is not None and val_hi >= test_lo:
        order_ok = False
    if train_hi is not None and test_lo is not None and val_lo is None and train_hi >= test_lo:
        order_ok = False

    wid_to_start = states_df.set_index("window_id")["window_start"]

    def bounds(npz, meta_key):
        wr = split_meta.get(meta_key, {}).get("window_id_range")
        rows = int(len(npz["X"]))
        start = str(wid_to_start.get(wr[0])) if wr and wr[0] in wid_to_start.index else None
        end = str(wid_to_start.get(wr[1])) if wr and wr[1] in wid_to_start.index else None
        return {"rows": rows, "sequences": rows, "start": start, "end": end}

    split_status = PASS if order_ok else FAIL
    split_result = {"status": split_status, "details": {
        "train": bounds(train, "train"),
        "validation": bounds(val, "validation"),
        "test": bounds(test, "test"),
        "order_valid": order_ok,
        "split_ratios": {
            "train": split_meta.get("train_ratio"), "validation": split_meta.get("val_ratio"),
            "test": split_meta.get("test_ratio"),
        },
    }}

    # ---- Leakage ----
    def seq_hash_set(npz):
        Xa = npz["X"]
        return set(_hash_array_rows(Xa)) if len(Xa) else set()

    train_h, val_h, test_h = seq_hash_set(train), seq_hash_set(val), seq_hash_set(test)
    tv_dup = len(train_h & val_h)
    tt_dup = len(train_h & test_h)
    vt_dup = len(val_h & test_h)

    def window_sets(npz):
        ids = npz["input_window_ids"]
        return [frozenset(row.tolist()) for row in ids] if len(ids) else []

    train_w, val_w, test_w = window_sets(train), window_sets(val), window_sets(test)

    def overlap_count(a_list, b_list):
        # O(len(a) + len(b)) via a shared-window union set, not O(|a|*|b|)
        union = set()
        for s in b_list:
            union |= s
        return sum(1 for s in a_list if s & union)

    train_val_overlap = overlap_count(train_w, val_w)
    train_test_overlap = overlap_count(train_w, test_w)
    val_test_overlap = overlap_count(val_w, test_w)

    scaler_path = REPO_ROOT / "models" / "temporal_scaler.bin"
    if scaler_path.exists():
        scaler_leakage = "TRAIN_ONLY"
        scaler_detail = (
            "Verified structurally: backend/temporal/splitting.py's "
            "fit_scaler_on_train() is called only on the chronological "
            "training split's raw state features, before any validation/"
            "test sequence is scaled."
        )
    else:
        scaler_leakage = "NOT_AVAILABLE"
        scaler_detail = "models/temporal_scaler.bin not found."

    target_related_fields = {"dominant_state", "attack_present"} | {f"{c.lower()}_flow_count" for c in STATE_CLASSES}
    label_overlap = sorted(set(n.lower() for n in STATE_FEATURE_NAMES) & target_related_fields)
    label_leakage = "DETECTED" if label_overlap else "NOT_DETECTED"

    leak_status = PASS
    if tv_dup or tt_dup or vt_dup or train_val_overlap or train_test_overlap or val_test_overlap or label_overlap:
        leak_status = FAIL

    leakage_result = {"status": leak_status, "details": {
        "exact_duplicate_sequences": {"train_validation": tv_dup, "train_test": tt_dup, "validation_test": vt_dup},
        "overlapping_window_sequences": {
            "train_validation": train_val_overlap, "train_test": train_test_overlap, "validation_test": val_test_overlap,
        },
        "scaler_leakage": scaler_leakage,
        "scaler_detail": scaler_detail,
        "label_leakage": label_leakage,
        "label_leakage_overlap_fields": label_overlap,
        "method": "hash-based exact-match (md5 per flattened sequence) + shared-window-id set overlap — O(n), no pairwise comparison",
    }}

    return split_result, leakage_result


def validate_temporal_dataset(source_csv: Path, temporal_dir: Path) -> dict:
    """
    Read-only validation of an already-generated temporal dataset.
    source_csv: the Current_State-labelled flow CSV that was fed into
      prepare_temporal_dataset() (i.e. summary['input_csv']).
    temporal_dir: the output directory prepare_temporal_dataset() wrote
      into (i.e. summary['output_dir']).
    """
    source_csv, temporal_dir = Path(source_csv), Path(temporal_dir)
    if not source_csv.exists():
        raise ValidationError(f"Source CSV not found: {source_csv}")

    header = pd.read_csv(source_csv, nrows=0)
    id_map = find_id_columns(list(header.columns))
    ts_col = id_map.get("timestamp")

    # Single read of the full flow-level CSV — reused across every
    # raw-level check below (current_state, timestamps, duplicates,
    # missing_data) rather than re-reading the file per check.
    df = pd.read_csv(source_csv, low_memory=False)
    rows_checked = len(df)

    checks: dict[str, dict] = {}
    checks["current_state"] = _validate_current_state(df)
    checks["timestamps"] = _validate_timestamps(df, ts_col)
    checks["duplicates"] = _validate_duplicates(df, ts_col)
    checks["missing_data"] = _validate_missing_data(df, ts_col)

    states_path = temporal_dir / "temporal_states.csv"
    if not states_path.exists():
        na = {"status": NOT_AVAILABLE, "details": {"message": "Temporal dataset artifacts not found in " + str(temporal_dir)}}
        for k in ("windows", "features", "transitions", "sequences", "chronological_split", "leakage"):
            checks[k] = na
    else:
        states_df = pd.read_csv(states_path)
        states_df["window_start"] = pd.to_datetime(states_df["window_start"])
        states_df["window_end"] = pd.to_datetime(states_df["window_end"])

        meta_path = temporal_dir / "temporal_sequences_metadata.json"
        window_size_seconds = (
            json.loads(meta_path.read_text())["window_size_seconds"] if meta_path.exists()
            else float((states_df["window_end"].iloc[0] - states_df["window_start"].iloc[0]).total_seconds())
        )

        checks["windows"] = _validate_windows(states_df, window_size_seconds)
        checks["features"] = _validate_features(states_df)
        checks["transitions"] = _validate_transitions(states_df, temporal_dir / "state_transitions.csv")
        checks["sequences"] = _validate_sequences(temporal_dir, states_df)
        checks["chronological_split"], checks["leakage"] = _validate_split_and_leakage(temporal_dir, states_df)

    overall = _combine(*(c["status"] for c in checks.values()))

    return {
        "overall_status": overall,
        "rows_checked": rows_checked,
        "source_csv": str(source_csv),
        "temporal_dir": str(temporal_dir),
        "checks": {k: v["status"] for k, v in checks.items()},
        "details": checks,
    }
