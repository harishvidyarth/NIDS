"""
Explicit validation checks for the temporal pipeline (task spec section
17, checks 1-10). Each check function returns (passed: bool, message: str)
rather than silently passing — the orchestrator (temporal_dataset.py)
collects all of them into a validation report and raises TemporalError on
any hard failure among checks 1-7; checks 8-10 are asserted with a plain
`assert` right where the relevant data is constructed, since they encode
invariants the code itself guarantees by construction (a violation there
would mean a real bug, not a data-quality issue).
"""
from __future__ import annotations

import pandas as pd

from .config import STATE_CLASSES


def check_timestamp_column_exists(columns: list[str], ts_col: str | None) -> tuple[bool, str]:
    if ts_col is None:
        return False, "No timestamp column found in the input CSV."
    return True, f"Timestamp column found: '{ts_col}'"


def check_timestamps_parseable(n_total: int, n_valid: int, n_invalid: int) -> tuple[bool, str]:
    if n_valid == 0:
        return False, f"None of {n_total} timestamp values could be parsed."
    msg = f"{n_valid}/{n_total} timestamps parsed successfully"
    if n_invalid:
        msg += f" ({n_invalid} invalid rows excluded)"
    return True, msg


def check_chronological_order(df: pd.DataFrame, ts_col: str) -> tuple[bool, str]:
    is_sorted = df[ts_col].is_monotonic_increasing
    return is_sorted, ("Rows are sorted chronologically" if is_sorted
                        else "Rows are NOT in chronological order")


def check_window_assignment(df: pd.DataFrame, window_size_seconds: int) -> tuple[bool, str]:
    """Every flow's window_start/window_end must actually bracket its own
    timestamp — i.e. the window really does contain the flow it claims to."""
    bad = df[(df["timestamp_parsed"] < df["window_start"]) | (df["timestamp_parsed"] >= df["window_end"])]
    ok = bad.empty
    return ok, ("Every flow's timestamp falls within its assigned window"
                if ok else f"{len(bad)} flow(s) assigned to the wrong window")


def check_no_future_leakage(states_df: pd.DataFrame) -> tuple[bool, str]:
    """window_start must be strictly non-decreasing across window_id
    order — a window can never be built from data timestamped after a
    later window's start."""
    ok = states_df["window_start"].is_monotonic_increasing
    return ok, ("No future window contributes to an earlier state"
                if ok else "Window ordering is inconsistent with time — possible future leakage")


def check_state_features_numeric(states_df: pd.DataFrame, feature_names: list[str]) -> tuple[bool, str]:
    bad_cols = [c for c in feature_names if not pd.api.types.is_numeric_dtype(states_df[c])]
    non_finite = states_df[feature_names].isna().to_numpy().sum() if not bad_cols else None
    ok = not bad_cols
    if not ok:
        return False, f"Non-numeric state feature column(s): {bad_cols}"
    return True, f"All {len(feature_names)} state features are numeric ({non_finite} NaN cell(s))"


def check_state_labels_valid(states_df: pd.DataFrame) -> tuple[bool, str]:
    invalid = set(states_df["dominant_state"].unique()) - set(STATE_CLASSES)
    ok = not invalid
    return ok, ("All dominant_state values are valid classes"
                if ok else f"Invalid state label(s) found: {invalid}")


def run_all_checks(context: dict) -> dict:
    """context carries whatever each check needs; see temporal_dataset.py
    for exactly what's passed in. Returns a dict of check_name -> (passed, message)."""
    results = {}
    results["1_timestamp_column_exists"] = check_timestamp_column_exists(
        context["columns"], context["ts_col"])
    results["2_timestamps_parseable"] = check_timestamps_parseable(
        context["n_total"], context["n_valid"], context["n_invalid"])
    results["3_chronological_order"] = check_chronological_order(
        context["sorted_df"], "timestamp_parsed")
    results["4_window_assignment"] = check_window_assignment(
        context["windowed_df"], context["window_size_seconds"])
    results["5_no_future_leakage"] = check_no_future_leakage(context["states_df"])
    results["6_state_features_numeric"] = check_state_features_numeric(
        context["states_df"], context["feature_names"])
    results["7_state_labels_valid"] = check_state_labels_valid(context["states_df"])
    return results
