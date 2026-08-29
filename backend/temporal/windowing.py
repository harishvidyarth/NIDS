"""
Timestamp parsing, chronological sorting, and fixed-size time-window
assignment. No timestamp is ever invented: rows whose timestamp can't be
parsed are reported explicitly (see TimestampParseResult) and excluded
from windowing, never silently defaulted to "now" or a neighboring row's
time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class TemporalError(Exception):
    pass


@dataclass
class TimestampParseResult:
    parsed: pd.Series          # datetime64, NaT for unparseable rows
    valid_mask: pd.Series      # bool
    n_total: int
    n_valid: int
    n_invalid: int
    invalid_row_indices: list  # original DataFrame index labels
    format_used: str


def parse_timestamps(series: pd.Series) -> TimestampParseResult:
    n_total = len(series)

    # Primary path: standard datetime parsing (handles the project's own
    # "YYYY-MM-DD HH:MM:SS" format and most common variants, including
    # timezone-aware strings if present).
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    format_used = "datetime"

    # Fallback: if datetime parsing found nothing at all but the column is
    # actually numeric, it may be a Unix epoch (seconds or milliseconds) —
    # a real, common timestamp representation, not a fabrication.
    if parsed.notna().sum() == 0:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > 0:
            unit = "ms" if numeric.dropna().median() > 1e12 else "s"
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
            format_used = f"unix_epoch_{unit}"

    # Normalize any timezone-aware values to naive UTC so arithmetic
    # against other rows is well-defined regardless of per-row tz mix.
    if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    valid_mask = parsed.notna()
    invalid_row_indices = series.index[~valid_mask].tolist()

    return TimestampParseResult(
        parsed=parsed,
        valid_mask=valid_mask,
        n_total=n_total,
        n_valid=int(valid_mask.sum()),
        n_invalid=int((~valid_mask).sum()),
        invalid_row_indices=invalid_row_indices,
        format_used=format_used,
    )


def sort_chronologically(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Sorted ascending by timestamp; ties broken by original row order
    (stable sort) so same-second flows keep a deterministic order."""
    return df.sort_values(by=ts_col, kind="stable").reset_index(drop=True)


def assign_windows(df: pd.DataFrame, ts_col: str, window_size_seconds: int) -> pd.DataFrame:
    """
    Adds window_id / window_start / window_end columns. df must already be
    sorted chronologically by ts_col with only valid (non-NaT) timestamps.
    window_id is derived purely from elapsed time since the first flow —
    windows with zero flows simply don't appear (no fabricated empty
    rows); consecutive windows in the output CSV may therefore be more
    than window_size_seconds apart, which is why state_transitions.csv
    carries an explicit time_delta_seconds rather than assuming a
    constant step.
    """
    if df.empty:
        raise TemporalError("Cannot assign windows: no valid timestamped rows.")

    t0 = df[ts_col].iloc[0]
    offsets_seconds = (df[ts_col] - t0).dt.total_seconds()
    window_id = np.floor(offsets_seconds / window_size_seconds).astype(int)

    out = df.copy()
    out["window_id"] = window_id
    out["window_start"] = t0 + pd.to_timedelta(window_id * window_size_seconds, unit="s")
    out["window_end"] = out["window_start"] + pd.Timedelta(seconds=window_size_seconds)
    return out
