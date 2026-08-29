"""
Aggregates the flows in each time window into a single deterministic
network-state vector, a dominant-state label, an attack_present flag, and
per-class flow counts. Never selects a single representative flow — every
number here is a genuine aggregate (sum/mean/min/max/var/nunique) over
every flow that fell in the window.
"""
from __future__ import annotations

import pandas as pd

from ..prediction.features import match_columns, TRAINING_FEATURES
from .config import STATE_CLASSES
from .schema import AGGREGATIONS, DST_PORT_TRAINING_FEATURE, STATE_FEATURE_NAMES, find_id_columns
from .windowing import TemporalError

_AGG_FUNC_MAP = {"sum": "sum", "mean": "mean", "min": "min", "max": "max", "var": "var"}


def build_temporal_states(
    df: pd.DataFrame, current_state_col: str, window_size_seconds: int
) -> tuple[pd.DataFrame, dict]:
    """
    df must already be sorted chronologically and have window_id/
    window_start/window_end (see windowing.assign_windows). Returns
    (temporal_states_df, warnings_dict).
    """
    warnings: dict = {}

    col_map = match_columns(list(df.columns))  # reuses the exact 77-feature matcher
    id_cols = find_id_columns(list(df.columns))

    if "src_ip" not in id_cols:
        warnings["src_ip"] = "No src_ip/Src IP column found; unique_src_ip_count is unavailable and recorded as zero."
    if "dst_ip" not in id_cols:
        warnings["dst_ip"] = "No dst_ip/Dst IP column found; unique_dst_ip_count is unavailable and recorded as zero."
    if "src_port" not in id_cols:
        warnings["src_port"] = "No src_port/Src Port column found; unique_src_port_count is unavailable and recorded as zero."

    dst_port_col = col_map.get(DST_PORT_TRAINING_FEATURE)  # always present in a valid ANN-schema CSV

    rows = []
    for window_id, g in df.groupby("window_id", sort=True):
        row = {
            "window_id": int(window_id),
            "window_start": g["window_start"].iloc[0],
            "window_end": g["window_end"].iloc[0],
            "flow_count": len(g),
        }

        row["unique_src_ip_count"] = int(g[id_cols["src_ip"]].nunique()) if "src_ip" in id_cols else 0
        row["unique_dst_ip_count"] = int(g[id_cols["dst_ip"]].nunique()) if "dst_ip" in id_cols else 0
        row["unique_src_port_count"] = int(g[id_cols["src_port"]].nunique()) if "src_port" in id_cols else 0
        row["unique_dst_port_count"] = int(g[dst_port_col].nunique()) if dst_port_col else 0

        for out_name, training_feature, func in AGGREGATIONS:
            src_col = col_map[training_feature]
            values = pd.to_numeric(g[src_col], errors="coerce")
            agg = getattr(values, _AGG_FUNC_MAP[func])()
            row[out_name] = float(agg) if pd.notna(agg) else 0.0

        row["total_packets"] = row["total_fwd_packets"] + row["total_bwd_packets"]
        row["total_bytes"] = row["total_fwd_bytes"] + row["total_bwd_bytes"]
        row["packets_per_second"] = row["total_packets"] / window_size_seconds
        row["bytes_per_second"] = row["total_bytes"] / window_size_seconds
        row["flows_per_second"] = row["flow_count"] / window_size_seconds

        class_counts = g.loc[g[current_state_col].isin(STATE_CLASSES), current_state_col].value_counts()
        for cls in STATE_CLASSES:
            row[f"{cls.lower()}_flow_count"] = int(class_counts.get(cls, 0))

        if class_counts.empty:
            continue
        dominant_state = class_counts.idxmax()  # dominant by scoreable flow count
        # Deterministic tie-break: if multiple classes share the max count,
        # prefer whichever is listed first in STATE_CLASSES so re-runs are
        # reproducible rather than depending on pandas' internal ordering.
        max_count = class_counts.max()
        tied = [c for c in STATE_CLASSES if class_counts.get(c, 0) == max_count]
        if len(tied) > 1:
            dominant_state = tied[0]
        row["dominant_state"] = dominant_state
        row["attack_present"] = int(dominant_state != "BENIGN" or any(
            row[f"{c.lower()}_flow_count"] > 0 for c in STATE_CLASSES if c != "BENIGN"
        ))

        rows.append(row)

    states_df = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)

    ordered_cols = (
        ["window_id", "window_start", "window_end"]
        + STATE_FEATURE_NAMES
        + [f"{c.lower()}_flow_count" for c in STATE_CLASSES]
        + ["dominant_state", "attack_present"]
    )
    states_df = states_df[ordered_cols]
    return states_df, warnings
