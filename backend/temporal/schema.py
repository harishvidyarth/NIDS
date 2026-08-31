"""
Column identification for the temporal pipeline. Nothing here duplicates
the existing 77-feature matching logic in backend/prediction/features.py —
`match_columns()` is reused as-is to locate every numeric source column
this module aggregates. This file only adds the small amount of
additional lookup that predict.py doesn't need: which raw column is the
timestamp, and (best-effort) which are the source/destination IP and
source port columns, so window state vectors can report unique-endpoint
counts.
"""
from __future__ import annotations

import re

from ..prediction.features import TRAINING_FEATURES, match_columns  # noqa: F401 (re-exported)

_ID_ALIASES = {
    "timestamp": {"timestamp"},
    "src_ip": {"srcip", "sourceip"},
    "dst_ip": {"dstip", "destinationip"},
    "src_port": {"srcport", "sourceport"},
    "flow_id": {"flowid"},
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_id_columns(columns: list[str]) -> dict[str, str]:
    """Best-effort map of role -> actual column name. A role missing from
    the result means that column genuinely isn't present in this CSV —
    callers must handle that explicitly (see state_builder.py), never
    invent a substitute."""
    found: dict[str, str] = {}
    for role, aliases in _ID_ALIASES.items():
        for col in columns:
            if _normalize(col) in aliases:
                found[role] = col
                break
    return found


# (output_name, training_feature_name, agg_func) — training_feature_name
# must be an exact entry of TRAINING_FEATURES so match_columns() resolves
# it to whatever the actual CSV calls it, regardless of CICFlowMeter
# naming convention (verbose / abbreviated / snake_case).
AGGREGATIONS = [
    ("total_fwd_packets", " Total Fwd Packets", "sum"),
    ("total_bwd_packets", " Total Backward Packets", "sum"),
    ("total_fwd_bytes", "Total Length of Fwd Packets", "sum"),
    ("total_bwd_bytes", " Total Length of Bwd Packets", "sum"),
    ("syn_count", " SYN Flag Count", "sum"),
    ("ack_count", " ACK Flag Count", "sum"),
    ("fin_count", "FIN Flag Count", "sum"),
    ("rst_count", " RST Flag Count", "sum"),
    ("psh_count", " PSH Flag Count", "sum"),
    ("urg_count", " URG Flag Count", "sum"),
    ("mean_iat", " Flow IAT Mean", "mean"),
    ("iat_variance", " Flow IAT Mean", "var"),
    ("min_iat", " Flow IAT Min", "min"),
    ("max_iat", " Flow IAT Max", "max"),
    ("mean_flow_duration", " Flow Duration", "mean"),
    ("max_flow_duration", " Flow Duration", "max"),
    ("mean_packet_size", " Average Packet Size", "mean"),
    ("max_packet_size", " Max Packet Length", "max"),
]

DST_PORT_TRAINING_FEATURE = " Destination Port"  # always present (part of the 77 required features)

# Final, documented order of the window-level state vector.
STATE_FEATURE_NAMES = [
    "flow_count",
    "unique_src_ip_count",
    "unique_dst_ip_count",
    "unique_src_port_count",
    "unique_dst_port_count",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_bytes",
    "packets_per_second",
    "bytes_per_second",
    "flows_per_second",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "mean_iat",
    "iat_variance",
    "min_iat",
    "max_iat",
    "mean_flow_duration",
    "max_flow_duration",
    "mean_packet_size",
    "max_packet_size",
]

# Additive XDR display/graph features. The trained forecasting models and
# their scalers continue to consume STATE_FEATURE_NAMES (the first 28 values).
XDR_FEATURE_NAMES = [
    "dns_entropy",
    "unique_sni",
    "beacon_score",
    "byte_asymmetry",
    "ja3_novelty",
    "http_error_rate",
]
STATE_FEATURE_NAMES_V2 = [*STATE_FEATURE_NAMES, *XDR_FEATURE_NAMES]
