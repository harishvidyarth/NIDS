"""
Canonical 77-feature schema the ANN (models/ISAA_ANN.h5) was trained on,
in the exact order used during training (see notebooks/NIDS - ANN.ipynb).

CICFlowMeter has shipped with at least two different CSV header styles over
the years:
  - "verbose" (the style CICIDS2017 itself was generated with, e.g.
    "Total Fwd Packets", "Flow Bytes/s", "Init_Win_bytes_forward")
  - "abbreviated" (newer CICFlowMeter/v4-style, e.g. "Tot Fwd Pkts",
    "Flow Byts/s", "Init Fwd Win Byts" — note the word order also changes)

The original repo's prediction notebook trusted column *position*, which
silently breaks if the installed CICFlowMeter emits either different names
or a different column order. `match_columns()` instead aligns by *meaning*:
every column name is reduced to a sorted set of canonical tokens (with
abbreviations expanded), so any CICFlowMeter build's output lines up with
the training schema regardless of naming style or column order.
"""
from __future__ import annotations

import re

# Exact training feature order (models were trained on these 77 columns,
# in this order, after MinMax scaling). Do not reorder.
TRAINING_FEATURES = [
    " Destination Port", " Flow Duration", " Total Fwd Packets",
    " Total Backward Packets", "Total Length of Fwd Packets",
    " Total Length of Bwd Packets", " Fwd Packet Length Max",
    " Fwd Packet Length Min", " Fwd Packet Length Mean",
    " Fwd Packet Length Std", "Bwd Packet Length Max",
    " Bwd Packet Length Min", " Bwd Packet Length Mean",
    " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s",
    " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min",
    "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max",
    " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std",
    " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
    " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length",
    " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s",
    " Min Packet Length", " Max Packet Length", " Packet Length Mean",
    " Packet Length Std", " Packet Length Variance", "FIN Flag Count",
    " SYN Flag Count", " RST Flag Count", " PSH Flag Count",
    " ACK Flag Count", " URG Flag Count", " CWE Flag Count",
    " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
    " Avg Fwd Segment Size", " Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    " Subflow Fwd Bytes", " Subflow Bwd Packets", " Subflow Bwd Bytes",
    "Init_Win_bytes_forward", " Init_Win_bytes_backward",
    " act_data_pkt_fwd", " min_seg_size_forward", "Active Mean",
    " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std",
    " Idle Max", " Idle Min",
]

# Non-feature identifier/label columns that may appear in a raw CICFlowMeter
# CSV and must never be fed to the model.
ID_COLUMNS_ALIASES = {
    "flowid", "srcip", "sourceip", "srcport", "sourceport",
    "dstip", "destinationip", "protocol", "timestamp", "label",
}

_SYNONYMS = {
    "tot": "total", "total": "total",
    "pkts": "pkt", "pkt": "pkt", "packets": "pkt", "packet": "pkt",
    "byts": "byte", "byte": "byte", "bytes": "byte",
    "len": "len", "length": "len",
    "cnt": "cnt", "count": "cnt",
    "var": "var", "variance": "var",
    "blk": "blk", "bulk": "blk",
    "seg": "seg", "segment": "seg",
    "avg": "avg", "average": "avg",
    "fwd": "fwd", "forward": "fwd",
    "bwd": "bwd", "backward": "bwd",
    "dst": "dst", "destination": "dst",
    "src": "src", "source": "src",
    "flag": "flag", "flags": "flag",
    "s": "s",  # trailing "/s" (per-second) token
    # CICIDS2017's original CICFlowMeter build labeled the TCP CWR flag
    # count column "CWE Flag Count" (a naming bug carried into the
    # published dataset and thus into this model's training schema).
    # Later CICFlowMeter builds correctly call it CWR. Same underlying
    # feature, so treat the two names as equivalent.
    "cwe": "cwr", "cwr": "cwr",
    "b": "blk",  # "Bytes/Bulk", "Packets/Bulk" -> *_b_avg in this build
}

# Concatenated abbreviations with no separator between their parts
# (e.g. "totlen_fwd_pkts") that the generic splitter can't break apart.
_COMPOUND_FIXES = {
    "totlen": "tot len",
}

_STOPWORDS = {"of"}


def _canonical_signature(name: str) -> tuple:
    lowered = name.strip().lower()
    for compound, expansion in _COMPOUND_FIXES.items():
        lowered = lowered.replace(compound, expansion)
    tokens = re.split(r"[^A-Za-z0-9]+", lowered)
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    tokens = [_SYNONYMS.get(t, t) for t in tokens]
    return tuple(sorted(tokens))


def match_columns(csv_columns: list[str]) -> dict[str, str]:
    """
    Map each TRAINING_FEATURES name -> the actual column name found in
    csv_columns (by canonical signature). Raises ValueError listing any
    training feature that could not be matched, instead of silently
    proceeding with misaligned columns.
    """
    by_signature: dict[tuple, str] = {}
    for col in csv_columns:
        sig = _canonical_signature(col)
        if sig not in by_signature:
            by_signature[sig] = col

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for feature in TRAINING_FEATURES:
        sig = _canonical_signature(feature)
        if sig in by_signature:
            mapping[feature] = by_signature[sig]
        else:
            missing.append(feature)

    if missing:
        raise ValueError(
            "CSV is missing "
            f"{len(missing)} required CICFlowMeter feature column(s) "
            f"(no match by name): {missing}"
        )
    return mapping


def is_id_or_label_column(name: str) -> bool:
    sig = re.sub(r"[^a-z0-9]", "", name.strip().lower())
    return sig in ID_COLUMNS_ALIASES
