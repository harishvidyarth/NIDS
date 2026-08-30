"""
Deterministic signature / heuristic layer, run alongside the frozen
77-feature ANN.

The ANN classifies each flow in isolation. A volumetric DDoS is a
*cross-flow* phenomenon — hundreds of short flows from many sources to one
victim — and in a mostly-benign capture the ANN often labels most of those
individual flows BENIGN, so the attack never surfaces. These rules look at
the whole flow table at once and at rows the ANN could not score
(INVALID_FEATURES), and emit an independent verdict. They never overwrite
an ANN label; `predict_csv` surfaces both and lets the UI show the
disagreement.

Thresholds are deliberately the same ones the MITRE behaviour-evidence
layer already uses (`backend/mitre/mapper.py::_behavior_evidence`):
flows/s >= 10, packets/s >= 100, SYN count >= 20, RST count >= 10.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nids.prediction")

# --- tunables (kept in one place so they can be traced in a report) ------
DDOS_MIN_SOURCE_FANIN = 20      # distinct src_ip hitting one dst_ip
DDOS_MIN_PKTS_PER_SEC = 100.0   # observed packets in one rolling second
DDOS_MIN_VICTIM_FLOW_RATIO = 0.5
DDOS_MAX_TOP_SOURCE_RATIO = 0.25
ROLLING_WINDOW_SECONDS = 1
DOS_MIN_SYN = 20                # syn_flag_cnt on a single flow
DOS_MAX_ACK_RATIO = 0.2         # ack_flag_cnt / syn_flag_cnt for a half-open flood
DOS_MIN_PKTS_PER_SEC = 100.0
PORTSCAN_MIN_UNIQUE_DPORTS = 20 # distinct dst_port from one src_ip to one dst_ip
PORTSCAN_MAX_MEAN_FWD_BYTES = 400.0  # scans send tiny probes

_BENIGN = "BENIGN"


def _num(series: Optional[pd.Series], n: int) -> pd.Series:
    if series is None:
        return pd.Series(np.zeros(n))
    return pd.to_numeric(series, errors="coerce").fillna(0.0).reset_index(drop=True)


def _port_access_pattern(ports: list[int]) -> str:
    """Classify how a set of destination ports was walked: 'sequential'
    (mostly +1 steps), 'strided' (a constant step > 1), or 'randomised'."""
    uniq = sorted({int(p) for p in ports if p and p > 0})
    if len(uniq) < 3:
        return "none"
    if np.all(np.diff(uniq) == 1):
        return "sequential"
    hit_order = [int(p) for p in ports if p and p > 0]
    steps = np.abs(np.diff(hit_order)) if len(hit_order) > 2 else np.array([])
    if steps.size and np.mean(steps == 1) >= 0.6:
        return "sequential"
    if steps.size and np.std(steps) < 1e-6 and steps[0] > 1:
        return "strided"
    return "randomised"


def flow_signatures(df: pd.DataFrame) -> dict:
    """Return per-row signature states aligned to ``df`` plus a summary.

    Result keys:
      states        list[str]  len == len(df); a CLASS_NAME or 'BENIGN'
      hits          list[dict] one per fired rule (rule, state, detail, flow_count)
      attack_class  str | None the most severe non-BENIGN state that fired
      counts        dict[str, int]
      port_scan     dict | None  {src, dst, unique_ports, pattern, ...}
    """
    n = len(df)
    states = [_BENIGN] * n
    hits: list[dict] = []
    port_scan: Optional[dict] = None

    try:
        cols = {c.strip().lower(): c for c in df.columns}

        def col(name: str) -> Optional[pd.Series]:
            real = cols.get(name)
            return df[real].reset_index(drop=True) if real is not None else None

        src_ip = col("src_ip")
        dst_ip = col("dst_ip")
        dst_port = col("dst_port")
        timestamp = next((value for value in (col("timestamp"), col("time"), col("starttime")) if value is not None), None)
        pkts_s = _num(col("flow_pkts_s"), n)
        total_packets = _num(col("tot_fwd_pkts"), n) + _num(col("tot_bwd_pkts"), n)
        syn = _num(col("syn_flag_cnt"), n)
        ack = _num(col("ack_flag_cnt"), n)
        fwd_bytes = _num(col("totlen_fwd_pkts"), n)

        idx = np.arange(n)

        distribution_evidence = []
        # ---- Rule 1: volumetric flood in a real rolling time bucket -----
        # Per-flow rates describe each flow's own duration and cannot be
        # added across an arbitrary capture. Use observed packet totals in a
        # fixed one-second bucket, then require victim and source distribution
        # evidence before distinguishing DDoS from a single-source DoS.
        if src_ip is not None and dst_ip is not None and timestamp is not None:
            parsed = pd.to_datetime(timestamp, errors="coerce")
            buckets = parsed.dt.floor(f"{ROLLING_WINDOW_SECONDS}s")
            valid_time = buckets.notna().to_numpy()
            for bucket in pd.unique(buckets[valid_time]):
                window_group = idx[(buckets == bucket).to_numpy()]
                window_flow_count = len(window_group)
                if not window_flow_count:
                    continue
                for victim in pd.unique(dst_ip.iloc[window_group]):
                    group = window_group[dst_ip.iloc[window_group].to_numpy() == victim]
                    source_counts = src_ip.iloc[group].astype(str).value_counts()
                    unique_sources = int(len(source_counts))
                    victim_ratio = float(len(group) / window_flow_count)
                    top_source_ratio = float(source_counts.iloc[0] / len(group)) if len(group) else 1.0
                    packet_rate = float(total_packets.iloc[group].sum() / ROLLING_WINDOW_SECONDS)
                    syn_count = float(syn.iloc[group].sum())
                    ack_count = float(ack.iloc[group].sum())
                    evidence = {
                        "window_start": pd.Timestamp(bucket).isoformat(),
                        "window_seconds": ROLLING_WINDOW_SECONDS,
                        "victim": str(victim),
                        "unique_sources": unique_sources,
                        "victim_flow_ratio": round(victim_ratio, 4),
                        "top_source_flow_ratio": round(top_source_ratio, 4),
                        "packet_rate": round(packet_rate, 4),
                        "syn_count": round(syn_count, 4),
                        "ack_to_syn_ratio": round(ack_count / syn_count, 4) if syn_count else None,
                        "rate_method": "observed_packet_count_per_fixed_window",
                    }
                    distribution_evidence.append(evidence)
                    distributed = (
                        unique_sources >= DDOS_MIN_SOURCE_FANIN
                        and victim_ratio >= DDOS_MIN_VICTIM_FLOW_RATIO
                        and top_source_ratio <= DDOS_MAX_TOP_SOURCE_RATIO
                    )
                    if distributed and packet_rate >= DDOS_MIN_PKTS_PER_SEC:
                        for row_index in group:
                            states[row_index] = "DDoS"
                        hits.append({
                            "rule": "rolling-source-fanin-flood", "state": "DDoS",
                            "detail": (
                                f"{unique_sources} concurrent sources -> {victim} at "
                                f"{packet_rate:.0f} observed packets/s"
                            ),
                            "flow_count": int(len(group)), "evidence": evidence,
                        })
                    elif unique_sources == 1 and packet_rate >= DOS_MIN_PKTS_PER_SEC:
                        ack_ratio = ack_count / syn_count if syn_count else 1.0
                        if syn_count >= DOS_MIN_SYN and ack_ratio <= DOS_MAX_ACK_RATIO:
                            for row_index in group:
                                if states[row_index] == _BENIGN:
                                    states[row_index] = "DoS"
                            hits.append({
                                "rule": "rolling-single-source-syn-flood", "state": "DoS",
                                "detail": f"single source -> {victim} at {packet_rate:.0f} observed packets/s",
                                "flow_count": int(len(group)), "evidence": evidence,
                            })

        # ---- Rule 2: SYN / half-open flood (per flow) -----------------
        syn_flood = (
            (syn >= DOS_MIN_SYN)
            & (ack <= syn * DOS_MAX_ACK_RATIO)
            & (pkts_s >= DOS_MIN_PKTS_PER_SEC)
        ).values
        if syn_flood.any():
            for i in idx[syn_flood]:
                if states[i] == _BENIGN:
                    states[i] = "DoS"
            hits.append({
                "rule": "syn-flood",
                "state": "DoS",
                "detail": (
                    f"{int(syn_flood.sum())} flow(s) with SYN>={DOS_MIN_SYN}, "
                    f"few ACKs, >={DOS_MIN_PKTS_PER_SEC:.0f} pkts/s"
                ),
                "flow_count": int(syn_flood.sum()),
            })

        # ---- Rule 3: port scan (one src sweeping many ports on one dst) -
        if src_ip is not None and dst_ip is not None and dst_port is not None:
            dports = _num(dst_port, n)
            pair = (src_ip.astype(str) + "|" + dst_ip.astype(str)).to_numpy()
            best = None
            for key in pd.unique(pair):
                g = idx[pair == key]
                s, d = key.split("|", 1)
                ports = [int(p) for p in dports.iloc[g].tolist() if p > 0]
                uniq = len(set(ports))
                mean_fwd = float(fwd_bytes.iloc[g].mean()) if len(g) else 0.0
                if uniq >= PORTSCAN_MIN_UNIQUE_DPORTS and mean_fwd <= PORTSCAN_MAX_MEAN_FWD_BYTES:
                    for i in g:
                        if states[i] == _BENIGN:
                            states[i] = "PortScan"
                    pattern = _port_access_pattern(ports)
                    hits.append({
                        "rule": "port-sweep",
                        "state": "PortScan",
                        "detail": (
                            f"{s} -> {d}: {uniq} distinct ports, "
                            f"{pattern} order, {mean_fwd:.0f} avg fwd bytes"
                        ),
                        "flow_count": int(len(g)),
                    })
                    if best is None or uniq > best["unique_ports"]:
                        best = {
                            "src": s, "dst": d, "unique_ports": uniq,
                            "pattern": pattern, "flow_count": int(len(g)),
                        }
            port_scan = best

    except Exception as exc:  # pragma: no cover - defensive
        logger.info(f"[Signatures] skipped ({exc})")
        return {
            "states": [_BENIGN] * n, "hits": [], "attack_class": None,
            "counts": {_BENIGN: n}, "port_scan": None, "distribution_evidence": [],
            "rate_method": "unavailable",
        }

    counts: dict[str, int] = {}
    for st in states:
        counts[st] = counts.get(st, 0) + 1
    severity = ["DDoS", "DoS", "PortScan"]
    attack_class = next((s for s in severity if counts.get(s, 0) > 0), None)
    return {
        "states": states,
        "hits": hits,
        "attack_class": attack_class,
        "counts": counts,
        "port_scan": port_scan,
        "distribution_evidence": distribution_evidence,
        "rate_method": "observed_packet_count_per_fixed_window" if timestamp is not None else "unavailable_no_timestamp",
    }
