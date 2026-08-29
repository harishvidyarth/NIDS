"""
Packet-level feature extraction (PCAP-derived), complementing the
flow-level features cicflowmeter produces.

The SIH "World Models" brief calls for *both* levels: flow-level features
capture aggregate behaviour (a SYN flood), while packet-level features
"expose timing and sequencing patterns (a slow reconnaissance scan
designed to evade flow-based thresholds)". cicflowmeter emits only flow
aggregates and — in this build — does not even emit source/destination IP
columns, so its CSV cannot be joined back to per-flow packet stats.

This module reads the PCAP directly with Scapy (already a dependency,
scapy==2.6.1) in a single streaming pass and computes, per bidirectional
5-tuple flow:

  ttl_mean, ttl_std          IP TTL distribution across the session
  tcp_window_mean/_min        TCP advertised receive window
  retransmission_count        repeated (seq, payload-len) in one direction
  ip_fragment_count           packets with MF set or frag-offset > 0
  payload_len_mean/_std       L4 payload size distribution
  packet_count                packets in the flow

It is standalone: `extract_packet_features()` returns a DataFrame and the
CLI writes a CSV. Nothing here is fed to the frozen 77-feature ANN.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


class PacketFeatureError(Exception):
    pass


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Direction-independent key so both halves of a conversation merge."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    lo, hi = sorted([a, b])
    return (lo[0], lo[1], hi[0], hi[1], proto)


def extract_packet_features(pcap_path: Path | str) -> pd.DataFrame:
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise PacketFeatureError(f"PCAP not found: {pcap_path}")
    if pcap_path.stat().st_size == 0:
        raise PacketFeatureError(f"PCAP is empty: {pcap_path}")

    try:
        from scapy.all import PcapReader, IP, IPv6, TCP, UDP
    except ImportError as exc:  # pragma: no cover
        raise PacketFeatureError(f"scapy is required: {exc}") from exc

    ttls = defaultdict(list)
    windows = defaultdict(list)
    payloads = defaultdict(list)
    frag = defaultdict(int)
    pkts = defaultdict(int)
    seen_seq = defaultdict(set)
    retrans = defaultdict(int)

    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                if IP in pkt:
                    ip = pkt[IP]
                    src, dst, ttl = ip.src, ip.dst, ip.ttl
                    is_frag = bool(ip.flags & 0x1) or ip.frag > 0
                elif IPv6 in pkt:
                    ip = pkt[IPv6]
                    src, dst, ttl = ip.src, ip.dst, ip.hlim
                    is_frag = False
                else:
                    continue

                if TCP in pkt:
                    l4 = pkt[TCP]
                    proto = "TCP"
                elif UDP in pkt:
                    l4 = pkt[UDP]
                    proto = "UDP"
                else:
                    continue

                key = _flow_key(src, dst, int(l4.sport), int(l4.dport), proto)
                pkts[key] += 1
                ttls[key].append(int(ttl))
                if is_frag:
                    frag[key] += 1
                payload = bytes(l4.payload)
                payloads[key].append(len(payload))
                if proto == "TCP":
                    windows[key].append(int(l4.window))
                    marker = (src, int(l4.sport), int(l4.seq), len(payload))
                    if len(payload) > 0 or (l4.flags & 0x02):  # data or SYN
                        if marker in seen_seq[key]:
                            retrans[key] += 1
                        else:
                            seen_seq[key].add(marker)
    except Exception as exc:  # pragma: no cover - corrupt pcap tail etc.
        if not pkts:
            raise PacketFeatureError(f"Could not read packets from {pcap_path}: {exc}") from exc

    if not pkts:
        raise PacketFeatureError("No IPv4/IPv6 TCP/UDP packets found in the capture.")

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))

    rows = []
    for key in pkts:
        s_ip, s_port, d_ip, d_port, proto = key
        w = windows.get(key, [])
        p = payloads.get(key, [])
        rows.append({
            "src_ip": s_ip, "src_port": s_port,
            "dst_ip": d_ip, "dst_port": d_port, "protocol": proto,
            "packet_count": pkts[key],
            "ttl_mean": round(_mean(ttls[key]), 3),
            "ttl_std": round(_std(ttls[key]), 3),
            "tcp_window_mean": round(_mean(w), 3),
            "tcp_window_min": float(min(w)) if w else 0.0,
            "retransmission_count": retrans.get(key, 0),
            "ip_fragment_count": frag.get(key, 0),
            "payload_len_mean": round(_mean(p), 3),
            "payload_len_std": round(_std(p), 3),
        })
    return pd.DataFrame(rows).sort_values("packet_count", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m backend.extraction.packet_features <pcap> <out.csv>")
        raise SystemExit(1)
    frame = extract_packet_features(sys.argv[1])
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sys.argv[2], index=False)
    print(f"Wrote {len(frame)} flow rows -> {sys.argv[2]}")
