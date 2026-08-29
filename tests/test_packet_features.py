"""Packet-level (PCAP-derived) feature extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.extraction.packet_features import (
    PacketFeatureError,
    _classify_port_order,
    extract_packet_features,
    packet_features_summary,
)

ROOT = Path(__file__).resolve().parents[1]
# One of the small demo captures extracted from captures.zip. Skip cleanly
# if it is not present (they are git-ignored under pcaps/).
DEMO_PCAPS = [
    ROOT / "pcaps" / "amp.TCP.syn.optionallyACK.optionallysamePort.pcapng",
    ROOT / "pcaps" / "amp.UDP.isakmp.pcap",
]
EXPECTED_COLUMNS = {
    "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packet_count",
    "ttl_mean", "ttl_std", "tcp_window_mean", "tcp_window_min",
    "retransmission_count", "ip_fragment_count", "payload_len_mean",
    "payload_len_std",
}


def _first_available():
    for p in DEMO_PCAPS:
        if p.exists():
            return p
    pytest.skip("no demo pcap present under pcaps/")


def test_extract_returns_expected_schema():
    pcap = _first_available()
    df = extract_packet_features(pcap)
    assert not df.empty
    assert set(df.columns) == EXPECTED_COLUMNS
    assert (df["packet_count"] >= 1).all()
    # TTL is a 1-byte IP field.
    assert df["ttl_mean"].between(0, 255).all()
    assert (df["ttl_std"] >= 0).all()
    assert (df["retransmission_count"] >= 0).all()
    assert (df["ip_fragment_count"] >= 0).all()
    assert (df["payload_len_mean"] >= 0).all()


def test_flow_key_is_direction_independent():
    pcap = _first_available()
    df = extract_packet_features(pcap)
    # No two rows should be exact A<->B mirrors of each other: the
    # direction-independent key must have merged them.
    seen = set()
    for _, r in df.iterrows():
        a = (r["src_ip"], r["src_port"], r["dst_ip"], r["dst_port"], r["protocol"])
        b = (r["dst_ip"], r["dst_port"], r["src_ip"], r["src_port"], r["protocol"])
        assert b not in seen
        seen.add(a)


def test_missing_file_raises():
    with pytest.raises(PacketFeatureError):
        extract_packet_features(ROOT / "pcaps" / "does_not_exist.pcap")


def test_classify_port_order():
    assert _classify_port_order(list(range(20, 45))) == "sequential"
    assert _classify_port_order([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]) == "strided"
    assert _classify_port_order([80, 443, 22]) == "none"  # too few
    wide = [11, 2222, 55, 9001, 333, 4004, 77, 6060, 8, 1234, 5678, 90, 4321,
            65000, 22, 8443, 1, 60000, 12, 40000, 7, 30000]
    assert _classify_port_order(wide) == "randomised"


def test_packet_features_summary_shape():
    pcap = _first_available()
    out = packet_features_summary(pcap)
    assert out["flow_count"] == len(out["flows"])
    assert set(out["columns"]) == EXPECTED_COLUMNS
    assert out["port_scan"]["pattern"] in {"none", "sequential", "strided", "randomised"}
