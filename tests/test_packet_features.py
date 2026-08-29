"""Packet-level (PCAP-derived) feature extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.extraction.packet_features import PacketFeatureError, extract_packet_features

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
