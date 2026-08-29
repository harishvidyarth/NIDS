"""Overall-state summary + capture drop parsing."""
from __future__ import annotations

from backend.capture.capture import _parse_capture_drops
from backend.prediction.predict import summarize_states


def test_dominant_state_majority_benign_is_benign():
    s = summarize_states({"BENIGN": 335, "DDoS": 13, "DoS": 0, "PortScan": 0}, 348)
    assert s["dominant_state"] == "BENIGN"          # was "MALICIOUS" before
    assert s["attack_flow_count"] == 13
    assert s["malicious_flow_ratio"] == round(13 / 348, 4)
    assert s["attack_present"] is True


def test_dominant_state_all_benign():
    s = summarize_states({"BENIGN": 100, "DDoS": 0, "DoS": 0, "PortScan": 0}, 100)
    assert s["dominant_state"] == "BENIGN"
    assert s["attack_flow_count"] == 0
    assert s["malicious_flow_ratio"] == 0.0
    assert s["attack_present"] is False


def test_dominant_state_attack_majority():
    s = summarize_states({"BENIGN": 5, "DDoS": 90, "DoS": 5, "PortScan": 0}, 100)
    assert s["dominant_state"] == "DDoS"
    assert s["attack_flow_count"] == 95
    assert s["malicious_flow_ratio"] == 0.95


def test_dominant_state_tie_prefers_benign():
    s = summarize_states({"BENIGN": 50, "DDoS": 50, "DoS": 0, "PortScan": 0}, 100)
    assert s["dominant_state"] == "BENIGN"          # tie -> earlier CLASS_NAMES


def test_summarize_states_no_scored_flows():
    s = summarize_states({"BENIGN": 0, "DDoS": 0, "DoS": 0, "PortScan": 0}, 0)
    assert s["malicious_flow_ratio"] == 0.0
    assert s["attack_present"] is False


def test_parse_drops_tcpdump():
    stderr = (
        "1234 packets captured\n"
        "5000 packets received by filter\n"
        "128 packets dropped by kernel\n"
    )
    received, dropped = _parse_capture_drops(stderr)
    assert received == 5000
    assert dropped == 128


def test_parse_drops_dumpcap():
    stderr = "Packets captured: 4096\nPackets received/dropped on interface '\\Device\\NPF_{X}': 12,000/512 (pcap:512/dumpcap:0)\n"
    received, dropped = _parse_capture_drops(stderr)
    assert received == 12000
    assert dropped == 512


def test_parse_drops_absent():
    assert _parse_capture_drops("") == (None, None)
    assert _parse_capture_drops("nothing useful here") == (None, None)
