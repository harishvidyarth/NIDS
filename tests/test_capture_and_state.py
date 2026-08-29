"""Overall-state summary + capture drop parsing."""
from __future__ import annotations

from backend.capture.capture import _parse_capture_drops
from backend.prediction.predict import summarize_states


def test_few_ann_ddos_flows_are_suspicious_not_attack():
    # 13 of 348 (3.7%) DDoS flows, ANN only, no signature hit -> below the
    # DDoS gate -> SUSPICIOUS, not a red ATTACK.
    s = summarize_states({"BENIGN": 335, "DDoS": 13, "DoS": 0, "PortScan": 0}, 348)
    assert s["dominant_state"] == "BENIGN"          # was "MALICIOUS" before
    assert s["attack_flow_count"] == 13
    assert s["attack_present"] is True
    assert s["attack_class"] == "DDoS"              # still disclosed
    assert s["verdict"] == "SUSPICIOUS — DDoS?"
    assert s["confidence"] == "low"


def test_signature_hit_clears_the_gate():
    # Same few flows, but the deterministic signature layer flagged DDoS.
    s = summarize_states(
        {"BENIGN": 335, "DDoS": 13, "DoS": 0, "PortScan": 0}, 348,
        signature_attack_class="DDoS",
    )
    assert s["verdict"] == "ATTACK — DDoS"
    assert s["confidence"] == "high"


def test_ddos_clears_gate_on_volume():
    # 30 of 100 (30%) DDoS flows -> clears MIN_ATTACK_FLOWS_DDOS/ratio.
    s = summarize_states({"BENIGN": 70, "DDoS": 30, "DoS": 0, "PortScan": 0}, 100)
    assert s["verdict"] == "ATTACK — DDoS"
    assert s["confidence"] == "high"


def test_verdict_benign_when_no_attack_flows():
    s = summarize_states({"BENIGN": 100, "DDoS": 0, "DoS": 0, "PortScan": 0}, 100)
    assert s["attack_class"] is None
    assert s["verdict"] == "BENIGN"
    assert s["confidence"] == "none"


def test_attack_class_ties_prefer_more_severe():
    # DoS and PortScan tie on count; DoS is the earlier (more severe) entry.
    # 20% attack ratio, 10 DoS flows -> clears the non-DDoS gate.
    s = summarize_states({"BENIGN": 80, "DDoS": 0, "DoS": 10, "PortScan": 10}, 100)
    assert s["dominant_state"] == "BENIGN"
    assert s["attack_class"] == "DoS"
    assert s["verdict"] == "ATTACK — DoS"


def test_one_off_portscan_flow_is_suspicious():
    s = summarize_states({"BENIGN": 99, "DDoS": 0, "DoS": 0, "PortScan": 1}, 100)
    assert s["attack_class"] == "PortScan"
    assert s["verdict"] == "SUSPICIOUS — PortScan?"
    assert s["confidence"] == "low"


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
