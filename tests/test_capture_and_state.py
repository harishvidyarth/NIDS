"""Overall-state summary + capture drop parsing."""
from __future__ import annotations

from backend.capture.capture import _parse_capture_drops
from backend.prediction.predict import summarize_states


def test_few_ann_ddos_flows_are_suspicious_not_attack():
    # 13 of 348 (3.7%) DDoS flows, ANN only, no signature hit -> below the
    # 20% DDoS gate -> SUSPICIOUS, not a red ATTACK.
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
    # 30 of 100 (30%) DDoS flows -> own share >= 20% -> ATTACK.
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
    # 25 DoS of 100 = 25% (its own share) >= 20% -> clears the gate.
    s = summarize_states({"BENIGN": 50, "DDoS": 0, "DoS": 25, "PortScan": 25}, 100)
    assert s["dominant_state"] == "BENIGN"
    assert s["attack_class"] == "DoS"
    assert s["attack_class_ratio"] == 0.25
    assert s["verdict"] == "ATTACK — DoS"


def test_minority_class_ratio_not_dragged_up_by_other_attack_class():
    # The reported live-dashboard capture: BENIGN 329, DDoS 9, DoS 10 of 348.
    # DoS's own share is 10/348 = 2.87% < 20% gate. Pre-fix the *combined*
    # non-BENIGN ratio (19/348 = 5.46%) wrongly cleared the gate and the
    # hero read a red "ATTACK — DoS".
    s = summarize_states({"BENIGN": 329, "DDoS": 9, "DoS": 10, "PortScan": 0}, 348)
    assert s["dominant_state"] == "BENIGN"
    assert s["attack_class"] == "DoS"
    assert s["attack_class_ratio"] == 0.0287
    assert s["verdict"] == "SUSPICIOUS — DoS?"
    assert s["confidence"] == "low"


def test_per_class_ratio_below_gate_is_suspicious_despite_combined_ratio():
    # DoS 12 of 100 = 12% < 20% gate. PortScan 12 pushes the combined
    # non-BENIGN ratio to 24%, but the gate ignores the combined figure.
    s = summarize_states({"BENIGN": 76, "DDoS": 0, "DoS": 12, "PortScan": 12}, 100)
    assert s["attack_class"] == "DoS"          # tie -> earlier / more severe
    assert s["attack_class_ratio"] == 0.12
    assert s["verdict"] == "SUSPICIOUS — DoS?"


def test_per_class_ratio_exactly_at_gate_is_attack():
    # DoS 20 of 100 = 20% == MIN_ATTACK_RATIO_OTHER. Boundary is inclusive.
    s = summarize_states({"BENIGN": 80, "DDoS": 0, "DoS": 20, "PortScan": 0}, 100)
    assert s["attack_class"] == "DoS"
    assert s["attack_class_ratio"] == 0.2
    assert s["verdict"] == "ATTACK — DoS"


def test_gate_is_percentage_only_no_absolute_floor():
    # 40 DoS flows but only 8% of a large capture -> SUSPICIOUS. A raw
    # flow count never clears the gate on its own.
    s = summarize_states({"BENIGN": 460, "DDoS": 0, "DoS": 40, "PortScan": 0}, 500)
    assert s["attack_class"] == "DoS"
    assert s["attack_class_ratio"] == 0.08
    assert s["verdict"] == "SUSPICIOUS — DoS?"


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
