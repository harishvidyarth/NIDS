from __future__ import annotations

import pytest

from backend.response.policy import PolicyError, evaluate_prediction


def _flow(src: str, dst: str = "203.0.113.20", state: str = "PortScan", port: int = 443):
    return {
        "Src IP": src,
        "Dst IP": dst,
        "Dst Port": port,
        "Protocol": "TCP",
        "signature_state": state,
        "predicted_state": "BENIGN",
    }


def test_signature_confirmed_portscan_is_executable():
    result = evaluate_prediction(
        {
            "signature_verdict": "PortScan",
            "signature_hits": [{"state": "PortScan", "rule": "fanout-port-scan"}],
            "port_scan_signature": {"src": "198.51.100.7", "dst": "203.0.113.20"},
            "flows": [_flow("198.51.100.7")],
        },
        ttl_minutes=15,
    )
    assert result.executable is True
    assert result.confidence_source == "deterministic_signature"
    assert result.targets[0].source_ip == "198.51.100.7"


def test_ann_only_and_forecast_results_are_recommendation_only():
    ann = evaluate_prediction(
        {"signature_verdict": None, "attack_class": "DoS", "confidence": "high", "flows": []},
        ttl_minutes=15,
    )
    forecast = evaluate_prediction(
        {"future_labels_are_forecasts": True, "horizons": [{"predicted_state": "DDoS"}]},
        ttl_minutes=15,
    )
    assert ann.executable is False
    assert "ANN-only" in " ".join(ann.limitations)
    assert forecast.executable is False
    assert "Forecast" in " ".join(forecast.limitations)


def test_ddos_more_than_64_sources_requires_upstream_mitigation():
    flows = [_flow(f"198.51.100.{index}", state="DDoS") for index in range(1, 66)]
    result = evaluate_prediction(
        {
            "signature_verdict": "DDoS",
            "signature_hits": [{"state": "DDoS", "rule": "rolling-source-fanin-flood"}],
            "flows": flows,
        },
        ttl_minutes=15,
    )
    assert result.executable is False
    assert result.upstream_recommendation
    narrowed = evaluate_prediction(
        {"signature_verdict": "DDoS", "signature_hits": [{"state": "DDoS"}], "flows": flows},
        ttl_minutes=15, selected_targets=[{"source_ip": "198.51.100.1"}],
    )
    assert narrowed.executable is False


def test_single_source_dos_is_scoped_to_victim_protocol_and_port():
    result = evaluate_prediction(
        {"signature_verdict": "DoS", "signature_hits": [{"state": "DoS"}],
         "flows": [_flow("198.51.100.9", state="DoS", port=8443)]}, ttl_minutes=15,
    )
    assert result.executable is True
    assert result.targets[0].victim_ip == "203.0.113.20"
    assert result.targets[0].protocol == "tcp"
    assert result.targets[0].destination_port == 8443


def test_ambiguous_dos_and_unattributed_ddos_are_not_executable():
    ambiguous = evaluate_prediction(
        {"signature_verdict": "DoS", "signature_hits": [{"state": "DoS"}],
         "flows": [_flow("198.51.100.9", state="DoS"), _flow("198.51.100.10", state="DoS")]},
        ttl_minutes=15,
    )
    missing = evaluate_prediction(
        {"signature_verdict": "DDoS", "signature_hits": [{"state": "DDoS"}],
         "flows": [_flow("198.51.100.9", state="DDoS"), {"signature_state": "DDoS", "dst_ip": "203.0.113.20"}]},
        ttl_minutes=15,
    )
    assert ambiguous.executable is False
    assert missing.executable is False


@pytest.mark.parametrize("source", ["127.0.0.1", "0.0.0.0", "224.0.0.1", "255.255.255.255"])
def test_unsafe_ip_classes_are_rejected(source):
    result = evaluate_prediction(
        {
            "signature_verdict": "PortScan",
            "signature_hits": [{"state": "PortScan"}],
            "port_scan_signature": {"src": source, "dst": "203.0.113.20"},
            "flows": [_flow(source)],
        },
        ttl_minutes=15,
    )
    assert result.executable is False
    assert result.targets == []


def test_management_allowlist_is_protected_and_injection_is_invalid():
    protected = evaluate_prediction(
        {
            "signature_verdict": "PortScan",
            "signature_hits": [{"state": "PortScan"}],
            "port_scan_signature": {"src": "198.51.100.7", "dst": "203.0.113.20"},
            "flows": [_flow("198.51.100.7")],
        }, ttl_minutes=15, protected_addresses={"198.51.100.7"},
    )
    assert protected.executable is False
    with pytest.raises(PolicyError):
        evaluate_prediction(
            {
                "signature_verdict": "PortScan",
                "signature_hits": [{"state": "PortScan"}],
                "port_scan_signature": {"src": "1.2.3.4; rm -rf /", "dst": "203.0.113.20"},
                "flows": [],
            }, ttl_minutes=15,
        )


@pytest.mark.parametrize("ttl", [0, 61])
def test_ttl_range_is_enforced(ttl):
    with pytest.raises(PolicyError):
        evaluate_prediction({}, ttl_minutes=ttl)
