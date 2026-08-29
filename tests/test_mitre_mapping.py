from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.mitre.mapper import AttackMetadataError, AttackMetadataStore, MitreAttackMapper


@pytest.fixture
def mapper():
    return MitreAttackMapper()


def strong_portscan_features():
    return {
        "flow_count": 20,
        "unique_dst_port_count": 18,
        "flows_per_second": 2.0,
        "syn_count": 16,
        "rst_count": 2,
        "packets_per_second": 30,
    }


def test_official_metadata_loads_and_has_pinned_provenance():
    store = AttackMetadataStore.load()
    assert store.version == "19.1"
    assert store.source_url.startswith("https://github.com/mitre-attack/attack-stix-data")
    assert {"T1046", "T1595", "T1498", "T1499"}.issubset(store.techniques)


def test_technique_ids_and_tactics_are_from_official_metadata(mapper):
    result = mapper.map_forecast(
        "PortScan",
        {"BENIGN": 0.05, "DDoS": 0.05, "DoS": 0.10, "PortScan": 0.80},
        strong_portscan_features(),
    )
    assert {item["technique_id"] for item in result["mitre_candidates"]} == {"T1046", "T1595"}
    assert {item["tactic"] for item in result["mitre_candidates"]} == {"Discovery", "Reconnaissance"}


def test_mapping_is_deterministic(mapper):
    arguments = (
        "PortScan",
        {"BENIGN": 0.05, "DDoS": 0.05, "DoS": 0.70, "PortScan": 0.20},
        strong_portscan_features(),
    )
    assert mapper.map_forecast(*arguments) == mapper.map_forecast(*arguments)


def test_benign_produces_no_forced_mapping(mapper):
    result = mapper.map_forecast(
        "BENIGN",
        {"BENIGN": 0.96, "DDoS": 0.01, "DoS": 0.02, "PortScan": 0.01},
        {},
    )
    assert result["attack_mapping"] is None
    assert result["mitre_candidates"] == []
    assert result["reason"] == "no attack evidence"


def test_low_evidence_is_not_asserted(mapper):
    result = mapper.map_forecast(
        "BENIGN",
        {"BENIGN": 0.45, "DDoS": 0.05, "DoS": 0.30, "PortScan": 0.20},
        {},
    )
    assert result["mapping_status"] == "INSUFFICIENT_EVIDENCE"
    assert all(item["mapping_status"] == "INSUFFICIENT_EVIDENCE" for item in result["mitre_candidates"])


def test_forecast_probability_is_separate_from_mapping_confidence(mapper):
    result = mapper.map_forecast(
        "DoS",
        {"BENIGN": 0.02, "DDoS": 0.02, "DoS": 0.91, "PortScan": 0.05},
        {"flows_per_second": 20, "syn_count": 50, "rst_count": 10, "packets_per_second": 100},
    )
    candidate = result["mitre_candidates"][0]
    assert result["forecast_probability"] == 0.91
    assert candidate["mapping_confidence"] != result["forecast_probability"]
    assert candidate["mapping_confidence_type"] == "deterministic_heuristic_not_calibrated"


def test_invalid_features_cannot_map(mapper):
    result = mapper.map_forecast(
        "INVALID_FEATURES",
        {"BENIGN": 0.01, "DDoS": 0.01, "DoS": 0.97, "PortScan": 0.01},
        strong_portscan_features(),
    )
    assert result["attack_mapping"] is None
    assert result["reason"] == "invalid features cannot support ATT&CK interpretation"


def test_mapper_supports_multiple_possible_techniques(mapper):
    result = mapper.map_forecast(
        "PortScan",
        {"BENIGN": 0.05, "DDoS": 0.05, "DoS": 0.75, "PortScan": 0.15},
        {**strong_portscan_features(), "flows_per_second": 30, "syn_count": 100},
    )
    assert {item["technique_id"] for item in result["mitre_candidates"]} == {"T1046", "T1595", "T1499"}


def test_mapper_operates_without_network_calls(monkeypatch, mapper):
    def fail(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert mapper.map_forecast(
        "DDoS",
        {"BENIGN": 0.05, "DDoS": 0.90, "DoS": 0.03, "PortScan": 0.02},
        {"flows_per_second": 50, "packets_per_second": 500, "syn_count": 200},
    )["mitre_candidates"]


def test_malformed_metadata_fails_safely(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"attack_version": "19.1", "techniques": [{"technique_id": "T9999"}]}))
    with pytest.raises(AttackMetadataError):
        AttackMetadataStore.load(path)


def test_no_fabricated_ids_in_mapping_rules(mapper):
    assert mapper.rule_technique_ids <= set(mapper.metadata.techniques)


def test_api_and_frontend_expose_mitre_context():
    api_source = Path("backend/lstm/training.py").read_text()
    html = Path("frontend/index.html").read_text()
    script = Path("frontend/app.js").read_text()
    assert "mitre_mapping" in api_source
    assert "MITRE ATT&amp;CK CONTEXT" in html
    assert "Possible ATT&amp;CK interpretation" in html
    assert "mapping_confidence" in script
