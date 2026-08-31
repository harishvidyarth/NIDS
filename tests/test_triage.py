from __future__ import annotations

import io
import json

from backend.triage.service import TriageService, build_triage


def _context() -> dict:
    return {
        "verdict": {"final_verdict": "PortScan", "confidence": 0.86},
        "forecast": {"maximum_attack_probability": 0.72},
        "mitre": {
            "mitre_candidates": [
                {
                    "technique_id": "T1046",
                    "technique_name": "Network Service Discovery",
                    "tactic": "Discovery",
                    "mapping_confidence": 0.65,
                    "mapping_status": "POSSIBLE",
                    "evidence": ["destination-port diversity is elevated"],
                },
                {
                    "technique_id": "T1595",
                    "technique_name": "Active Scanning",
                    "tactic": "Reconnaissance",
                    "mapping_confidence": 0.55,
                    "mapping_status": "POSSIBLE",
                    "evidence": ["observed current network state is PortScan"],
                },
            ],
            "operator_guidance": [
                "Verify whether the scan was authorized.",
                "Review exposed services.",
                "Preserve flow evidence.",
            ],
        },
        "campaign_score": 0.61,
        "enrichment": {"beacon_score_max": 0.92, "ja3_novelty": 0.70},
        "deception_events": [],
    }


def test_template_triage_is_deterministic_and_structured():
    first = build_triage(_context())
    second = build_triage(_context())

    assert first == second
    assert first["source"] == "deterministic_template"
    assert first["advisory_only"] is True
    assert first["confidence"] == "HIGH"
    assert first["ranked_techniques"][0]["technique_id"] == "T1046"
    assert 3 <= len(first["playbook"]) <= 5
    assert "PortScan" in first["summary"]
    assert any("periodic" in reason for reason in first["why_flagged"])


def test_deception_event_forces_high_confidence_triage():
    context = _context()
    context["verdict"] = {"final_verdict": "BENIGN", "confidence": 0.10}
    context["forecast"] = {}
    context["campaign_score"] = 0.0
    context["deception_events"] = [{"event_type": "DECEPTION_CANARY_HIT"}]

    result = build_triage(context)

    assert result["confidence"] == "HIGH"
    assert result["confidence_score"] == 0.99
    assert "deception canary" in result["why_flagged"][0].lower()
    assert result["ranked_techniques"][0]["technique_id"] == "T1046"


def test_deception_supplies_ranked_technique_when_mapper_has_none():
    result = build_triage({
        "verdict": {"final_verdict": "BENIGN"},
        "deception_events": [{"event_type": "DECEPTION_CANARY_HIT"}],
    })
    assert result["ranked_techniques"][0]["technique_id"] == "T1555"


def test_configured_endpoint_uses_structured_prompt_and_completion():
    completion = {
        "summary": "Local model summary.",
        "ranked_techniques": [],
        "playbook": ["Validate.", "Preserve.", "Monitor."],
        "confidence": "MEDIUM",
        "confidence_score": 0.6,
        "why_flagged": ["Graph anomaly."],
    }
    observed = {}

    def transport(outbound, timeout):
        observed["url"] = outbound.full_url
        observed["timeout"] = timeout
        observed["payload"] = json.loads(outbound.data)
        return io.BytesIO(json.dumps({"response": json.dumps(completion)}).encode())

    result = TriageService("http://127.0.0.1:11434/api/generate", transport=transport).summarize(_context())

    assert result["source"] == "configured_llm"
    assert result["advisory_only"] is True
    assert result["summary"] == "Local model summary."
    assert observed["url"] == "http://127.0.0.1:11434/api/generate"
    assert "deterministic_draft" in json.loads(observed["payload"]["prompt"])


def test_invalid_endpoint_result_degrades_to_template():
    def transport(_outbound, _timeout):
        return io.BytesIO(b'{"response":"not-json"}')

    result = TriageService("http://127.0.0.1:11434/api/generate", transport=transport).summarize(_context())

    assert result["source"] == "deterministic_template"
    assert result["llm_error"].startswith("Configured triage endpoint")


def test_non_loopback_endpoint_is_never_called():
    called = False

    def transport(_outbound, _timeout):
        nonlocal called
        called = True

    result = TriageService("https://example.com/generate", transport=transport).summarize(_context())
    assert result["source"] == "deterministic_template"
    assert called is False
