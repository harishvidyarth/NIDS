"""NIDS 4-class state -> operator-facing MITRE ATT&CK kill-chain stage.

Ported from NAF `src/model/label_map.py` + `src/explainability/mitre.py`.
NIDS only classifies BENIGN / DDoS / DoS / PortScan, so the ladder it can
actually express is Reconnaissance -> ... -> Impact; the full ladder is
still returned so the UI can show where the current/forecast stage sits.
"""
from __future__ import annotations

MITRE_TACTICS = {
    "Reconnaissance": "TA0043",
    "Initial Access": "TA0001",
    "Command and Control": "TA0011",
    "Lateral Movement": "TA0008",
    "Exfiltration": "TA0010",
    "Impact": "TA0040",
}

# Kill-chain order shown in the UI ladder.
PROGRESS_STAGES = (
    "Reconnaissance",
    "Initial Access",
    "Command and Control",
    "Lateral Movement",
    "Exfiltration",
    "Impact",
)

# The 4 states NIDS can output.
STATE_TO_STAGE = {
    "BENIGN": "Benign",
    "PortScan": "Reconnaissance",
    "DoS": "Impact",
    "DDoS": "Impact",
}

_TERMINAL = {"Impact"}


def state_to_stage(state: str) -> str:
    return STATE_TO_STAGE.get(str(state), "Unknown")


def presentation_state(stage: str, risk: str | None = None) -> dict:
    """Stage + tactic id + the full ladder with the active rung flagged."""
    stage = str(stage or "Unknown")
    terminal = stage in _TERMINAL
    return {
        "predicted_stage": stage,
        "tactic_id": MITRE_TACTICS.get(stage),
        "progress_stages": [
            {"name": name, "tactic_id": MITRE_TACTICS[name], "active": name == stage}
            for name in PROGRESS_STAGES
        ],
        "terminal_impact_alert": terminal and risk == "high",
        "terminal_tactic_id": "TA0040" if terminal else None,
        "disclaimer": (
            "Stage is derived from the forecast state class, not a per-technique "
            "detection. NIDS classes cover Reconnaissance and Impact only."
        ),
    }


def risk_level(infiltration_probability: float) -> str:
    if infiltration_probability >= 0.80:
        return "high"
    if infiltration_probability >= 0.50:
        return "medium"
    return "low"
