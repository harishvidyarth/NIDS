from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..lstm.config import FORECAST_CLASSES

DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "data" / "enterprise_attack_v19_1_subset.json"
VALID_TACTICS = {"Reconnaissance", "Discovery", "Impact"}
TECHNIQUE_ID_PATTERN = re.compile(r"^T\d{4}(?:\.\d{3})?$")
NETWORK_LIMITATION = "Network-flow evidence alone cannot confirm adversary intent or host/process behavior."
UNCALIBRATED_LIMITATION = "Mapping confidence is a deterministic heuristic evidence score, not a calibrated probability."


class AttackMetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class AttackMetadataStore:
    version: str
    data_modified: str
    source_url: str
    source_sha256: str
    techniques: dict[str, dict]
    detection_strategies: tuple[dict, ...]
    mitigations: tuple[dict, ...]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_METADATA_PATH) -> "AttackMetadataStore":
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AttackMetadataError(f"Unable to load local ATT&CK metadata: {error}") from error
        required_root = {"schema", "attack_version", "data_modified", "source", "collection", "techniques"}
        if not required_root.issubset(payload) or payload.get("schema") != "mitre-attack-enterprise-subset/v1":
            raise AttackMetadataError("Malformed ATT&CK metadata root.")
        source = payload.get("source") or {}
        if source.get("publisher") != "The MITRE Corporation" or not source.get("url") or not source.get("stix_bundle_sha256"):
            raise AttackMetadataError("ATT&CK metadata provenance is missing or invalid.")
        techniques: dict[str, dict] = {}
        for item in payload.get("techniques", []):
            required = {"stix_id", "technique_id", "technique_name", "tactics", "url", "modified"}
            if not required.issubset(item):
                raise AttackMetadataError("Malformed ATT&CK technique metadata.")
            technique_id = item["technique_id"]
            if not TECHNIQUE_ID_PATTERN.fullmatch(technique_id):
                raise AttackMetadataError(f"Invalid ATT&CK technique ID: {technique_id}")
            expected_suffix = "/" + technique_id.replace(".", "/")
            if not item["url"].endswith(expected_suffix):
                raise AttackMetadataError(f"Technique URL does not match ID: {technique_id}")
            if not item["tactics"] or not set(item["tactics"]).issubset(VALID_TACTICS):
                raise AttackMetadataError(f"Invalid tactic reference for {technique_id}")
            if technique_id in techniques:
                raise AttackMetadataError(f"Duplicate ATT&CK technique ID: {technique_id}")
            techniques[technique_id] = item
        if not techniques:
            raise AttackMetadataError("ATT&CK metadata contains no techniques.")
        return cls(
            version=str(payload["attack_version"]),
            data_modified=str(payload["data_modified"]),
            source_url=str(source["url"]),
            source_sha256=str(source["stix_bundle_sha256"]),
            techniques=techniques,
            detection_strategies=tuple(payload.get("detection_strategies", [])),
            mitigations=tuple(payload.get("mitigations", [])),
        )


class MitreAttackMapper:
    STATE_TECHNIQUES = {
        "PortScan": ("T1595", "T1046"),
        "DDoS": ("T1498", "T1498.001", "T1498.002"),
        "DoS": ("T1499",),
    }

    def __init__(self, metadata: AttackMetadataStore | None = None):
        self.metadata = metadata or AttackMetadataStore.load()
        self.rule_technique_ids = {
            technique_id for technique_ids in self.STATE_TECHNIQUES.values() for technique_id in technique_ids
        }
        missing = self.rule_technique_ids - set(self.metadata.techniques)
        if missing:
            raise AttackMetadataError(f"Mapping rules reference unknown ATT&CK IDs: {sorted(missing)}")

    @staticmethod
    def _number(features: Mapping[str, object], name: str) -> float:
        try:
            value = float(features.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _behavior_evidence(self, state: str, features: Mapping[str, object]) -> list[str]:
        flow_count = self._number(features, "flow_count")
        destination_ports = self._number(features, "unique_dst_port_count")
        flows_per_second = self._number(features, "flows_per_second")
        packets_per_second = self._number(features, "packets_per_second")
        syn_count = self._number(features, "syn_count")
        rst_count = self._number(features, "rst_count")
        evidence = []
        if state == "PortScan":
            if destination_ports >= 10:
                evidence.append(f"destination-port diversity is elevated ({destination_ports:g} unique ports)")
            if flow_count > 0 and destination_ports / flow_count >= 0.5:
                evidence.append("destination-port diversity spans at least half of observed flows")
            if syn_count >= max(5.0, flow_count * 0.5):
                evidence.append(f"repeated SYN behavior is present ({syn_count:g} SYN flags)")
        else:
            if flows_per_second >= 10:
                evidence.append(f"connection rate is elevated ({flows_per_second:g} flows/s)")
            if packets_per_second >= 100:
                evidence.append(f"packet rate is elevated ({packets_per_second:g} packets/s)")
            if syn_count >= 20:
                evidence.append(f"high SYN volume is present ({syn_count:g} SYN flags)")
            if rst_count >= 10:
                evidence.append(f"high RST volume is present ({rst_count:g} RST flags)")
        return evidence

    def _candidate(self, technique_id: str, status: str, confidence: float, evidence: list[str], limitation: str) -> dict:
        technique = self.metadata.techniques[technique_id]
        return {
            "technique_id": technique_id,
            "technique_name": technique["technique_name"],
            "tactic": technique["tactics"][0],
            "mapping_status": status,
            "mapping_confidence": confidence,
            "mapping_confidence_type": "deterministic_heuristic_not_calibrated",
            "evidence": evidence,
            "limitations": [NETWORK_LIMITATION, UNCALIBRATED_LIMITATION, limitation],
        }

    @staticmethod
    def _guidance(state: str | None) -> tuple[str, list[str], list[str]]:
        if state == "DDoS":
            return "high", [
                "Validate that sources are distributed within the same rolling window.",
                "Identify the concentrated victim IP, service, and business owner.",
                "Engage the ISP, CDN, or DDoS provider for upstream filtering.",
                "Apply risk-reviewed rate limits or protocol/port filtering.",
                "Preserve the capture, flow export, service telemetry, and response timeline.",
            ], [
                "Concurrent source and victim concentration across rolling windows.",
                "Victim service saturation, latency, error rate, and interface utilization.",
                "Protocol details sufficient to distinguish direct flood from reflection/amplification.",
            ]
        if state == "DoS":
            return "high", [
                "Inspect the affected service and host health.",
                "Check connection-table, thread, socket, and SYN-backlog exhaustion.",
                "Apply service-specific rate limits and validate recovery.",
                "Preserve flow and endpoint evidence before blocking confirmed sources.",
            ], ["Service health and resource exhaustion telemetry.", "Connection state and rate-limit logs."]
        if state == "PortScan":
            return "medium", [
                "Verify whether the scan was authorized.",
                "Review exposed services and recent configuration changes.",
                "Monitor or block confirmed unauthorized sources using change control.",
                "Preserve flow and firewall evidence.",
            ], ["Scanner ownership and authorization.", "Destination service inventory and firewall logs."]
        return "informational", [
            "Continue monitoring and compare traffic with the established baseline.",
            "Preserve the capture and analysis result.",
            "Collect endpoint and service evidence if operational symptoms appear.",
        ], ["Additional rolling-window traffic.", "Endpoint, service, and network-device telemetry."]

    def map_forecast(
        self,
        current_state: str,
        probabilities: Mapping[str, float],
        temporal_features: Mapping[str, object] | None = None,
    ) -> dict:
        temporal_features = temporal_features or {}
        normalized = {label: float(probabilities.get(label, 0.0)) for label in FORECAST_CLASSES}
        predicted_state = max(FORECAST_CLASSES, key=lambda label: normalized[label])
        forecast_probability = normalized[predicted_state]
        result = {
            "current_state": current_state,
            "predicted_next_state": predicted_state,
            "next_state_probabilities": normalized,
            "forecast_probability": forecast_probability,
            "attack_present": current_state in self.STATE_TECHNIQUES,
            "attack_mapping": None,
            "mapping_status": "INSUFFICIENT_EVIDENCE",
            "mitre_candidates": [],
            "attack_version": self.metadata.version,
            "attack_data_modified": self.metadata.data_modified,
            "metadata_source": self.metadata.source_url,
        }
        guidance_state = current_state if current_state in self.STATE_TECHNIQUES else (
            predicted_state if predicted_state in self.STATE_TECHNIQUES and forecast_probability >= 0.50 else None
        )
        severity, operator_guidance, evidence_needed = self._guidance(guidance_state)
        result.update({
            "severity": severity,
            "operator_guidance": operator_guidance,
            "evidence_needed": evidence_needed,
            "action_provenance": {
                "source": "local_risk_matched_playbook",
                "mitigations": ["M1037"] if guidance_state in {"DDoS", "DoS"} else [],
                "detection_strategies": ["DET0518"] if guidance_state == "DDoS" else [],
                "requires_operator_validation": True,
            },
        })
        if current_state == "INVALID_FEATURES":
            result["reason"] = "invalid features cannot support ATT&CK interpretation"
            return result
        attack_probability = 1.0 - normalized["BENIGN"]
        if current_state == "BENIGN" and predicted_state == "BENIGN" and attack_probability < 0.50:
            result["reason"] = "no attack evidence"
            return result

        states = []
        if current_state in self.STATE_TECHNIQUES:
            states.append(current_state)
        if predicted_state in self.STATE_TECHNIQUES and predicted_state not in states:
            states.append(predicted_state)
        if not states:
            states = [max(self.STATE_TECHNIQUES, key=lambda label: normalized[label])]

        candidates = []
        for state in states:
            probability = normalized[state]
            behavior = self._behavior_evidence(state, temporal_features)
            state_evidence = []
            if current_state == state:
                state_evidence.append(f"observed current network state is {state}")
            if probability >= 0.25:
                state_evidence.append(f"LSTM next-state probability for {state} is {probability:.3f}")
            possible = bool(behavior) and (current_state == state or probability >= 0.50)
            status = "POSSIBLE" if possible else "INSUFFICIENT_EVIDENCE"
            confidence = 0.65 if possible and current_state == state and probability >= 0.50 else 0.55 if possible else 0.25
            for technique_id in self.STATE_TECHNIQUES[state]:
                if technique_id == "T1595":
                    limitation = "Flow data cannot establish whether scanning occurred before compromise."
                elif technique_id == "T1046":
                    limitation = "Flow data cannot establish whether scanning occurred from a compromised internal host."
                elif technique_id == "T1498":
                    limitation = "Flow statistics cannot identify distributed sources or a specific network-flood sub-technique here."
                elif technique_id == "T1498.001":
                    limitation = "Direct-flood context requires protocol and source validation beyond aggregate flow statistics."
                elif technique_id == "T1498.002":
                    limitation = "Reflection amplification requires spoofing, reflector, and amplification-protocol evidence."
                else:
                    limitation = "Flow statistics cannot identify an endpoint exhaustion or exploitation sub-technique."
                candidate_status = status if technique_id in {"T1498", "T1046", "T1595", "T1499"} else "INSUFFICIENT_EVIDENCE"
                candidate_confidence = confidence if candidate_status == status else min(confidence, 0.25)
                candidates.append(self._candidate(technique_id, candidate_status, candidate_confidence, state_evidence + behavior, limitation))

        candidates.sort(key=lambda item: (-item["mapping_confidence"], item["technique_id"]))
        result["mitre_candidates"] = candidates
        result["mapping_status"] = "POSSIBLE" if any(
            item["mapping_status"] == "POSSIBLE" for item in candidates
        ) else "INSUFFICIENT_EVIDENCE"
        result["attack_mapping"] = "possible_attack_context" if result["mapping_status"] == "POSSIBLE" else None
        result["reason"] = (
            "network-flow evidence supports possible ATT&CK context"
            if result["mapping_status"] == "POSSIBLE"
            else "traffic-state evidence is insufficient for an ATT&CK technique assertion"
        )
        return result
