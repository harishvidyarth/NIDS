"""Deterministic and optional local-LLM XDR triage.

The output is advisory data only.  This module deliberately has no import from
``backend.response`` and cannot execute or schedule response actions.
"""
from __future__ import annotations

import json
import ipaddress
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib import request
from urllib.parse import urlsplit


Transport = Callable[[request.Request, float], Any]
MAX_COMPLETION_BYTES = 1_000_000


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("triage endpoint redirects are disabled")


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _verdict_label(verdict: Mapping[str, object]) -> str:
    for key in ("effective_attack_class", "attack_class", "final_verdict", "verdict", "predicted_class", "prediction", "label"):
        value = verdict.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _verdict_confidence(verdict: Mapping[str, object]) -> float:
    for key in ("effective_confidence", "confidence", "probability", "score"):
        value = verdict.get(key)
        if value is not None:
            if isinstance(value, str) and value.lower() in {"high", "low", "none"}:
                return {"high": 0.9, "low": 0.5, "none": 0.0}[value.lower()]
            return min(1.0, max(0.0, _number(value)))
    return 0.0


def _forecast_probability(forecast: Mapping[str, object]) -> float:
    direct = forecast.get("maximum_attack_probability")
    if direct is not None:
        return min(1.0, max(0.0, _number(direct)))
    horizons = forecast.get("horizons")
    if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)):
        return 0.0
    return max(
        (_number(item.get("attack_probability")) for item in horizons if isinstance(item, Mapping)),
        default=0.0,
    )


def _techniques(mitre: Mapping[str, object]) -> list[dict]:
    raw = mitre.get("mitre_candidates", mitre.get("techniques", []))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    ranked = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        evidence = item.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, Sequence):
            evidence = []
        ranked.append({
            "technique_id": str(item.get("technique_id", item.get("id", "UNKNOWN"))),
            "technique_name": str(item.get("technique_name", item.get("name", "Unspecified technique"))),
            "tactic": str(item.get("tactic", "Unknown")),
            "confidence": min(1.0, max(0.0, _number(
                item.get("mapping_confidence", item.get("confidence", 0.0))
            ))),
            "status": str(item.get("mapping_status", item.get("status", "ADVISORY"))),
            "evidence": [str(value) for value in evidence if str(value).strip()],
        })
    return sorted(ranked, key=lambda item: (-item["confidence"], item["technique_id"]))


def _enrichment_flags(enrichment: Mapping[str, object]) -> list[str]:
    checks = (
        ("beacon_score_max", 0.80, "highly periodic connection timing"),
        ("dns_query_entropy_mean", 3.50, "high-entropy DNS queries"),
        ("nxdomain_ratio", 0.30, "an elevated NXDOMAIN ratio"),
        ("ja3_novelty", 0.50, "novel TLS fingerprints"),
        ("byte_asymmetry_max", 0.90, "strong byte-direction asymmetry"),
    )
    return [sentence for name, threshold, sentence in checks if _number(enrichment.get(name)) >= threshold]


def build_triage(context: Mapping[str, object]) -> dict:
    """Build a stable, offline triage summary from normalized XDR evidence."""
    verdict = context.get("verdict", {})
    forecast = context.get("forecast", {})
    mitre = context.get("mitre", context.get("mitre_mapping", {}))
    enrichment = context.get("enrichment", {})
    deception_events = context.get("deception_events", [])
    if not isinstance(verdict, Mapping):
        verdict = {}
    if not isinstance(forecast, Mapping):
        forecast = {}
    if not isinstance(mitre, Mapping):
        mitre = {}
    if not isinstance(enrichment, Mapping):
        enrichment = {}
    if not isinstance(deception_events, Sequence) or isinstance(deception_events, (str, bytes)):
        deception_events = []

    label = _verdict_label(verdict)
    verdict_score = _verdict_confidence(verdict)
    forecast_score = _forecast_probability(forecast)
    campaign_score = min(1.0, max(0.0, _number(context.get("campaign_score"))))
    has_canary = any(isinstance(event, Mapping) for event in deception_events)
    ranked = _techniques(mitre)
    flags = _enrichment_flags(enrichment)
    if has_canary and not ranked:
        ranked = [{
            "technique_id": "T1555",
            "technique_name": "Credentials from Password Stores",
            "tactic": "Credential Access",
            "confidence": 0.8,
            "status": "POSSIBLE",
            "evidence": ["A deliberately planted credential honeytoken was accessed."],
        }]

    why_flagged = []
    if label not in {"BENIGN", "NONE", "UNKNOWN"}:
        why_flagged.append(f"The current detector classified the session as {label}.")
    if forecast_score >= 0.50:
        why_flagged.append(f"The forecast attack probability reaches {forecast_score:.0%}.")
    if campaign_score >= 0.40:
        why_flagged.append(f"Communication-graph campaign score is {campaign_score:.2f}.")
    why_flagged.extend(f"Sensor fusion observed {flag}." for flag in flags)
    if has_canary:
        why_flagged.insert(0, "A deception canary was accessed; this is direct high-confidence evidence.")
    if not why_flagged:
        why_flagged.append("No high-confidence attack evidence is currently present.")

    score = max(verdict_score, forecast_score, campaign_score)
    if has_canary:
        score = max(score, 0.99)
    if score >= 0.80 or has_canary:
        confidence = "HIGH"
    elif score >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if has_canary:
        summary = (
            f"The {label} session requires prompt investigation because a deception canary was accessed. "
            f"Graph and forecast context score {campaign_score:.2f} and {forecast_score:.0%}, respectively; "
            "validate the originating host and preserve corroborating network and endpoint evidence."
        )
    elif label not in {"BENIGN", "NONE", "UNKNOWN"}:
        summary = (
            f"The detector reports {label} with {verdict_score:.0%} confidence. "
            f"The short-horizon attack probability is {forecast_score:.0%} and the graph campaign score is "
            f"{campaign_score:.2f}; review the listed evidence before taking any containment action."
        )
    else:
        summary = (
            f"The current session is {label}, with a short-horizon attack probability of {forecast_score:.0%} "
            f"and a graph campaign score of {campaign_score:.2f}. Continue monitoring and validate any new "
            "sensor-fusion anomalies before escalation."
        )

    guidance = mitre.get("operator_guidance", [])
    if not isinstance(guidance, Sequence) or isinstance(guidance, (str, bytes)):
        guidance = []
    playbook = [str(item) for item in guidance if str(item).strip()][:5]
    defaults = [
        "Validate the source, destination, time range, and business context against the raw evidence.",
        "Preserve the capture, enriched logs, graph snapshot, and detection timeline.",
        "Inspect affected hosts and services for corroborating endpoint or application evidence.",
        "Draft a scoped, time-limited containment action for explicit operator review.",
        "Monitor recovery and document the result before closing the incident.",
    ]
    for step in defaults:
        if len(playbook) >= 5:
            break
        if step not in playbook:
            playbook.append(step)
    if len(playbook) < 3:
        playbook.extend(defaults[:3 - len(playbook)])

    return {
        "summary": summary,
        "ranked_techniques": ranked,
        "playbook": playbook[:5],
        "confidence": confidence,
        "confidence_score": round(score, 4),
        "why_flagged": why_flagged,
        "source": "deterministic_template",
        "advisory_only": True,
    }


class TriageService:
    """Produce deterministic triage, optionally refined by a configured endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_seconds: float = 5.0,
        transport: Transport | None = None,
    ):
        self.endpoint = endpoint.strip() if endpoint else None
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._urlopen

    @staticmethod
    def _urlopen(outbound: request.Request, timeout: float):
        return request.build_opener(_NoRedirect).open(outbound, timeout=timeout)

    def summarize(self, context: Mapping[str, object]) -> dict:
        fallback = build_triage(context)
        if not self.endpoint:
            return fallback
        try:
            generated = self._request_completion(context, fallback)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            fallback["llm_error"] = "Configured triage endpoint was unavailable or returned invalid data."
            return fallback
        generated["source"] = "configured_llm"
        generated["advisory_only"] = True
        return generated

    def _request_completion(self, context: Mapping[str, object], fallback: Mapping[str, object]) -> dict:
        parsed = urlsplit(self.endpoint or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("triage endpoint must be an HTTP(S) loopback URL")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback:
            raise ValueError("triage endpoint must resolve explicitly to loopback")
        prompt = {
            "instruction": (
                "Return JSON only. Summarize this network-security evidence without asserting facts not in "
                "the evidence. Response actions are advisory and require operator validation."
            ),
            "required_schema": {
                "summary": "string",
                "ranked_techniques": "array",
                "playbook": "array of 3-5 strings",
                "confidence": "HIGH, MEDIUM, or LOW",
                "confidence_score": "number from 0 to 1",
                "why_flagged": "array of strings",
            },
            "evidence": context,
            "deterministic_draft": fallback,
        }
        payload = json.dumps({"prompt": json.dumps(prompt, sort_keys=True), "stream": False}).encode("utf-8")
        outbound = request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        response = self._transport(outbound, self.timeout_seconds)
        if hasattr(response, "__enter__"):
            with response as opened:
                raw = opened.read(MAX_COMPLETION_BYTES + 1)
        else:
            raw = response.read(MAX_COMPLETION_BYTES + 1)
        if len(raw) > MAX_COMPLETION_BYTES:
            raise ValueError("triage completion exceeds the size limit")
        envelope = json.loads(raw.decode("utf-8"))
        completion = envelope.get("response", envelope)
        if isinstance(completion, str):
            completion = json.loads(completion)
        return self._validate_completion(completion)

    @staticmethod
    def _validate_completion(completion: object) -> dict:
        if not isinstance(completion, Mapping):
            raise ValueError("completion must be an object")
        required = {"summary", "ranked_techniques", "playbook", "confidence", "why_flagged"}
        if not required.issubset(completion):
            raise ValueError("completion is missing required fields")
        if not isinstance(completion["summary"], str):
            raise ValueError("summary must be text")
        playbook = completion["playbook"]
        if not isinstance(playbook, list) or not 3 <= len(playbook) <= 5:
            raise ValueError("playbook must contain three to five steps")
        if completion["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("confidence must use the supported scale")
        result = dict(completion)
        result["confidence_score"] = min(1.0, max(0.0, _number(result.get("confidence_score"))))
        return result
