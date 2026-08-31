from __future__ import annotations

import ipaddress
import json
import platform
import re
import socket
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import PolicyDecision, ResponseTarget


class PolicyError(ValueError):
    pass


SUPPORTED_SIGNATURES = {"PortScan", "DoS", "DDoS"}
MAX_DDOS_SOURCES = 64
_BROADCAST_V4 = ipaddress.ip_address("255.255.255.255")


def discover_system_protected_addresses() -> set[str]:
    """Best-effort, read-only discovery; configured allowlists remain authoritative."""
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None):
            addresses.add(str(ipaddress.ip_address(item[4][0])))
    except (OSError, ValueError):
        pass
    resolv = Path("/etc/resolv.conf")
    try:
        for line in resolv.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                addresses.add(str(ipaddress.ip_address(parts[1])))
    except (OSError, ValueError):
        pass
    route = Path("/proc/net/route")
    try:
        for line in route.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 2 and fields[1] == "00000000":
                addresses.add(str(ipaddress.ip_address(socket.inet_ntoa(struct.pack("<L", int(fields[2], 16))))))
    except (OSError, ValueError, struct.error):
        pass

    def add(value: str) -> None:
        try:
            addresses.add(str(ipaddress.ip_address(value.strip().split("%")[0])))
        except ValueError:
            pass

    def read(argv: list[str]) -> str:
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    system = platform.system().lower()
    if system == "linux":
        executable = next((str(path) for path in (Path("/usr/sbin/ip"), Path("/sbin/ip"), Path("/usr/bin/ip")) if path.is_file()), None)
        if executable:
            try:
                for interface in json.loads(read([executable, "-j", "address", "show"])):
                    for item in interface.get("addr_info", []):
                        if item.get("local"):
                            add(item["local"])
            except (json.JSONDecodeError, TypeError):
                pass
    elif system == "darwin":
        output = read(["/sbin/ifconfig"])
        for match in re.finditer(r"^\s*inet6?\s+([^\s]+)", output, re.MULTILINE):
            add(match.group(1))
        route_output = read(["/sbin/route", "-n", "get", "default"])
        gateway = re.search(r"^\s*gateway:\s*([^\s]+)", route_output, re.MULTILINE)
        if gateway:
            add(gateway.group(1))
        for match in re.finditer(r"nameserver\[[0-9]+\]\s*:\s*([^\s]+)", read(["/usr/sbin/scutil", "--dns"])):
            add(match.group(1))
    elif system == "windows":
        output = read(["C:/Windows/System32/ipconfig.exe", "/all"])
        for line in output.splitlines():
            if any(label in line for label in ("IPv4 Address", "IPv6 Address", "Default Gateway", "DHCP Server", "DNS Servers")):
                value = line.split(":", 1)[-1].replace("(Preferred)", "").strip()
                add(value)
    return addresses


def _field(mapping: dict[str, Any], *names: str) -> Any:
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in mapping.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value not in (None, "", "N/A"):
            return value
    return None


def _parse_ip(value: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise PolicyError(f"Invalid IP address in response evidence: {value!r}") from exc


def _unsafe(address: ipaddress._BaseAddress, protected: set[str]) -> str | None:
    if str(address) in protected:
        return "configured protected/local infrastructure address"
    if address.is_unspecified:
        return "unspecified address"
    if address.is_loopback:
        return "loopback address"
    if address.is_multicast:
        return "multicast address"
    if address.is_link_local:
        return "link-local address"
    if address == _BROADCAST_V4:
        return "broadcast address"
    return None


def _protocol(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return {"6": "tcp", "17": "udp"}.get(text, text if text in {"tcp", "udp"} else None)


def _port(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        port = int(float(value))
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65535 else None


def _matching_flows(prediction: dict[str, Any], attack_class: str) -> list[dict[str, Any]]:
    return [
        flow for flow in prediction.get("flows", [])
        if str(_field(flow, "signature_state") or "") == attack_class
    ]


def _validated_target(
    source: Any, victim: Any, protocol: Any, port: Any, protected: set[str], warnings: list[str]
) -> ResponseTarget | None:
    address = _parse_ip(source)
    reason = _unsafe(address, protected)
    if reason:
        warnings.append(f"Rejected source {address}: {reason}.")
        return None
    victim_text = None
    if victim not in (None, "", "N/A"):
        victim_address = _parse_ip(victim)
        victim_text = str(victim_address)
    return ResponseTarget(str(address), victim_text, _protocol(protocol), _port(port))


def evaluate_prediction(
    prediction: dict[str, Any], *, ttl_minutes: int = 15,
    protected_addresses: Iterable[str] = (), selected_targets: Sequence[dict[str, Any]] | None = None,
) -> PolicyDecision:
    if not 1 <= ttl_minutes <= 60:
        raise PolicyError("TTL must be between 1 and 60 minutes.")
    protected = {str(_parse_ip(value)) for value in protected_addresses}

    if prediction.get("future_labels_are_forecasts") or prediction.get("horizons"):
        return PolicyDecision(
            False, None, "forecast", limitations=[
                "Forecast results are recommendation-only and can never execute firewall rules."
            ], upstream_recommendation="Prepare monitoring and upstream mitigation; wait for current deterministic evidence.",
        )

    attack_class = prediction.get("signature_verdict")
    hits = [hit for hit in prediction.get("signature_hits", []) if hit.get("state") == attack_class]
    if attack_class not in SUPPORTED_SIGNATURES or not hits:
        return PolicyDecision(
            False, prediction.get("attack_class"), "ann_only", evidence={"prediction": prediction.get("verdict")},
            limitations=["ANN-only detections are lower-confidence recommendations and are not executable in v1."],
        )

    warnings: list[str] = []
    flows = _matching_flows(prediction, attack_class)
    candidates: list[ResponseTarget] = []

    if attack_class == "PortScan":
        signature = prediction.get("port_scan_signature") or {}
        source = _field(signature, "src", "source", "src_ip")
        if source is None and flows:
            source = _field(flows[0], "src_ip", "Src IP")
        victim = _field(signature, "dst", "victim", "dst_ip")
        if source is not None:
            target = _validated_target(source, None, None, None, protected, warnings)
            if target:
                candidates.append(target)
    else:
        by_source: dict[str, list[dict[str, Any]]] = {}
        missing_source_count = 0
        for flow in flows:
            source = _field(flow, "src_ip", "Src IP")
            if source is None:
                missing_source_count += 1
                continue
            by_source.setdefault(str(source), []).append(flow)
        if attack_class == "DDoS" and (missing_source_count or len(by_source) > MAX_DDOS_SOURCES):
            if missing_source_count:
                warnings.append(f"DDoS has {missing_source_count} signature flow(s) without source attribution.")
            if len(by_source) > MAX_DDOS_SOURCES:
                warnings.append(f"DDoS has {len(by_source)} attributable sources; the executable limit is {MAX_DDOS_SOURCES}.")
        elif attack_class == "DoS" and len(by_source) != 1:
            warnings.append("Single-source DoS attribution is missing or ambiguous.")
        else:
            for source, source_flows in by_source.items():
                victims = {_field(flow, "dst_ip", "Dst IP") for flow in source_flows}
                protocols = {_protocol(_field(flow, "protocol")) for flow in source_flows}
                ports = {_port(_field(flow, "dst_port", "Dst Port")) for flow in source_flows}
                victim = next(iter(victims)) if len(victims) == 1 else None
                protocol = next(iter(protocols)) if len(protocols) == 1 else None
                port = next(iter(ports)) if len(ports) == 1 else None
                target = _validated_target(source, victim, protocol, port, protected, warnings)
                if target:
                    candidates.append(target)

    # Operator selection can narrow deterministic evidence, never introduce a new source.
    original_ddos_gate_failed = attack_class == "DDoS" and any(
        "without source attribution" in warning or "executable limit" in warning for warning in warnings
    )
    if selected_targets is not None and not original_ddos_gate_failed:
        selected_sources = {str(_parse_ip(_field(item, "source_ip", "src_ip"))) for item in selected_targets}
        detected_sources = {target.source_ip for target in candidates}
        if not selected_sources <= detected_sources:
            raise PolicyError("Selected targets must be a subset of deterministic signature evidence.")
        candidates = [target for target in candidates if target.source_ip in selected_sources]

    upstream = None
    limitations: list[str] = []
    if original_ddos_gate_failed:
        candidates = []
        upstream = "Use ISP/CDN/WAF upstream filtering; a host firewall cannot recover a saturated link."
    elif attack_class == "DDoS":
        limitations.append("Local source blocks do not guarantee mitigation of link-saturating DDoS traffic.")
        upstream = "Escalate to ISP/CDN/WAF filtering if link saturation is observed."

    executable = bool(candidates)
    if not executable and not warnings:
        warnings.append("Reliable, safe source-address attribution is unavailable.")
    return PolicyDecision(
        executable, attack_class, "deterministic_signature", candidates,
        evidence={"signature_hits": hits, "source_count": len(candidates), "ttl_minutes": ttl_minutes},
        warnings=warnings, limitations=limitations, upstream_recommendation=upstream,
    )
