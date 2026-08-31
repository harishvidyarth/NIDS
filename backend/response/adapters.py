from __future__ import annotations

import hashlib
import json
import platform as platform_module
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

from .models import ResponseTarget


ReadRunner = Callable[[list[str]], dict[str, Any]]
Helper = Callable[[str, dict[str, Any]], dict[str, Any]]


def _default_read_runner(argv: list[str]) -> dict[str, Any]:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class FirewallAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> dict[str, Any]: ...

    @abstractmethod
    def scan(self) -> dict[str, Any]: ...

    @abstractmethod
    def render(self, action_id: str, targets: Sequence[ResponseTarget], ttl_minutes: int) -> dict[str, Any]: ...

    @abstractmethod
    def apply(self, plan: dict[str, Any]) -> list[str]: ...

    @abstractmethod
    def verify(self, action: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def rollback(self, action: dict[str, Any]) -> dict[str, Any]: ...


class _HelperAdapter(FirewallAdapter):
    engine = "unsupported"
    platform = "unsupported"

    def __init__(self, read_runner: ReadRunner | None = None, helper: Helper | None = None):
        self.read_runner = read_runner or _default_read_runner
        self.helper = helper

    def capabilities(self) -> dict[str, Any]:
        return {
            "platform": self.platform, "engine": self.engine,
            "privilege_ready": self.helper is not None,
            "supported_actions": ["scan", "render"] + (["apply", "verify", "rollback"] if self.helper else []),
            "mutation_boundary": "typed_privileged_helper",
        }

    def _helper(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.helper is None:
            raise RuntimeError("A configured privileged helper is required for firewall mutation.")
        if operation not in {"apply_plan", "verify_action", "rollback_action"}:
            raise RuntimeError("Unsupported privileged helper operation.")
        return self.helper(operation, payload)

    def apply(self, plan: dict[str, Any]) -> list[str]:
        result = self._helper("apply_plan", plan)
        return list(result.get("native_identifiers", []))

    def verify(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._helper("verify_action", action)

    def rollback(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._helper("rollback_action", action)


class LinuxNftablesAdapter(_HelperAdapter):
    engine = "nftables"
    platform = "linux"

    def _nft_path(self) -> str | None:
        candidate = shutil.which("nft")
        return str(Path(candidate).resolve()) if candidate else None

    def scan(self) -> dict[str, Any]:
        executable = self._nft_path()
        if not executable:
            namespace = {"available": False, "reason": "nft executable not found"}
        else:
            result = self.read_runner([executable, "-j", "list", "table", "inet", "nids_response"])
            namespace = {"available": True, "returncode": result["returncode"], "stdout": result["stdout"]}
        conflicts = [name for name in ("ufw", "iptables") if shutil.which(name)]
        return {
            "active": bool(executable), "engine": self.engine, "conflicts": conflicts,
            "namespace_healthy": bool(executable and namespace.get("returncode") == 0),
            "nids_owned_rules": [], "fingerprint": _fingerprint(namespace),
            "read_only": True,
        }

    def render(self, action_id: str, targets: Sequence[ResponseTarget], ttl_minutes: int) -> dict[str, Any]:
        suffix = action_id.replace("-", "")
        nft = self._nft_path() or "/usr/sbin/nft"
        commands = []
        prerequisites = [
            [nft, "add", "table", "inet", "nids_response"],
            [nft, "add", "chain", "inet", "nids_response", "input",
             "{", "type", "filter", "hook", "input", "priority", "0", ";", "policy", "accept", ";", "}"],
        ]
        for family, nft_type in (("ip", "ipv4_addr"), ("ip6", "ipv6_addr")):
            family_targets = [target for target in targets if (":" in target.source_ip) == (family == "ip6")]
            if not family_targets:
                continue
            for index, target in enumerate(family_targets):
                set_name = f"action_{suffix}_{family}_{index}"
                commands.append([nft, "add", "set", "inet", "nids_response", set_name,
                                 "{", "type", nft_type, ";", "flags", "timeout", ";", "}"])
                commands.append([nft, "add", "element", "inet", "nids_response", set_name,
                                 "{", target.source_ip, "timeout", f"{ttl_minutes}m", "}"])
                rule = [nft, "add", "rule", "inet", "nids_response", "input", family,
                        "saddr", f"@{set_name}"]
                if target.victim_ip:
                    rule += [family, "daddr", target.victim_ip]
                if target.protocol:
                    rule += [target.protocol]
                if target.destination_port and target.protocol in {"tcp", "udp"}:
                    rule += ["dport", str(target.destination_port)]
                rule += ["counter", "drop", "comment", f"nids:{action_id}:{index}"]
                commands.append(rule)
        return {
            "engine": self.engine, "namespace": "inet nids_response", "commands": commands,
            "namespace_prerequisites": prerequisites,
            "affected_traffic": [target.to_dict() for target in targets],
            "ttl_minutes": ttl_minutes,
            "rollback": "Delete only nft rule handles returned for this action; never flush the table or global ruleset.",
        }


class WindowsNetSecurityAdapter(_HelperAdapter):
    engine = "Windows Firewall / NetSecurity"
    platform = "windows"

    def scan(self) -> dict[str, Any]:
        powershell = str(Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"))
        result = self.read_runner([powershell, "-NoProfile", "-NonInteractive", "-Command",
                                   "Get-NetFirewallRule -Group 'NIDS Response' | Select-Object Name,Enabled | ConvertTo-Json -Compress"])
        namespace = {"returncode": result["returncode"], "stdout": result["stdout"]}
        return {"active": result["returncode"] == 0, "engine": self.engine, "conflicts": [],
                "namespace_healthy": result["returncode"] == 0, "nids_owned_rules": [],
                "fingerprint": _fingerprint(namespace), "read_only": True}

    def render(self, action_id: str, targets: Sequence[ResponseTarget], ttl_minutes: int) -> dict[str, Any]:
        commands = []
        for index, target in enumerate(targets):
            name = f"NIDS-{action_id}-{index}"
            command = ["New-NetFirewallRule", "-DisplayName", name, "-Group", "NIDS Response",
                       "-Direction", "Inbound", "-Action", "Block", "-RemoteAddress", target.source_ip]
            if target.protocol:
                command += ["-Protocol", target.protocol.upper()]
            if target.destination_port:
                command += ["-LocalPort", str(target.destination_port)]
            commands.append(command)
        return {"engine": self.engine, "group": "NIDS Response", "commands": commands,
                "affected_traffic": [target.to_dict() for target in targets], "ttl_minutes": ttl_minutes,
                "rollback": "Remove only NIDS Response rules whose names contain this action UUID."}


class MacOSPFAdapter(_HelperAdapter):
    engine = "PF"
    platform = "macos"

    def scan(self) -> dict[str, Any]:
        info = self.read_runner(["/sbin/pfctl", "-s", "info"])
        anchors = self.read_runner(["/sbin/pfctl", "-s", "Anchors"])
        result = self.read_runner(["/sbin/pfctl", "-a", "com.nids.response", "-sr"])
        namespace = {"returncode": result["returncode"], "stdout": result["stdout"],
                     "pf_enabled": "enabled" in info.get("stdout", "").lower(),
                     "anchors": anchors.get("stdout", "")}
        anchor_ready = bool(namespace["pf_enabled"] and anchors["returncode"] == 0
                            and "com.nids.response" in anchors.get("stdout", "") and result["returncode"] == 0)
        return {"active": bool(namespace["pf_enabled"]), "engine": self.engine, "conflicts": [],
                "namespace_healthy": anchor_ready, "anchor_integration_available": anchor_ready,
                "nids_owned_rules": [], "fingerprint": _fingerprint(namespace), "read_only": True}

    def render(self, action_id: str, targets: Sequence[ResponseTarget], ttl_minutes: int) -> dict[str, Any]:
        rules = []
        for target in targets:
            rule = ["block", "in", "quick"]
            if target.protocol:
                rule += ["proto", target.protocol]
            rule += ["from", target.source_ip]
            if target.victim_ip:
                rule += ["to", target.victim_ip]
            if target.destination_port:
                rule += ["port", str(target.destination_port)]
            rule += ["label", f"nids:{action_id}"]
            rules.append(rule)
        action_anchor = f"com.nids.response/{action_id.replace('-', '')}"
        return {"engine": self.engine, "anchor": action_anchor, "commands": rules,
                "affected_traffic": [target.to_dict() for target in targets], "ttl_minutes": ttl_minutes,
                "rollback": f"Flush only the action sub-anchor {action_anchor}; never flush the shared or global PF ruleset.",
                "requires_anchor_integration": True}


class UnsupportedAdapter(_HelperAdapter):
    def scan(self) -> dict[str, Any]:
        value = {"platform": platform_module.system().lower(), "supported": False}
        return {"active": False, "engine": "unsupported", "conflicts": [], "namespace_healthy": False,
                "nids_owned_rules": [], "fingerprint": _fingerprint(value), "read_only": True}

    def render(self, action_id: str, targets: Sequence[ResponseTarget], ttl_minutes: int) -> dict[str, Any]:
        return {"engine": "unsupported", "commands": [], "affected_traffic": [],
                "ttl_minutes": ttl_minutes, "rollback": "No executable firewall target is available."}


def adapter_for_platform(system: str | None = None, *, helper: Helper | None = None) -> FirewallAdapter:
    name = (system or platform_module.system()).lower()
    if name == "linux":
        return LinuxNftablesAdapter(helper=helper)
    if name == "windows":
        return WindowsNetSecurityAdapter(helper=helper)
    if name in {"darwin", "macos"}:
        return MacOSPFAdapter(helper=helper)
    return UnsupportedAdapter(helper=helper)
