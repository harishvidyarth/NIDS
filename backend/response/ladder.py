"""Prototype XDR response ladder.

This module is intentionally dry-run only.  It validates a typed target, renders
the command that an external enforcement integration could use, and records the
decision.  It never invokes a subprocess.
"""
from __future__ import annotations

import ipaddress
import json
import os
import platform
import shlex
import threading
import time
import uuid
from pathlib import Path

LADDER = ("NONE", "ALERT", "RATE_LIMIT", "FIREWALL_RULE", "QUARANTINE")


class LadderError(ValueError):
    pass


def _command(step: str, target: str, action_id: str, system: str | None = None) -> tuple[str, str]:
    family = "ip6" if ipaddress.ip_address(target).version == 6 else "ip"
    system = (system or platform.system()).lower()
    if step == "NONE":
        return "no action (recommendation only)", "no rollback required"
    if step == "ALERT":
        return "log alert only (no network change)", "close alert"
    if "darwin" in system:
        anchor = f"com.nids.response/{action_id.replace('-', '')}"
        apply = ["/sbin/pfctl", "-a", anchor, "-f", f"/var/run/nids/{action_id}.conf"]
        revert = ["/sbin/pfctl", "-a", anchor, "-F", "rules"]
    elif "windows" in system:
        apply = ["powershell.exe", "-NoProfile", "-Command", "New-NetFirewallRule", "-DisplayName",
                 f"NIDS-{action_id}", "-Group", "NIDS Response", "-Direction", "Inbound",
                 "-Action", "Block", "-RemoteAddress", target]
        revert = ["powershell.exe", "-NoProfile", "-Command", "Remove-NetFirewallRule",
                  "-DisplayName", f"NIDS-{action_id}"]
    elif step == "RATE_LIMIT":
        protocol = "ipv6" if family == "ip6" else "ip"
        preference = str(1000 + int(action_id.replace("-", "")[:6], 16) % 30_000)
        apply = ["/sbin/tc", "filter", "add", "dev", "eth0", "parent", "ffff:",
                 "protocol", protocol, "pref", preference, "flower", "src_ip", target,
                 "action", "police", "rate", "1mbit", "drop"]
        revert = ["/sbin/tc", "filter", "del", "dev", "eth0", "parent", "ffff:",
                  "protocol", protocol, "pref", preference]
    else:
        table = f"nids_{action_id.replace('-', '')[:12]}"
        apply = ["/usr/sbin/nft", "-f", f"/var/run/nids/{action_id}.nft"]
        revert = ["/usr/sbin/nft", "delete", "table", "inet", table]
    return shlex.join(apply), shlex.join(revert)


def _ruleset(step: str, target: str, action_id: str, ttl_seconds: int, system: str | None = None) -> str:
    """Exact file/command content paired with the displayed dry-run command."""
    system = (system or platform.system()).lower()
    if step == "NONE":
        return "# recommendation only; no native rule"
    if step == "ALERT":
        return "# local audit alert only; no native rule"
    if "darwin" in system:
        rules = [f'block in quick from {target} to any label "nids:{action_id}"']
        if step == "QUARANTINE":
            rules.append(f'block out quick from any to {target} label "nids:{action_id}"')
        return "\n".join(rules) + "\n"
    if "windows" in system or step == "RATE_LIMIT":
        return _command(step, target, action_id, system)[0]
    family = "ip6" if ipaddress.ip_address(target).version == 6 else "ip"
    table = f"nids_{action_id.replace('-', '')[:12]}"
    lines = [
        f"add table inet {table}",
        f"add chain inet {table} input {{ type filter hook input priority 10; policy accept; }}",
        f'add rule inet {table} input {family} saddr {target} counter drop comment "nids:{action_id};ttl={ttl_seconds}s"',
    ]
    if step == "QUARANTINE":
        lines.extend([
            f"add chain inet {table} output {{ type filter hook output priority 10; policy accept; }}",
            f'add rule inet {table} output {family} daddr {target} counter drop comment "nids:{action_id};ttl={ttl_seconds}s"',
        ])
    return "\n".join(lines) + "\n"


def propose_step(verdict: dict, forecast: dict | None = None) -> str:
    """Choose a conservative demo ladder step from current, not forecast-only, evidence."""
    forecast = forecast or {}
    attack = str(verdict.get("current_state") or verdict.get("predicted_class") or "BENIGN")
    signature = bool(verdict.get("signature_confirmed") or verdict.get("signature", {}).get("confirmed"))
    if attack == "BENIGN":
        return "ALERT" if float(forecast.get("maximum_attack_probability", 0.0) or 0.0) >= 0.5 else "NONE"
    if not signature:
        return "ALERT"
    if attack == "PortScan":
        return "RATE_LIMIT"
    if attack == "DoS":
        return "FIREWALL_RULE"
    if attack == "DDoS":
        return "QUARANTINE"
    return "ALERT"


class DryRunResponseService:
    def __init__(self, audit_path: Path, protected_addresses=(), system: str | None = None):
        self.audit_path = Path(audit_path)
        self.protected = {str(ipaddress.ip_address(value)) for value in protected_addresses}
        self.system = system
        self._plans: dict[str, dict] = {}
        self._actions: dict[str, dict] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.RLock()
        self._rehydrate()

    def _read_audit(self) -> list[dict]:
        if not self.audit_path.exists():
            return []
        events = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def _rehydrate(self) -> None:
        overdue = []
        with self._lock:
            for event in self._read_audit():
                if event.get("event") == "PLAN" and event.get("plan_id"):
                    self._plans[event["plan_id"]] = {key: value for key, value in event.items() if key != "event"}
                if event.get("event") in {"APPLY", "ROLLBACK"} and event.get("action_id"):
                    self._actions[event["action_id"]] = {key: value for key, value in event.items() if key != "event"}
            now = time.time()
            for action_id, action in self._actions.items():
                if action.get("status") != "DRY_RUN_APPLIED":
                    continue
                remaining = float(action.get("applied_at", now)) + int(action.get("ttl_seconds", 0)) - now
                if remaining <= 0:
                    overdue.append(action_id)
                    continue
                timer = threading.Timer(remaining, self.rollback, args=(action_id, "TTL_EXPIRED"))
                timer.daemon = True
                self._timers[action_id] = timer
                timer.start()
        for action_id in overdue:
            self.rollback(action_id, "TTL_EXPIRED_AFTER_RESTART")

    def _append(self, event: dict) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True) + "\n"
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            os.chmod(self.audit_path, 0o600)

    def plan(self, verdict: dict, forecast: dict | None, target: str, ttl_seconds: int = 900) -> dict:
        try:
            normalized = str(ipaddress.ip_address(target))
        except ValueError as error:
            raise LadderError("Target must be a valid IP address.") from error
        if normalized in self.protected:
            raise LadderError("Target is protected and cannot receive a response action.")
        if not 1 <= int(ttl_seconds) <= 3600:
            raise LadderError("TTL must be between 1 and 3600 seconds.")
        step = propose_step(verdict, forecast)
        plan_id = str(uuid.uuid4())
        action_id = str(uuid.uuid4())
        apply_command, revert_command = _command(step, normalized, action_id, self.system)
        plan = {
            "plan_id": plan_id, "action_id": action_id, "step": step, "target": normalized,
            "ttl_seconds": int(ttl_seconds), "dry_run": True, "command": apply_command,
            "ruleset": _ruleset(step, normalized, action_id, int(ttl_seconds), self.system),
            "revert_command": revert_command, "requires_operator_ack": LADDER.index(step) > LADDER.index("RATE_LIMIT"),
            "created_at": time.time(),
        }
        with self._lock:
            self._plans[plan_id] = plan
        self._append({"event": "PLAN", **plan})
        return dict(plan)

    def apply(self, plan_id: str, operator_ack: bool = False) -> dict:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise LadderError("Unknown response plan.")
            if plan["step"] == "NONE":
                raise LadderError("A NONE recommendation has no response action to apply.")
            if plan["action_id"] in self._actions:
                raise LadderError("Response plan has already been applied.")
            if plan["requires_operator_ack"] and not operator_ack:
                raise LadderError("Explicit operator acknowledgement is required for this step.")
            action = {**plan, "status": "DRY_RUN_APPLIED", "applied_at": time.time(), "operator_ack": operator_ack}
            self._actions[plan["action_id"]] = action
            self._append({"event": "APPLY", **action})
            timer = threading.Timer(plan["ttl_seconds"], self.rollback, args=(plan["action_id"], "TTL_EXPIRED"))
            timer.daemon = True
            self._timers[plan["action_id"]] = timer
            timer.start()
            return dict(action)

    def rollback(self, action_id: str, reason: str = "OPERATOR") -> dict:
        with self._lock:
            action = self._actions.get(action_id)
            if action is None:
                raise LadderError("Unknown response action.")
            if action["status"] == "DRY_RUN_ROLLED_BACK":
                return dict(action)
            action["status"] = "DRY_RUN_ROLLED_BACK"
            action["rollback_reason"] = reason
            action["rolled_back_at"] = time.time()
            timer = self._timers.pop(action_id, None)
            if timer and reason != "TTL_EXPIRED":
                timer.cancel()
            self._append({"event": "ROLLBACK", **action})
            return dict(action)

    def audit(self) -> list[dict]:
        with self._lock:
            return self._read_audit()
