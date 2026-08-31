from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from .adapters import FirewallAdapter
from .models import ResponseTarget
from .policy import PolicyError, discover_system_protected_addresses, evaluate_prediction
from .store import ResponseStore


class ConflictError(RuntimeError):
    pass


class NotFoundError(LookupError):
    pass


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


PLAN_VALIDITY_MINUTES = 5


class ResponseService:
    def __init__(self, store: ResponseStore, adapter: FirewallAdapter, protected_addresses: set[str] | None = None):
        self.store = store
        self.adapter = adapter
        self.protected_addresses = discover_system_protected_addresses() | (protected_addresses or set())
        self._lock = threading.RLock()
        self._last_scan_monotonic = 0.0

    def capabilities(self) -> dict[str, Any]:
        return self.adapter.capabilities()

    def scan(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if now - self._last_scan_monotonic < 1.0:
                raise ConflictError("Firewall scans are limited to one per second.")
            self._last_scan_monotonic = now
            before = self.adapter.scan()
            after = self.adapter.scan()
            if before.get("fingerprint") != after.get("fingerprint"):
                raise ConflictError("Read-only firewall scan observed drift while scanning; retry.")
            payload = {**after, "fingerprint_unchanged": True}
            return self.store.save_scan(str(uuid.uuid4()), payload)

    def create_plan(
        self, prediction: dict[str, Any], *, prediction_reference: dict[str, Any], ttl_minutes: int = 15,
        selected_targets: Sequence[dict[str, Any]] | None = None, actor: str | None = None,
    ) -> dict[str, Any]:
        decision = evaluate_prediction(
            prediction, ttl_minutes=ttl_minutes, protected_addresses=self.protected_addresses,
            selected_targets=selected_targets,
        )
        plan_id = str(uuid.uuid4())
        action_id = str(uuid.uuid4())
        prediction_digest = _canonical_hash(prediction)
        plan_expires_at = (datetime.now(timezone.utc) + timedelta(minutes=PLAN_VALIDITY_MINUTES)).isoformat()
        scan = self.adapter.scan()
        rendered = self.adapter.render(action_id, decision.targets, ttl_minutes) if decision.executable else {
            "engine": self.adapter.capabilities().get("engine"), "commands": [], "affected_traffic": [],
            "ttl_minutes": ttl_minutes, "rollback": "No executable change is proposed.",
        }
        immutable = {
            "plan_id": plan_id, "action_id": action_id, "prediction_reference": prediction_reference,
            "eligibility": decision.to_dict(), "evidence": decision.evidence,
            "targets": [target.to_dict() for target in decision.targets], "ttl_minutes": ttl_minutes,
            "native_changes": rendered, "warnings": decision.warnings,
            "limitations": decision.limitations, "upstream_recommendation": decision.upstream_recommendation,
            "scan_fingerprint": scan["fingerprint"], "prediction_digest": prediction_digest,
            "plan_expires_at": plan_expires_at,
        }
        plan_hash = _canonical_hash(immutable)
        payload = {**immutable, "plan_hash": plan_hash, "state": "PROPOSED"}
        self.store.create_plan_action(plan_id=plan_id, action_id=action_id, plan_hash=plan_hash,
                                      scan_fingerprint=scan["fingerprint"], payload=payload, actor=actor)
        return payload

    def apply_plan(
        self, plan_id: str, plan_hash: str, *, confirmed: bool, actor: str,
        current_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not confirmed:
            raise ConflictError("Explicit operator confirmation is required.")
        with self._lock:
            plan = self.store.get_plan(plan_id)
            if plan is None:
                raise NotFoundError("Response plan not found.")
            if plan_hash != plan["plan_hash"]:
                raise ConflictError("Displayed plan hash is stale or does not match the immutable plan.")
            if datetime.fromisoformat(plan["plan_expires_at"]) <= datetime.now(timezone.utc):
                raise ConflictError("Response plan expired; create a fresh preview from current evidence.")
            if current_prediction is None or _canonical_hash(current_prediction) != plan["prediction_digest"]:
                raise ConflictError("Referenced prediction changed after preview; create a fresh plan.")
            if not plan.get("eligibility", {}).get("executable"):
                raise ConflictError("This recommendation has no executable firewall plan.")
            action = self.store.get_action(plan["action_id"])
            if action["state"] != "PROPOSED":
                raise ConflictError(f"Cannot apply an action in state {action['state']}.")
            scan = self.adapter.scan()
            if scan["fingerprint"] != plan["scan_fingerprint"]:
                raise ConflictError("Firewall drifted after preview; scan and create a new plan.")
            if not scan.get("namespace_healthy"):
                raise ConflictError("The isolated NIDS firewall namespace/anchor is not ready for safe mutation.")
            if not self.store.transition(action["action_id"], expected={"PROPOSED"}, state="APPROVED", actor=actor,
                                         event_payload={"plan_hash": plan_hash}):
                raise ConflictError("The action lifecycle changed concurrently.")
            self.store.transition(action["action_id"], expected={"APPROVED"}, state="APPLYING", actor=actor)
            helper_plan = {"action_id": action["action_id"], "plan_hash": plan_hash,
                           "targets": plan["targets"], "ttl_minutes": plan["ttl_minutes"],
                           "native_changes": plan["native_changes"]}
            try:
                identifiers = self.adapter.apply(helper_plan)
                if not identifiers:
                    raise RuntimeError("Privileged helper returned no native rule identifiers.")
            except Exception as exc:
                self.store.transition(action["action_id"], expected={"APPLYING"}, state="APPLY_FAILED", actor=actor,
                                      failure={"kind": type(exc).__name__, "message": str(exc)})
                raise
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=int(plan["ttl_minutes"]))).isoformat()
            self.store.transition(action["action_id"], expected={"APPLYING"}, state="APPLIED", actor=actor,
                                  native_identifiers=identifiers, expires_at=expires_at,
                                  event_payload={"native_identifiers": identifiers, "expires_at": expires_at})
            return self.get_action(action["action_id"])

    def verify_action(self, action_id: str, *, actor: str = "operator") -> dict[str, Any]:
        with self._lock:
            action = self._action(action_id)
            if action["state"] not in {"APPLIED", "VERIFIED", "VERIFY_FAILED"}:
                raise ConflictError(f"Cannot verify an action in state {action['state']}.")
            if not self.store.transition(action_id, expected={action["state"]}, state="VERIFYING", actor=actor):
                raise ConflictError("The action lifecycle changed concurrently.")
            action = self._action(action_id)
            try:
                result = self.adapter.verify(action)
                if not result.get("present"):
                    raise RuntimeError("Expected NIDS-owned firewall rules are not present.")
            except Exception as exc:
                self.store.transition(action_id, expected={"VERIFYING"}, state="VERIFY_FAILED", actor=actor,
                                      failure={"kind": type(exc).__name__, "message": str(exc)})
                return self.get_action(action_id)
            self.store.transition(action_id, expected={"VERIFYING"}, state="VERIFIED", actor=actor,
                                  verification=result, event_payload=result)
            return self.get_action(action_id)

    def rollback_action(self, action_id: str, *, actor: str = "operator", expired: bool = False) -> dict[str, Any]:
        with self._lock:
            action = self._action(action_id)
            allowed = {"APPLIED", "VERIFIED", "VERIFY_FAILED", "ROLLBACK_FAILED"}
            if action["state"] not in allowed:
                raise ConflictError(f"Cannot rollback an action in state {action['state']}.")
            if not self.store.transition(action_id, expected={action["state"]}, state="ROLLING_BACK", actor=actor,
                                         event_payload={"expiry": expired}):
                raise ConflictError("The action lifecycle changed concurrently.")
            action = self._action(action_id)
            try:
                result = self.adapter.rollback(action)
            except Exception as exc:
                self.store.transition(action_id, expected={"ROLLING_BACK"}, state="ROLLBACK_FAILED", actor=actor,
                                      failure={"kind": type(exc).__name__, "message": str(exc)})
                return self.get_action(action_id)
            final_state = "EXPIRED" if expired else "ROLLED_BACK"
            self.store.transition(action_id, expected={"ROLLING_BACK"}, state=final_state, actor=actor,
                                  event_payload=result)
            return self.get_action(action_id)

    def reconcile(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        reconciled = []
        for action in self.store.list_actions({"APPLIED", "VERIFIED", "VERIFY_FAILED"}):
            expiry = datetime.fromisoformat(action["expires_at"]) if action.get("expires_at") else None
            if expiry and expiry <= now:
                reconciled.append(self.rollback_action(action["action_id"], actor="system-expiry", expired=True))
            else:
                reconciled.append(self.verify_action(action["action_id"], actor="system-reconcile"))
        return reconciled

    def expire_due(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        expired = []
        for action in self.store.list_actions({"APPLIED", "VERIFIED", "VERIFY_FAILED"}):
            expiry = datetime.fromisoformat(action["expires_at"]) if action.get("expires_at") else None
            if expiry and expiry <= now:
                expired.append(self.rollback_action(action["action_id"], actor="system-expiry", expired=True))
        return expired

    def list_actions(self) -> list[dict[str, Any]]:
        return self.store.list_actions()

    def get_action(self, action_id: str) -> dict[str, Any]:
        return self._action(action_id)

    def _action(self, action_id: str) -> dict[str, Any]:
        action = self.store.get_action(action_id)
        if action is None:
            raise NotFoundError("Response action not found.")
        return action
