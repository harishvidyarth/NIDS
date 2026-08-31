from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.response.adapters import FirewallAdapter
from backend.response.service import ConflictError, ResponseService, _canonical_hash
from backend.response.store import ResponseStore


PREDICTION = {
    "signature_verdict": "PortScan",
    "signature_hits": [{"state": "PortScan", "rule": "fanout-port-scan"}],
    "port_scan_signature": {"src": "198.51.100.7", "dst": "203.0.113.20"},
    "flows": [{"src_ip": "198.51.100.7", "dst_ip": "203.0.113.20", "dst_port": 443,
               "protocol": "TCP", "signature_state": "PortScan"}],
}


class FakeAdapter(FirewallAdapter):
    def __init__(self):
        self.fingerprint = "clean"
        self.rules = set()

    def capabilities(self):
        return {"platform": "test", "engine": "fake", "privilege_ready": True,
                "supported_actions": ["scan", "apply", "verify", "rollback"]}

    def scan(self):
        return {"active": True, "conflicts": [], "namespace_healthy": True,
                "fingerprint": self.fingerprint, "nids_owned_rules": sorted(self.rules)}

    def render(self, action_id, targets, ttl_minutes):
        return {"commands": [["fake", "block", target.source_ip] for target in targets],
                "affected_traffic": [target.to_dict() for target in targets],
                "rollback": "Remove only fake identifiers for this action."}

    def apply(self, plan):
        identifier = f"fake:{plan['action_id']}"
        self.rules.add(identifier)
        self.fingerprint = "owned"
        return [identifier]

    def verify(self, action):
        return {"present": all(item in self.rules for item in action["native_identifiers"]),
                "identifiers": action["native_identifiers"]}

    def rollback(self, action):
        for item in action["native_identifiers"]:
            self.rules.discard(item)
        self.fingerprint = "clean"
        return {"removed": action["native_identifiers"]}


def _service(tmp_path):
    return ResponseService(ResponseStore(tmp_path / "response.sqlite3"), FakeAdapter())


def test_plan_hash_is_stable_and_full_lifecycle_is_audited(tmp_path):
    service = _service(tmp_path)
    plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"}, ttl_minutes=15)
    assert plan["state"] == "PROPOSED"
    assert len(plan["plan_hash"]) == 64
    action = service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    assert action["state"] == "APPLIED"
    with pytest.raises(ConflictError):
        service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    assert service.verify_action(action["action_id"])["state"] == "VERIFIED"
    assert service.rollback_action(action["action_id"])["state"] == "ROLLED_BACK"
    detail = service.get_action(action["action_id"])
    assert [event["state"] for event in detail["events"]] == [
        "DETECTED", "PROPOSED", "APPROVED", "APPLYING", "APPLIED",
        "VERIFYING", "VERIFIED", "ROLLING_BACK", "ROLLED_BACK",
    ]


def test_stale_hash_duplicate_apply_and_firewall_drift_are_conflicts(tmp_path):
    service = _service(tmp_path)
    plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"}, ttl_minutes=15)
    with pytest.raises(ConflictError):
        service.apply_plan(plan["plan_id"], "0" * 64, confirmed=True, actor="operator", current_prediction=PREDICTION)
    service.adapter.fingerprint = "drifted"
    with pytest.raises(ConflictError):
        service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)


def test_expiry_reconciliation_removes_only_action_rules(tmp_path):
    service = _service(tmp_path)
    plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"}, ttl_minutes=1)
    action = service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    past = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    service.store.set_expires_at(action["action_id"], past)
    service.reconcile()
    assert service.get_action(action["action_id"])["state"] == "EXPIRED"
    assert service.adapter.rules == set()


def test_canonical_plan_hash_is_order_independent():
    assert _canonical_hash({"b": 2, "a": [1]}) == _canonical_hash({"a": [1], "b": 2})


def test_verify_failure_and_duplicate_rollback_are_audited(tmp_path):
    service = _service(tmp_path)
    plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"})
    action = service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    service.adapter.rules.clear()
    failed = service.verify_action(action["action_id"])
    assert failed["state"] == "VERIFY_FAILED"
    rolled_back = service.rollback_action(action["action_id"])
    assert rolled_back["state"] == "ROLLED_BACK"
    with pytest.raises(ConflictError):
        service.rollback_action(action["action_id"])


def test_apply_failure_is_structured_and_terminal(tmp_path):
    service = _service(tmp_path)
    plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"})
    service.adapter.apply = lambda payload: (_ for _ in ()).throw(RuntimeError("helper unavailable"))
    with pytest.raises(RuntimeError, match="helper unavailable"):
        service.apply_plan(plan["plan_id"], plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    action = service.get_action(plan["action_id"])
    assert action["state"] == "APPLY_FAILED"
    assert action["failure"]["kind"] == "RuntimeError"


def test_rollback_removes_only_one_actions_identifiers(tmp_path):
    service = _service(tmp_path)
    first_plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"})
    first = service.apply_plan(first_plan["plan_id"], first_plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    second_plan = service.create_plan(PREDICTION, prediction_reference={"mode": "live"})
    second = service.apply_plan(second_plan["plan_id"], second_plan["plan_hash"], confirmed=True, actor="operator", current_prediction=PREDICTION)
    service.rollback_action(first["action_id"])
    assert f"fake:{first['action_id']}" not in service.adapter.rules
    assert f"fake:{second['action_id']}" in service.adapter.rules
