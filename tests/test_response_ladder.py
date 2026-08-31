import time

import pytest

from backend.response.ladder import DryRunResponseService, LadderError, propose_step


def test_ladder_is_conservative_for_forecast_and_ann_only():
    assert propose_step({"current_state": "BENIGN"}, {"maximum_attack_probability": 0.8}) == "ALERT"
    assert propose_step({"current_state": "DoS", "signature_confirmed": False}) == "ALERT"
    assert propose_step({"current_state": "DoS", "signature_confirmed": True}) == "FIREWALL_RULE"


def test_none_is_never_applicable(tmp_path):
    service = DryRunResponseService(tmp_path / "audit.jsonl", system="Linux")
    plan = service.plan({"current_state": "BENIGN"}, {}, "198.51.100.4")
    assert plan["step"] == "NONE"
    assert plan["command"] == "no action (recommendation only)"
    with pytest.raises(LadderError, match="no response action"):
        service.apply(plan["plan_id"])


def test_protected_address_and_ack_gate(tmp_path):
    service = DryRunResponseService(tmp_path / "audit.jsonl", ["10.0.0.1"], system="Linux")
    with pytest.raises(LadderError, match="protected"):
        service.plan({"current_state": "DoS"}, {}, "10.0.0.1")
    plan = service.plan({"current_state": "DoS", "signature_confirmed": True}, {}, "10.0.0.9")
    with pytest.raises(LadderError, match="acknowledgement"):
        service.apply(plan["plan_id"])
    action = service.apply(plan["plan_id"], operator_ack=True)
    assert action["dry_run"] is True
    assert "/usr/sbin/nft" in action["command"]
    assert "add table inet nids_" in action["ruleset"]
    assert "10.0.0.9" in action["ruleset"]
    assert "delete table inet nids_" in action["revert_command"]


def test_ttl_rolls_back_and_appends_audit(tmp_path):
    service = DryRunResponseService(tmp_path / "audit.jsonl", system="Linux")
    plan = service.plan({"current_state": "PortScan", "signature_confirmed": True}, {}, "198.51.100.8", 1)
    action = service.apply(plan["plan_id"])
    deadline = time.time() + 2
    while time.time() < deadline and service.audit()[-1]["event"] != "ROLLBACK":
        time.sleep(0.02)
    events = service.audit()
    assert [event["event"] for event in events] == ["PLAN", "APPLY", "ROLLBACK"]
    assert events[-1]["rollback_reason"] == "TTL_EXPIRED"
    assert events[-1]["action_id"] == action["action_id"]


def test_command_injection_target_is_rejected(tmp_path):
    service = DryRunResponseService(tmp_path / "audit.jsonl")
    with pytest.raises(LadderError, match="valid IP"):
        service.plan({"current_state": "DoS"}, {}, "1.2.3.4; touch /tmp/pwned")


def test_restart_rehydrates_and_expires_pending_action(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = DryRunResponseService(path, system="Linux")
    plan = first.plan({"current_state": "PortScan", "signature_confirmed": True}, {}, "198.51.100.9", 1)
    action = first.apply(plan["plan_id"])
    first._timers[action["action_id"]].cancel()
    time.sleep(1.05)
    restored = DryRunResponseService(path, system="Linux")
    assert restored.audit()[-1]["event"] == "ROLLBACK"
    assert restored.audit()[-1]["rollback_reason"] == "TTL_EXPIRED_AFTER_RESTART"


def test_preview_rollbacks_are_action_scoped(tmp_path):
    mac = DryRunResponseService(tmp_path / "mac.jsonl", system="Darwin")
    plan = mac.plan({"current_state": "DoS", "signature_confirmed": True}, {}, "198.51.100.3")
    assert plan["action_id"].replace("-", "") in plan["revert_command"]
    linux = DryRunResponseService(tmp_path / "linux.jsonl", system="Linux")
    rate = linux.plan({"current_state": "PortScan", "signature_confirmed": True}, {}, "198.51.100.7")
    assert "src_ip" in rate["command"] and " pref " in rate["revert_command"]
