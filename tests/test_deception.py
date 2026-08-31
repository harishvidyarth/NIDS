from __future__ import annotations

from datetime import datetime, timezone

from backend.deception.service import HONEYTOKEN_PATH, CanaryStore
from backend.triage.service import build_triage


def test_canary_hit_is_recorded_with_request_evidence():
    instant = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = CanaryStore(clock=lambda: instant)

    hit = store.record_hit("192.0.2.44", "xdr-demo/1.0")

    assert store.list_hits() == [hit]
    assert hit["source_ip"] == "192.0.2.44"
    assert hit["user_agent"] == "xdr-demo/1.0"
    assert hit["timestamp"] == "2026-08-30T12:00:00+00:00"
    assert hit["honeytoken_path"] == "backend/deception/honeytoken_credentials.txt"
    assert HONEYTOKEN_PATH.is_file()


def test_hit_surfaces_as_high_confidence_triage_event_and_campaign_boost():
    store = CanaryStore()
    store.record_hit("198.51.100.7", "curl/8")

    events = store.high_confidence_events()
    triage = build_triage({
        "verdict": {"final_verdict": "BENIGN"},
        "forecast": {},
        "mitre": {},
        "campaign_score": store.campaign_score_boost(),
        "enrichment": {},
        "deception_events": events,
    })

    assert events[0]["event_type"] == "DECEPTION_CANARY_HIT"
    assert events[0]["confidence"] == 1.0
    assert store.campaign_score_boost() == 0.20
    assert triage["confidence"] == "HIGH"
    assert "deception canary" in triage["summary"].lower()


def test_canary_store_returns_copies_and_bounds_campaign_boost():
    store = CanaryStore()
    for index in range(5):
        store.record_hit(f"192.0.2.{index}", None)

    copy = store.list_hits()
    copy[0]["source_ip"] = "changed"

    assert store.list_hits()[0]["source_ip"] != "changed"
    assert store.campaign_score_boost() == 0.35
