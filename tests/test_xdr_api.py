from fastapi import HTTPException
from starlette.requests import Request
from types import SimpleNamespace

from backend.api import main


def test_xdr_routes_are_registered_and_pipeline_shape_is_unchanged():
    paths = {route.path for route in main.app.routes}
    assert {"/api/ingest/zeek", "/api/ingest/zeek/dir", "/api/graph", "/api/triage",
            "/api/response/plan", "/api/response/apply", "/api/response/rollback",
            "/api/response/audit", "/api/deception/canary", "/api/deception/hits"}.issubset(paths)
    assert set(main.pipeline_state()) == {"stage", "error", "capture", "extraction", "prediction", "timings"}


def test_ingest_endpoint_round_trip_and_feature_gate(monkeypatch):
    monkeypatch.setattr(main, "_xdr_demo_enabled", False)
    monkeypatch.setattr(main, "load_config", lambda: {"xdr": {"enabled": False}})
    with __import__("pytest").raises(HTTPException) as disabled:
        main.ingest_zeek(main.ZeekIngestRequest(session_id="api-test", logs=[{
            "_path": "dns", "ts": 1, "query": "example.test", "rcode_name": "NOERROR",
        }]))
    assert disabled.value.status_code == 404

    monkeypatch.setattr(main, "_xdr_demo_enabled", True)
    result = main.ingest_zeek(main.ZeekIngestRequest(session_id="api-test", logs=[{
        "_path": "dns", "ts": 1, "query": "highentropyvalue.example", "rcode_name": "NXDOMAIN",
    }]))
    assert result["accepted"] == 1
    assert result["enrichment"]["nxdomain_ratio"] == 1.0


def test_requested_session_prediction_is_not_mixed(monkeypatch):
    wanted = SimpleNamespace(prediction_result={"session": "wanted"})
    monkeypatch.setattr(main.upload_mgr, "get_session", lambda value: wanted if value == "wanted" else None)
    assert main._current_xdr_prediction("wanted") == {"session": "wanted"}
    assert main._current_xdr_prediction("missing") == {}


def test_cross_site_canary_trigger_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "_xdr_demo_enabled", True)
    request = Request({
        "type": "http", "method": "GET", "path": "/api/deception/canary",
        "scheme": "http", "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 50000),
        "headers": [(b"host", b"127.0.0.1:8765"), (b"sec-fetch-site", b"cross-site")],
    })
    with __import__("pytest").raises(HTTPException) as rejected:
        main.deception_canary(request)
    assert rejected.value.status_code == 403


def test_prototype_never_wires_environment_firewall_helper(monkeypatch):
    from backend.response.helper_client import configured_helper

    monkeypatch.setenv("NIDS_FIREWALL_HELPER", "/tmp/should-not-run")
    monkeypatch.setattr(main, "_response_service", None)
    assert configured_helper() is None
    assert main._get_response_service().capabilities()["privilege_ready"] is False
