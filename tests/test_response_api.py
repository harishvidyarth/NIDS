from __future__ import annotations

import pytest
from fastapi import HTTPException
from types import SimpleNamespace

from backend.api import main
from backend.response.api import (
    LOCAL_AUTHORIZATION_TOKEN, ApplyRequest, PlanRequest, PredictionReference,
    require_local_authorization,
)
from backend.response.service import ResponseService
from backend.response.store import ResponseStore
from test_response_service import FakeAdapter, PREDICTION


def _endpoint(path: str, method: str):
    return next(route.endpoint for route in main.app.routes
                if getattr(route, "path", None) == path and method in getattr(route, "methods", set()))


def _request(host="127.0.0.1", host_header="127.0.0.1:8765", origin=None):
    headers = {"host": host_header}
    if origin is not None:
        headers["origin"] = origin
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


class _Response:
    def __init__(self):
        self.headers = {}


def test_response_api_requires_token_and_exposes_lifecycle(tmp_path, monkeypatch):
    service = ResponseService(ResponseStore(tmp_path / "api.sqlite3"), FakeAdapter())
    monkeypatch.setattr(main, "_response_service", service)
    monkeypatch.setattr(main.state, "prediction_result", PREDICTION)
    with pytest.raises(HTTPException) as denied:
        require_local_authorization(_request(), None)
    assert denied.value.status_code == 403
    with pytest.raises(HTTPException):
        require_local_authorization(_request("192.0.2.5"), LOCAL_AUTHORIZATION_TOKEN)
    with pytest.raises(HTTPException):
        require_local_authorization(_request(host_header="evil.example"), LOCAL_AUTHORIZATION_TOKEN)
    with pytest.raises(HTTPException):
        require_local_authorization(_request(origin="https://evil.example"), LOCAL_AUTHORIZATION_TOKEN)
    require_local_authorization(_request(), LOCAL_AUTHORIZATION_TOKEN)

    capabilities = _endpoint("/api/response/capabilities", "GET")(_request(), _Response())
    assert capabilities["local_authorization_token"] == LOCAL_AUTHORIZATION_TOKEN
    plan = _endpoint("/api/response/plans", "POST")(
        PlanRequest(prediction_reference=PredictionReference(mode="live"), ttl_minutes=15)
    )
    applied = _endpoint("/api/response/plans/{plan_id}/apply", "POST")(
        plan["plan_id"], ApplyRequest(plan_hash=plan["plan_hash"], confirmed=True)
    )
    assert applied["state"] == "APPLIED"


def test_stale_plan_returns_structured_409(tmp_path, monkeypatch):
    service = ResponseService(ResponseStore(tmp_path / "api.sqlite3"), FakeAdapter())
    monkeypatch.setattr(main, "_response_service", service)
    monkeypatch.setattr(main.state, "prediction_result", PREDICTION)
    plan = _endpoint("/api/response/plans", "POST")(
        PlanRequest(prediction_reference=PredictionReference(mode="live"))
    )
    with pytest.raises(HTTPException) as conflict:
        _endpoint("/api/response/plans/{plan_id}/apply", "POST")(
            plan["plan_id"], ApplyRequest(plan_hash="0" * 64, confirmed=True)
        )
    assert conflict.value.status_code == 409
    assert conflict.value.detail["code"] == "RESPONSE_CONFLICT"
