from __future__ import annotations

import secrets
import ipaddress
import getpass
from urllib.parse import urlsplit
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .policy import PolicyError
from .service import ConflictError, NotFoundError, ResponseService


LOCAL_AUTHORIZATION_TOKEN = secrets.token_urlsafe(32)
LOCAL_ACTOR = getpass.getuser() or "local-operator"
TRUSTED_HOSTS = {"127.0.0.1", "localhost", "::1"}


class PredictionReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["live", "upload"]
    session_id: str | None = None


class SelectedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ip: str


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_reference: PredictionReference
    selected_targets: list[SelectedTarget] | None = None
    ttl_minutes: int = Field(default=15, ge=1, le=60)


class ApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_hash: str = Field(min_length=64, max_length=64)
    confirmed: bool


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool


def _trusted_local_request(request: Request) -> bool:
    try:
        local_client = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local_client = False
    host_header = request.headers.get("host", "")
    try:
        host = urlsplit(f"//{host_header}").hostname
    except ValueError:
        host = None
    if not local_client or host not in TRUSTED_HOSTS:
        return False
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or parsed.netloc != host_header:
            return False
    return True


def require_local_authorization(
    request: Request,
    token: str | None = Header(default=None, alias="X-NIDS-Response-Token"),
) -> None:
    if not _trusted_local_request(request) or token is None or not secrets.compare_digest(token, LOCAL_AUTHORIZATION_TOKEN):
        raise HTTPException(status_code=403, detail={"code": "LOCAL_AUTH_REQUIRED", "message": "Local response authorization is required."})


def create_response_router(
    service_getter: Callable[[], ResponseService],
    prediction_resolver: Callable[[dict[str, Any]], dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/api/response", tags=["response"])

    def service() -> ResponseService:
        return service_getter()

    def translate(exc: Exception) -> HTTPException:
        if isinstance(exc, NotFoundError):
            return HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": str(exc)})
        if isinstance(exc, (ConflictError, PolicyError)):
            return HTTPException(status_code=409, detail={"code": "RESPONSE_CONFLICT", "message": str(exc)})
        return HTTPException(status_code=503, detail={"code": "HELPER_UNAVAILABLE", "message": str(exc)})

    @router.get("/capabilities")
    def capabilities(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if not _trusted_local_request(request):
            raise HTTPException(status_code=403, detail={"code": "LOCAL_ONLY", "message": "Response capabilities are available only to the same-origin loopback UI."})
        return {**service().capabilities(), "local_authorization_token": LOCAL_AUTHORIZATION_TOKEN,
                "local_actor": LOCAL_ACTOR,
                "token_scope": "current backend process; send only in X-NIDS-Response-Token"}

    @router.post("/scan", dependencies=[Depends(require_local_authorization)])
    def scan():
        try:
            return service().scan()
        except Exception as exc:
            raise translate(exc) from exc

    @router.get("/scans/{scan_id}", dependencies=[Depends(require_local_authorization)])
    def get_scan(scan_id: str):
        result = service().store.get_scan(scan_id)
        if result is None:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Firewall scan not found."})
        return result

    @router.post("/plans", dependencies=[Depends(require_local_authorization)])
    def create_plan(request: PlanRequest):
        try:
            reference = request.prediction_reference.model_dump()
            prediction = prediction_resolver(reference)
            targets = [item.model_dump() for item in request.selected_targets] if request.selected_targets is not None else None
            return service().create_plan(prediction, prediction_reference=reference,
                                         selected_targets=targets, ttl_minutes=request.ttl_minutes, actor=LOCAL_ACTOR)
        except Exception as exc:
            raise translate(exc) from exc

    @router.post("/plans/{plan_id}/apply", dependencies=[Depends(require_local_authorization)])
    def apply_plan(plan_id: str, request: ApplyRequest):
        try:
            saved_plan = service().store.get_plan(plan_id)
            if saved_plan is None:
                raise NotFoundError("Response plan not found.")
            current_prediction = prediction_resolver(saved_plan["prediction_reference"])
            return service().apply_plan(plan_id, request.plan_hash, confirmed=request.confirmed,
                                        actor=LOCAL_ACTOR, current_prediction=current_prediction)
        except Exception as exc:
            raise translate(exc) from exc

    @router.post("/actions/{action_id}/verify", dependencies=[Depends(require_local_authorization)])
    def verify_action(action_id: str, request: ActionRequest):
        if not request.confirmed:
            raise HTTPException(status_code=409, detail={"code": "CONFIRMATION_REQUIRED", "message": "Explicit verification confirmation is required."})
        try:
            return service().verify_action(action_id, actor=LOCAL_ACTOR)
        except Exception as exc:
            raise translate(exc) from exc

    @router.post("/actions/{action_id}/rollback", dependencies=[Depends(require_local_authorization)])
    def rollback_action(action_id: str, request: ActionRequest):
        if not request.confirmed:
            raise HTTPException(status_code=409, detail={"code": "CONFIRMATION_REQUIRED", "message": "Explicit rollback confirmation is required."})
        try:
            return service().rollback_action(action_id, actor=LOCAL_ACTOR)
        except Exception as exc:
            raise translate(exc) from exc

    @router.get("/actions", dependencies=[Depends(require_local_authorization)])
    def list_actions():
        return {"actions": service().list_actions()}

    @router.get("/actions/{action_id}", dependencies=[Depends(require_local_authorization)])
    def get_action(action_id: str):
        try:
            return service().get_action(action_id)
        except Exception as exc:
            raise translate(exc) from exc

    return router
