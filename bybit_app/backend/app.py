from __future__ import annotations

import time
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bybit_app.utils.ai.kill_switch import clear_pause, get_state as get_kill_switch_state, set_pause
from bybit_app.utils.background import get_guardian_state, get_preflight_snapshot, get_ws_snapshot
from bybit_app.utils.envs import get_api_client

app = FastAPI(title="BybitBot Backend", version="1.0.0")


class KillSwitchRequest(BaseModel):
    minutes: float | None = None
    reason: str | None = None


class OrderRequest(BaseModel):
    category: str = "spot"
    symbol: str
    side: str
    orderType: str = "Market"
    qty: float | None = None
    price: float | None = None
    timeInForce: str | None = None

    class Config:
        extra = "allow"


def _ensure_mapping(payload: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if payload is None:
        return {}
    if hasattr(payload, "dict"):
        try:
            result = payload.dict()
            if isinstance(result, Mapping):
                return result
        except Exception:  # pragma: no cover - defensive guard
            pass
    try:
        return dict(payload)  # type: ignore[arg-type]
    except Exception:
        return {}


@app.get("/health")
def health() -> Mapping[str, Any]:
    state = get_kill_switch_state()
    return {
        "status": "ok",
        "timestamp": time.time(),
        "killSwitch": {
            "paused": state.paused,
            "until": state.until,
            "reason": state.reason,
            "manual": getattr(state, "manual", False),
        },
    }


@app.get("/state/guardian")
def guardian_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_guardian_state())


@app.get("/state/ws")
def websocket_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_ws_snapshot())


@app.get("/state/preflight")
def preflight_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_preflight_snapshot())


@app.get("/state/summary")
def state_summary() -> Mapping[str, Any]:
    return {
        "guardian": guardian_state(),
        "websocket": websocket_state(),
        "preflight": preflight_state(),
        "killSwitch": health()["killSwitch"],
    }


@app.post("/kill-switch/pause")
def pause_kill_switch(request: KillSwitchRequest) -> Mapping[str, Any]:
    reason = (request.reason or "Paused via API").strip() or "Paused via API"
    until = set_pause(request.minutes, reason)
    return {"status": "paused", "until": until, "reason": reason}


@app.post("/kill-switch/resume")
def resume_kill_switch() -> Mapping[str, Any]:
    clear_pause()
    return {"status": "resumed"}


@app.post("/orders/place")
def place_order(request: OrderRequest) -> Mapping[str, Any]:
    try:
        client = get_api_client()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=503, detail="API client unavailable") from exc

    payload = request.dict(exclude_unset=True)
    try:
        result = client.place_order(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _ensure_mapping(result)


def create_app() -> FastAPI:
    """Return a configured FastAPI application."""

    return app
