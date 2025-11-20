from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel

from bybit_app.utils.ai.kill_switch import clear_pause, get_state as get_kill_switch_state, set_pause
from bybit_app.utils.background import get_guardian_state, get_preflight_snapshot, get_ws_snapshot
from bybit_app.utils.envs import get_api_client, get_settings

AUTH_HEADER = "Authorization"
SIGNATURE_HEADER = "X-Bybit-Signature"
TIMESTAMP_HEADER = "X-Bybit-Timestamp"
TIMESTAMP_WINDOW_SECONDS = 300
FUTURE_TIMESTAMP_GRACE_SECONDS = 10
SIGNATURE_CACHE_SIZE = 1024
FAILURE_TRACKER_TTL_SECONDS = 60
FAILURE_TRACKER_MAX_ATTEMPTS = 3

app = FastAPI(title="BybitBot Backend", version="1.0.0")
logger = logging.getLogger(__name__)


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
        extra = "forbid"


class _SignatureLRU:
    def __init__(self, ttl_seconds: float, max_items: int = SIGNATURE_CACHE_SIZE) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        while self._items:
            key, seen_at = next(iter(self._items.items()))
            if seen_at < cutoff:
                self._items.popitem(last=False)
            else:
                break

    def is_replay(self, key: str, *, now: float) -> bool:
        with self._lock:
            self._purge_expired(now)

            seen_at = self._items.get(key)
            if seen_at is not None and now - seen_at <= self.ttl_seconds:
                return True

            self._items[key] = now
            self._items.move_to_end(key)

            if len(self._items) > self.max_items:
                self._items.popitem(last=False)

            return False


_signature_cache = _SignatureLRU(ttl_seconds=TIMESTAMP_WINDOW_SECONDS)


class _FailureTracker:
    def __init__(self, ttl_seconds: float, max_attempts: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_attempts = max_attempts
        self._attempts: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._lock = Lock()

    def _purge_expired(self, now: float) -> None:
        while self._attempts:
            key, (_, expires_at) = next(iter(self._attempts.items()))
            if expires_at < now:
                self._attempts.popitem(last=False)
            else:
                break

    def register_failure(self, key: str, *, now: float) -> bool:
        with self._lock:
            self._purge_expired(now)

            count, expires_at = self._attempts.get(key, (0, now + self.ttl_seconds))
            if now > expires_at:
                count = 0
                expires_at = now + self.ttl_seconds

            count += 1
            self._attempts[key] = (count, expires_at)
            self._attempts.move_to_end(key)
            return count >= self.max_attempts


_failure_tracker = _FailureTracker(
    ttl_seconds=FAILURE_TRACKER_TTL_SECONDS, max_attempts=FAILURE_TRACKER_MAX_ATTEMPTS
)


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


async def verify_backend_auth(
    request: Request,
    authorization: str | None = Header(default=None, alias=AUTH_HEADER),
    signature: str | None = Header(default=None, alias=SIGNATURE_HEADER),
    timestamp: str | None = Header(default=None, alias=TIMESTAMP_HEADER),
) -> None:
    settings = get_settings()
    secret = getattr(settings, "backend_auth_token", "").strip()

    if not secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Backend authentication is not configured",
        )

    def _match_bearer(token: str | None) -> bool:
        if not token:
            return False
        if not token.lower().startswith("bearer "):
            return False
        provided = token.split(" ", 1)[1].strip()
        return hmac.compare_digest(provided, secret)

    def _failure_key() -> str:
        client_host = request.client.host if request.client else "unknown"
        if signature:
            return f"sig:{signature}"
        return f"ip:{client_host}"

    def _maybe_throttle() -> None:
        if _failure_tracker.register_failure(_failure_key(), now=time.time()):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts",
            )

    if authorization and authorization.lower().startswith("bearer "):
        if _match_bearer(authorization):
            return
        _maybe_throttle()
        logger.warning(
            "backend_auth_invalid_token",
            extra={
                "event": "backend_auth_invalid_token",
                "client": request.client.host if request.client else "unknown",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
        )

    if not signature and not timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
        )

    if not signature or not timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signature authentication requires X-Bybit-Signature and X-Bybit-Timestamp headers",
        )

    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid timestamp format",
        ) from None

    now = time.time()
    if timestamp_value - now > FUTURE_TIMESTAMP_GRACE_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Timestamp cannot be in the future",
        )

    if now - timestamp_value > TIMESTAMP_WINDOW_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Timestamp is too old",
        )

    body = await request.body()
    method = request.method
    raw_path = request.scope.get("raw_path")
    query_string = request.scope.get("query_string", b"")

    if isinstance(raw_path, (bytes, bytearray)):
        path = raw_path.decode("latin-1")
    else:
        path = raw_path or request.url.path

    if query_string:
        query = query_string.decode("latin-1") if isinstance(query_string, (bytes, bytearray)) else str(query_string)
        if "?" not in path:
            path = f"{path}?{query}"

    payload = f"{timestamp}.{method}.{path}.".encode() + body
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        _maybe_throttle()
        logger.warning(
            "backend_auth_invalid_signature",
            extra={
                "event": "backend_auth_invalid_signature",
                "client": request.client.host if request.client else "unknown",
                "method": method,
                "path": path,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid signature",
        )

    if _signature_cache.is_replay(signature, now=now):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Duplicate signature detected",
        )

    return


@app.get("/health", dependencies=[Depends(verify_backend_auth)])
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


@app.get("/state/guardian", dependencies=[Depends(verify_backend_auth)])
def guardian_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_guardian_state())


@app.get("/state/ws", dependencies=[Depends(verify_backend_auth)])
def websocket_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_ws_snapshot())


@app.get("/state/preflight", dependencies=[Depends(verify_backend_auth)])
def preflight_state() -> Mapping[str, Any]:
    return _ensure_mapping(get_preflight_snapshot())


@app.get("/state/summary", dependencies=[Depends(verify_backend_auth)])
def state_summary() -> Mapping[str, Any]:
    return {
        "guardian": guardian_state(),
        "websocket": websocket_state(),
        "preflight": preflight_state(),
        "killSwitch": health()["killSwitch"],
    }


@app.post("/kill-switch/pause", dependencies=[Depends(verify_backend_auth)])
def pause_kill_switch(request: KillSwitchRequest) -> Mapping[str, Any]:
    reason = (request.reason or "Paused via API").strip() or "Paused via API"
    until = set_pause(request.minutes, reason)
    return {"status": "paused", "until": until, "reason": reason}


@app.post("/kill-switch/resume", dependencies=[Depends(verify_backend_auth)])
def resume_kill_switch() -> Mapping[str, Any]:
    clear_pause()
    return {"status": "resumed"}


@app.post("/orders/place", dependencies=[Depends(verify_backend_auth)])
def place_order(request: OrderRequest) -> Mapping[str, Any]:
    try:
        client = get_api_client()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=503, detail="API client unavailable") from exc

    payload = {
        "category": request.category,
        "symbol": request.symbol,
        "side": request.side,
        "orderType": request.orderType,
    }
    if request.qty is not None:
        payload["qty"] = request.qty
    if request.price is not None:
        payload["price"] = request.price
    if request.timeInForce is not None:
        payload["timeInForce"] = request.timeInForce
    try:
        result = client.place_order(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _ensure_mapping(result)


def create_app() -> FastAPI:
    """Return a configured FastAPI application."""

    return app
