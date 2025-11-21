from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Mapping
from urllib.parse import parse_qsl, quote, urlencode

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from enum import Enum

from pydantic import BaseModel, root_validator, validator

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
FAILURE_TRACKER_MAX_SIZE = 1024
MAX_BACKEND_BODY_BYTES = 5 * 1024 * 1024

app = FastAPI(title="BybitBot Backend", version="1.0.0")
logger = logging.getLogger(__name__)


class KillSwitchRequest(BaseModel):
    minutes: float | None = None
    reason: str | None = None


class CategoryEnum(str, Enum):
    SPOT = "spot"
    LINEAR = "linear"
    INVERSE = "inverse"
    OPTION = "option"


class SideEnum(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderTypeEnum(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class TimeInForceEnum(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "PostOnly"


class OrderRequest(BaseModel):
    category: CategoryEnum = CategoryEnum.SPOT
    symbol: str
    side: SideEnum
    orderType: OrderTypeEnum = OrderTypeEnum.MARKET
    qty: float
    price: float | None = None
    timeInForce: TimeInForceEnum | None = None

    class Config:
        extra = "forbid"

    @validator("qty")
    def _qty_required(cls, value: float) -> float:
        if value is None:  # pragma: no cover - guarded by typing
            raise ValueError("qty is required")
        if value <= 0:
            raise ValueError("qty must be positive")
        return value

    @validator("price")
    def _positive_price(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("price must be positive")
        return value

    @root_validator
    def _price_required_for_limit(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
        order_type = values.get("orderType")
        price = values.get("price")

        if order_type == OrderTypeEnum.LIMIT and price is None:
            raise ValueError("price is required for limit orders")

        return values


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
    def __init__(self, ttl_seconds: float, max_attempts: int, max_items: int = FAILURE_TRACKER_MAX_SIZE) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_attempts = max_attempts
        self.max_items = max_items
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

            if len(self._attempts) > self.max_items:
                self._attempts.popitem(last=False)
            return count >= self.max_attempts

    def clear(self, key: str, *, now: float) -> None:
        with self._lock:
            self._purge_expired(now)
            self._attempts.pop(key, None)


_failure_tracker = _FailureTracker(
    ttl_seconds=FAILURE_TRACKER_TTL_SECONDS,
    max_attempts=FAILURE_TRACKER_MAX_ATTEMPTS,
    max_items=FAILURE_TRACKER_MAX_SIZE,
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


async def _read_request_body(request: Request) -> bytes:
    header_value = request.headers.get("content-length")
    if header_value is not None:
        try:
            content_length = int(header_value)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header",
            ) from None
        if content_length < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header",
            )
        if content_length > MAX_BACKEND_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request body too large",
            )

    body_chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BACKEND_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request body too large",
            )
        body_chunks.append(chunk)

    body = b"".join(body_chunks)
    request._body = body  # type: ignore[attr-defined]
    request._stream_consumed = True  # type: ignore[attr-defined]
    return body


def _strip_port(host: str | None) -> str:
    if not host:
        return ""
    host = host.strip().strip("\"")
    if not host:
        return ""
    if host.startswith("[") and "]" in host:
        host = host[1 : host.index("]")]
        return host
    if ":" in host:
        host = host.split(":", 1)[0]
    return host


def _trusted_proxy_hosts(settings: Any) -> set[str]:
    raw = getattr(settings, "trusted_proxy_hosts", "") or ""
    hosts = {_strip_port(item).lower() for item in str(raw).split(",") if _strip_port(item)}
    return hosts


def _forwarded_client_host(request: Request) -> str | None:
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        for part in x_forwarded_for.split(","):
            cleaned = _strip_port(part)
            if cleaned:
                return cleaned

    forwarded = request.headers.get("Forwarded")
    if forwarded:
        for entry in forwarded.split(","):
            for segment in entry.split(";"):
                segment = segment.strip()
                if segment.lower().startswith("for="):
                    candidate = segment.split("=", 1)[1].strip().strip("\"")
                    cleaned = _strip_port(candidate)
                    if cleaned:
                        return cleaned
    return None


def _client_host(request: Request, settings: Any) -> str:
    direct_host = _strip_port(request.client.host if request.client else "")
    trust_all = bool(getattr(settings, "trust_proxy_headers", False))
    trusted = _trusted_proxy_hosts(settings)

    if trust_all or (direct_host and direct_host.lower() in trusted):
        forwarded = _forwarded_client_host(request)
        if forwarded:
            return forwarded

    return direct_host or "unknown"


async def verify_backend_auth(
    request: Request,
    authorization: str | None = Header(default=None, alias=AUTH_HEADER),
    signature: str | None = Header(default=None, alias=SIGNATURE_HEADER),
    timestamp: str | None = Header(default=None, alias=TIMESTAMP_HEADER),
) -> None:
    settings = get_settings()
    secret = getattr(settings, "backend_auth_token", "").strip()
    client_host = _client_host(request, settings)

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
        host = client_host or "unknown"
        return f"ip:{host}"

    failure_key = _failure_key()

    def _maybe_throttle() -> None:
        if _failure_tracker.register_failure(failure_key, now=time.time()):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts",
            )

    if authorization and authorization.lower().startswith("bearer "):
        if _match_bearer(authorization):
            _failure_tracker.clear(failure_key, now=time.time())
            return
        _maybe_throttle()
        logger.warning(
            "backend_auth_invalid_token",
            extra={
                "event": "backend_auth_invalid_token",
                "client": client_host,
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

    body = await _read_request_body(request)
    method = request.method
    prefix = getattr(settings, "backend_path_prefix", "").strip()
    path = request.url.path
    if prefix:
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        if path.startswith(prefix):
            path = path[len(prefix) :] or "/"
    normalized_path = quote(path, safe="/-._~") or "/"

    query_string = request.url.query
    normalized_query = ""
    if query_string:
        query_params = parse_qsl(query_string, keep_blank_values=True)
        normalized_query = urlencode(query_params, doseq=True, safe="-._~")

    signable_path = normalized_path
    if normalized_query:
        signable_path = f"{normalized_path}?{normalized_query}"

    payload = f"{timestamp}.{method}.{signable_path}.".encode() + body
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        _maybe_throttle()
        logger.warning(
            "backend_auth_invalid_signature",
            extra={
                "event": "backend_auth_invalid_signature",
                "client": client_host,
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

    _failure_tracker.clear(failure_key, now=now)
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
