
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import threading
import time

import requests
from datetime import datetime, timezone

from .envs import get_api_client
from .http_client import configure_http_session
from .log import log

_SERVER_TIME_KEYS: tuple[str, ...] = (
    "timeNano",
    "timeNs",
    "timeSecond",
    "serverTime",
    "currentTime",
    "time",
    "timestamp",
)


_CLOCKS: dict[tuple[str, bool], "_SyncedClock"] = {}
_CLOCK_LOCK = threading.Lock()


@dataclass(frozen=True)
class SyncedTimestamp:
    """Snapshot of the current exchange-adjusted time."""

    value_ms: int
    offset_ms: float
    latency_ms: float

    def as_int(self) -> int:
        """Return the timestamp as an ``int`` for backwards compatibility."""

        return int(self.value_ms)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_epoch(raw_value: float) -> float:
    """Normalise Bybit timestamps to seconds."""

    if raw_value > 1e15:  # nanoseconds
        return raw_value / 1_000_000_000.0
    if raw_value > 1e12:  # milliseconds
        return raw_value / 1_000.0
    return raw_value


def extract_server_epoch(payload: Dict[str, Any]) -> Optional[float]:
    """Extract unix timestamp in seconds from Bybit server time payload."""

    if not isinstance(payload, dict):
        return None

    sources: Iterable[Dict[str, Any]] = []
    result_obj = payload.get("result")
    if isinstance(result_obj, dict):
        sources = (result_obj, payload)
    else:
        sources = (payload,)

    for source in sources:
        for key in _SERVER_TIME_KEYS:
            value = _safe_float(source.get(key))
            if value is not None:
                return _normalise_epoch(value)

    return None


def extract_server_datetime(payload: Dict[str, Any]) -> Optional[datetime]:
    """Return server time as an aware ``datetime`` in UTC."""

    epoch = extract_server_epoch(payload)
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def check_time_drift_seconds() -> float:
    """Return ``local_time - server_time`` in seconds.

    Falls back to ``0.0`` when the server response cannot be parsed or the
    request fails, logging the corresponding error for diagnostics.
    """

    api = get_api_client()
    try:
        payload = api.server_time()
    except Exception as exc:  # pragma: no cover - network/runtime errors
        log("time.drift.error", err=str(exc))
        return 0.0

    server_epoch = extract_server_epoch(payload)
    if server_epoch is None:
        log("time.drift.error", err="invalid payload", payload=payload)
        return 0.0

    local_dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
    server_dt = datetime.fromtimestamp(server_epoch, tz=timezone.utc)
    return (local_dt - server_dt).total_seconds()


class _SyncedClock:
    """Track drift-adjusted timestamps for a specific REST endpoint."""

    def __init__(self) -> None:
        self._offset_ms: float = 0.0
        self._latency_ms: float = 0.0
        self._expiry: float = 0.0
        self._lock = threading.Lock()

    def snapshot(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout: float = 5.0,
        verify: bool = True,
        force_refresh: bool = False,
    ) -> SyncedTimestamp:
        now = time.time()
        if force_refresh or now >= self._expiry:
            self._refresh(
                base_url,
                session=session,
                timeout=timeout,
                verify=verify,
            )
        value = int(time.time() * 1000 + self._offset_ms)
        return SyncedTimestamp(value_ms=value, offset_ms=self._offset_ms, latency_ms=self._latency_ms)

    def _refresh(
        self,
        base_url: str,
        *,
        session: requests.Session | None,
        timeout: float,
        verify: bool,
    ) -> None:
        with self._lock:
            now = time.time()
            if now < self._expiry:
                return

            url = base_url.rstrip("/") + "/v5/market/time"
            close_http = False
            if session is None:
                http = requests.Session()
                configure_http_session(http)
                close_http = True
            else:
                http = session
            start = time.time()
            try:
                response = http.get(url, timeout=timeout, verify=verify)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - defensive guard
                log("time.sync.error", err=str(exc), base=url)
                self._expiry = time.time() + 5.0
                return
            finally:
                if close_http:
                    http.close()

            server_epoch = extract_server_epoch(payload)
            if server_epoch is None:
                log(
                    "time.sync.error",
                    err="invalid payload",
                    base=url,
                    payload=payload,
                )
                self._expiry = time.time() + 5.0
                return

            end = time.time()
            # Estimate mid-point between request/response to compensate latency.
            latency = max((end - start) / 2.0, 0.0)
            local_epoch = end - latency
            self._offset_ms = server_epoch * 1000.0 - local_epoch * 1000.0
            self._latency_ms = latency * 1000.0
            # Refresh at most once per minute unless explicitly forced.
            self._expiry = time.time() + 60.0


def _clock_key(base_url: str, verify: bool) -> tuple[str, bool]:
    return (base_url.rstrip("/"), bool(verify))


def synced_timestamp(
    base_url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 5.0,
    verify: bool = True,
    force_refresh: bool = False,
) -> SyncedTimestamp:
    """Return a snapshot of the exchange-synchronised clock."""

    key = _clock_key(base_url, verify)
    with _CLOCK_LOCK:
        clock = _CLOCKS.setdefault(key, _SyncedClock())
    return clock.snapshot(
        base_url,
        session=session,
        timeout=timeout,
        verify=verify,
        force_refresh=force_refresh,
    )


def synced_timestamp_ms(
    base_url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 5.0,
    verify: bool = True,
    force_refresh: bool = False,
) -> int:
    """Return a drift-adjusted unix timestamp in milliseconds."""

    snapshot = synced_timestamp(
        base_url,
        session=session,
        timeout=timeout,
        verify=verify,
        force_refresh=force_refresh,
    )
    return snapshot.as_int()


def invalidate_synced_clock(base_url: str | None = None, *, verify: bool | None = None) -> None:
    """Drop cached offsets for the given base URL (or all if omitted)."""

    with _CLOCK_LOCK:
        if base_url is None:
            _CLOCKS.clear()
            return
        key = _clock_key(base_url, verify if verify is not None else True)
        _CLOCKS.pop(key, None)
        if verify is None:
            # Remove both verify=True/False variants when verify is unspecified.
            _CLOCKS.pop(_clock_key(base_url, True), None)
            _CLOCKS.pop(_clock_key(base_url, False), None)


# Backwards compatibility for modules importing the private helper.
_extract_server_epoch = extract_server_epoch

__all__ = [
    "SyncedTimestamp",
    "extract_server_epoch",
    "extract_server_datetime",
    "check_time_drift_seconds",
    "synced_timestamp",
    "synced_timestamp_ms",
    "invalidate_synced_clock",
]
