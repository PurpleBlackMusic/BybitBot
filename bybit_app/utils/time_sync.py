
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import time

from .envs import get_api_client
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

    local_epoch = time.time()
    return local_epoch - server_epoch


# Backwards compatibility for modules importing the private helper.
_extract_server_epoch = extract_server_epoch

__all__ = ["extract_server_epoch", "check_time_drift_seconds"]
