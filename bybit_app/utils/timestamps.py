"""Helpers for working with execution timestamps across heterogeneous payloads."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

__all__ = ["extract_exec_timestamp", "normalise_exec_time"]


def _coerce_epoch_to_ms(value: float) -> int | None:
    """Return the most plausible millisecond epoch for a raw numeric value."""

    if not math.isfinite(value):
        return None

    magnitude = abs(value)
    if magnitude >= 1e18:  # nanoseconds
        seconds = value / 1e9
    elif magnitude >= 1e15:  # microseconds
        seconds = value / 1e6
    elif magnitude >= 1e12:  # milliseconds
        seconds = value / 1e3
    elif magnitude >= 1e9:  # seconds
        seconds = value
    else:
        return None

    return int(seconds * 1000)


def normalise_exec_time(raw: object) -> int | None:
    """Return unix timestamp in milliseconds for heterogeneous ``execTime`` values."""

    if raw is None:
        return None

    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            numeric_value = float(text)
        except (TypeError, ValueError):
            numeric_value = None
        else:
            coerced = _coerce_epoch_to_ms(numeric_value)
            if coerced is not None:
                return coerced
        try:
            parsed = pd.to_datetime(text, utc=True, errors="coerce")
        except (ValueError, TypeError):
            parsed = pd.NaT
        if pd.notna(parsed):
            return int(parsed.to_pydatetime().timestamp() * 1000)
        return None

    if isinstance(raw, (int, float)):
        coerced = _coerce_epoch_to_ms(float(raw))
        if coerced is not None:
            return coerced
        return None

    try:
        numeric_value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    return _coerce_epoch_to_ms(numeric_value)


def extract_exec_timestamp(
    payload: Mapping[str, object],
    *,
    candidates: Iterable[str]
    | None = (
        "execTime",
        "execTimeNs",
        "ts",
        "timestamp",
        "transactTime",
        "transactionTime",
        "createdTime",
        "updatedTime",
    ),
) -> int | None:
    """Return the first normalised execution timestamp from *payload*.

    The helper iterates over the provided *candidates* (by default the most
    common Bybit timestamp fields) and returns the first value that can be
    normalised to a millisecond Unix epoch. ``None`` is returned when no
    candidate contains a usable timestamp.
    """

    if candidates is None:
        return None

    for key in candidates:
        value = payload.get(key)
        if value is None:
            continue
        normalised = normalise_exec_time(value)
        if normalised is not None:
            return normalised

    return None
