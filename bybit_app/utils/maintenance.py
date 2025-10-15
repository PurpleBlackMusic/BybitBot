"""Helpers for applying maintenance stop-lists to trading symbols."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

Symbol = str


def _normalise_timestamp(value: object) -> Optional[float]:
    """Convert various timestamp representations to epoch seconds."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number <= 0:
        return None

    # Values from JSON may be in milliseconds. Treat large numbers as ms.
    if number > 1e12:
        number /= 1000.0
    elif number > 1e10:
        number /= 1000.0

    return number


@dataclass(frozen=True)
class _StopEntry:
    symbols: Tuple[Symbol, ...]
    start: float
    end: Optional[float]
    reason: Optional[str]

    def matches(self, symbol: Symbol, now: float) -> bool:
        if now < self.start:
            return False
        if self.end is not None and now > self.end:
            return False
        upper = symbol.upper()
        if "*" in self.symbols:
            return True
        return upper in self.symbols


class MaintenanceStoplist:
    """Represents a collection of maintenance windows for tradable symbols."""

    def __init__(self, entries: Sequence[_StopEntry]) -> None:
        self._entries = tuple(entries)

    def is_blocked(self, symbol: Symbol, *, now: Optional[float] = None) -> bool:
        blocked, _ = self.check(symbol, now=now)
        return blocked

    def check(self, symbol: Symbol, *, now: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        current = time.time() if now is None else float(now)
        upper = symbol.upper()
        reason: Optional[str] = None
        for entry in self._entries:
            if not entry.matches(upper, current):
                continue
            reason = entry.reason
            return True, reason
        return False, reason


def _parse_symbols(value: object) -> Tuple[Symbol, ...]:
    if isinstance(value, str):
        cleaned = value.strip().upper()
        return (cleaned,) if cleaned else tuple()
    if isinstance(value, Sequence):
        result = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip().upper()
                if cleaned:
                    result.append(cleaned)
        if result:
            return tuple(dict.fromkeys(result))
    return tuple()


def _parse_stop_entry(data: Mapping[str, object]) -> Optional[_StopEntry]:
    symbols_raw = data.get("symbols")
    if symbols_raw is None and "symbol" in data:
        symbols_raw = data.get("symbol")
    symbols = _parse_symbols(symbols_raw)
    if not symbols:
        return None

    start = _normalise_timestamp(
        data.get("start")
        or data.get("from")
        or data.get("start_time")
        or data.get("startTime")
    )
    if start is None:
        start = float("-inf")

    end = _normalise_timestamp(
        data.get("end")
        or data.get("until")
        or data.get("end_time")
        or data.get("endTime")
        or data.get("expires_at")
        or data.get("expiry")
    )

    reason_value = data.get("reason")
    reason = None
    if isinstance(reason_value, str):
        cleaned = reason_value.strip()
        if cleaned:
            reason = cleaned

    return _StopEntry(symbols=symbols, start=start, end=end, reason=reason)


def _parse_stoplist_mapping(mapping: Mapping[str, object]) -> Iterable[_StopEntry]:
    for symbol, payload in mapping.items():
        symbols = _parse_symbols(symbol)
        if not symbols:
            continue
        start = float("-inf")
        end: Optional[float] = None
        reason: Optional[str] = None

        if isinstance(payload, Mapping):
            start = _normalise_timestamp(
                payload.get("start")
                or payload.get("from")
                or payload.get("start_time")
                or payload.get("startTime")
            ) or float("-inf")
            end = _normalise_timestamp(
                payload.get("until")
                or payload.get("end")
                or payload.get("expires_at")
                or payload.get("expiry")
                or payload.get("endTime")
            )
            reason_value = payload.get("reason")
            if isinstance(reason_value, str):
                cleaned = reason_value.strip()
                if cleaned:
                    reason = cleaned
        else:
            end = _normalise_timestamp(payload)

        yield _StopEntry(symbols=symbols, start=start, end=end, reason=reason)


def load_maintenance_stoplist(data_dir: Path | str) -> MaintenanceStoplist:
    """Load maintenance stop-list information from disk."""

    base = Path(data_dir)
    path = base / "config" / "maintenance.json"
    entries: list[_StopEntry] = []
    if not path.exists():
        return MaintenanceStoplist(entries)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return MaintenanceStoplist(entries)

    if isinstance(payload, Mapping):
        windows = payload.get("windows")
        if isinstance(windows, Sequence):
            for window in windows:
                if not isinstance(window, Mapping):
                    continue
                parsed = _parse_stop_entry(window)
                if parsed is not None:
                    entries.append(parsed)

        stoplist = payload.get("stoplist")
        if isinstance(stoplist, Mapping):
            entries.extend(_parse_stoplist_mapping(stoplist))

    return MaintenanceStoplist(entries)
