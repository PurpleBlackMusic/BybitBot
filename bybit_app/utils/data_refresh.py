"""Helpers for keeping cached market data fresh."""

from __future__ import annotations

import time
import csv
import io
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .envs import Settings, get_api_client, get_settings
from .file_io import atomic_write_text, ensure_directory
from .log import log
from .paths import DATA_DIR

__all__ = ["ensure_hourly_ohlcv"]


_HOURLY_MS = 60 * 60 * 1000
_MAX_CACHE_HOURS = 24 * 120  # keep roughly four months of cached candles


def _normalise_row(row: Iterable[object]) -> Optional[dict[str, object]]:
    items = list(row)
    if len(items) < 7:
        return None
    try:
        start = int(float(items[0]))
    except (TypeError, ValueError):
        return None
    return {
        "start": start,
        "open": float(items[1]),
        "high": float(items[2]),
        "low": float(items[3]),
        "close": float(items[4]),
        "volume": float(items[5]),
        "turnover": float(items[6]),
    }


def _parse_kline_payload(payload: Mapping[str, object]) -> list[dict[str, object]]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        rows = result.get("list")
    else:
        rows = payload.get("list")

    if not isinstance(rows, Iterable):
        return []

    normalised: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, Mapping):
            candidates = _normalise_row(row.values())
        else:
            candidates = _normalise_row(row)  # type: ignore[arg-type]
        if candidates is not None:
            normalised.append(candidates)
    return normalised


def _read_cached_rows(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}

    rows: dict[int, dict[str, float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                try:
                    start = int(float(raw.get("start", 0)))
                except (TypeError, ValueError):
                    continue
                if start <= 0:
                    continue
                try:
                    rows[start] = {
                        "start": float(start),
                        "open": float(raw.get("open", 0.0)),
                        "high": float(raw.get("high", 0.0)),
                        "low": float(raw.get("low", 0.0)),
                        "close": float(raw.get("close", 0.0)),
                        "volume": float(raw.get("volume", 0.0)),
                        "turnover": float(raw.get("turnover", 0.0)),
                    }
                except (TypeError, ValueError):
                    continue
    except Exception:  # pragma: no cover - defensive fallback
        return {}
    return rows


def _serialise_rows(rows: Mapping[int, Mapping[str, float]]) -> str:
    ordered_keys = sorted(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["start", "open", "high", "low", "close", "volume", "turnover"])
    for key in ordered_keys:
        row = rows[key]
        writer.writerow(
            [
                int(round(float(row.get("start", key)))),
                f"{float(row.get('open', 0.0)):.10f}",
                f"{float(row.get('high', 0.0)):.10f}",
                f"{float(row.get('low', 0.0)):.10f}",
                f"{float(row.get('close', 0.0)):.10f}",
                f"{float(row.get('volume', 0.0)):.10f}",
                f"{float(row.get('turnover', 0.0)):.10f}",
            ]
        )
    return buffer.getvalue()


def ensure_hourly_ohlcv(
    symbol: str,
    *,
    data_dir: Path = DATA_DIR,
    max_age_minutes: float = 120.0,
    settings: Optional[Settings] = None,
    api=None,
) -> Path:
    """Make sure the cached hourly OHLCV file for ``symbol`` is reasonably fresh."""

    target_dir = Path(data_dir) / "ohlcv" / "spot" / symbol.upper()
    ensure_directory(target_dir)
    path = target_dir / f"{symbol.upper()}_1h.csv"

    existing_rows = _read_cached_rows(path)
    last_cached_start = max(existing_rows) if existing_rows else None

    now_ms = int(time.time() * 1000)
    max_age_seconds = max(float(max_age_minutes), 1.0) * 60.0
    needs_refresh = True
    if path.exists():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        age = time.time() - mtime
        if age <= max_age_seconds:
            needs_refresh = False

    if not needs_refresh:
        if existing_rows and _MAX_CACHE_HOURS:
            prune_before = now_ms - int(_MAX_CACHE_HOURS * _HOURLY_MS)
            if prune_before > 0:
                filtered = {k: v for k, v in existing_rows.items() if k >= prune_before}
            else:
                filtered = dict(existing_rows)
            if filtered and len(filtered) != len(existing_rows):
                atomic_write_text(path, _serialise_rows(filtered))
        return path

    if api is None:
        runtime_settings = settings
        if runtime_settings is None:
            try:
                runtime_settings = get_settings()
            except Exception:
                runtime_settings = None
        if runtime_settings is None:
            return path
        try:
            client = get_api_client(settings=runtime_settings)
        except Exception:
            return path
    else:
        client = api

    end = now_ms
    default_window = 200 * _HOURLY_MS
    start = max(end - default_window, 0)
    if last_cached_start is not None:
        candidate = last_cached_start + _HOURLY_MS
        if candidate < end:
            start = max(candidate, end - default_window)
        else:
            start = candidate

    try:
        bars = int((end - start) / _HOURLY_MS) + 2
        payload = client.kline(
            category="spot",
            symbol=symbol.upper(),
            interval=60,
            limit=min(200, max(bars, 1)),
            start=start,
            end=end,
        )
    except Exception as exc:  # pragma: no cover - API failure fallback
        log("data_refresh.ohlcv.fetch_error", symbol=symbol, err=str(exc))
        return path

    rows = _parse_kline_payload(payload if isinstance(payload, Mapping) else {})
    if not rows:
        log("data_refresh.ohlcv.empty", symbol=symbol)
        return path

    merged: dict[int, dict[str, float]] = dict(existing_rows)
    newly_added = False
    last_seen = last_cached_start
    for row in sorted(rows, key=lambda item: int(item["start"])):
        start_value = int(row["start"])
        if last_seen is not None and start_value <= last_seen:
            continue
        merged[start_value] = {
            "start": float(start_value),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "turnover": float(row.get("turnover", 0.0)),
        }
        newly_added = True
        last_seen = start_value if last_seen is None else max(last_seen, start_value)

    if not newly_added and not merged:
        return path

    filtered_rows = merged
    if _MAX_CACHE_HOURS and merged:
        prune_before = end - int(_MAX_CACHE_HOURS * _HOURLY_MS)
        if prune_before > 0:
            filtered_rows = {k: v for k, v in merged.items() if k >= prune_before}

    if newly_added or len(filtered_rows) != len(existing_rows):
        payload_text = _serialise_rows(filtered_rows)
        atomic_write_text(path, payload_text)

    return path

