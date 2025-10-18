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
    end = int(time.time() * 1000)
    start = end - int(200 * 60 * 60 * 1000)

    try:
        payload = client.kline(
            category="spot",
            symbol=symbol.upper(),
            interval=60,
            limit=200,
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

    deduplicated: dict[int, dict[str, object]] = {}
    for row in rows:
        start_value = int(row["start"])
        deduplicated[start_value] = {
            "start": start_value,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "turnover": float(row.get("turnover", 0.0)),
        }

    ordered_rows = [
        deduplicated[key] for key in sorted(deduplicated.keys())
    ]
    if not ordered_rows:
        return path

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["start", "open", "high", "low", "close", "volume", "turnover"])
    for row in ordered_rows:
        writer.writerow(
            [
                row["start"],
                f"{row['open']:.10f}",
                f"{row['high']:.10f}",
                f"{row['low']:.10f}",
                f"{row['close']:.10f}",
                f"{row['volume']:.10f}",
                f"{row['turnover']:.10f}",
            ]
        )

    atomic_write_text(path, buffer.getvalue())
    return path

