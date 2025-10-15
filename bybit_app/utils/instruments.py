"""Helpers for discovering available trading instruments."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterable, List, Set

import requests

from .file_io import ensure_directory, tail_lines
from .log import log
from .paths import DATA_DIR

_TESTNET_URL = "https://api-testnet.bybit.com/v5/market/instruments-info"
_MAINNET_URL = "https://api.bybit.com/v5/market/instruments-info"

_CACHE: dict[str, Set[str]] = {}
_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}

_HISTORY_DIR = DATA_DIR / "cache" / "instrument_history"


def _history_path(testnet: bool) -> Path:
    suffix = "testnet" if testnet else "mainnet"
    return _HISTORY_DIR / f"spot_symbols_{suffix}.jsonl"


def _normalise_symbol_set(symbols: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        text = str(raw or "").strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    cleaned.sort()
    return cleaned


def _normalise_timestamp(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    # Handle milliseconds/nanoseconds that occasionally appear in payloads.
    if ts > 1e15:
        ts /= 1e9
    elif ts > 1e12:
        ts /= 1e3
    return ts


def _read_history_snapshot(path: Path, *, as_of: float | None = None) -> Set[str]:
    if not path.exists():
        return set()

    if as_of is None:
        lines = tail_lines(path, 1, drop_blank=True)
    else:
        limit = 64
        target = _normalise_timestamp(as_of)
        while True:
            lines = tail_lines(path, limit, drop_blank=True)
            for raw in reversed(lines):
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = _normalise_timestamp(payload.get("ts"))
                if ts is None or target is None or ts <= target:
                    symbols = payload.get("symbols")
                    if isinstance(symbols, list):
                        return set(_normalise_symbol_set(symbols))
            if not lines or len(lines) < limit:
                break
            if limit >= 4096:
                break
            limit *= 2
        return set()

    if not lines:
        return set()

    for raw in reversed(lines):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        symbols = payload.get("symbols")
        if isinstance(symbols, list):
            return set(_normalise_symbol_set(symbols))
    return set()


def _persist_history(symbols: Set[str], *, testnet: bool) -> None:
    path = _history_path(testnet)
    try:
        current_snapshot = set(_normalise_symbol_set(symbols))
        previous = _read_history_snapshot(path)
        if previous == current_snapshot:
            return
        ensure_directory(path.parent)
        payload = {"ts": time.time(), "symbols": sorted(current_snapshot)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        log(
            "instruments.history.persist_failed",
            scope="spot",
            testnet=testnet,
            err=str(exc),
        )


def get_listed_spot_symbols_at(
    as_of: float | int,
    *,
    testnet: bool = True,
) -> Set[str]:
    """Return the Bybit spot listings snapshot closest to ``as_of``."""

    normalised = _normalise_timestamp(as_of)
    if normalised is None:
        return set()

    snapshot = _read_history_snapshot(_history_path(testnet), as_of=normalised)
    return snapshot


def _fetch_spot_symbols(*, testnet: bool = True, timeout: float = 5.0) -> Set[str]:
    url = _TESTNET_URL if testnet else _MAINNET_URL

    def _fetch(url: str) -> Set[str]:
        cursor: str | None = None
        seen_cursors: Set[str] = set()
        symbols: Set[str] = set()

        while True:
            params = {"category": "spot"}
            if cursor:
                params["cursor"] = cursor

            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result") or {}
            rows = result.get("list") or []
            for item in rows:
                symbol = str(item.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                symbols.add(symbol)

            next_cursor_raw = (
                result.get("nextPageCursor")
                or result.get("nextPageToken")
                or payload.get("nextPageCursor")
                or payload.get("nextPageToken")
            )
            next_cursor = str(next_cursor_raw).strip() if next_cursor_raw else ""
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return symbols

    try:
        return _fetch(url)
    except requests.exceptions.RequestException as exc:
        if not testnet:
            raise

        if isinstance(exc, requests.exceptions.HTTPError):
            log("instruments.fetch.testnet_http_error", scope="spot", err=str(exc))
            try:
                fallback_symbols = _fetch(_MAINNET_URL)
            except requests.exceptions.RequestException as fallback_exc:
                log(
                    "instruments.fetch.catalogue_unavailable",
                    scope="spot",
                    testnet=True,
                    err=str(fallback_exc),
                )
                with _LOCK:
                    cached = set(_CACHE.get("spot_testnet") or set())
                if cached:
                    log(
                        "instruments.fetch.testnet_cache_fallback",
                        scope="spot",
                        count=len(cached),
                    )
                    return cached
                return set()

            log(
                "instruments.fetch.testnet_mainnet_fallback",
                scope="spot",
                count=len(fallback_symbols),
                err=str(exc),
            )
            return fallback_symbols

        log("instruments.fetch.testnet_failed", scope="spot", err=str(exc))
        with _LOCK:
            cached = set(_CACHE.get("spot_testnet") or set())
        if cached:
            log(
                "instruments.fetch.testnet_cache_fallback",
                scope="spot",
                count=len(cached),
            )
            return cached
        log(
            "instruments.fetch.catalogue_unavailable",
            scope="spot",
            testnet=True,
            err=str(exc),
        )
        return set()


def get_listed_spot_symbols(
    *, testnet: bool = True, force_refresh: bool = False, timeout: float = 5.0
) -> Set[str]:
    """Return cached spot symbols listed on Bybit.

    The helper fetches the instrument catalogue once per process and keeps the
    result in memory so repeated lookups stay fast. When the HTTP request fails
    the function falls back to the last successful snapshot (if any) and
    returns an empty set otherwise.
    """

    cache_key = "spot_testnet" if testnet else "spot_mainnet"
    fetch_required = False
    event: threading.Event | None = None

    with _LOCK:
        if not force_refresh and cache_key in _CACHE:
            return set(_CACHE[cache_key])

        event = _IN_FLIGHT.get(cache_key)
        if event is None:
            event = threading.Event()
            _IN_FLIGHT[cache_key] = event
            fetch_required = True

    if not fetch_required:
        assert event is not None  # for type-checkers
        event.wait()
        with _LOCK:
            cached = _CACHE.get(cache_key)
        return set(cached or set())

    try:
        symbols = _fetch_spot_symbols(testnet=testnet, timeout=timeout)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        log("instruments.fetch.error", scope="spot", testnet=testnet, err=str(exc))
        with _LOCK:
            cached = _CACHE.get(cache_key)
            in_flight = _IN_FLIGHT.pop(cache_key, None)
        if in_flight is not None:
            in_flight.set()
        return set(cached or set())

    snapshot = set(symbols)
    with _LOCK:
        _CACHE[cache_key] = set(snapshot)
        in_flight = _IN_FLIGHT.pop(cache_key, None)
    if in_flight is not None:
        in_flight.set()
    _persist_history(snapshot, testnet=testnet)
    log(
        "instruments.fetch.success",
        scope="spot",
        testnet=testnet,
        count=len(snapshot),
    )
    return set(snapshot)


def filter_listed_spot_symbols(
    symbols: Iterable[str],
    *,
    testnet: bool = True,
    force_refresh: bool = False,
    as_of: float | int | None = None,
) -> List[str]:
    """Filter the provided iterable to Bybit-listed spot symbols."""

    if as_of is not None:
        listed = get_listed_spot_symbols_at(as_of, testnet=testnet)
    else:
        listed = get_listed_spot_symbols(
            testnet=testnet, force_refresh=force_refresh
        )
    if not listed:
        return [
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        ]

    filtered: List[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if symbol in listed and symbol not in filtered:
            filtered.append(symbol)
    return filtered

