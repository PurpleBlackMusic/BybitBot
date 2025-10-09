"""Helpers for discovering available trading instruments."""

from __future__ import annotations

import threading
from typing import Iterable, List, Set

import requests

from .log import log

_TESTNET_URL = "https://api-testnet.bybit.com/v5/market/instruments-info"
_MAINNET_URL = "https://api.bybit.com/v5/market/instruments-info"

_CACHE: dict[str, Set[str]] = {}
_LOCK = threading.Lock()


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
        log("instruments.fetch.testnet_failed", scope="spot", err=str(exc))
        return _fetch(_MAINNET_URL)


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
    with _LOCK:
        if not force_refresh and cache_key in _CACHE:
            return set(_CACHE[cache_key])

    try:
        symbols = _fetch_spot_symbols(testnet=testnet, timeout=timeout)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        log("instruments.fetch.error", scope="spot", testnet=testnet, err=str(exc))
        with _LOCK:
            cached = _CACHE.get(cache_key)
        return set(cached or set())

    with _LOCK:
        _CACHE[cache_key] = set(symbols)
    log(
        "instruments.fetch.success",
        scope="spot",
        testnet=testnet,
        count=len(symbols),
    )
    return set(symbols)


def filter_listed_spot_symbols(
    symbols: Iterable[str], *, testnet: bool = True, force_refresh: bool = False
) -> List[str]:
    """Filter the provided iterable to Bybit-listed spot symbols."""

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

