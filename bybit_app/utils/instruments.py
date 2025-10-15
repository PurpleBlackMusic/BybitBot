"""Helpers for discovering available trading instruments."""

from __future__ import annotations

import threading
from typing import Iterable, List, Mapping, Set

import requests
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .log import log

_TESTNET_URL = "https://api-testnet.bybit.com/v5/market/instruments-info"
_MAINNET_URL = "https://api.bybit.com/v5/market/instruments-info"

_CACHE: dict[str, Set[str]] = {}
_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}


class _InstrumentRetryError(RuntimeError):
    """Internal marker for retryable instrument catalogue failures."""

    def __init__(self, message: str, *, meta: Mapping[str, object] | None = None):
        super().__init__(message)
        self.meta = dict(meta or {})


def _fetch_spot_symbols(*, testnet: bool = True, timeout: float = 5.0) -> Set[str]:
    url = _TESTNET_URL if testnet else _MAINNET_URL

    def _fetch(url: str) -> Set[str]:
        cursor: str | None = None
        seen_cursors: Set[str] = set()
        symbols: Set[str] = set()

        max_attempts = 3
        wait_strategy = wait_random_exponential(multiplier=0.5, max=5.0)

        def _log_retry(retry_state: RetryCallState) -> None:
            if retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                meta: dict[str, object] = {
                    "attempt": retry_state.attempt_number,
                    "maxAttempts": max_attempts,
                    "scope": "spot",
                    "url": url,
                }
                if isinstance(exc, _InstrumentRetryError):
                    meta.update(exc.meta)
                else:
                    meta["err"] = str(exc)
                log("instruments.fetch.retry", **meta)

        @retry(
            reraise=True,
            wait=wait_strategy,
            stop=stop_after_attempt(max_attempts),
            retry=retry_if_exception_type((_InstrumentRetryError, requests.exceptions.RequestException)),
            after=_log_retry,
        )
        def _request_page(query: Mapping[str, object]) -> dict[str, object]:
            try:
                response = requests.get(url, params=query, timeout=timeout)
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                log(
                    "instruments.fetch.http_error",
                    scope="spot",
                    url=url,
                    status=status_code,
                    err=str(exc),
                )
                raise
            except requests.exceptions.RequestException as exc:
                log("instruments.fetch.network_error", scope="spot", url=url, err=str(exc))
                raise

            try:
                payload = response.json()
            except ValueError as exc:
                meta = {"status": response.status_code, "err": "invalid_json"}
                log("instruments.fetch.invalid_json", scope="spot", url=url, **meta)
                raise _InstrumentRetryError("invalid JSON response", meta=meta) from exc

            ret_code_raw = payload.get("retCode")
            ret_code_text = str(ret_code_raw).strip() if ret_code_raw is not None else ""
            if ret_code_text and ret_code_text != "0":
                ret_msg = str(
                    payload.get("retMsg")
                    or payload.get("ret_message")
                    or payload.get("retMessage")
                    or payload.get("message")
                    or ""
                ).strip()
                meta = {"retCode": ret_code_text, "retMsg": ret_msg or None}
                log("instruments.fetch.retcode_error", scope="spot", url=url, **meta)
                raise _InstrumentRetryError(
                    f"retCode {ret_code_text}: {ret_msg or 'unknown error'}",
                    meta=meta,
                )

            return payload

        while True:
            params = {"category": "spot"}
            if cursor:
                params["cursor"] = cursor

            payload = _request_page(params)
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
    except (_InstrumentRetryError, requests.exceptions.RequestException) as exc:
        root_exc = exc.__cause__ if isinstance(exc, _InstrumentRetryError) and exc.__cause__ else exc
        if not testnet:
            if isinstance(exc, _InstrumentRetryError):
                if root_exc is not exc:
                    raise RuntimeError(str(exc)) from root_exc
                raise RuntimeError(str(exc))
            raise

        if isinstance(root_exc, requests.exceptions.HTTPError):
            log("instruments.fetch.testnet_http_error", scope="spot", err=str(root_exc))
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
                err=str(root_exc),
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

    with _LOCK:
        _CACHE[cache_key] = set(symbols)
        in_flight = _IN_FLIGHT.pop(cache_key, None)
    if in_flight is not None:
        in_flight.set()
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

