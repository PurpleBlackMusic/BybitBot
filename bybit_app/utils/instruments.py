"""Helpers for discovering available trading instruments."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from concurrent.futures import Future
from typing import Any, Coroutine, Iterable, List, Sequence, Set

import httpx

from .file_io import ensure_directory, tail_lines
from .log import log
from .paths import DATA_DIR

_TESTNET_URL = "https://api-testnet.bybit.com/v5/market/instruments-info"
_MAINNET_URL = "https://api.bybit.com/v5/market/instruments-info"

_CACHE: dict[str, Set[str]] = {}
_CACHE_TIMESTAMPS: dict[str, float] = {}
_BACKGROUND_REFRESHES: dict[str, Future[Set[str]]] = {}
_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}
_ASYNC_WORKER_LOCK = threading.Lock()
_ASYNC_WORKER: "_AsyncWorker | None" = None

_DEFAULT_REFRESH_INTERVAL = 300.0

_HISTORY_DIR = DATA_DIR / "cache" / "instrument_history"


class _RetryableInstrumentFetchError(RuntimeError):
    """Internal marker for retryable catalogue fetch failures."""


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
    except OSError as exc:  # pragma: no cover - defensive logging
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


def _retry_delay(attempt: int, *, base: float = 0.5, maximum: float = 5.0) -> float:
    delay = base * (2 ** max(0, attempt - 1))
    return float(min(delay, maximum))


async def _request_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    cursor: str | None,
    timeout: float,
    testnet: bool,
    max_attempts: int = 3,
) -> dict[str, object]:
    params = {"category": "spot"}
    if cursor:
        params["cursor"] = cursor

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError as exc:
            if attempt >= max_attempts:
                raise _RetryableInstrumentFetchError(str(exc)) from exc
            log(
                "instruments.fetch.retry",
                scope="spot",
                testnet=testnet,
                cursor=cursor or None,
                attempt=attempt,
                maxAttempts=max_attempts,
                err=str(exc),
            )
            await asyncio.sleep(_retry_delay(attempt))
            continue

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive guard
            if attempt >= max_attempts:
                raise _RetryableInstrumentFetchError("invalid JSON response") from exc
            log(
                "instruments.fetch.retry",
                scope="spot",
                testnet=testnet,
                cursor=cursor or None,
                attempt=attempt,
                maxAttempts=max_attempts,
                err="invalid JSON response",
            )
            await asyncio.sleep(_retry_delay(attempt))
            continue

        if not isinstance(payload, dict):
            if attempt >= max_attempts:
                raise _RetryableInstrumentFetchError("unexpected payload shape")
            log(
                "instruments.fetch.retry",
                scope="spot",
                testnet=testnet,
                cursor=cursor or None,
                attempt=attempt,
                maxAttempts=max_attempts,
                err="unexpected payload shape",
            )
            await asyncio.sleep(_retry_delay(attempt))
            continue

        ret_code = payload.get("retCode")
        if ret_code not in (None, 0, "0"):
            ret_msg_raw = (
                payload.get("retMsg")
                or payload.get("ret_message")
                or payload.get("message")
            )
            ret_msg = str(ret_msg_raw).strip() if ret_msg_raw else None
            log(
                "instruments.fetch.retcode_error",
                scope="spot",
                testnet=testnet,
                cursor=cursor or None,
                retCode=ret_code,
                retMsg=ret_msg,
            )
            if attempt >= max_attempts:
                raise _RetryableInstrumentFetchError(
                    f"retCode {ret_code}: {ret_msg or 'unknown error'}"
                )
            await asyncio.sleep(_retry_delay(attempt))
            continue

        return payload

    raise _RetryableInstrumentFetchError("maximum retry attempts exhausted")


async def _fetch_catalogue(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float,
    testnet: bool,
) -> Set[str]:
    cursor: str | None = None
    seen_cursors: Set[str] = set()
    symbols: Set[str] = set()

    while True:
        payload = await _request_page(
            client,
            url,
            cursor=cursor,
            timeout=timeout,
            testnet=testnet,
        )
        result = payload.get("result") or {}
        rows: Sequence[object] = result.get("list") or []
        for item in rows:
            if not isinstance(item, dict):
                continue
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
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return symbols


def _create_async_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout))


def _get_async_worker() -> "_AsyncWorker":
    global _ASYNC_WORKER
    with _ASYNC_WORKER_LOCK:
        worker = _ASYNC_WORKER
        if worker is None or not worker.is_active:
            worker = _AsyncWorker()
            _ASYNC_WORKER = worker
    return worker


def _run_async(coro: Coroutine[Any, Any, Set[str]]) -> Set[str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    worker = _get_async_worker()
    future = worker.submit(coro)
    try:
        return future.result()
    except Exception:
        future.cancel()
        raise


async def _fetch_spot_symbols_async(
    *, testnet: bool = True, timeout: float = 5.0
) -> Set[str]:
    url = _TESTNET_URL if testnet else _MAINNET_URL

    async with _create_async_client(timeout) as client:
        try:
            return await _fetch_catalogue(client, url, timeout=timeout, testnet=testnet)
        except httpx.HTTPStatusError as http_exc:
            if not testnet:
                raise
            log(
                "instruments.fetch.testnet_http_error",
                scope="spot",
                err=str(http_exc),
            )
            try:
                fallback = await _fetch_catalogue(
                    client,
                    _MAINNET_URL,
                    timeout=timeout,
                    testnet=False,
                )
            except (httpx.HTTPStatusError, _RetryableInstrumentFetchError) as fallback_exc:
                _log_catalogue_unavailable(fallback_exc)
                return _load_cached_snapshot()

            log(
                "instruments.fetch.testnet_mainnet_fallback",
                scope="spot",
                count=len(fallback),
                err=str(http_exc),
            )
            return fallback
        except _RetryableInstrumentFetchError as exc:
            if not testnet:
                raise
            log("instruments.fetch.testnet_failed", scope="spot", err=str(exc))
            cached = _load_cached_snapshot()
            if cached:
                log(
                    "instruments.fetch.testnet_cache_fallback",
                    scope="spot",
                    count=len(cached),
                )
                return cached
            _log_catalogue_unavailable(exc)
            return set()


def _load_cached_snapshot() -> Set[str]:
    with _LOCK:
        cached = set(_CACHE.get("spot_testnet") or set())
    return cached


def _log_catalogue_unavailable(exc: BaseException) -> None:
    log(
        "instruments.fetch.catalogue_unavailable",
        scope="spot",
        testnet=True,
        err=str(exc),
    )


def _normalise_refresh_interval(value: float | int | None) -> float | None:
    if value is None:
        return _DEFAULT_REFRESH_INTERVAL
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_REFRESH_INTERVAL
    if numeric <= 0:
        return None
    return float(numeric)


def _finalise_fetch_success(
    cache_key: str,
    snapshot: Set[str],
    *,
    testnet: bool,
    event: threading.Event | None,
    background: bool,
) -> None:
    payload = {
        "scope": "spot",
        "testnet": testnet,
        "count": len(snapshot),
    }
    if background:
        payload["background"] = True

    with _LOCK:
        _CACHE[cache_key] = set(snapshot)
        _CACHE_TIMESTAMPS[cache_key] = time.monotonic()
        inflight = _IN_FLIGHT.pop(cache_key, None)
    signal = event or inflight
    if signal is not None:
        signal.set()

    _persist_history(snapshot, testnet=testnet)
    log("instruments.fetch.success", **payload)


def _finalise_fetch_failure(
    cache_key: str,
    *,
    testnet: bool,
    event: threading.Event | None,
    exc: BaseException,
    background: bool,
) -> Set[str]:
    payload = {
        "scope": "spot",
        "testnet": testnet,
        "err": str(exc),
    }
    if background:
        payload["background"] = True

    log("instruments.fetch.error", **payload)

    with _LOCK:
        cached = set(_CACHE.get(cache_key) or set())
        inflight = _IN_FLIGHT.pop(cache_key, None)

    signal = event or inflight
    if signal is not None:
        signal.set()

    return cached


def _schedule_background_refresh(
    cache_key: str,
    *,
    testnet: bool,
    timeout: float,
    event: threading.Event,
) -> None:
    def _on_complete(fut: Future[Set[str]]) -> None:
        try:
            symbols = fut.result()
        except Exception as exc:  # pragma: no cover - defensive join
            _finalise_fetch_failure(
                cache_key,
                testnet=testnet,
                event=event,
                exc=exc,
                background=True,
            )
        else:
            snapshot = set(symbols)
            _finalise_fetch_success(
                cache_key,
                snapshot,
                testnet=testnet,
                event=event,
                background=True,
            )
        finally:
            with _LOCK:
                _BACKGROUND_REFRESHES.pop(cache_key, None)

    worker = _get_async_worker()
    future = worker.submit(
        _fetch_spot_symbols_async(testnet=testnet, timeout=timeout)
    )
    future.add_done_callback(_on_complete)
    with _LOCK:
        _BACKGROUND_REFRESHES[cache_key] = future


def _perform_blocking_fetch(
    cache_key: str,
    *,
    testnet: bool,
    timeout: float,
    event: threading.Event,
) -> Set[str]:
    try:
        symbols = _fetch_spot_symbols(testnet=testnet, timeout=timeout)
    except (
        httpx.HTTPError,
        _RetryableInstrumentFetchError,
        OSError,
        asyncio.TimeoutError,
    ) as exc:  # pragma: no cover - network/runtime guard
        return _finalise_fetch_failure(
            cache_key,
            testnet=testnet,
            event=event,
            exc=exc,
            background=False,
        )

    snapshot = set(symbols)
    _finalise_fetch_success(
        cache_key,
        snapshot,
        testnet=testnet,
        event=event,
        background=False,
    )
    return snapshot


def _fetch_spot_symbols(*, testnet: bool = True, timeout: float = 5.0) -> Set[str]:
    return _run_async(_fetch_spot_symbols_async(testnet=testnet, timeout=timeout))


def get_listed_spot_symbols(
    *,
    testnet: bool = True,
    force_refresh: bool = False,
    timeout: float = 5.0,
    refresh_interval: float | int | None = None,
) -> Set[str]:
    """Return cached spot symbols listed on Bybit.

    The helper fetches the instrument catalogue once per process and keeps the
    result in memory so repeated lookups stay fast. When the HTTP request fails
    the function falls back to the last successful snapshot (if any) and
    returns an empty set otherwise. Pass ``refresh_interval`` to control when a
    cached snapshot should be refreshed in the background. Provide a non-positive
    interval to disable automatic refreshes.
    """

    cache_key = "spot_testnet" if testnet else "spot_mainnet"
    refresh_seconds = _normalise_refresh_interval(refresh_interval)
    now = time.monotonic()
    fetch_required = False
    wait_event: threading.Event | None = None
    background_event: threading.Event | None = None
    schedule_background = False
    cached_return: Set[str] | None = None

    with _LOCK:
        cached = _CACHE.get(cache_key)
        last_update = _CACHE_TIMESTAMPS.get(cache_key)
        inflight = _IN_FLIGHT.get(cache_key)

        if cached and not force_refresh:
            stale = (
                refresh_seconds is not None
                and last_update is not None
                and now - last_update >= refresh_seconds
            )
            if not stale:
                return set(cached)

            cached_return = set(cached)
            if inflight is None:
                background_event = threading.Event()
                _IN_FLIGHT[cache_key] = background_event
                schedule_background = True
        else:
            if inflight is None:
                wait_event = threading.Event()
                _IN_FLIGHT[cache_key] = wait_event
                fetch_required = True
            else:
                wait_event = inflight

    if cached_return is not None:
        if schedule_background and background_event is not None:
            _schedule_background_refresh(
                cache_key,
                testnet=testnet,
                timeout=timeout,
                event=background_event,
            )
        return cached_return

    if wait_event is None:
        # No cached value and no event means the schedule above created an
        # event solely for background refresh; return the latest cache copy.
        with _LOCK:
            cached_snapshot = _CACHE.get(cache_key)
        return set(cached_snapshot or set())

    if fetch_required:
        return _perform_blocking_fetch(
            cache_key,
            testnet=testnet,
            timeout=timeout,
            event=wait_event,
        )

    wait_event.wait()
    with _LOCK:
        cached_snapshot = _CACHE.get(cache_key)
    return set(cached_snapshot or set())


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


class _AsyncWorker:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._stopped.set()

    @property
    def is_active(self) -> bool:
        return self._thread.is_alive() and not self._stopped.is_set()

    def submit(self, coro: Coroutine[Any, Any, Set[str]]) -> Future[Set[str]]:
        future: Future[Set[str]] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future

    def close(self) -> None:
        if self._stopped.is_set():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
        self._stopped.wait(timeout=1.0)


def _reset_async_worker_for_tests() -> None:
    global _ASYNC_WORKER
    with _ASYNC_WORKER_LOCK:
        worker = _ASYNC_WORKER
        _ASYNC_WORKER = None
    if worker is not None:
        worker.close()

