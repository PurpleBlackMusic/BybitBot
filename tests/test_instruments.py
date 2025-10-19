import json
import threading
from pathlib import Path

import httpx
import pytest

import bybit_app.utils.instruments as instruments


class _FakeAsyncResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - behaviourless stub
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses, *, before_request=None):
        self._responses = list(responses)
        self._before_request = before_request
        self.calls: list[dict[str, object]] = []
        self._index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - behaviourless stub
        return False

    async def get(self, url, params=None, timeout=None):
        if self._before_request is not None:
            self._before_request(url, params, timeout)
        if self._index >= len(self._responses):  # pragma: no cover - defensive guard
            raise AssertionError("unexpected request")
        entry = self._responses[self._index]
        self._index += 1
        call = {"url": url, "params": dict(params or {}), "timeout": timeout}
        self.calls.append(call)
        if isinstance(entry, Exception):
            raise entry
        status = int(entry.get("status", 200))
        payload = entry.get("payload")
        if status >= 400:
            request = httpx.Request("GET", url, params=params)
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)
        return _FakeAsyncResponse(payload)


@pytest.fixture(autouse=True)
def _reset_instrument_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instruments._CACHE.clear()
    instruments._CACHE_TIMESTAMPS.clear()
    instruments._BACKGROUND_REFRESHES.clear()
    instruments._IN_FLIGHT.clear()
    monkeypatch.setattr(instruments, "_HISTORY_DIR", tmp_path / "history")
    instruments._reset_async_worker_for_tests()
    yield
    instruments._CACHE.clear()
    instruments._CACHE_TIMESTAMPS.clear()
    instruments._BACKGROUND_REFRESHES.clear()
    instruments._IN_FLIGHT.clear()
    instruments._reset_async_worker_for_tests()


def test_get_listed_spot_symbols_fetches_all_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {
            "result": {
                "list": [
                    {"symbol": "ETHUSDT"},
                ],
                "nextPageCursor": "cursor-1",
            }
        },
        {
            "result": {
                "list": [
                    {"symbol": "ADAUSDT"},
                    {"symbol": "XRPUSDT"},
                ],
                "nextPageCursor": "",
            }
        },
    ]

    client = _FakeAsyncClient(
        [{"payload": payload} for payload in payloads],
    )
    monkeypatch.setattr(instruments, "_create_async_client", lambda timeout: client)

    result = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)

    assert result == {"ETHUSDT", "ADAUSDT", "XRPUSDT"}
    assert len(client.calls) == 2
    assert client.calls[0]["params"] == {"category": "spot"}
    assert client.calls[1]["params"] == {"category": "spot", "cursor": "cursor-1"}


def test_get_listed_spot_symbols_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "BTCUSDT"},
            ],
        }
    }

    client = _FakeAsyncClient([{"payload": payload}])
    monkeypatch.setattr(instruments, "_create_async_client", lambda timeout: client)

    first = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)
    second = instruments.get_listed_spot_symbols(testnet=True)

    assert first == {"BTCUSDT"}
    assert second == {"BTCUSDT"}
    assert len(client.calls) == 1


def test_get_listed_spot_symbols_testnet_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "MAINUSDT"},
            ]
        }
    }

    events: list[tuple[str, dict]] = []

    def fake_log(event: str, **payload):
        events.append((event, dict(payload)))

    client = _FakeAsyncClient(
        [
            {"status": 403, "payload": {}},
            {"payload": payload},
        ]
    )
    monkeypatch.setattr(instruments, "_create_async_client", lambda timeout: client)
    monkeypatch.setattr(instruments, "log", fake_log)

    result = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)

    assert result == {"MAINUSDT"}
    assert [call["url"] for call in client.calls] == [
        instruments._TESTNET_URL,
        instruments._MAINNET_URL,
    ]

    fallback_events = [payload for event, payload in events if event == "instruments.fetch.testnet_mainnet_fallback"]
    assert fallback_events and fallback_events[0]["count"] == 1

    second = instruments.get_listed_spot_symbols(testnet=True)
    assert second == {"MAINUSDT"}
    assert [call["url"] for call in client.calls] == [
        instruments._TESTNET_URL,
        instruments._MAINNET_URL,
    ]


def test_get_listed_spot_symbols_concurrent_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "SOLUSDT"},
            ]
        }
    }

    first_request_started = threading.Event()
    allow_response = threading.Event()

    def before_request(url, params, timeout):
        if not first_request_started.is_set():
            first_request_started.set()
            allow_response.wait(timeout=5.0)

    client = _FakeAsyncClient([{"payload": payload}], before_request=before_request)
    monkeypatch.setattr(instruments, "_create_async_client", lambda timeout: client)

    results: list[set[str] | None] = [None, None]
    errors: list[Exception] = []

    def _call(index: int) -> None:
        try:
            results[index] = instruments.get_listed_spot_symbols(testnet=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            errors.append(exc)

    threads = [threading.Thread(target=_call, args=(idx,)) for idx in range(2)]
    for thread in threads:
        thread.start()

    first_request_started.wait(timeout=5.0)
    allow_response.set()

    for thread in threads:
        thread.join()

    assert not errors
    assert results == [{"SOLUSDT"}, {"SOLUSDT"}]
    assert len(client.calls) == 1


def test_get_listed_spot_symbols_triggers_background_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_key = "spot_testnet"
    instruments._CACHE[cache_key] = {"STALE"}
    instruments._CACHE_TIMESTAMPS[cache_key] = instruments.time.monotonic() - 1_000.0

    triggered: dict[str, object] = {}

    def fake_schedule(
        cache_key_arg: str,
        *,
        testnet: bool,
        timeout: float,
        event: threading.Event,
    ) -> None:
        triggered["cache_key"] = cache_key_arg
        triggered["testnet"] = testnet
        triggered["timeout"] = timeout
        instruments._finalise_fetch_success(  # type: ignore[attr-defined]
            cache_key_arg,
            {"FRESH"},
            testnet=testnet,
            event=event,
            background=True,
        )

    monkeypatch.setattr(instruments, "_schedule_background_refresh", fake_schedule)

    result = instruments.get_listed_spot_symbols(
        testnet=True, refresh_interval=1.0
    )

    assert result == {"STALE"}
    assert triggered["cache_key"] == cache_key
    assert triggered["testnet"] is True
    assert instruments._CACHE[cache_key] == {"FRESH"}
    assert cache_key not in instruments._IN_FLIGHT


def test_run_async_uses_background_worker_when_loop_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):  # pragma: no cover - signature parity
            return self._value

        def cancel(self) -> bool:  # pragma: no cover - behaviourless stub
            return False

    class _DummyWorker:
        def __init__(self):
            self.submitted: list[object] = []

        @property
        def is_active(self) -> bool:
            return True

        def submit(self, coro):
            self.submitted.append(coro)
            coro.close()
            return _DummyFuture({"OK"})

    worker = _DummyWorker()
    monkeypatch.setattr(instruments, "_get_async_worker", lambda: worker)
    monkeypatch.setattr(instruments.asyncio, "get_running_loop", lambda: object())

    def _fail_run(_):  # pragma: no cover - safety net
        raise AssertionError("asyncio.run should not be called when loop is running")

    monkeypatch.setattr(instruments.asyncio, "run", _fail_run)

    async def _coro():  # pragma: no cover - dummy coroutine
        return {"unexpected"}

    result = instruments._run_async(_coro())

    assert result == {"OK"}
    assert worker.submitted, "background worker was not used"


def test_get_listed_spot_symbols_persists_history(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {"result": {"list": [{"symbol": "AAAUSDT"}]}},
        {"result": {"list": [{"symbol": "AAAUSDT"}, {"symbol": "BBBUSDT"}]}}
    ]

    client = _FakeAsyncClient([{"payload": payload} for payload in payloads])
    time_values = iter([1_000_000.0, 1_000_100.0, 1_000_100.0])

    def fake_time() -> float:
        try:
            return next(time_values)
        except StopIteration:  # pragma: no cover - defensive guard
            return 1_000_100.0

    monkeypatch.setattr(instruments, "_create_async_client", lambda timeout: client)
    monkeypatch.setattr(instruments.time, "time", fake_time)

    first = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)
    second = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)

    assert first == {"AAAUSDT"}
    assert second == {"AAAUSDT", "BBBUSDT"}

    history_path = instruments._history_path(True)
    assert history_path.exists()
    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    entries = [json.loads(line) for line in lines]
    assert entries[0]["symbols"] == ["AAAUSDT"]
    assert entries[1]["symbols"] == ["AAAUSDT", "BBBUSDT"]


def test_get_listed_spot_symbols_at_returns_snapshot(tmp_path: Path) -> None:
    history_path = instruments._history_path(True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": 1_000_000.0, "symbols": ["AAAUSDT"]},
        {"ts": 1_000_200.0, "symbols": ["AAAUSDT", "BBBUSDT"]},
    ]
    history_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    before_listing = instruments.get_listed_spot_symbols_at(500_000.0, testnet=True)
    assert before_listing == set()

    mid_snapshot = instruments.get_listed_spot_symbols_at(1_000_100.0, testnet=True)
    assert mid_snapshot == {"AAAUSDT"}

    ms_query = instruments.get_listed_spot_symbols_at(1_000_250_000.0, testnet=True)
    assert ms_query == {"AAAUSDT", "BBBUSDT"}
