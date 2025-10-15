import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import bybit_app.utils.instruments as instruments


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - behaviourless stub
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_instrument_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instruments._CACHE.clear()
    instruments._IN_FLIGHT.clear()
    monkeypatch.setattr(instruments, "_HISTORY_DIR", tmp_path / "history")
    yield
    instruments._CACHE.clear()
    instruments._IN_FLIGHT.clear()


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

    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        index = len(calls)
        calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        return _FakeResponse(payloads[index])

    monkeypatch.setattr(instruments.requests, "get", fake_get)

    result = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)

    assert result == {"ETHUSDT", "ADAUSDT", "XRPUSDT"}
    assert len(calls) == 2
    assert calls[0]["params"] == {"category": "spot"}
    assert calls[1]["params"] == {"category": "spot", "cursor": "cursor-1"}


def test_get_listed_spot_symbols_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "BTCUSDT"},
            ],
        }
    }

    calls = SimpleNamespace(count=0)

    def fake_get(url, params=None, timeout=None):
        calls.count += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(instruments.requests, "get", fake_get)

    first = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)
    second = instruments.get_listed_spot_symbols(testnet=True)

    assert first == {"BTCUSDT"}
    assert second == {"BTCUSDT"}
    assert calls.count == 1


def test_get_listed_spot_symbols_testnet_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "MAINUSDT"},
            ]
        }
    }

    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if "api-testnet" in url:
            raise instruments.requests.exceptions.HTTPError("403 Client Error")
        assert url == instruments._MAINNET_URL
        return _FakeResponse(payload)

    events: list[tuple[str, dict]] = []

    def fake_log(event: str, **payload):
        events.append((event, dict(payload)))

    monkeypatch.setattr(instruments.requests, "get", fake_get)
    monkeypatch.setattr(instruments, "log", fake_log)

    result = instruments.get_listed_spot_symbols(testnet=True, force_refresh=True)

    assert result == {"MAINUSDT"}
    assert calls == [instruments._TESTNET_URL, instruments._MAINNET_URL]

    fallback_events = [payload for event, payload in events if event == "instruments.fetch.testnet_mainnet_fallback"]
    assert fallback_events and fallback_events[0]["count"] == 1

    second = instruments.get_listed_spot_symbols(testnet=True)
    assert second == {"MAINUSDT"}
    assert calls == [instruments._TESTNET_URL, instruments._MAINNET_URL]


def test_get_listed_spot_symbols_concurrent_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "result": {
            "list": [
                {"symbol": "SOLUSDT"},
            ]
        }
    }

    calls = SimpleNamespace(count=0)

    first_request_started = threading.Event()
    allow_response = threading.Event()

    def fake_get(url, params=None, timeout=None):
        calls.count += 1
        first_request_started.set()
        allow_response.wait(timeout=5.0)
        return _FakeResponse(payload)

    monkeypatch.setattr(instruments.requests, "get", fake_get)

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
    assert calls.count == 1


def test_get_listed_spot_symbols_persists_history(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {"result": {"list": [{"symbol": "AAAUSDT"}]}},
        {"result": {"list": [{"symbol": "AAAUSDT"}, {"symbol": "BBBUSDT"}]}}
    ]

    call_index = SimpleNamespace(value=0)

    def fake_get(url, params=None, timeout=None):
        idx = min(call_index.value, len(payloads) - 1)
        call_index.value += 1
        return _FakeResponse(payloads[idx])

    time_values = iter([1_000_000.0, 1_000_100.0, 1_000_100.0])

    def fake_time() -> float:
        try:
            return next(time_values)
        except StopIteration:  # pragma: no cover - defensive guard
            return 1_000_100.0

    monkeypatch.setattr(instruments.requests, "get", fake_get)
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
