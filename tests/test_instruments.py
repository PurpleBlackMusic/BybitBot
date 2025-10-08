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
def _reset_instrument_cache():
    instruments._CACHE.clear()
    yield
    instruments._CACHE.clear()


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
