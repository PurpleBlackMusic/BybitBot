from __future__ import annotations

from types import SimpleNamespace

import pytest

import bybit_app.utils.live_signal as live_signal_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.live_signal import LiveSignalFetcher


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_opportunity(symbol: str = "BTCUSDT") -> dict[str, object]:
    return {
        "symbol": symbol,
        "probability": 0.72,
        "ev_bps": 18.0,
        "turnover_usd": 8_000_000.0,
        "spread_bps": 12.0,
        "change_pct": 2.5,
        "trend": "buy",
    }


def test_live_signal_fetcher_reuses_cache_within_ttl(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=1_000.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    calls = {"count": 0}

    def fake_scan(api, **kwargs):
        calls["count"] += 1
        return [_make_opportunity()]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", fake_scan)

    fetcher = LiveSignalFetcher(
        settings=Settings(ai_min_ev_bps=10.0), data_dir=tmp_path, cache_ttl=30.0
    )

    first = fetcher.fetch()
    second = fetcher.fetch()

    assert calls["count"] == 1
    assert second == first
    assert second["status_source"] == "live"


def test_live_signal_fetcher_serves_stale_cache_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=2_000.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    call_order: list[str] = []

    def first_scan(api, **kwargs):
        call_order.append("first")
        return [_make_opportunity("ETHUSDT")]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", first_scan)

    fetcher = LiveSignalFetcher(
        settings=Settings(ai_min_ev_bps=8.0),
        data_dir=tmp_path,
        cache_ttl=20.0,
    )

    initial = fetcher.fetch()
    assert initial["symbol"] == "ETHUSDT"

    def failing_scan(api, **kwargs):
        call_order.append("fail")
        return []

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", failing_scan)

    clock.advance(30.0)  # exceed cache ttl but stay within stale grace window
    fallback = fetcher.fetch()

    assert call_order == ["first", "fail"]
    assert fallback == initial
