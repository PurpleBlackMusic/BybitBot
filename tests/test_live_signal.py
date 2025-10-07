from __future__ import annotations

from types import SimpleNamespace

import pytest

import bybit_app.utils.live_signal as live_signal_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.live_signal import LiveSignalError, LiveSignalFetcher


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
        settings=Settings(ai_live_only=False, ai_min_ev_bps=10.0), data_dir=tmp_path, cache_ttl=30.0
    )

    first = fetcher.fetch()
    second = fetcher.fetch()

    assert calls["count"] == 1
    assert second == first
    assert second["status_source"] == "live"


def test_live_signal_fetcher_reports_failure_when_scan_empty(
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
        settings=Settings(ai_live_only=False, ai_min_ev_bps=8.0),
        data_dir=tmp_path,
        cache_ttl=20.0,
    )

    initial = fetcher.fetch()
    assert initial["symbol"] == "ETHUSDT"

    def failing_scan(api, **kwargs):
        call_order.append("fail")
        return []

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", failing_scan)

    clock.advance(30.0)

    with pytest.raises(LiveSignalError) as exc:
        fetcher.fetch()

    assert "не вернул" in str(exc.value).lower()
    assert call_order == ["first", "fail"]
    assert fetcher._cached_status == initial


def test_live_signal_fetcher_raises_when_api_missing(monkeypatch, tmp_path) -> None:

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(live_signal_module, "get_api_client", _raise)

    fetcher = LiveSignalFetcher(settings=Settings(ai_live_only=False), data_dir=tmp_path, cache_ttl=0.0)

    with pytest.raises(LiveSignalError) as exc:
        fetcher.fetch()

    assert "api" in str(exc.value).lower()


def test_live_signal_fetcher_live_only_disables_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=3_000.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    calls = {"count": 0}

    def fake_scan(api, **kwargs):
        calls["count"] += 1
        symbol = "SOLUSDT" if calls["count"] == 1 else "DOGEUSDT"
        return [_make_opportunity(symbol)]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", fake_scan)

    fetcher = LiveSignalFetcher(
        settings=Settings(ai_live_only=True, ai_min_ev_bps=5.0), data_dir=tmp_path, cache_ttl=45.0
    )

    first = fetcher.fetch()
    clock.advance(5.0)
    second = fetcher.fetch()

    assert calls["count"] == 2
    assert first["symbol"] == "SOLUSDT"
    assert second["symbol"] == "DOGEUSDT"
