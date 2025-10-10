from __future__ import annotations

from types import SimpleNamespace

import pytest

import bybit_app.utils.live_signal as live_signal_module
import bybit_app.utils.market_scanner as market_scanner_module
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


def test_live_signal_fetcher_refreshes_snapshot_after_ttl(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=4_000.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(market_scanner_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    stale_snapshot = {
        "ts": clock.time() - 1_000.0,
        "rows": [
            {
                "symbol": "AAAUSDT",
                "price24hPcnt": 2.0,
                "turnover24h": 2_000_000.0,
                "bestBidPrice": 1.0,
                "bestAskPrice": 1.01,
                "volume24h": 1_000_000.0,
            }
        ],
    }
    market_scanner_module.save_market_snapshot(stale_snapshot, data_dir=tmp_path)

    def fake_feature_bundle(raw: dict[str, object]) -> dict[str, object]:
        return {
            "blended_change_pct": raw.get("price24hPcnt", 0.0),
            "volatility_pct": 5.0,
            "volatility_windows": {},
            "volume_spike_score": 0.0,
            "volume_impulse": {},
            "depth_imbalance": 0.0,
            "correlations": {},
            "correlation_strength": 0.0,
        }

    monkeypatch.setattr(market_scanner_module, "build_feature_bundle", fake_feature_bundle)
    monkeypatch.setattr(market_scanner_module, "ensure_market_model", lambda **_: None)

    fetch_calls = {"count": 0}

    def fake_fetch_snapshot(api, category: str = "spot") -> dict[str, object]:
        fetch_calls["count"] += 1
        return {
            "ts": clock.time(),
            "rows": [
                {
                    "symbol": "AAAUSDT",
                    "price24hPcnt": 3.5,
                    "turnover24h": 3_000_000.0,
                    "bestBidPrice": 2.0,
                    "bestAskPrice": 2.01,
                    "volume24h": 1_500_000.0,
                }
            ],
        }

    monkeypatch.setattr(
        market_scanner_module, "fetch_market_snapshot", fake_fetch_snapshot
    )

    fetcher = LiveSignalFetcher(
        settings=Settings(ai_live_only=False, ai_min_ev_bps=5.0),
        data_dir=tmp_path,
        cache_ttl=30.0,
    )

    result = fetcher.fetch()

    assert fetch_calls["count"] == 1
    assert result["symbol"] == "AAAUSDT"


def test_live_signal_fetcher_threads_network_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=1_500.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    captured: list[bool] = []

    def fake_scan(api, **kwargs):
        captured.append(bool(kwargs.get("testnet")))
        symbol = "TESTCOINUSDT" if kwargs.get("testnet") else "MAINCOINUSDT"
        return [_make_opportunity(symbol)]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", fake_scan)

    base_kwargs = dict(ai_live_only=False, ai_min_ev_bps=5.0)

    main_fetcher = LiveSignalFetcher(
        settings=Settings(testnet=False, **base_kwargs),
        data_dir=tmp_path,
        cache_ttl=0.0,
    )
    main_status = main_fetcher.fetch()

    test_fetcher = LiveSignalFetcher(
        settings=Settings(testnet=True, **base_kwargs),
        data_dir=tmp_path,
        cache_ttl=0.0,
    )
    test_status = test_fetcher.fetch()

    assert captured == [False, True]
    assert main_status["symbol"] == "MAINCOINUSDT"
    assert test_status["symbol"] == "TESTCOINUSDT"


def test_live_signal_fetcher_returns_cached_when_scan_empty_within_grace(
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
    assert initial["status_source"] == "live"

    def failing_scan(api, **kwargs):
        call_order.append("fail")
        return []

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", failing_scan)

    window = fetcher.cache_ttl + fetcher.stale_grace
    clock.advance(window - 5.0)

    fallback = fetcher.fetch()

    assert fallback["symbol"] == "ETHUSDT"
    assert fallback["status_source"] == "live_cached"
    assert call_order == ["first", "fail"]
    assert fetcher._cached_status == initial


def test_live_signal_fetcher_raises_when_scan_empty_and_cache_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=2_500.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    call_order: list[str] = []

    def initial_scan(api, **kwargs):
        call_order.append("first")
        return [_make_opportunity("LTCUSDT")]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", initial_scan)

    fetcher = LiveSignalFetcher(
        settings=Settings(ai_live_only=False, ai_min_ev_bps=6.0),
        data_dir=tmp_path,
        cache_ttl=15.0,
    )

    _ = fetcher.fetch()

    def failing_scan(api, **kwargs):
        call_order.append("fail")
        return []

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", failing_scan)

    window = fetcher.cache_ttl + fetcher.stale_grace
    clock.advance(window + 1.0)

    with pytest.raises(LiveSignalError) as exc:
        fetcher.fetch()

    assert "не вернул" in str(exc.value).lower()
    assert call_order == ["first", "fail"]


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


def test_live_signal_fetcher_runtime_live_only_forces_rescan(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    clock = Clock(start=4_500.0)
    monkeypatch.setattr(live_signal_module, "time", clock)
    monkeypatch.setattr(live_signal_module, "get_api_client", lambda: SimpleNamespace())

    calls = {"count": 0}

    def fake_scan(api, **kwargs):
        calls["count"] += 1
        symbol = "ADAUSDT" if calls["count"] == 1 else "XRPUSDT"
        return [_make_opportunity(symbol)]

    monkeypatch.setattr(live_signal_module, "scan_market_opportunities", fake_scan)

    monkeypatch.setattr(
        live_signal_module,
        "get_settings",
        lambda: Settings(ai_live_only=True, ai_min_ev_bps=5.0),
    )

    fetcher = LiveSignalFetcher(data_dir=tmp_path, cache_ttl=60.0)

    first = fetcher.fetch()
    clock.advance(1.0)
    second = fetcher.fetch()

    assert calls["count"] == 2
    assert first["symbol"] == "ADAUSDT"
    assert second["symbol"] == "XRPUSDT"
    assert fetcher.cache_ttl == 0.0
    assert fetcher.stale_grace == 0.0
