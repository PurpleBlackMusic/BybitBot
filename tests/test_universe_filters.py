import pytest

import bybit_app.utils.universe as universe_module


def test_filter_available_spot_pairs_uses_listed_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols: [symbol for symbol in symbols if symbol == "ETHUSDT"],
    )

    result = universe_module.filter_available_spot_pairs([
        "ETHUSDT",
        "FAKEUSDT",
        "ADAUSDT",
    ])

    assert result == ["ETHUSDT"]


def test_filter_available_spot_pairs_falls_back_when_listing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(universe_module, "filter_listed_spot_symbols", lambda symbols: [])

    result = universe_module.filter_available_spot_pairs(["ETHUSDT", "ADAUSDT"])

    assert result == ["ETHUSDT", "ADAUSDT"]


def test_filter_available_spot_pairs_excludes_blacklisted_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols: list(symbols),
    )

    result = universe_module.filter_available_spot_pairs(
        [
            "BULLUSDT",
            "BBUSDT",
            "1000SUSDT",
            "BTCUSDT",
            "SOLUSDT",
        ]
    )

    assert result == ["BTCUSDT", "SOLUSDT"]


def test_filter_blacklisted_symbols_ignores_non_strings() -> None:
    assert universe_module.filter_blacklisted_symbols([
        "ETHUSDT",
        None,
        1234,
        "",
    ]) == ["ETHUSDT"]


def test_resolve_liquidity_filters_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        ai_min_turnover_usd = 1_000_000.0
        ai_max_spread_bps = 40.0

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())

    min_turnover, max_spread = universe_module._resolve_liquidity_filters(None, None)

    assert min_turnover == 2_000_000.0
    assert max_spread == 25.0


def test_resolve_liquidity_filters_respects_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        ai_min_turnover_usd = 5_000_000.0
        ai_max_spread_bps = 15.0

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())

    min_turnover, max_spread = universe_module._resolve_liquidity_filters(3_500_000.0, 10.0)

    assert min_turnover == 3_500_000.0
    assert max_spread == 10.0


def test_build_universe_skips_blacklisted_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        ai_min_turnover_usd = 2_000_000.0
        ai_max_spread_bps = 25.0

    class DummyAPI:
        def _safe_req(self, method, path, params=None, body=None, signed=False):
            return {
                "result": {
                    "list": [
                        {
                            "symbol": "BULLUSDT",
                            "turnover24h": "10000000",
                            "bestBidPrice": "1",
                            "bestAskPrice": "1.0005",
                        },
                        {
                            "symbol": "BTCUSDT",
                            "turnover24h": "15000000",
                            "bestBidPrice": "30000",
                            "bestAskPrice": "30010",
                        },
                    ]
                }
            }

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols: list(symbols),
    )

    result = universe_module.build_universe(DummyAPI(), size=5)

    assert result == ["BTCUSDT"]


def test_build_universe_retains_size_after_listing_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        ai_min_turnover_usd = 2_000_000.0
        ai_max_spread_bps = 25.0

    class DummyAPI:
        def _safe_req(self, method, path, params=None, body=None, signed=False):
            return {
                "result": {
                    "list": [
                        {
                            "symbol": "FAKEUSDT",
                            "turnover24h": "5000000",
                            "bestBidPrice": "1",
                            "bestAskPrice": "1.0005",
                        },
                        {
                            "symbol": "ETHUSDT",
                            "turnover24h": "15000000",
                            "bestBidPrice": "3000",
                            "bestAskPrice": "3000.3",
                        },
                        {
                            "symbol": "SOLUSDT",
                            "turnover24h": "10000000",
                            "bestBidPrice": "20",
                            "bestAskPrice": "20.02",
                        },
                    ]
                }
            }

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols: [symbol for symbol in symbols if symbol != "FAKEUSDT"],
    )

    result = universe_module.build_universe(DummyAPI(), size=2)

    assert result == ["ETHUSDT", "SOLUSDT"]
