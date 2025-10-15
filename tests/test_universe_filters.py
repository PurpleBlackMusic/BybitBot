import pytest

import bybit_app.utils.universe as universe_module


def test_filter_available_spot_pairs_uses_listed_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: [symbol for symbol in symbols if symbol == "ETHUSDT"],
    )

    result = universe_module.filter_available_spot_pairs([
        "ETHUSDT",
        "FAKEUSDT",
        "ADAUSDT",
    ])

    assert result == ["ETHUSDT"]


def test_filter_available_spot_pairs_respects_mainnet(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        testnet = False

    captured = {}

    def _fake_filter(symbols, *, testnet, **kwargs):
        captured["testnet"] = testnet
        return [
            symbol
            for symbol in symbols
            if symbol in {"BTCUSDT", "MAINNETONLYUSDT"}
        ]

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(universe_module, "filter_listed_spot_symbols", _fake_filter)

    result = universe_module.filter_available_spot_pairs([
        "BTCUSDT",
        "MAINNETONLYUSDT",
        "TESTNETONLYUSDT",
    ])

    assert captured["testnet"] is False
    assert result == ["BTCUSDT", "MAINNETONLYUSDT"]


def test_filter_available_spot_pairs_falls_back_when_listing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: [],
    )

    result = universe_module.filter_available_spot_pairs(["ETHUSDT", "ADAUSDT"])

    assert result == ["ETHUSDT", "ADAUSDT"]


def test_filter_available_spot_pairs_excludes_blacklisted_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: list(symbols),
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


def test_filter_available_spot_pairs_supports_multiple_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: list(symbols),
    )

    result = universe_module.filter_available_spot_pairs(
        [
            "ETHUSDT",
            "BTCUSDC",
            "SOLUSDT",
            "XRPUSDC",
            "DOGEUSD",
        ],
        quote_assets=["USDT", "USDC"],
    )

    assert result == ["ETHUSDT", "BTCUSDC", "SOLUSDT", "XRPUSDC"]


def test_filter_available_spot_pairs_defaults_to_usdt_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: list(symbols),
    )

    result = universe_module.filter_available_spot_pairs(
        ["BTCUSDT", "ETHUSDT", "SOLUSDC"],
    )

    assert result == ["BTCUSDT", "ETHUSDT"]


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

    assert min_turnover == 1_000_000.0
    assert max_spread == 40.0


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

    instrument_meta = {
        "BULLUSDT": {"symbol": "BULLUSDT", "age_days": 180.0, "meanSpreadBps": 8.0, "volatilityRank": "medium"},
        "BTCUSDT": {"symbol": "BTCUSDT", "age_days": 365.0, "meanSpreadBps": 5.0, "volatilityRank": "medium"},
    }

    class DummyAPI:
        def _safe_req(self, method, path, params=None, body=None, signed=False):
            if path == "/v5/market/tickers":
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
            if path == "/v5/market/instruments-info":
                if params and params.get("symbol"):
                    symbol = params["symbol"]
                    meta = instrument_meta.get(symbol)
                    return {"result": {"list": [meta] if meta else []}}
                return {"result": {"list": list(instrument_meta.values())}}
            raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: list(symbols),
    )
    monkeypatch.setattr(universe_module, "_INSTRUMENT_META_CACHE", {})

    result = universe_module.build_universe(DummyAPI(), size=5)

    assert result == ["BTCUSDT"]


def test_build_universe_retains_size_after_listing_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        ai_min_turnover_usd = 2_000_000.0
        ai_max_spread_bps = 25.0

    instrument_meta = {
        "FAKEUSDT": {"symbol": "FAKEUSDT", "age_days": 120.0, "meanSpreadBps": 25.0, "volatilityRank": "medium"},
        "ETHUSDT": {"symbol": "ETHUSDT", "age_days": 400.0, "meanSpreadBps": 6.0, "volatilityRank": "medium"},
        "SOLUSDT": {"symbol": "SOLUSDT", "age_days": 250.0, "meanSpreadBps": 9.0, "volatilityRank": "medium"},
    }

    class DummyAPI:
        def _safe_req(self, method, path, params=None, body=None, signed=False):
            if path == "/v5/market/tickers":
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
            if path == "/v5/market/instruments-info":
                if params and params.get("symbol"):
                    symbol = params["symbol"]
                    meta = instrument_meta.get(symbol)
                    return {"result": {"list": [meta] if meta else []}}
                return {"result": {"list": list(instrument_meta.values())}}
            raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: [symbol for symbol in symbols if symbol != "FAKEUSDT"],
    )
    monkeypatch.setattr(universe_module, "_INSTRUMENT_META_CACHE", {})

    result = universe_module.build_universe(DummyAPI(), size=2)

    assert result == ["ETHUSDT", "SOLUSDT"]


def test_build_universe_applies_age_spread_and_volatility_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySettings:
        ai_min_turnover_usd = 0.0
        ai_max_spread_bps = 100.0

    instrument_meta = {
        "LEGACYUSDT": {"symbol": "LEGACYUSDT", "age_days": 200.0, "meanSpreadBps": 12.0, "volatilityRank": "medium"},
        "YOUNGUSDT": {"symbol": "YOUNGUSDT", "age_days": 10.0, "meanSpreadBps": 8.0, "volatilityRank": "medium"},
        "WIDESPREADUSDT": {"symbol": "WIDESPREADUSDT", "age_days": 150.0, "meanSpreadBps": 45.0, "volatilityRank": "medium"},
        "HYPERVOLUSDT": {"symbol": "HYPERVOLUSDT", "age_days": 160.0, "meanSpreadBps": 10.0, "volatilityRank": "high"},
    }

    class DummyAPI:
        def _safe_req(self, method, path, params=None, body=None, signed=False):
            if path == "/v5/market/tickers":
                return {
                    "result": {
                        "list": [
                            {
                                "symbol": "LEGACYUSDT",
                                "turnover24h": "5000000",
                                "bestBidPrice": "10",
                                "bestAskPrice": "10.012",
                            },
                            {
                                "symbol": "YOUNGUSDT",
                                "turnover24h": "7000000",
                                "bestBidPrice": "2",
                                "bestAskPrice": "2.002",
                            },
                            {
                                "symbol": "WIDESPREADUSDT",
                                "turnover24h": "8000000",
                                "bestBidPrice": "5",
                                "bestAskPrice": "5.005",
                            },
                            {
                                "symbol": "HYPERVOLUSDT",
                                "turnover24h": "6000000",
                                "bestBidPrice": "1",
                                "bestAskPrice": "1.001",
                                "price24hPcnt": "0.25",
                            },
                        ]
                    }
                }
            if path == "/v5/market/instruments-info":
                if params and params.get("symbol"):
                    symbol = params["symbol"]
                    meta = instrument_meta.get(symbol)
                    return {"result": {"list": [meta] if meta else []}}
                return {"result": {"list": list(instrument_meta.values())}}
            raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(universe_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols, **kwargs: list(symbols),
    )
    monkeypatch.setattr(universe_module, "_INSTRUMENT_META_CACHE", {})

    scored = universe_module.build_universe_scored(
        DummyAPI(), size=0, min_turnover=0.0, max_spread_bps=100.0
    )

    assert [symbol for symbol, _ in scored] == ["LEGACYUSDT"]
