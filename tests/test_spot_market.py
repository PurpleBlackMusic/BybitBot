from decimal import Decimal

import pytest

from bybit_app.utils import spot_market as spot_market_module
from bybit_app.utils.spot_market import place_spot_market_with_tolerance


class DummyAPI:
    def __init__(self, instrument_payload, ticker_payload=None):
        self.instrument_payload = instrument_payload
        self.ticker_payload = ticker_payload
        self.place_calls: list[dict] = []
        self.info_calls = 0
        self.ticker_calls = 0

    def instruments_info(self, category="spot", symbol: str | None = None):
        self.info_calls += 1
        return self.instrument_payload

    def place_order(self, **kwargs):
        self.place_calls.append(kwargs)
        return {"ok": True, "body": kwargs}

    def tickers(self, category="spot", symbol: str | None = None):
        self.ticker_calls += 1
        if isinstance(self.ticker_payload, Exception):
            raise self.ticker_payload
        return self.ticker_payload or {}


def setup_function(_):
    spot_market_module._INSTRUMENT_CACHE.clear()
    spot_market_module._PRICE_CACHE.clear()


def test_place_spot_market_enforces_min_notional():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {
                        "minOrderAmt": "10",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    api = DummyAPI(payload)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=3.0,
        unit="quoteCoin",
        tol_value=0.5,
    )

    assert response["ok"] is True
    assert api.place_calls[0]["qty"] == "10"
    assert api.place_calls[0]["slippageTolerance"] == "1.0000"
    # repeated call should reuse cached instrument data
    place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=12.0,
        unit="quoteCoin",
    )
    assert api.info_calls == 1


def test_place_spot_market_respects_available_balance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {
                        "minOrderAmt": "10",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    api = DummyAPI(payload)

    with pytest.raises(RuntimeError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=3.0,
            unit="quoteCoin",
            tol_value=0.5,
            max_quote=Decimal("6"),
        )

    assert "Недостаточно свободного баланса" in str(excinfo.value)
    assert api.place_calls == []


def test_place_spot_market_base_unit_checks_available_balance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "lotSizeFilter": {
                        "minOrderQty": "0.001",
                        "qtyStep": "0.001",
                        "minOrderAmt": "0",
                    },
                }
            ]
        }
    }
    ticker = {"result": {"list": [{"symbol": "ETHUSDT", "bestBidPrice": "2000"}]}}
    api = DummyAPI(payload, ticker_payload=ticker)

    with pytest.raises(RuntimeError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="ETHUSDT",
            side="Buy",
            qty=0.01,
            unit="baseCoin",
            max_quote=Decimal("10"),
        )

    assert "Недостаточно свободного баланса" in str(excinfo.value)
    assert api.place_calls == []
    assert api.ticker_calls == 1


def test_place_spot_market_balance_guard_includes_tolerance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {
                        "minOrderAmt": "10",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    api = DummyAPI(payload)

    with pytest.raises(RuntimeError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=10,
            unit="quoteCoin",
            tol_value=1.2,
            max_quote=Decimal("11"),
        )

    assert "Недостаточно свободного баланса" in str(excinfo.value)
    assert api.place_calls == []


def test_place_spot_market_raises_when_no_instrument():
    payload = {"result": {"list": []}}
    api = DummyAPI(payload)

    with pytest.raises(RuntimeError):
        place_spot_market_with_tolerance(
            api,
            symbol="FOOUSDT",
            side="Buy",
            qty=6.0,
            unit="quoteCoin",
        )


def test_place_spot_market_adjusts_base_qty_with_price_snapshot():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "lotSizeFilter": {
                        "minOrderQty": "0.01",
                        "qtyStep": "0.001",
                        "minOrderAmt": "10",
                    },
                }
            ]
        }
    }
    ticker = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "bestAskPrice": "2500",
                }
            ]
        }
    }
    api = DummyAPI(payload, ticker_payload=ticker)

    response = place_spot_market_with_tolerance(
        api,
        symbol="ETHUSDT",
        side="Buy",
        qty=0.003,
        unit="baseCoin",
        tol_value=0.3,
    )

    assert response["ok"] is True
    assert api.place_calls[0]["marketUnit"] == "baseCoin"
    assert api.place_calls[0]["qty"] == "0.01"
    assert api.place_calls[0]["slippageTolerance"] == "1.0000"
    # ensure price snapshot queried once and cached
    assert api.ticker_calls == 1

    # second call reuses caches and keeps qty above minimum
    place_spot_market_with_tolerance(
        api,
        symbol="ETHUSDT",
        side="Buy",
        qty=0.02,
        unit="baseCoin",
    )
    assert api.info_calls == 1
    assert api.ticker_calls == 1


def test_place_spot_market_base_unit_requires_price():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "SOLUSDT",
                    "lotSizeFilter": {
                        "minOrderQty": "0.01",
                        "qtyStep": "0.001",
                        "minOrderAmt": "20",
                    },
                }
            ]
        }
    }
    ticker = {"result": {"list": [{"symbol": "SOLUSDT"}]}}
    api = DummyAPI(payload, ticker_payload=ticker)

    with pytest.raises(RuntimeError):
        place_spot_market_with_tolerance(
            api,
            symbol="SOLUSDT",
            side="Buy",
            qty=0.01,
            unit="baseCoin",
        )
