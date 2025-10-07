from decimal import Decimal

import pytest

from bybit_app.utils import spot_market as spot_market_module
from bybit_app.utils.spot_market import (
    place_spot_market_with_tolerance,
    prepare_spot_trade_snapshot,
)


class DummyAPI:
    def __init__(self, instrument_payload, ticker_payload=None, wallet_payload=None):
        self.instrument_payload = instrument_payload
        self.ticker_payload = ticker_payload
        self.wallet_payload = wallet_payload or {
            "result": {"list": [{"coin": [{"coin": "USDT", "availableBalance": "100"}]}]}
        }
        self.place_calls: list[dict] = []
        self.info_calls = 0
        self.ticker_calls = 0
        self.wallet_calls = 0

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

    def wallet_balance(self, accountType="UNIFIED"):
        self.wallet_calls += 1
        if isinstance(self.wallet_payload, Exception):
            raise self.wallet_payload
        payload = self.wallet_payload
        if isinstance(payload, dict) and "result" not in payload:
            lookup_key = (accountType or "").upper()
            payload = payload.get(lookup_key, payload.get("default"))
        return payload


def setup_function(_):
    spot_market_module._INSTRUMENT_CACHE.clear()
    spot_market_module._PRICE_CACHE.clear()
    spot_market_module._BALANCE_CACHE.clear()


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


def test_place_spot_market_requires_quote_currency_balance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDC",
                    "quoteCoin": "USDC",
                    "baseCoin": "BBSOL",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    wallet = {
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "USDT", "availableBalance": "120"},
                    ]
                }
            ]
        }
    }
    api = DummyAPI(payload, wallet_payload=wallet)

    with pytest.raises(RuntimeError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BBSOLUSDC",
            side="Buy",
            qty=15.0,
            unit="quoteCoin",
        )

    message = str(excinfo.value)
    assert "USDC" in message
    assert "USDT" in message
    assert api.place_calls == []
    assert api.wallet_calls >= 1


def test_place_spot_market_allows_matching_quote_balance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDC",
                    "quoteCoin": "USDC",
                    "baseCoin": "BBSOL",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    wallet = {
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "USDC", "availableBalance": "50"},
                        {"coin": "USDT", "availableBalance": "10"},
                    ]
                }
            ]
        }
    }
    api = DummyAPI(payload, wallet_payload=wallet)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BBSOLUSDC",
        side="Buy",
        qty=15.0,
        unit="quoteCoin",
    )

    assert response["ok"] is True
    assert api.place_calls
    assert api.wallet_calls >= 1


def test_place_spot_market_uses_spot_wallet_when_unified_empty():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDT",
                    "quoteCoin": "USDT",
                    "baseCoin": "BBSOL",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    wallet_payload = {
        "UNIFIED": {
            "result": {
                "list": [
                    {"coin": [{"coin": "USDT", "availableBalance": "0"}]}
                ]
            }
        },
        "SPOT": {
            "result": {
                "list": [
                    {"coin": [{"coin": "USDT", "availableBalance": "200"}]}
                ]
            }
        },
    }
    api = DummyAPI(payload, wallet_payload=wallet_payload)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BBSOLUSDT",
        side="Buy",
        qty=10.0,
        unit="quoteCoin",
    )

    assert response["ok"] is True
    assert api.place_calls
    # wallet is queried twice: once for UNIFIED and once for SPOT fallback
    assert api.wallet_calls == 2


def test_place_spot_market_ignores_spot_wallet_account_type_error():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDT",
                    "quoteCoin": "USDT",
                    "baseCoin": "BBSOL",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }

    class FallbackErrorAPI(DummyAPI):
        def wallet_balance(self, accountType="UNIFIED"):
            if accountType and accountType.upper() != "UNIFIED":
                self.wallet_calls += 1
                raise RuntimeError(
                    "Bybit error 10001: accountType only support UNIFIED (/v5/account/wallet-balance)"
                )
            return super().wallet_balance(accountType=accountType)

    wallet_payload = {
        "UNIFIED": {
            "result": {
                "list": [
                    {
                        "coin": [
                            {"coin": "USDT", "availableBalance": "0"},
                        ]
                    }
                ]
            }
        }
    }

    api = FallbackErrorAPI(payload, wallet_payload=wallet_payload)

    with pytest.raises(RuntimeError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BBSOLUSDT",
            side="Buy",
            qty=10.0,
            unit="quoteCoin",
        )

    message = str(excinfo.value)
    assert "Недостаточно" in message
    assert "accountType only support UNIFIED" not in message
    # unified balance queried successfully, spot fallback failure is ignored
    assert api.wallet_calls == 2


def test_place_spot_market_accepts_prefetched_resources():
    limits = {
        "min_order_amt": Decimal("5"),
        "quote_step": Decimal("0.01"),
        "min_order_qty": Decimal("0"),
        "qty_step": Decimal("0.00000001"),
        "base_coin": "BBSOL",
        "quote_coin": "USDT",
    }
    balances = {"USDT": Decimal("50")}
    api = DummyAPI({}, ticker_payload=RuntimeError("ticker should not be called"), wallet_payload=RuntimeError("wallet should not be called"))

    response = place_spot_market_with_tolerance(
        api,
        symbol="BBSOLUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        tol_value=0.5,
        price_snapshot=Decimal("15"),
        balances=balances,
        limits=limits,
    )

    assert response["ok"] is True
    assert api.info_calls == 0
    assert api.ticker_calls == 0
    assert api.wallet_calls == 0


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


def test_prepare_spot_trade_snapshot_prefetches_resources():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDT",
                    "quoteCoin": "USDT",
                    "baseCoin": "BBSOL",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    ticker = {
        "result": {
            "list": [
                {
                    "symbol": "BBSOLUSDT",
                    "bestAskPrice": "2.5",
                }
            ]
        }
    }
    wallet = {
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "USDT", "availableBalance": "75"},
                    ]
                }
            ]
        }
    }
    api = DummyAPI(payload, ticker_payload=ticker, wallet_payload=wallet)

    snapshot = prepare_spot_trade_snapshot(api, "BBSOLUSDT")

    assert snapshot.symbol == "BBSOLUSDT"
    assert snapshot.price == Decimal("2.5")
    assert snapshot.balances == {"USDT": Decimal("75")}
    assert snapshot.limits is not None
    assert api.info_calls == 1
    assert api.ticker_calls == 1
    assert api.wallet_calls == 1

    response = place_spot_market_with_tolerance(
        api,
        symbol="BBSOLUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        **snapshot.as_kwargs(),
    )

    assert response["ok"] is True
    assert api.info_calls == 1
    assert api.ticker_calls == 1
    assert api.wallet_calls == 1


def test_prepare_spot_trade_snapshot_force_refresh_invalidates_cache():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "quoteCoin": "USDT",
                    "baseCoin": "BTC",
                    "lotSizeFilter": {
                        "minOrderAmt": "10",
                        "minOrderAmtIncrement": "0.1",
                    },
                }
            ]
        }
    }
    ticker = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "30000",
                }
            ]
        }
    }
    wallet = {
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "USDT", "availableBalance": "120"},
                    ]
                }
            ]
        }
    }
    api = DummyAPI(payload, ticker_payload=ticker, wallet_payload=wallet)

    first_snapshot = prepare_spot_trade_snapshot(api, "BTCUSDT")
    assert first_snapshot.price == Decimal("30000")
    assert api.info_calls == 1
    assert api.ticker_calls == 1
    assert api.wallet_calls == 1

    second_snapshot = prepare_spot_trade_snapshot(api, "BTCUSDT")
    assert second_snapshot.price == Decimal("30000")
    assert api.info_calls == 1
    assert api.ticker_calls == 1
    assert api.wallet_calls == 1

    refreshed_snapshot = prepare_spot_trade_snapshot(api, "BTCUSDT", force_refresh=True)
    assert refreshed_snapshot.price == Decimal("30000")
    assert api.info_calls == 2
    assert api.ticker_calls == 2
    assert api.wallet_calls == 2
