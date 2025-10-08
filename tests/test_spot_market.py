import time
from decimal import Decimal

import pytest

from bybit_app.utils import spot_market as spot_market_module
from bybit_app.utils.spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    prepare_spot_trade_snapshot,
    resolve_trade_symbol,
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

    def instruments_info(self, category="spot", symbol: str | None = None, **kwargs):
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
    spot_market_module._SYMBOL_CACHE.clear()


def _universe_payload(entries: list[dict]) -> dict:
    return {"result": {"list": entries}}


def test_resolve_trade_symbol_handles_delimiters():
    payload = _universe_payload([
        {"symbol": "SOLUSDT", "quoteCoin": "USDT", "status": "Trading"},
    ])
    api = DummyAPI(payload)

    resolved, meta = resolve_trade_symbol("sol/usdt", api=api)

    assert resolved == "SOLUSDT"
    assert meta["reason"] == "exact"


def test_resolve_trade_symbol_uses_alias_mapping():
    payload = _universe_payload([
        {
            "symbol": "WBTCUSDT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "alias": "BTC",
            "baseCoin": "WBTC",
        }
    ])
    api = DummyAPI(payload)

    resolved, meta = resolve_trade_symbol("btc", api=api)

    assert resolved == "WBTCUSDT"
    assert meta["reason"] == "alias_match"
    assert meta["alias"] == "BTC"


def test_resolve_trade_symbol_refreshes_when_cache_stale():
    payload = _universe_payload([
        {"symbol": "NEWUSDT", "quoteCoin": "USDT", "status": "Trading"},
    ])
    api = DummyAPI(payload)

    spot_market_module._SYMBOL_CACHE.set(
        "spot_usdt",
        {"symbols": {}, "by_base": {}, "aliases": {}, "ts": time.time()},
    )

    resolved, meta = resolve_trade_symbol("NEWUSDT", api=api)

    assert resolved == "NEWUSDT"
    assert meta["reason"] == "exact"
    assert meta["cache_state"] == "refreshed"


def test_resolve_trade_symbol_prefers_canonical_symbol_for_alias():
    payload = _universe_payload([
        {
            "symbol": "PEPEUSDT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "baseCoin": "PEPE",
        },
        {
            "symbol": "PEPE3LUSDT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "baseCoin": "PEPE",
        },
    ])
    api = DummyAPI(payload)

    resolved, meta = resolve_trade_symbol("PEPE", api=api)

    assert resolved == "PEPEUSDT"
    assert meta["reason"] == "alias_match"


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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=3.0,
            unit="quoteCoin",
            tol_value=0.5,
        )

    message = str(excinfo.value)
    assert "Минимальный объём" in message
    assert not api.place_calls


def test_place_spot_market_omits_slippage_when_zero_tolerance():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.01",
                    },
                }
            ]
        }
    }
    api = DummyAPI(payload)

    response = place_spot_market_with_tolerance(
        api,
        symbol="ETHUSDT",
        side="Buy",
        qty=10.0,
        unit="quoteCoin",
        tol_value=0,
    )

    assert response["ok"] is True
    placed = api.place_calls[0]
    assert "slippageTolerance" not in placed
    assert "slippageToleranceType" not in placed


def test_wallet_available_balances_use_wallet_balance_when_withdraw_zero():
    wallet_payload = {
        "result": {
            "list": [
                {
                    "accountType": "UNIFIED",
                    "coin": [
                        {
                            "coin": "USDT",
                            "totalAvailableBalance": "0",
                            "availableToWithdraw": "0",
                            "walletBalance": "26603.12",
                        }
                    ],
                }
            ]
        }
    }
    api = DummyAPI({}, wallet_payload=wallet_payload)

    balances = spot_market_module._wallet_available_balances(api, account_type="UNIFIED")

    assert balances["USDT"] == Decimal("26603.12")


def test_wallet_available_balances_pick_positive_amount_if_present():
    wallet_payload = {
        "result": {
            "list": [
                {
                    "accountType": "UNIFIED",
                    "coin": [
                        {
                            "coin": "USDT",
                            "availableToWithdraw": "0",
                            "availableBalance": "-5",
                            "walletBalance": "10",
                        }
                    ],
                }
            ]
        }
    }
    api = DummyAPI({}, wallet_payload=wallet_payload)

    balances = spot_market_module._wallet_available_balances(api, account_type="UNIFIED")

    assert balances["USDT"] == Decimal("10")


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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=12.0,
            unit="quoteCoin",
            tol_value=0.5,
            max_quote=Decimal("6"),
        )

    assert "Недостаточно свободного капитала" in str(excinfo.value)
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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="ETHUSDT",
            side="Buy",
            qty=0.01,
            unit="baseCoin",
            max_quote=Decimal("10"),
        )

    assert "Недостаточно свободного капитала" in str(excinfo.value)
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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=10,
            unit="quoteCoin",
            tol_value=1.2,
            max_quote=Decimal("10.05"),
        )

    assert "Недостаточно свободного капитала" in str(excinfo.value)
    assert api.place_calls == []


def test_place_spot_market_accepts_percent_string():
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
        qty=10,
        unit="quoteCoin",
        tol_value="0.75%",
    )

    assert response["ok"] is True
    placed = api.place_calls[0]
    assert placed["slippageToleranceType"] == "Percent"
    assert placed["slippageTolerance"] == "0.7500"


def test_place_spot_market_accepts_bps_suffix():
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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=10,
            unit="quoteCoin",
            tol_value="50bps",
            max_quote=Decimal("10.04"),
        )

    assert "Недостаточно свободного капитала" in str(excinfo.value)
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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BBSOLUSDC",
            side="Buy",
            qty=15.0,
            unit="quoteCoin",
        )

    details = getattr(excinfo.value, "details", {})
    assert details.get("asset") == "USDC"
    alt_usdt = details.get("alt_usdt")
    if alt_usdt is not None:
        assert "120" in alt_usdt
    assert api.place_calls == []
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

    with pytest.raises(OrderValidationError) as excinfo:
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


def test_place_spot_market_ignores_account_type_error_from_non_runtime_exception():
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

    class ValueErrorFallbackAPI(DummyAPI):
        def wallet_balance(self, accountType="UNIFIED"):
            if accountType and accountType.upper() != "UNIFIED":
                self.wallet_calls += 1
                raise ValueError(
                    "Bybit error 10001: accountType only support UNIFIED. (/v5/account/wallet-balance)!"
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

    api = ValueErrorFallbackAPI(payload, wallet_payload=wallet_payload)

    with pytest.raises(OrderValidationError) as excinfo:
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
    assert api.wallet_calls == 2


def test_place_spot_market_ignores_account_type_error_with_supports_phrase():
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

    class ValueErrorSupportsFallbackAPI(DummyAPI):
        def wallet_balance(self, accountType="UNIFIED"):
            if accountType and accountType.upper() != "UNIFIED":
                self.wallet_calls += 1
                raise ValueError(
                    "Bybit error 10001: accountType only supports UNIFIED trading account."
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

    api = ValueErrorSupportsFallbackAPI(payload, wallet_payload=wallet_payload)

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BBSOLUSDT",
            side="Buy",
            qty=10.0,
            unit="quoteCoin",
        )

    message = str(excinfo.value)
    assert "Недостаточно" in message
    assert "accountType only supports UNIFIED" not in message
    assert api.wallet_calls == 2


def test_place_spot_market_ignores_account_type_error_with_http_status_code():
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

    class HTTPStatusAccountTypeError(Exception):
        def __init__(self):
            super().__init__(
                "400 Client Error: Bad Request for url: https://api.bybit.com/v5/account/wallet-balance"
                " - accountType only supports UNIFIED trading account"
            )
            self.status_code = 400

    class HTTPStatusFallbackAPI(DummyAPI):
        def wallet_balance(self, accountType="UNIFIED"):
            if accountType and accountType.upper() != "UNIFIED":
                self.wallet_calls += 1
                raise HTTPStatusAccountTypeError()
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

    api = HTTPStatusFallbackAPI(payload, wallet_payload=wallet_payload)

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="BBSOLUSDT",
            side="Buy",
            qty=10.0,
            unit="quoteCoin",
        )

    message = str(excinfo.value)
    assert "Недостаточно" in message
    assert "accountType only supports UNIFIED" not in message
    assert api.wallet_calls == 2


def test_place_spot_market_ignores_structured_account_type_error():
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

    class StructuredAccountTypeError(Exception):
        def __init__(self):
            super().__init__()
            self.retCode = 10001
            self.retMsg = "accountType only support UNIFIED"

        def __str__(self):
            return "Bybit structured error"

    class StructuredFallbackAPI(DummyAPI):
        def wallet_balance(self, accountType="UNIFIED"):
            if accountType and accountType.upper() != "UNIFIED":
                self.wallet_calls += 1
                raise StructuredAccountTypeError()
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

    api = StructuredFallbackAPI(payload, wallet_payload=wallet_payload)

    with pytest.raises(OrderValidationError) as excinfo:
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

    with pytest.raises(OrderValidationError):
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

    with pytest.raises(OrderValidationError) as excinfo:
        place_spot_market_with_tolerance(
            api,
            symbol="ETHUSDT",
            side="Buy",
            qty=0.003,
            unit="baseCoin",
            tol_value=0.3,
        )

    assert "минимального лота" in str(excinfo.value)
    assert api.place_calls == []


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

    with pytest.raises(OrderValidationError):
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
