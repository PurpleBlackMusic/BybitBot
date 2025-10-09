import time
from decimal import Decimal

import pytest

from bybit_app.utils import spot_market as spot_market_module
from bybit_app.utils import validators
from bybit_app.utils.envs import Settings
from bybit_app.utils.spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    prepare_spot_trade_snapshot,
    resolve_trade_symbol,
)


class DummyAPI:
    def __init__(
        self,
        instrument_payload,
        ticker_payload=None,
        wallet_payload=None,
        orderbook_payload=None,
        creds=None,
        exchange_asset_payload=None,
    ):
        self.instrument_payload = instrument_payload
        self.ticker_payload = ticker_payload
        self.wallet_payload = wallet_payload or {
            "result": {"list": [{"coin": [{"coin": "USDT", "availableBalance": "100"}]}]}
        }
        self.orderbook_payload = orderbook_payload or {
            "result": {
                "a": [["101.0", "2"], ["102.0", "3"]],
                "b": [["100.0", "2"], ["99.0", "3"]],
            }
        }
        self.exchange_asset_payload = exchange_asset_payload
        self.place_calls: list[dict] = []
        self.info_calls = 0
        self.ticker_calls = 0
        self.wallet_calls = 0
        self.asset_info_calls = 0
        if creds is not None:
            self.creds = creds

    def instruments_info(self, category="spot", symbol: str | None = None, **kwargs):
        self.info_calls += 1
        return self.instrument_payload

    def place_order(self, **kwargs):
        self.place_calls.append(kwargs)
        return {"ok": True, "body": kwargs}

    def orderbook(self, category="spot", symbol: str | None = None, limit: int = 50):
        return self.orderbook_payload

    def tickers(self, category="spot", symbol: str | None = None):
        self.ticker_calls += 1
        if isinstance(self.ticker_payload, Exception):
            raise self.ticker_payload
        if self.ticker_payload is None:
            mark = self._default_mark_price()
            requested = symbol or "BTCUSDT"
            return {"result": {"list": [{"symbol": requested, "markPrice": mark}]}}
        return self.ticker_payload

    def _default_mark_price(self) -> str:
        result = self.orderbook_payload.get("result") if isinstance(self.orderbook_payload, dict) else {}
        asks = (result.get("a") or []) if isinstance(result, dict) else []
        bids = (result.get("b") or []) if isinstance(result, dict) else []
        if asks and isinstance(asks[0], (list, tuple)) and asks[0]:
            return str(asks[0][0])
        if bids and isinstance(bids[0], (list, tuple)) and bids[0]:
            return str(bids[0][0])
        return "100.0"

    def wallet_balance(self, accountType="UNIFIED"):
        self.wallet_calls += 1
        if isinstance(self.wallet_payload, Exception):
            raise self.wallet_payload
        payload = self.wallet_payload
        if isinstance(payload, dict) and "result" not in payload:
            lookup_key = (accountType or "").upper()
            payload = payload.get(lookup_key, payload.get("default"))
        return payload

    def asset_exchange_query_asset_info(self):
        self.asset_info_calls += 1
        if isinstance(self.exchange_asset_payload, Exception):
            raise self.exchange_asset_payload
        return self.exchange_asset_payload or {}


def setup_function(_):
    spot_market_module._INSTRUMENT_CACHE.clear()
    spot_market_module._PRICE_CACHE.clear()
    spot_market_module._BALANCE_CACHE.clear()
    spot_market_module._SYMBOL_CACHE.clear()


def _universe_payload(entries: list[dict]) -> dict:
    return {"result": {"list": entries}}


def _basic_limits() -> dict:
    return {
        "min_order_amt": "5",
        "quote_step": "0.01",
        "min_order_qty": "0.00000001",
        "qty_step": "0.00000001",
        "quote_coin": "USDT",
        "base_coin": "BTC",
        "tick_size": "0.1",
    }


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
        spot_market_module._symbol_cache_key(api),
        {"symbols": {}, "by_base": {}, "aliases": {}, "ts": time.time()},
    )

    resolved, meta = resolve_trade_symbol("NEWUSDT", api=api)

    assert resolved == "NEWUSDT"
    assert meta["reason"] == "exact"
    assert meta["cache_state"] == "refreshed"


def test_tradable_universe_cache_scoped_by_network():
    payload_testnet = _universe_payload([
        {"symbol": "TNTUSDT", "quoteCoin": "USDT", "status": "Trading"},
    ])
    payload_mainnet = _universe_payload([
        {"symbol": "MAINUSDT", "quoteCoin": "USDT", "status": "Trading"},
    ])

    testnet_creds = type("Creds", (), {"testnet": True})()
    mainnet_creds = type("Creds", (), {"testnet": False})()

    testnet_api = DummyAPI(payload_testnet, creds=testnet_creds)
    mainnet_api = DummyAPI(payload_mainnet, creds=mainnet_creds)

    resolved_testnet, meta_testnet = resolve_trade_symbol("TNTUSDT", api=testnet_api)
    assert resolved_testnet == "TNTUSDT"
    assert meta_testnet["cache_state"] == "cached"

    resolved_mainnet, meta_mainnet = resolve_trade_symbol("MAINUSDT", api=mainnet_api)
    assert resolved_mainnet == "MAINUSDT"
    assert meta_mainnet["cache_state"] == "cached"

    # cached lookup should return without refetching but remain scoped per network
    cached_testnet, meta_testnet_cached = resolve_trade_symbol("TNTUSDT", api=testnet_api)
    assert cached_testnet == "TNTUSDT"
    assert meta_testnet_cached["cache_state"] == "cached"

    cached_mainnet, meta_mainnet_cached = resolve_trade_symbol("MAINUSDT", api=mainnet_api)
    assert cached_mainnet == "MAINUSDT"
    assert meta_mainnet_cached["cache_state"] == "cached"


def test_latest_price_cache_scoped_by_network():
    payload = _universe_payload([
        {"symbol": "BTCUSDT", "quoteCoin": "USDT", "status": "Trading"},
    ])

    testnet_creds = type("Creds", (), {"testnet": True})()
    mainnet_creds = type("Creds", (), {"testnet": False})()

    testnet_ticker = {"result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "101"}]}}
    mainnet_ticker = {"result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "202"}]}}

    testnet_api = DummyAPI(payload, ticker_payload=testnet_ticker, creds=testnet_creds)
    mainnet_api = DummyAPI(payload, ticker_payload=mainnet_ticker, creds=mainnet_creds)

    testnet_price = spot_market_module._latest_price(testnet_api, "BTCUSDT")
    assert testnet_price == Decimal("101")
    assert testnet_api.ticker_calls == 1

    mainnet_price = spot_market_module._latest_price(mainnet_api, "BTCUSDT")
    assert mainnet_price == Decimal("202")
    assert mainnet_api.ticker_calls == 1

    # Subsequent lookups should remain cached per network without extra API calls.
    repeat_testnet_price = spot_market_module._latest_price(testnet_api, "BTCUSDT")
    assert repeat_testnet_price == Decimal("101")
    assert testnet_api.ticker_calls == 1

    repeat_mainnet_price = spot_market_module._latest_price(mainnet_api, "BTCUSDT")
    assert repeat_mainnet_price == Decimal("202")
    assert mainnet_api.ticker_calls == 1


def test_instrument_limits_cache_scoped_by_network():
    def build_payload(min_amount: str, qty_step: str) -> dict:
        return {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "Trading",
                        "baseCoin": "BTC",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "minOrderAmt": min_amount,
                            "minOrderQty": "0.0001",
                            "qtyStep": qty_step,
                        },
                    }
                ]
            }
        }

    testnet_payload = build_payload("6", "0.0001")
    mainnet_payload = build_payload("11", "0.001")

    testnet_creds = type("Creds", (), {"testnet": True})()
    mainnet_creds = type("Creds", (), {"testnet": False})()

    testnet_api = DummyAPI(testnet_payload, creds=testnet_creds)
    mainnet_api = DummyAPI(mainnet_payload, creds=mainnet_creds)

    limits_testnet = spot_market_module._instrument_limits(testnet_api, "BTCUSDT")
    limits_mainnet = spot_market_module._instrument_limits(mainnet_api, "BTCUSDT")

    assert limits_testnet["min_order_amt"] == Decimal("6")
    assert limits_mainnet["min_order_amt"] == Decimal("11")

    # alternating requests should continue to return environment-specific limits
    limits_testnet_again = spot_market_module._instrument_limits(testnet_api, "BTCUSDT")
    limits_mainnet_again = spot_market_module._instrument_limits(mainnet_api, "BTCUSDT")

    assert limits_testnet_again["min_order_amt"] == Decimal("6")
    assert limits_mainnet_again["min_order_amt"] == Decimal("11")

    assert testnet_api.info_calls == 1
    assert mainnet_api.info_calls == 1


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


def test_wallet_available_balances_cache_scoped_by_network():
    def wallet_payload(amount: str) -> dict:
        return {
            "result": {
                "list": [
                    {
                        "coin": [
                            {
                                "coin": "USDT",
                                "availableBalance": amount,
                                "availableToWithdraw": amount,
                            }
                        ]
                    }
                ]
            }
        }

    def exchange_payload(amount: str) -> dict:
        return {
            "result": {
                "list": [
                    {
                        "coin": [
                            {
                                "coin": "USDT",
                                "availableBalance": amount,
                                "availableToWithdraw": amount,
                            }
                        ]
                    }
                ]
            }
        }

    testnet_creds = type("Creds", (), {"testnet": True})()
    mainnet_creds = type("Creds", (), {"testnet": False})()

    testnet_api = DummyAPI(
        {},
        wallet_payload={"UNIFIED": wallet_payload("0"), "SPOT": {}},
        creds=testnet_creds,
        exchange_asset_payload=exchange_payload("50"),
    )
    mainnet_api = DummyAPI(
        {},
        wallet_payload={"UNIFIED": wallet_payload("0"), "SPOT": {}},
        creds=mainnet_creds,
        exchange_asset_payload=exchange_payload("75"),
    )

    testnet_balances = spot_market_module._wallet_available_balances(
        testnet_api, account_type="UNIFIED"
    )
    assert testnet_balances["USDT"] == Decimal("50")

    mainnet_balances = spot_market_module._wallet_available_balances(
        mainnet_api, account_type="UNIFIED"
    )
    assert mainnet_balances["USDT"] == Decimal("75")

    # alternating requests should continue to serve the network-scoped snapshot
    testnet_api.exchange_asset_payload = exchange_payload("60")
    mainnet_api.exchange_asset_payload = exchange_payload("90")

    cached_testnet = spot_market_module._wallet_available_balances(
        testnet_api, account_type="UNIFIED"
    )
    cached_mainnet = spot_market_module._wallet_available_balances(
        mainnet_api, account_type="UNIFIED"
    )

    assert cached_testnet["USDT"] == Decimal("50")
    assert cached_mainnet["USDT"] == Decimal("75")


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
    orderbook = {"result": {"a": [["2000", "5"]], "b": [["1999", "5"]]}}
    api = DummyAPI(payload, ticker_payload=ticker, orderbook_payload=orderbook)

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
    assert placed["orderType"] == "Limit"
    assert placed["timeInForce"] == "GTC"
    audit = response.get("_local", {}).get("order_audit", {})
    assert audit.get("tolerance_value") == "0.7500"
    assert "slippageTolerance" not in placed


def test_place_spot_market_clamps_percent_tolerance_range():
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

    response_high = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        tol_value=8.0,
    )

    placed_high = api.place_calls[-1]
    assert placed_high["orderType"] == "Limit"
    audit_high = response_high.get("_local", {}).get("order_audit", {})
    assert audit_high.get("tolerance_value") == "5.0000"

    api.place_calls.clear()

    response_low = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        tol_value=0.01,
    )

    placed_low = api.place_calls[-1]
    assert placed_low["orderType"] == "Limit"
    audit_low = response_low.get("_local", {}).get("order_audit", {})
    assert audit_low.get("tolerance_value") == "0.0500"


def test_place_spot_market_clamps_bps_tolerance_range():
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

    response_high = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        tol_type="Bps",
        tol_value=800,
    )

    audit_high = response_high.get("_local", {}).get("order_audit", {})
    assert audit_high.get("tolerance_type") == "Bps"
    assert audit_high.get("tolerance_value") == "500.0000"

    api.place_calls.clear()

    response_low = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=10,
        unit="quoteCoin",
        tol_type="Bps",
        tol_value=1,
    )

    audit_low = response_low.get("_local", {}).get("order_audit", {})
    assert audit_low.get("tolerance_type") == "Bps"
    assert audit_low.get("tolerance_value") == "5.0000"


def test_place_spot_market_guard_reduces_qty_on_tolerance():
    payload = _universe_payload(
        [
            {
                "symbol": "BTCUSDT",
                "quoteCoin": "USDT",
                "status": "Trading",
                "lotSizeFilter": {
                    "minOrderAmt": "5",
                    "minOrderQty": "0.0001",
                    "qtyStep": "0.0001",
                },
                "priceFilter": {"tickSize": "0.1"},
            }
        ]
    )
    orderbook = {"result": {"a": [["100.03", "5"]], "b": [["99.5", "5"]]}}
    api = DummyAPI(payload, orderbook_payload=orderbook)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=100,
        unit="quoteCoin",
        tol_type="Percent",
        tol_value=0.0,
    )

    assert response["ok"] is True
    placed = api.place_calls[0]
    audit = response.get("_local", {}).get("order_audit", {})
    assert Decimal(placed["qty"]) == Decimal(audit.get("order_qty_base"))
    assert Decimal(audit.get("order_notional")) <= Decimal("100")


def test_prepare_spot_market_allows_price_within_mark_tolerance_bps():
    orderbook = {"result": {"a": [["104", "5"]], "b": [["99", "5"]]}}
    api = DummyAPI({}, orderbook_payload=orderbook)

    prepared = spot_market_module.prepare_spot_market_order(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=Decimal("100"),
        unit="quoteCoin",
        tol_type="Bps",
        tol_value=500,
        price_snapshot=Decimal("100"),
        balances={"USDT": Decimal("200")},
        limits=_basic_limits(),
    )

    assert prepared.payload["price"] == "104.0"
    audit = prepared.audit
    assert audit.get("price_used") == "100"
    assert audit.get("limit_price") == "104"


def test_prepare_spot_market_sorts_descending_asks_to_best_price():
    orderbook = {
        "result": {
            "a": [["110", "5"], ["100", "5"]],
            "b": [["99", "5"], ["98", "5"]],
        }
    }
    api = DummyAPI({}, orderbook_payload=orderbook)

    prepared = spot_market_module.prepare_spot_market_order(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=Decimal("100"),
        unit="quoteCoin",
        tol_type="Percent",
        tol_value=0,
        price_snapshot=Decimal("100"),
        balances={"USDT": Decimal("500")},
        limits=_basic_limits(),
    )

    audit = prepared.audit
    assert prepared.payload["price"] == "100.0"
    assert audit.get("limit_price") == "100"
    assert audit.get("price_used") == "100"
    consumed = audit.get("consumed_levels") or []
    assert consumed and consumed[0]["price"] == "100"


def test_prepare_spot_market_target_quote_min_notional_adjustment():
    orderbook = {"result": {"a": [["1.001", "20"]], "b": [["0.999", "20"]]}}
    limits = {
        "min_order_amt": "10",
        "quote_step": "0.01",
        "min_order_qty": "0",
        "qty_step": "0.00000001",
        "quote_coin": "USDT",
        "base_coin": "TEST",
        "tick_size": "0.1",
    }
    api = DummyAPI({}, orderbook_payload=orderbook)

    prepared = spot_market_module.prepare_spot_market_order(
        api,
        symbol="TESTUSDT",
        side="Buy",
        qty=Decimal("10"),
        unit="quoteCoin",
        tol_value=10,
        price_snapshot=Decimal("1.1"),
        limits=limits,
    )

    audit = prepared.audit
    assert audit.get("validator_ok") is True
    assert not audit.get("validator_reasons")
    assert Decimal(audit.get("limit_notional")) >= Decimal("10")
    assert Decimal(audit.get("order_qty_base")) > Decimal("0")


def test_buy_tick_and_validation_ceiling_respects_worst_ask():
    asks = [(Decimal("100.03"), Decimal("0.4"))]
    bids = [(Decimal("99.5"), Decimal("0.4"))]
    target_quote = Decimal("10")
    qty_step = Decimal("0.05")
    min_qty = Decimal("0.01")
    tick_size = Decimal("0.1")

    worst_price, qty_base, _, consumed = spot_market_module._plan_limit_ioc_order(
        asks=asks,
        bids=bids,
        side="buy",
        target_quote=target_quote,
        target_base=None,
        qty_step=qty_step,
        min_qty=min_qty,
    )

    limit_price = spot_market_module._apply_tick(worst_price, tick_size, "buy")

    instrument = {
        "priceFilter": {"tickSize": str(tick_size)},
        "lotSizeFilter": {
            "qtyStep": str(qty_step),
            "minOrderQty": str(min_qty),
            "minNotional": "0",
            "minOrderAmt": "0",
        },
    }

    qty_base_raw = target_quote / limit_price
    validated = validators.validate_spot_rules(
        instrument=instrument,
        price=limit_price,
        qty=qty_base_raw,
        side="buy",
    )

    total_consumed = sum(qty for _, qty in consumed)

    assert limit_price >= worst_price
    assert validated.qty >= total_consumed


def test_sell_tick_and_validation_floor_retained():
    asks = [(Decimal("101.2"), Decimal("0.4"))]
    bids = [(Decimal("99.97"), Decimal("0.4"))]
    target_base = Decimal("0.11")
    qty_step = Decimal("0.05")
    min_qty = Decimal("0.01")
    tick_size = Decimal("0.1")

    worst_price, qty_base, _, consumed = spot_market_module._plan_limit_ioc_order(
        asks=asks,
        bids=bids,
        side="sell",
        target_quote=None,
        target_base=target_base,
        qty_step=qty_step,
        min_qty=min_qty,
    )

    limit_price = spot_market_module._apply_tick(worst_price, tick_size, "sell")

    instrument = {
        "priceFilter": {"tickSize": str(tick_size)},
        "lotSizeFilter": {
            "qtyStep": str(qty_step),
            "minOrderQty": str(min_qty),
            "minNotional": "0",
            "minOrderAmt": "0",
        },
    }

    validated = validators.validate_spot_rules(
        instrument=instrument,
        price=limit_price,
        qty=target_base,
        side="sell",
    )

    total_consumed = sum(qty for _, qty in consumed)

    assert limit_price <= worst_price
    assert validated.qty <= total_consumed


def test_prepare_spot_market_blocks_price_outside_mark_tolerance():
    orderbook = {"result": {"a": [["104", "5"]], "b": [["99", "5"]]}}
    api = DummyAPI({}, orderbook_payload=orderbook)

    with pytest.raises(OrderValidationError) as excinfo:
        spot_market_module.prepare_spot_market_order(
            api,
            symbol="BTCUSDT",
            side="Buy",
            qty=Decimal("100"),
            unit="quoteCoin",
            tol_type="Bps",
            tol_value=100,
            price_snapshot=Decimal("100"),
            balances={"USDT": Decimal("200")},
            limits=_basic_limits(),
        )

    err = excinfo.value
    assert getattr(err, "code", None) == "price_deviation"
    details = getattr(err, "details", {}) or {}
    assert details.get("max_allowed") == "101"
    assert details.get("mark_price") == "100"


def test_place_spot_market_twap_splits_quantity_on_price_deviation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(twap_enabled=True, twap_slices=6)
    api = DummyAPI(_universe_payload([]))

    requested: list[Decimal] = []

    def fake_prepare(api_obj, symbol, side, chunk_qty, **kwargs):
        qty_decimal = Decimal(str(chunk_qty))
        requested.append(qty_decimal)
        if qty_decimal > Decimal("40"):
            raise OrderValidationError("too wide", code="price_deviation")
        payload = {
            "category": "spot",
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": format(qty_decimal, "f"),
            "price": "1",
            "timeInForce": "GTC",
            "orderFilter": "Order",
            "accountType": "UNIFIED",
            "marketUnit": "quoteCoin",
        }
        audit = {"quote_step": "0.01", "qty_step": "0.00000001"}
        return spot_market_module.PreparedSpotMarketOrder(payload=payload, audit=audit)

    place_payloads: list[dict] = []

    def fake_place_order(**payload):
        place_payloads.append(payload)
        qty_value = Decimal(str(payload.get("qty", "0")))
        return {
            "result": {
                "cumExecQty": format(qty_value, "f"),
                "cumExecValue": format(qty_value, "f"),
                "avgPrice": "1",
            }
        }

    monkeypatch.setattr(spot_market_module, "prepare_spot_market_order", fake_prepare)
    monkeypatch.setattr(api, "place_order", fake_place_order)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=Decimal("120"),
        unit="quoteCoin",
        settings=settings,
    )

    assert requested[0] == Decimal("120")
    # After TWAP activation requests are split evenly across six slices
    assert all(q == Decimal("20") for q in requested[1:])
    assert len(place_payloads) == 6
    assert all(payload.get("qty") == "20" for payload in place_payloads)

    local_meta = response.get("_local", {}) if isinstance(response, dict) else {}
    twap_meta = local_meta.get("twap", {})
    assert twap_meta.get("active") is True
    assert twap_meta.get("target_slices") == 6
    assert twap_meta.get("orders_sent") == 6
    adjustments = twap_meta.get("adjustments") or []
    assert adjustments and adjustments[0].get("action") == "activate"


def test_place_spot_market_twap_scales_slices_from_price_deviation_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(twap_enabled=True, twap_slices=2)
    api = DummyAPI(_universe_payload([]))

    requested: list[Decimal] = []

    def fake_prepare(api_obj, symbol, side, chunk_qty, **kwargs):
        qty_decimal = Decimal(str(chunk_qty))
        requested.append(qty_decimal)
        if qty_decimal > Decimal("15"):
            raise OrderValidationError(
                "too wide",
                code="price_deviation",
                details={"limit_price": "10", "max_allowed": "1"},
            )
        payload = {
            "category": "spot",
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": format(qty_decimal, "f"),
            "price": "1",
            "timeInForce": "GTC",
            "orderFilter": "Order",
            "accountType": "UNIFIED",
            "marketUnit": "quoteCoin",
        }
        audit = {"quote_step": "0.01", "qty_step": "0.00000001"}
        return spot_market_module.PreparedSpotMarketOrder(payload=payload, audit=audit)

    place_payloads: list[dict] = []

    def fake_place_order(**payload):
        place_payloads.append(payload)
        qty_value = Decimal(str(payload.get("qty", "0")))
        return {
            "result": {
                "cumExecQty": format(qty_value, "f"),
                "cumExecValue": format(qty_value, "f"),
                "avgPrice": "1",
            }
        }

    monkeypatch.setattr(spot_market_module, "prepare_spot_market_order", fake_prepare)
    monkeypatch.setattr(api, "place_order", fake_place_order)

    response = place_spot_market_with_tolerance(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=Decimal("120"),
        unit="quoteCoin",
        settings=settings,
    )

    assert requested[0] == Decimal("120")
    # The first reattempt uses the scaled slice count derived from deviation details.
    assert requested[1] == Decimal("12")
    assert len(place_payloads) == 10
    assert all(payload.get("qty") == "12" for payload in place_payloads)

    local_meta = response.get("_local", {}) if isinstance(response, dict) else {}
    twap_meta = local_meta.get("twap", {})
    assert twap_meta.get("active") is True
    assert twap_meta.get("target_slices") == 10
    adjustments = twap_meta.get("adjustments") or []
    assert adjustments and adjustments[0].get("action") == "activate"
    assert adjustments[0].get("target_slices") == 10
    assert adjustments[0].get("ratio") == "10"


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


def test_place_spot_market_uses_quote_market_unit_with_tick_gap():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "FOOUSDT",
                    "lotSizeFilter": {
                        "minOrderAmt": "5",
                        "minOrderAmtIncrement": "0.01",
                        "minOrderQty": "0.001",
                        "qtyStep": "0.001",
                    },
                    "priceFilter": {"tickSize": "1"},
                }
            ]
        }
    }
    ticker = {"result": {"list": [{"symbol": "FOOUSDT", "markPrice": "1"}]}}
    orderbook = {"result": {"a": [["0.95", "100"]], "b": [["0.90", "100"]]}}
    api = DummyAPI(payload, ticker_payload=ticker, orderbook_payload=orderbook)

    response = place_spot_market_with_tolerance(
        api,
        symbol="FOOUSDT",
        side="Buy",
        qty=Decimal("10"),
        unit="quoteCoin",
        tol_value=Decimal("0"),
    )

    assert response["ok"] is True
    assert api.place_calls
    placed = api.place_calls[0]
    assert "marketUnit" not in placed
    audit = response.get("_local", {}).get("order_audit", {})
    assert Decimal(placed["qty"]) == Decimal(audit.get("order_qty_base"))
    assert placed["qty"] == audit.get("qty_payload")
    assert Decimal(audit.get("limit_notional")) >= Decimal("9")


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


def test_load_wallet_balances_returns_empty_when_exchange_fallback_empty():
    payload = {}

    class SpotUnsupportedAPI(DummyAPI):
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
                    {"coin": [{"coin": "USDT", "availableBalance": "0"}]}
                ]
            }
        }
    }

    api = SpotUnsupportedAPI(payload, wallet_payload=wallet_payload, exchange_asset_payload={})

    balances = spot_market_module._load_wallet_balances(api, account_type="SPOT")

    assert balances == {}
    assert api.wallet_calls == 1
    assert api.asset_info_calls >= 1


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
    assert api.asset_info_calls >= 1


def test_place_spot_market_uses_exchange_asset_info_when_spot_wallet_rejected():
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

    class FallbackExchangeAPI(DummyAPI):
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
                    {"coin": [{"coin": "USDT", "availableBalance": "0"}]}
                ]
            }
        }
    }

    exchange_asset_payload = {
        "result": {
            "spot": [
                {
                    "details": [
                        {"coin": "USDT", "availableToWithdraw": "150", "walletBalance": "150"}
                    ]
                }
            ]
        }
    }

    api = FallbackExchangeAPI(
        payload,
        wallet_payload=wallet_payload,
        exchange_asset_payload=exchange_asset_payload,
    )

    response = place_spot_market_with_tolerance(
        api,
        symbol="BBSOLUSDT",
        side="Buy",
        qty=10.0,
        unit="quoteCoin",
    )

    assert response["ok"] is True
    audit = response.get("_local", {}).get("order_audit", {})
    assert Decimal(audit.get("order_notional")) > Decimal("0")
    assert api.wallet_calls == 2
    assert api.asset_info_calls >= 1


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
    assert api.asset_info_calls >= 1


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
    assert api.asset_info_calls >= 1


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
    assert api.asset_info_calls >= 1


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
    orderbook = {"result": {"a": [["15", "5"]], "b": [["14.9", "5"]]}}
    api = DummyAPI(
        {},
        ticker_payload=RuntimeError("ticker should not be called"),
        wallet_payload=RuntimeError("wallet should not be called"),
        orderbook_payload=orderbook,
    )

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


def test_place_spot_market_sell_rounds_qty_down_to_step():
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "SOLUSDT",
                    "quoteCoin": "USDT",
                    "baseCoin": "SOL",
                    "lotSizeFilter": {
                        "minOrderQty": "0.1",
                        "qtyStep": "0.1",
                        "minOrderAmt": "5",
                    },
                    "priceFilter": {"tickSize": "0.01"},
                }
            ]
        }
    }
    orderbook = {
        "result": {
            "a": [["20.10", "5"], ["20.20", "5"]],
            "b": [["20.00", "1"], ["19.90", "1"]],
        }
    }
    wallet = {
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "SOL", "availableBalance": "2"},
                        {"coin": "USDT", "availableBalance": "0"},
                    ]
                }
            ]
        }
    }
    ticker = {"result": {"list": [{"symbol": "SOLUSDT", "bestBidPrice": "20.0"}]}}
    api = DummyAPI(payload, ticker_payload=ticker, wallet_payload=wallet, orderbook_payload=orderbook)

    response = place_spot_market_with_tolerance(
        api,
        symbol="SOLUSDT",
        side="Sell",
        qty=Decimal("1.234"),
        unit="baseCoin",
    )

    assert response["ok"] is True
    placed = api.place_calls[0]
    assert placed["qty"] == "1.2"
    assert "marketUnit" not in placed


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
    orderbook = {"result": {"a": [["2.5", "10"]], "b": [["2.49", "10"]]}}
    api = DummyAPI(payload, ticker_payload=ticker, wallet_payload=wallet, orderbook_payload=orderbook)

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
