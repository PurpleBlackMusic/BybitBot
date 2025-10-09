import copy
import time
from decimal import Decimal
from typing import Optional

import pytest

import bybit_app.utils.signal_executor as signal_executor_module

from bybit_app.utils.envs import Settings
from bybit_app.utils.signal_executor import (
    AutomationLoop,
    ExecutionResult,
    SignalExecutor,
)
from bybit_app.utils.spot_market import OrderValidationError, SpotTradeSnapshot


@pytest.fixture(autouse=True)
def _stub_ws_autostart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "autostart",
        lambda include_private=True: (False, False),
    )


class StubBot:
    def __init__(
        self,
        summary: dict,
        settings: Settings | None = None,
        fingerprint: str | None = None,
    ) -> None:
        self._summary = summary
        self.settings = settings or Settings()
        self._fingerprint = fingerprint

    def status_summary(self) -> dict:
        return copy.deepcopy(self._summary)

    def status_fingerprint(self) -> str | None:
        return self._fingerprint


class StubAPI:
    def __init__(self, total: float = 0.0, available: float = 0.0) -> None:
        self._total = total
        self._available = available
        self.orders: list[dict[str, object]] = []

    def wallet_balance(self) -> dict:
        return {
            "result": {
                "list": [
                    {
                        "totalEquity": str(self._total),
                        "availableBalance": str(self._available),
                    }
                ]
            }
        }

    def place_order(self, **payload: object) -> dict[str, object]:
        self.orders.append(dict(payload))
        order_id = f"stub-{len(self.orders)}"
        return {
            "retCode": 0,
            "result": {
                "orderLinkId": payload.get("orderLinkId"),
                "orderId": order_id,
            },
        }

    def instruments_info(self, category: str = "spot", symbol: str | None = None, **_: object) -> dict:
        symbol_text = symbol or "BTCUSDT"
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "symbol": symbol_text,
                        "status": "Trading",
                        "baseCoin": symbol_text[:-4] if symbol_text.endswith("USDT") else "BTC",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "minOrderAmt": "5",
                            "minOrderQty": "0.0001",
                            "qtyStep": "0.0001",
                            "minOrderQtyIncrement": "0.0001",
                            "minOrderAmtIncrement": "0.01",
                        },
                        "priceFilter": {
                            "tickSize": "0.0001",
                            "minPrice": "0",
                            "maxPrice": "0",
                        },
                    }
                ]
            },
        }


def patch_tp_sources(
    monkeypatch: pytest.MonkeyPatch,
    filled: Decimal | None = None,
    reserved: Decimal = Decimal("0"),
) -> None:
    def fake_collect(self, symbol: str, **kwargs: object) -> Decimal:
        if filled is not None:
            return filled
        executed_base = kwargs.get("executed_base")
        if isinstance(executed_base, Decimal):
            return executed_base
        return Decimal("0")

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_filled_base_total",
        fake_collect,
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_open_sell_reserved",
        lambda self, symbol, rows=None: reserved,
    )


def test_signal_executor_skips_when_not_actionable() -> None:
    bot = StubBot({"actionable": False}, Settings(ai_enabled=True))
    executor = SignalExecutor(bot)
    result = executor.execute_once()
    assert isinstance(result, ExecutionResult)
    assert result.status == "skipped"


def test_signal_executor_auto_exit_sells_old_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    ledger_rows = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": "2000",
            "execQty": "1",
            "execFee": "0",
            "execTime": now - 7200,
        }
    ]

    summary = {"actionable": False, "mode": "wait", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=True,
        ai_max_hold_minutes=60,
        ai_min_exit_bps=-50,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "read_ledger",
        lambda n=2000, **_: list(ledger_rows),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        lambda api_obj, symbol, include_limits=False, include_balances=False, include_price=True: SpotTradeSnapshot(
            symbol=symbol,
            price=Decimal("1800"),
            balances=None,
            limits=None,
        ),
    )
    monkeypatch.setattr(signal_executor_module, "send_telegram", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api=None, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "status",
        lambda: {"private": {"age_seconds": 0, "running": True, "connected": True}},
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["side"] == "Sell"
    assert result.order["symbol"] == "ETHUSDT"
    assert result.context is not None
    auto_exit = result.context.get("auto_exit") if isinstance(result.context, dict) else None
    assert auto_exit is not None
    assert auto_exit.get("symbol") == "ETHUSDT"
    assert "pnl_limit" in auto_exit.get("reasons", [])


def test_signal_executor_respects_disabled_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    bot = StubBot(summary, Settings(ai_enabled=False))
    monkeypatch.setattr(
        signal_executor_module,
        "get_api_client",
        lambda: StubAPI(total=500.0, available=400.0),
    )
    executor = SignalExecutor(bot)
    result = executor.execute_once()
    assert result.status == "disabled"


def test_signal_executor_dry_run_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings)
    monkeypatch.setattr(
        signal_executor_module,
        "get_api_client",
        lambda: StubAPI(total=1000.0, available=800.0),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    executor = SignalExecutor(bot)
    result = executor.execute_once()
    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "ETHUSDT"
    assert result.order["side"] == "Buy"


def test_signal_executor_maps_usdc_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ADAUSDC"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=300.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)

    captured: dict[str, str] = {}

    def fake_resolve(symbol: str, api, allow_nearest: bool = True):
        captured["symbol"] = symbol
        return symbol, {"reason": "exact"}

    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        fake_resolve,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert captured["symbol"] == "ADAUSDT"
    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "ADAUSDT"


def test_signal_executor_places_market_order(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "actionable": True,
        "mode": "buy",
        "symbol": "BTCUSDT",
        "candidate_symbols": ["BTCUSDT", "ETHUSDT"],
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.5,
        spot_cash_reserve_pct=10.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict | None = None

    def fake_place(api_obj, **kwargs):
        nonlocal captured
        captured = kwargs
        return {"status": "ok", "result": {"orderId": "test"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured is not None
    assert captured["symbol"] == "BTCUSDT"
    assert captured["side"] == "Buy"
    assert captured["unit"] == "quoteCoin"
    assert pytest.approx(captured["qty"], rel=1e-3) == 15.0


def test_signal_executor_scales_buy_notional_for_slippage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=25.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
        ai_max_slippage_bps=500,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=120.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs):
        multiplier, _, _, _ = signal_executor_module._resolve_slippage_tolerance(
            "Percent", kwargs["tol_value"]
        )
        guard = kwargs.get("max_quote")
        qty = Decimal(str(kwargs["qty"]))
        multiplier_decimal = Decimal(str(multiplier))
        if guard is not None:
            guard_decimal = Decimal(str(guard))
            if qty * multiplier_decimal > guard_decimal + Decimal("1e-9"):
                raise OrderValidationError("max quote exceeded", code="max_quote")
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "scaled"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured

    usable_after_reserve = 120.0 - 500.0 * 0.02
    multiplier, _, _, _ = signal_executor_module._resolve_slippage_tolerance(
        "Percent", settings.ai_max_slippage_bps / 100.0
    )
    expected_qty = Decimal(str(usable_after_reserve)) / Decimal(str(multiplier))

    assert float(captured["qty"]) == pytest.approx(float(expected_qty), rel=1e-6)
    assert captured["max_quote"] == pytest.approx(usable_after_reserve)


def test_signal_executor_sell_ignores_max_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "ETHUSDT",
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=10.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=100.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict | None = None

    def fake_place(api_obj, **kwargs):
        nonlocal captured
        captured = dict(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-1"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured is not None
    assert captured["side"] == "Sell"
    assert captured.get("max_quote") is None
    assert captured["qty"] > 0


def test_signal_executor_sell_uses_balance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "BTCUSDT",
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=50.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=100.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"BTC": Decimal("0.001")},
        limits={"min_order_amt": Decimal("5"), "base_coin": "BTC"},
    )
    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        lambda api_obj, symbol, **_: snapshot,
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-fallback"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert captured["side"] == "Sell"
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 25.0

    assert result.context is not None
    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert pytest.approx(fallback.get("available_base"), rel=1e-9) == 0.001
    assert pytest.approx(fallback.get("quote_notional"), rel=1e-9) == 25.0
    assert pytest.approx(fallback.get("min_order_amt"), rel=1e-9) == 5.0


def test_signal_executor_places_tp_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="50,100",
        spot_tp_ladder_split_pct="60,40",
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.75"))

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.75",
            "cumExecValue": "75",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.01",
                "min_order_qty": "0.01",
                "quote_step": "0.01",
                "limit_price": "100.00",
            }
        },
    }

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 2
    first, second = api.orders
    assert first["side"] == "Sell"
    assert second["side"] == "Sell"
    assert first["orderType"] == "Limit"
    assert second["orderType"] == "Limit"
    assert Decimal(first["qty"]) == Decimal("0.45")
    assert Decimal(second["qty"]) == Decimal("0.30")
    assert Decimal(first["price"]) == Decimal("100.5")
    assert Decimal(second["price"]) == Decimal("101")
    assert result.order is not None
    assert result.order.get("take_profit_orders")
    assert result.context is not None
    execution_payload = result.context.get("execution", {})
    assert Decimal(execution_payload.get("avg_price")) == Decimal("100")


def test_signal_executor_coalesces_tp_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="5,9",
        spot_tp_ladder_split_pct="50,50",
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1500.0, available=1200.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch)

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "1",
            "cumExecValue": "100",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.1",
                "min_order_qty": "0.1",
                "quote_step": "0.01",
                "price_payload": "0.1",
            }
        },
    }

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 1
    order = api.orders[0]
    assert order["qty"] == "1.0"
    assert order["price"] == "100.1"
    assert result.order is not None
    ladder = result.order.get("take_profit_orders")
    assert ladder is not None
    assert ladder[0]["profit_bps"] == "5,9"
    assert ladder[0]["orderId"] == "stub-1"
    assert result.order["execution"]["sell_budget_base"] == "1.0"


def test_signal_executor_tp_ladder_respects_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="50,100",
        spot_tp_ladder_split_pct="60,40",
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.5"), reserved=Decimal("0.5"))

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.5",
            "cumExecValue": "50",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.01",
                "min_order_qty": "0.01",
                "quote_step": "0.01",
                "limit_price": "100.00",
            }
        },
    }

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert api.orders == []
    assert result.order is not None
    execution = result.order.get("execution")
    assert execution is not None
    assert Decimal(execution.get("sell_budget_base")) == Decimal("0")
    assert Decimal(execution.get("open_sell_reserved")) == Decimal("0.5")
    assert Decimal(execution.get("filled_base_total")) == Decimal("0.5")


def test_signal_executor_tp_ladder_falls_back_to_execution_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="50,100",
        spot_tp_ladder_split_pct="60,40",
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_filled_base_from_private_ws",
        lambda self, symbol, **_: Decimal("0"),
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_filled_base_from_ledger",
        lambda self, symbol, **_: Decimal("0"),
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_open_sell_reserved",
        lambda self, symbol, rows=None: Decimal("0"),
    )
    monkeypatch.setattr(signal_executor_module, "read_ledger", lambda n=2000, **_: [])
    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", lambda events=None: {})
    monkeypatch.setattr(signal_executor_module, "send_telegram", lambda *args, **kwargs: None)

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.75",
            "cumExecValue": "75",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.01",
                "min_order_qty": "0.01",
                "quote_step": "0.01",
                "limit_price": "100.00",
            }
        },
    }

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 2
    first, second = api.orders
    assert Decimal(first["qty"]) == Decimal("0.45")
    assert Decimal(second["qty"]) == Decimal("0.30")


def test_signal_executor_tp_ladder_respects_price_band(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="50",
        spot_tp_ladder_split_pct="100",
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=2000.0, available=1500.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.2"))

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.2",
            "cumExecValue": "20",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.1",
                "min_order_qty": "0.1",
                "quote_step": "0.01",
                "limit_price": "100.00",
            }
        },
    }

    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda _api, _symbol: {
            "qty_step": Decimal("0.1"),
            "min_order_qty": Decimal("0.1"),
            "quote_step": Decimal("0.01"),
            "tick_size": Decimal("0.5"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("105"),
            "max_price": Decimal("0"),
            "base_coin": "BTC",
            "quote_coin": "USDT",
        },
    )

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 1
    order = api.orders[0]
    assert Decimal(order["price"]) == Decimal("105")
    assert Decimal(order["qty"]) == Decimal("0.2")


def test_signal_executor_open_sell_reserved_prefers_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = SignalExecutor(StubBot({"actionable": False}, Settings(ai_enabled=True)))

    rows = [
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Limit",
            "orderStatus": "New",
            "orderId": "order-1",
            "qty": "0.5",
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Limit",
            "orderStatus": "New",
            "orderId": "order-1",
            "leavesQty": "0.2",
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Limit",
            "orderStatus": "New",
            "orderId": "order-2",
            "qty": "0.1",
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Market",
            "orderStatus": "New",
            "orderId": "order-3",
            "qty": "0.3",
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderType": "Limit",
            "orderStatus": "Cancelled",
            "orderId": "order-4",
            "qty": "0.4",
        },
    ]

    def fake_rows(self, topic_keyword: str):
        assert topic_keyword == "order"
        return rows

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_realtime_private_rows",
        fake_rows,
    )

    reserved = executor._resolve_open_sell_reserved("BTCUSDT")
    assert reserved == Decimal("0.3")


def test_signal_executor_tp_ladder_uses_local_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=2000.0, available=1500.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.75"))

    response_payload = {
        "retCode": 0,
        "result": {"orderId": "primary"},
        "_local": {
            "order_audit": {
                "qty_step": "0.01",
                "min_order_qty": "0.01",
                "quote_step": "0.01",
                "limit_price": "100.00",
            },
            "attempts": [
                {"executed_base": "0.50", "executed_quote": "50"},
                {"executed_base": "0.25", "executed_quote": "25"},
            ],
        },
    }

    def fake_place(api_obj, **kwargs):
        return response_payload

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 3
    qtys = [Decimal(order["qty"]) for order in api.orders]
    prices = [Decimal(order["price"]) for order in api.orders]
    assert qtys == [Decimal("0.37"), Decimal("0.22"), Decimal("0.16")]
    assert prices == [Decimal("100.35"), Decimal("100.7"), Decimal("101.1")]
    assert result.order is not None
    execution = result.order.get("execution")
    assert execution is not None
    assert execution.get("executed_base") == "0.75"
    assert Decimal(execution.get("avg_price")) == Decimal("100")
    assert execution.get("sell_budget_base") == "0.75"
    assert execution.get("filled_base_total") == "0.75"

def test_signal_executor_allows_zero_slippage(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_max_slippage_bps=0,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict | None = None

    def fake_place(api_obj, **kwargs):
        nonlocal captured
        captured = kwargs
        return {"status": "ok", "result": {"orderId": "zero-slip"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured is not None
    assert captured["tol_value"] == 0.0
    assert captured["tol_type"] == "Percent"


def test_signal_executor_clamps_percent_slippage(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_max_slippage_bps=800,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict | None = None

    def fake_place(api_obj, **kwargs):
        nonlocal captured
        captured = kwargs
        return {"status": "ok", "result": {"orderId": "clamped-slip"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured is not None
    assert captured["tol_type"] == "Percent"
    assert captured["tol_value"] == pytest.approx(5.0)


def test_signal_executor_prefers_trade_candidate_over_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "buy",
        "trade_candidates": [
            {"symbol": "ADAUSDT", "actionable": True, "holding": True, "priority": 1},
            {
                "symbol": "SOLUSDT",
                "actionable": True,
                "holding": False,
                "priority": 2,
                "edge_score": 4.2,
            },
        ],
        "candidate_symbols": ["ADAUSDT", "SOLUSDT"],
    }
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=400.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    autostart_calls: dict[str, object] = {}

    def record_autostart(include_private: bool = True):
        autostart_calls["include_private"] = include_private
        return True, True

    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "autostart",
        record_autostart,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "SOLUSDT"
    meta = result.order.get("symbol_meta")
    assert meta is not None
    candidate_meta = meta.get("candidate") if isinstance(meta, dict) else None
    assert isinstance(candidate_meta, dict)
    assert candidate_meta.get("source") == "trade_candidates"
    assert autostart_calls.get("include_private") is True


def test_signal_executor_sends_telegram_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        telegram_notify=True,
        tg_trade_notifs=True,
        tg_trade_notifs_min_notional=10.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    response_payload = {
        "ok": True,
        "result": {
            "orderId": "filled-order",
            "cumExecQty": "0.5",
            "cumExecValue": "50.5",
            "avgPrice": "101.0",
        },
        "_local": {
            "order_audit": {
                "qty_step": "0.0001",
                "price_payload": "101.0",
                "limit_price": "101.0",
            },
            "order_payload": {
                "qty": "0.5",
                "price": "101.0",
            },
        },
    }

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: response_payload,
    )

    def fake_tp(self, api_obj, settings_obj, symbol, side, response, **kwargs):
        return (
            [{"price": "105.0", "qty": "0.3", "profit_bps": "35"}],
            {
                "executed_base": "0.5",
                "executed_quote": "50.5",
                "avg_price": "101.0",
                "sell_budget_base": "0.3",
            },
        )

    monkeypatch.setattr(SignalExecutor, "_place_tp_ladder", fake_tp)

    ledger_rows = [
        {"symbol": "BTCUSDT", "category": "spot", "side": "Buy", "execPrice": 100.0, "execQty": 0.2, "execFee": 0.0},
        {"symbol": "BTCUSDT", "category": "spot", "side": "Sell", "execPrice": 120.0, "execQty": 0.1, "execFee": 0.0},
    ]
    monkeypatch.setattr(signal_executor_module, "read_ledger", lambda n=2000, **_: ledger_rows)

    captured: dict[str, str] = {}

    def fake_send(text: str):
        captured["text"] = text
        return {"ok": True}

    monkeypatch.setattr(signal_executor_module, "send_telegram", fake_send)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert "text" in captured
    message = captured["text"]
    assert message.startswith("куплено 0.5000 BTC по 101.00000000")
    assert "цель 105.00000000" in message
    assert "продано 0.3000" in message
    assert "PnL" in message and "+2.00" in message


def test_signal_executor_uses_execution_stats_for_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        telegram_notify=True,
        tg_trade_notifs=True,
        tg_trade_notifs_min_notional=5.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    response_payload = {
        "ok": True,
        "result": {"orderId": "delayed-fill"},
        "_local": {
            "order_audit": {
                "qty_step": "0.001",
                "price_payload": "50.0",
            },
            "order_payload": {"qty": "0.25", "price": "50.0"},
        },
    }

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: response_payload,
    )

    exec_stats = {
        "executed_base": "0.25",
        "executed_quote": "0",
        "avg_price": "50.0",
        "sell_budget_base": "0.25",
        "filled_base_total": "0.25",
    }

    monkeypatch.setattr(SignalExecutor, "_place_tp_ladder", lambda *args, **kwargs: ([], exec_stats))
    monkeypatch.setattr(signal_executor_module, "read_ledger", lambda n=2000, **_: [])
    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", lambda events: {})

    captured: dict[str, str] = {}

    def fake_send(text: str):
        captured["text"] = text
        return {"ok": True}

    monkeypatch.setattr(signal_executor_module, "send_telegram", fake_send)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert "text" in captured
    message = captured["text"]
    assert message.startswith("куплено 0.250 ETH по 50.00000000") or message.startswith("куплено 0.250 ET")
    assert "по 50" in message

def test_signal_executor_skips_on_min_notional(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_place(*_args, **_kwargs):
        raise OrderValidationError("Минимальный объём ордера не достигнут.", code="min_notional")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None and "min_notional" in result.reason


def test_signal_executor_notifies_on_price_deviation(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        telegram_notify=True,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_place(*_args, **_kwargs):
        raise OrderValidationError("слишком сильное отклонение цены", code="price_deviation")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    captured: list[str] = []

    def fake_send(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(signal_executor_module, "send_telegram", fake_send)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert captured, "telegram notification should be sent"
    assert "price_deviation" in captured[0]
    assert "ETHUSDT" in captured[0]


def test_signal_executor_does_not_notify_validation_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        telegram_notify=False,
        tg_trade_notifs=False,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_place(*_args, **_kwargs):
        raise OrderValidationError("слишком сильное отклонение цены", code="price_deviation")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    called = {"value": False}

    def fake_send(_text: str) -> None:
        called["value"] = True

    monkeypatch.setattr(signal_executor_module, "send_telegram", fake_send)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert called["value"] is False


def test_signal_executor_scales_position_with_signal_strength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "buy",
        "symbol": "ETHUSDT",
        "probability": 0.55,
        "ev_bps": 13.0,
        "primary_watch": {"symbol": "ETHUSDT", "edge_score": 0.5},
        "thresholds": {"min_ev_bps": 12.0},
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=True,
        ai_buy_threshold=0.6,
        ai_min_ev_bps=12.0,
        ai_risk_per_trade_pct=2.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=600.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "ETHUSDT"
    assert result.order["notional_quote"] < 20.0
    assert result.order["notional_quote"] == pytest.approx(6.62, rel=1e-3)


def test_signal_executor_pauses_when_private_ws_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=True,
        ws_watchdog_max_age_sec=10.0,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    class DummyManager:
        def status(self) -> dict:
            return {
                "private": {
                    "running": True,
                    "connected": True,
                    "age_seconds": 42.0,
                }
            }

    dummy_manager = DummyManager()
    monkeypatch.setattr(signal_executor_module, "ws_manager", dummy_manager)

    api_called = {"value": False}

    def fake_api() -> StubAPI:
        api_called["value"] = True
        return StubAPI(total=1000.0, available=900.0)

    monkeypatch.setattr(signal_executor_module, "get_api_client", fake_api)
    logged_events: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **payload: object) -> None:
        logged_events.append((event, dict(payload)))

    monkeypatch.setattr(signal_executor_module, "log", fake_log)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "disabled"
    assert result.reason is not None and "WebSocket" in result.reason
    assert api_called["value"] is False
    assert all(event != "guardian.auto.ws.autostart.error" for event, _ in logged_events)


def test_automation_loop_skips_repeated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-1")

    monkeypatch.setattr(
        signal_executor_module,
        "get_api_client",
        lambda: StubAPI(total=1000.0, available=900.0),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    executor = SignalExecutor(bot)

    call_count = {"value": 0}

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        return ExecutionResult(status="dry_run")

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(executor, poll_interval=0.0, success_cooldown=0.0)

    first_delay = loop._tick()
    assert call_count["value"] == 1
    assert first_delay == 0.0

    second_delay = loop._tick()
    assert call_count["value"] == 1
    assert second_delay == 0.0

    bot._fingerprint = "sig-2"
    third_delay = loop._tick()
    assert call_count["value"] == 2
    assert third_delay == 0.0


def test_automation_loop_retries_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-err")

    monkeypatch.setattr(
        signal_executor_module,
        "get_api_client",
        lambda: StubAPI(total=1000.0, available=900.0),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    executor = SignalExecutor(bot)

    call_count = {"value": 0}

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        return ExecutionResult(status="error", reason="network")

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(executor, poll_interval=0.0, success_cooldown=1.0, error_backoff=0.0)

    loop._tick()
    loop._tick()
    assert call_count["value"] == 2


def test_automation_loop_caches_skipped_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": False, "mode": "wait", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-skip")

    executor = SignalExecutor(bot)

    call_count = {"value": 0}

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        return ExecutionResult(status="skipped", reason="not actionable")

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(
        executor,
        poll_interval=3.5,
        success_cooldown=1.0,
        error_backoff=0.0,
    )

    first_delay = loop._tick()
    assert call_count["value"] == 1
    assert first_delay == 1.0

    second_delay = loop._tick()
    assert call_count["value"] == 1
    assert second_delay == 3.5

    bot._fingerprint = "sig-skip-new"
    third_delay = loop._tick()
    assert call_count["value"] == 2
    assert third_delay == 1.0


def test_automation_loop_reacts_to_ai_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=False, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-toggle")

    executor = SignalExecutor(bot)

    call_count = {"value": 0}

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        if bot.settings.ai_enabled:
            return ExecutionResult(status="dry_run")
        return ExecutionResult(status="disabled", reason="AI disabled")

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(
        executor,
        poll_interval=2.0,
        success_cooldown=1.5,
        error_backoff=0.0,
    )

    first_delay = loop._tick()
    assert call_count["value"] == 1
    assert first_delay == 1.5

    second_delay = loop._tick()
    assert call_count["value"] == 1
    assert second_delay == 2.0

    bot.settings.ai_enabled = True
    third_delay = loop._tick()
    assert call_count["value"] == 2
    assert third_delay == 1.5


def test_automation_loop_emits_results_via_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-callback")

    monkeypatch.setattr(
        signal_executor_module,
        "get_api_client",
        lambda: StubAPI(total=1000.0, available=900.0),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    executor = SignalExecutor(bot)
    captured: list[tuple[str, Optional[str], tuple[bool, bool, bool]]] = []

    def on_cycle(result: ExecutionResult, signature, marker) -> None:
        captured.append((result.status, signature, marker))

    loop = AutomationLoop(
        executor,
        poll_interval=0.0,
        success_cooldown=0.0,
        error_backoff=0.0,
        on_cycle=on_cycle,
    )

    delay = loop._tick()
    assert delay == 0.0
    assert captured
    status, signature, marker = captured[-1]
    assert status == "dry_run"
    assert signature == "sig-callback"
    assert isinstance(marker, tuple)
    assert loop._last_result is not None
    assert loop._last_result.status == "dry_run"

