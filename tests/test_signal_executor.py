import copy
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple, Union

import time

import pytest

import bybit_app.utils.pnl as pnl_module
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

    def portfolio_overview(self) -> dict:
        return {}

    def trade_statistics(self, limit: Optional[int] = None) -> dict:
        return {}


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


def test_signal_executor_force_exit_uses_trade_stats_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": False, "mode": "buy", "symbol": "ADAUSDT"}
    settings = Settings(ai_enabled=True, ai_max_hold_minutes=0.0, ai_min_exit_bps=None)
    bot = StubBot(summary, settings)

    stats_calls: dict[str, int] = {}

    def fake_trade_statistics(limit: Optional[int] = None) -> dict:
        stats_calls["count"] = stats_calls.get("count", 0) + 1
        return {
            "auto_exit_defaults": {
                "hold_minutes": 15.0,
                "hold_sample_count": 12,
                "exit_bps": -20.0,
                "bps_sample_count": 12,
            }
        }

    monkeypatch.setattr(bot, "trade_statistics", fake_trade_statistics)

    positions = {
        "ADAUSDT": {
            "qty": 10.0,
            "avg_cost": 1.0,
            "realized_pnl": 0.0,
            "hold_seconds": 1200.0,
            "price": 0.95,
            "pnl_value": -0.5,
            "pnl_bps": -5.0,
            "quote_notional": 9.5,
        }
    }

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_open_positions",
        lambda self, settings, summary: positions,
    )

    events: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **payload: object) -> None:
        events.append((event, dict(payload)))

    monkeypatch.setattr(signal_executor_module, "log", fake_log)

    executor = SignalExecutor(bot)
    forced_summary, metadata = executor._maybe_force_exit(summary, settings)

    assert stats_calls.get("count", 0) >= 1
    assert forced_summary is not None
    assert forced_summary["mode"] == "sell"
    assert forced_summary["actionable"] is True
    assert forced_summary["symbol"] == "ADAUSDT"
    assert metadata is not None
    assert metadata.get("hold_threshold_minutes") == pytest.approx(15.0)
    triggers = metadata.get("triggers") or []
    assert any(trigger.get("type") == "hold_time" for trigger in triggers)
    assert any(event == "guardian.auto.force_exit.defaults" for event, _ in events)


def test_signal_executor_force_exit_skips_positive_exit_bps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": False, "mode": "buy", "symbol": "ADAUSDT"}
    settings = Settings(ai_enabled=True, ai_max_hold_minutes=0.0, ai_min_exit_bps=None)
    bot = StubBot(summary, settings)

    def fake_trade_statistics(limit: Optional[int] = None) -> dict:
        return {
            "auto_exit_defaults": {
                "hold_minutes": 15.0,
                "hold_sample_count": 12,
                "exit_bps": 15.0,
                "bps_sample_count": 12,
            }
        }

    monkeypatch.setattr(bot, "trade_statistics", fake_trade_statistics)

    positions = {
        "ADAUSDT": {
            "qty": 10.0,
            "avg_cost": 1.0,
            "realized_pnl": 0.0,
            "hold_seconds": 60.0,
            "price": 1.02,
            "pnl_value": 0.2,
            "pnl_bps": 5.0,
            "quote_notional": 10.2,
        }
    }

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_open_positions",
        lambda self, settings, summary: positions,
    )

    executor = SignalExecutor(bot)
    forced_summary, metadata = executor._maybe_force_exit(summary, settings)

    assert forced_summary is None
    assert metadata is None


def test_signal_executor_force_exit_ignores_stale_summary_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    entry_ts = now - 10.0

    summary = {
        "actionable": False,
        "mode": "buy",
        "symbol": "BTCUSDT",
        "price": 80.0,
        "status": {"updated_ts": entry_ts - 300.0},
        "age_seconds": 600.0,
    }

    settings = Settings(
        ai_enabled=True,
        ai_max_hold_minutes=60.0,
        ai_min_exit_bps=-10.0,
        ai_max_slippage_bps=200,
    )

    portfolio_snapshot = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "qty": 2.0,
                "avg_cost": 100.0,
                "last_price": 100.0,
                "updated_ts": entry_ts + 1.0,
            }
        ]
    }

    bot = StubBot(summary, settings)
    bot.portfolio_overview = lambda: copy.deepcopy(portfolio_snapshot)

    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execQty": "2",
            "execPrice": "100",
            "execFee": "0",
            "execTime": entry_ts,
        }
    ]

    monkeypatch.setattr(
        signal_executor_module,
        "read_ledger",
        lambda limit, settings=None: list(events),
    )

    def fake_inventory(*, events=None, settings=None, **_):
        return {"BTCUSDT": {"position_qty": 2.0, "avg_cost": 100.0, "realized_pnl": 0.0}}

    monkeypatch.setattr(
        signal_executor_module,
        "spot_inventory_and_pnl",
        fake_inventory,
    )

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_current_time",
        lambda self: now,
    )

    executor = SignalExecutor(bot)

    positions = executor._collect_open_positions(settings, summary)
    assert "BTCUSDT" in positions
    btc_position = positions["BTCUSDT"]
    assert btc_position["price_stale"] is True
    assert btc_position["price_source"] in {"portfolio", "execution", "avg_cost"}
    assert btc_position["price"] == pytest.approx(100.0)
    assert btc_position["pnl_bps"] == pytest.approx(0.0)

    forced_summary, metadata = executor._maybe_force_exit(summary, settings)

    assert forced_summary is None
    assert metadata is None


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


def test_signal_executor_guard_forces_sell_on_time_and_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "25000",
            "execQty": "0.01",
            "execFee": "0.0",
            "execTime": now - 7200,
        }
    ]

    def fake_read_ledger(
        n: int = 5000, *, settings: object | None = None, network: object | None = None
    ) -> list[dict[str, object]]:
        return list(events)

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)

    sent_messages: list[str] = []

    def fake_send(text: str):
        sent_messages.append(text)
        return {"ok": True}

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

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

    summary = {
        "actionable": False,
        "mode": "wait",
        "symbol": "BTCUSDT",
        "prices": {"BTCUSDT": 24000.0},
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=True,
        ai_max_hold_minutes=30.0,
        ai_min_exit_bps=-50.0,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=0.0,
    )

    class ForceExitBot(StubBot):
        def portfolio_overview(self) -> dict:
            return {
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "qty": 0.01,
                        "avg_cost": 25000.0,
                        "realized_pnl": 0.0,
                    }
                ]
            }

    bot = ForceExitBot(summary, settings)
    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["side"] == "Sell"
    assert result.order["symbol"] == "BTCUSDT"
    assert pytest.approx(result.order["notional_quote"], rel=1e-3) == 240.0

    assert result.context is not None
    forced = result.context.get("forced_exit")
    assert isinstance(forced, dict)
    assert forced.get("symbol") == "BTCUSDT"
    assert forced.get("pnl_bps") is not None and forced["pnl_bps"] < 0
    assert forced.get("hold_minutes") is not None and forced["hold_minutes"] >= 30.0
    assert any(trigger.get("type") == "pnl" for trigger in forced.get("triggers", []))
    assert any(trigger.get("type") == "hold_time" for trigger in forced.get("triggers", []))

    assert sent_messages and "BTCUSDT" in sent_messages[0]


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
    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        lambda *args, **kwargs: None,
    )

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

    def fake_rows(topic_keyword: str, *, snapshot=None):
        assert topic_keyword == "order"
        assert snapshot is None
        return rows

    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "realtime_private_rows",
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

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert "text" in captured
    message = captured["text"]
    assert message.startswith("🟢 BTCUSDT: открытие 0.5000 BTC по 101.00000000")
    assert "(цели: 105.00000000)" in message
    assert "PnL" not in message


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

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert "text" in captured
    message = captured["text"]
    assert message.startswith("🟢 ETHUSDT: открытие 0.250 ETH по 50.00000000")
    assert "(цели: -)" in message


def test_signal_executor_sell_notification_reports_realized_pnl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "sell", "symbol": "ETHUSDT"}
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
        "result": {
            "orderId": "sell-filled",
            "cumExecQty": "0.4",
            "cumExecValue": "48.0",
            "avgPrice": "120.0",
        },
        "_local": {
            "order_audit": {"qty_step": "0.0001", "limit_price": "120.0"},
            "order_payload": {"qty": "0.4", "price": "120.0"},
        },
    }

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: response_payload,
    )

    before_rows = [
        {
            "symbol": "ETHUSDT",
            "category": "spot",
            "side": "Buy",
            "execPrice": "100.0",
            "execQty": "0.4",
            "execFee": "0.0",
            "marker": "before",
        }
    ]
    after_rows = before_rows + [
        {
            "symbol": "ETHUSDT",
            "category": "spot",
            "side": "Sell",
            "execPrice": "120.0",
            "execQty": "0.4",
            "execFee": "0.0",
            "marker": "after",
        }
    ]

    snapshots = [before_rows, after_rows]

    def fake_snapshot(self, limit: int = 2000, *, settings: Settings | None = None):
        if snapshots:
            return [dict(entry) for entry in snapshots.pop(0)]
        return [dict(entry) for entry in after_rows]

    monkeypatch.setattr(SignalExecutor, "_ledger_rows_snapshot", fake_snapshot)

    helper_calls: list[tuple[int, int]] = []

    def fake_realized_delta(
        self,
        rows_before,
        rows_after,
        symbol,
        *,
        new_rows=None,
    ):
        helper_calls.append(
            (
                len(rows_before or []),
                0 if new_rows is None else len(new_rows),
            )
        )
        return Decimal("4")

    monkeypatch.setattr(SignalExecutor, "_realized_delta", fake_realized_delta)

    captured: dict[str, str] = {}

    def fake_send(text: str):
        captured["text"] = text
        return {"ok": True}

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert "text" in captured
    message = captured["text"]
    assert message.startswith("🔴 ETHUSDT: закрытие 0.4000 ETH по 120.00000000")
    assert "PnL сделки +4.00 USDT" in message
    assert helper_calls == [(len(before_rows), 1)]

def test_signal_executor_incremental_pnl_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        telegram_notify=True,
        tg_trade_notifs=True,
        tg_trade_notifs_min_notional=0.0,
    )
    bot = StubBot({"actionable": True, "mode": "sell", "symbol": "BTCUSDT"}, settings)
    executor = SignalExecutor(bot)

    filler_rows = [
        {
            "symbol": "ETHUSDT",
            "category": "spot",
            "side": "Buy",
            "execPrice": 10.0 + (idx % 3),
            "execQty": 0.5,
            "execFee": 0.0,
        }
        for idx in range(300)
    ]
    btc_buys = [
        {
            "symbol": "BTCUSDT",
            "category": "spot",
            "side": "Buy",
            "execPrice": 100.0,
            "execQty": 0.1,
            "execFee": 0.0,
        }
        for _ in range(10)
    ]
    existing_sell = {
        "symbol": "BTCUSDT",
        "category": "spot",
        "side": "Sell",
        "execPrice": 110.0,
        "execQty": 0.2,
        "execFee": 0.0,
    }
    ledger_rows_before = filler_rows + btc_buys + [existing_sell]
    new_sell = {
        "symbol": "BTCUSDT",
        "category": "spot",
        "side": "Sell",
        "execPrice": 120.0,
        "execQty": 0.3,
        "execFee": 0.0,
    }
    ledger_rows_after = list(ledger_rows_before) + [new_sell]

    response_payload = {
        "result": {
            "orderId": "sell-order",
            "cumExecQty": "0.3",
            "cumExecValue": "36",
            "avgPrice": "120",
        }
    }
    execution_stats = {
        "executed_base": "0.3",
        "executed_quote": "36",
        "avg_price": "120",
    }
    audit_payload = {"qty_step": "0.01", "limit_price": "120"}

    messages: list[str] = []

    def fake_send(message: str) -> None:
        messages.append(message)

    monkeypatch.setattr(signal_executor_module, "enqueue_telegram_message", fake_send)

    original_realized = SignalExecutor._realized_delta
    helper_calls: list[Decimal] = []
    helper_new_rows: list[int] = []

    def tracking_realized(
        self,
        rows_before,
        rows_after,
        symbol,
        *,
        new_rows=None,
    ):
        helper_new_rows.append(0 if new_rows is None else len(new_rows))
        result = original_realized(
            self,
            rows_before,
            rows_after,
            symbol,
            new_rows=new_rows,
        )
        helper_calls.append(result)
        return result

    monkeypatch.setattr(SignalExecutor, "_realized_delta", tracking_realized)

    executor._maybe_notify_trade(
        settings=settings,
        symbol="BTCUSDT",
        side="sell",
        response=response_payload,
        ladder_orders=None,
        execution_stats=execution_stats,
        audit=audit_payload,
        ledger_rows_before=ledger_rows_before,
        ledger_rows_after=ledger_rows_after,
    )

    assert helper_calls, "helper should be invoked"
    assert helper_new_rows == [1]
    assert helper_calls[0].quantize(Decimal("0.01")) == Decimal("6.00")

    assert messages, "notification should be sent"
    assert "+6.00 USDT" in messages[0]

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

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

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

    monkeypatch.setattr(
        signal_executor_module,
        "enqueue_telegram_message",
        fake_send,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert called["value"] is False


def test_signal_executor_rotates_symbol_after_repeated_price_deviation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "buy",
        "trade_candidates": [
            {"symbol": "ADAUSDT", "actionable": True},
            {"symbol": "XRPUSDT", "actionable": True},
        ],
    }
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

    attempts: list[str] = []
    error_counter = {"value": 0}

    def fake_place(api, symbol, **kwargs):
        attempts.append(symbol)
        if symbol == "ADAUSDT" and error_counter["value"] < 2:
            error_counter["value"] += 1
            raise OrderValidationError(
                "слишком сильное отклонение цены", code="price_deviation"
            )
        return {
            "retCode": 0,
            "result": {"orderId": f"ok-{len(attempts)}"},
            "_local": {"order_audit": {}},
        }

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )
    monkeypatch.setattr(signal_executor_module, "read_ledger", lambda n=2000, **_: [])
    monkeypatch.setattr(
        signal_executor_module, "spot_inventory_and_pnl", lambda events: {}
    )
    monkeypatch.setattr(
        SignalExecutor, "_place_tp_ladder", lambda *args, **kwargs: ([], None)
    )

    executor = SignalExecutor(bot)

    first = executor.execute_once()
    second = executor.execute_once()
    third = executor.execute_once()

    assert first.status == "skipped"
    assert second.status == "skipped"
    assert third.status == "filled"

    assert attempts[0] == "ADAUSDT"
    assert attempts[1] == "ADAUSDT"
    assert attempts[2] == "XRPUSDT"


def test_signal_executor_restores_quarantine_state(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
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

    call_counter = {"value": 0}

    def fake_place(*_args, **_kwargs):
        call_counter["value"] += 1
        raise OrderValidationError("слишком сильное отклонение цены", code="price_deviation")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    base_time = 1700000000.0
    monkeypatch.setattr(SignalExecutor, "_current_time", lambda self: base_time)

    executor = SignalExecutor(bot)
    first_attempt = executor.execute_once()
    second_attempt = executor.execute_once()

    assert first_attempt.status == "skipped"
    assert second_attempt.status == "skipped"
    assert executor._is_symbol_quarantined("ETHUSDT")

    state = executor.export_state()

    restored_executor = SignalExecutor(StubBot(summary, settings))
    restored_executor.restore_state(state)

    assert restored_executor._is_symbol_quarantined("ETHUSDT")

    call_counter["value"] = 0
    restart_result = restored_executor.execute_once()
    assert restart_result.status == "skipped"
    assert call_counter["value"] == 0


def test_signal_executor_skips_price_limit_liquidity(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
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

    call_counter = {"value": 0}
    call_records: list[dict[str, object]] = []

    def fake_place(*_args, **_kwargs):
        call_counter["value"] += 1
        call_records.append(dict(_kwargs))
        raise OrderValidationError(
            "Недостаточная глубина стакана для заданного объёма в котировочной валюте.",
            code="insufficient_liquidity",
            details={
                "requested_quote": "100.0",
                "available_quote": "42.0",
                "side": "buy",
                "price_cap": "123.45",
                "price_limit_hit": True,
            },
        )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    base_time = 1700000000.0
    time_state = {"value": base_time}

    def fake_now(self):
        current = time_state["value"]
        time_state["value"] += 30.0
        return current

    monkeypatch.setattr(SignalExecutor, "_current_time", fake_now)

    executor = SignalExecutor(bot)

    first_result = executor.execute_once()

    assert first_result.status == "skipped"
    assert first_result.reason is not None
    assert "ждём восстановления ликвидности" in first_result.reason
    assert first_result.context is not None
    assert first_result.context.get("validation_code") == "insufficient_liquidity"
    assert call_counter["value"] == 1
    assert call_records and "qty" in call_records[0]
    initial_qty = float(call_records[0]["qty"])
    initial_tol = float(call_records[0]["tol_value"])
    backoff_meta = first_result.context.get("price_limit_backoff")
    assert isinstance(backoff_meta, dict)
    assert backoff_meta.get("retries") == 1
    assert first_result.context.get("quarantine_ttl") == pytest.approx(
        signal_executor_module._PRICE_LIMIT_LIQUIDITY_TTL
    )

    quarantine_until = executor._symbol_quarantine.get("ETHUSDT")
    assert quarantine_until is not None
    assert quarantine_until >= base_time + signal_executor_module._PRICE_LIMIT_LIQUIDITY_TTL

    backoff_state = executor._price_limit_backoff.get("ETHUSDT")
    assert backoff_state is not None
    assert backoff_state.get("retries") == 1

    # advance time beyond quarantine to allow a second attempt
    time_state["value"] = quarantine_until + 120.0

    second_result = executor.execute_once()

    assert second_result.status == "skipped"
    assert second_result.reason is not None
    assert call_counter["value"] == 2
    assert len(call_records) == 2
    followup_qty = float(call_records[1]["qty"])
    followup_tol = float(call_records[1]["tol_value"])
    assert followup_qty < initial_qty
    assert followup_tol > initial_tol
    assert executor._price_limit_backoff["ETHUSDT"]["retries"] == 2
    assert (
        executor._price_limit_backoff["ETHUSDT"]["quarantine_ttl"]
        > signal_executor_module._PRICE_LIMIT_LIQUIDITY_TTL
    )
    assert second_result.context is not None
    followup_backoff = second_result.context.get("price_limit_backoff")
    assert isinstance(followup_backoff, dict)
    assert followup_backoff.get("retries") == 2

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


def test_signal_executor_blocks_after_daily_loss_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day_key = time.strftime("%Y-%m-%d", time.gmtime())
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=2.5)
    bot = StubBot(summary, settings)

    stub_api = StubAPI(total=1000.0, available=800.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: stub_api)

    def fake_daily_pnl(**_: object) -> dict[str, dict[str, dict[str, float]]]:
        return {
            day_key: {
                "BTCUSDT": {
                    "spot_pnl": -150.0,
                    "fees": 5.0,
                }
            }
        }

    monkeypatch.setattr(signal_executor_module, "daily_pnl", fake_daily_pnl)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "disabled"
    assert result.reason is not None and "Дневной убыток" in result.reason
    assert result.context is not None
    assert result.context.get("guard") == "daily_loss_limit"
    assert result.context.get("loss_value") == pytest.approx(155.0)
    assert result.context.get("loss_percent") == pytest.approx(15.5)
    assert stub_api.orders == []


def test_signal_executor_daily_loss_ignores_derivatives(monkeypatch: pytest.MonkeyPatch) -> None:
    day_key = time.strftime("%Y-%m-%d", time.gmtime())
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=1.0)
    bot = StubBot(summary, settings)

    stub_api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: stub_api)

    def fake_daily_pnl(**_: object) -> dict[str, dict[str, dict[str, float]]]:
        return {
            day_key: {
                "BTCUSDT": {
                    "categories": ["linear"],
                    "spot_pnl": 0.0,
                    "spot_fees": 0.0,
                    "spot_net": 0.0,
                    "fees": -45.0,
                    "derivatives_fees": -45.0,
                }
            }
        }

    monkeypatch.setattr(signal_executor_module, "daily_pnl", fake_daily_pnl)

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status != "disabled"
    assert (result.context or {}).get("guard") != "daily_loss_limit"
    assert stub_api.orders == []


def test_daily_loss_guard_uses_cached_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=1.0)
    bot = StubBot(summary, settings)

    pnl_module.invalidate_daily_pnl_cache()

    summary_dir = tmp_path / "pnl"
    summary_dir.mkdir(exist_ok=True)
    summary_path = summary_dir / "pnl_daily.json"
    monkeypatch.setattr(pnl_module, "_SUMMARY", summary_path)

    now_ts = time.time()
    ledger_rows = [
        {
            "execTime": now_ts,
            "symbol": "BTCUSDT",
            "side": "sell",
            "execPrice": 100.0,
            "execQty": 1.0,
            "execFee": 1.0,
            "category": "spot",
        },
        {
            "execTime": now_ts,
            "symbol": "BTCUSDT",
            "side": "buy",
            "execPrice": 200.0,
            "execQty": 1.0,
            "execFee": 1.0,
            "category": "spot",
        },
    ]

    read_calls = {"count": 0}

    def fake_read_ledger(
        n: Optional[int] = 5000,
        *,
        settings: object | None = None,
        network: object | None = None,
        ledger_path: Optional[Union[str, Path]] = None,
        last_exec_id: Optional[str] = None,
        return_meta: bool = False,
    ):
        read_calls["count"] += 1
        rows = [dict(row) for row in ledger_rows]
        if return_meta:
            return rows, None, True
        return rows

    monkeypatch.setattr(pnl_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(signal_executor_module, "daily_pnl", pnl_module.daily_pnl)

    def fake_resolve_wallet(
        self, *, require_success: bool
    ) -> Tuple[Optional[object], Tuple[float, float]]:
        return None, (1000.0, 1000.0)

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        fake_resolve_wallet,
    )

    executor = SignalExecutor(bot)

    try:
        first = executor.execute_once()
        second = executor.execute_once()

        assert read_calls["count"] == 1
        assert first.status == "disabled"
        assert second.status == "disabled"
    finally:
        pnl_module.invalidate_daily_pnl_cache()


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

