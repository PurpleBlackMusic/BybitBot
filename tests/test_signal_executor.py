import copy
import math
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import json
import time

import pytest

import bybit_app.utils.pnl as pnl_module
import bybit_app.utils.signal_executor as signal_executor_module
import bybit_app.utils.ws_manager as ws_manager_module
import bybit_app.utils.spot_pnl as spot_pnl_module
import bybit_app.utils.spot_fifo as spot_fifo_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.signal_executor import (
    AutomationLoop,
    ExecutionResult,
    SignalExecutor,
)
from bybit_app.utils.ai.kill_switch import KillSwitchState
from bybit_app.utils.self_learning import TradePerformanceSnapshot
from bybit_app.utils.spot_market import (
    OrderValidationError,
    SpotTradeSnapshot,
    PreparedSpotMarketOrder,
    _BALANCE_CACHE,
)
from bybit_app.utils.ws_manager import WSManager
from tests.test_spot_market import DummyAPI as SpotDummyAPI, _universe_payload


@pytest.fixture(autouse=True)
def _stub_ws_autostart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "autostart",
        lambda include_private=True: (False, False),
    )


@pytest.fixture(autouse=True)
def _clear_wallet_cache() -> None:
    _BALANCE_CACHE.clear()
    yield
    _BALANCE_CACHE.clear()


@pytest.fixture(autouse=True)
def _kill_switch_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"paused": False, "until": None, "reason": None}

    def fake_set_pause(minutes: float, reason: str) -> float:
        try:
            duration = float(minutes)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0.0:
            state.update(paused=False, until=None, reason=None)
            return time.time()
        until = time.time() + duration * 60.0
        state.update(paused=True, until=until, reason=reason)
        return until

    def fake_state() -> KillSwitchState:
        until = state.get("until")
        paused = bool(state.get("paused"))
        if paused and isinstance(until, (int, float)) and until <= time.time():
            state.update(paused=False, until=None, reason=None)
            paused = False
            until = None
        return KillSwitchState(paused=paused, until=until, reason=state.get("reason"))

    monkeypatch.setattr(signal_executor_module, "activate_kill_switch", fake_set_pause)
    monkeypatch.setattr(signal_executor_module, "kill_switch_state", fake_state)
    yield


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
    def __init__(
        self,
        total: float = 0.0,
        available: float = 0.0,
        *,
        orderbook_payload: Optional[dict[str, object]] = None,
    ) -> None:
        self._total = total
        self._available = available
        self.orders: list[dict[str, object]] = []
        self.amendments: list[dict[str, object]] = []
        self.cancellations: list[dict[str, object]] = []
        self._orderbook_payload = orderbook_payload or {
            "result": {
                "a": [["100.0", "25"]],
                "b": [["99.9", "25"]],
            }
        }

    def wallet_balance(self, *args: object, **kwargs: object) -> dict:
        return {
            "result": {
                "list": [
                    {
                        "totalEquity": str(self._total),
                        "availableBalance": str(self._available),
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": str(self._available),
                                "availableToWithdraw": str(self._available),
                                "availableBalance": str(self._available),
                            }
                        ],
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

    def amend_order(self, **payload: object) -> dict[str, object]:
        self.amendments.append(dict(payload))
        return {"retCode": 0, "result": {}}

    def cancel_order(self, **payload: object) -> dict[str, object]:
        self.cancellations.append(dict(payload))
        return {"retCode": 0, "result": {}}

    def orderbook(
        self,
        *,
        category: str = "spot",
        symbol: str | None = None,
        limit: int = 1,
    ) -> dict[str, object]:
        assert category == "spot"
        result = copy.deepcopy(self._orderbook_payload)
        if limit <= 0 or not isinstance(result, dict):
            return result
        payload_result = result.get("result") if isinstance(result, dict) else None
        if isinstance(payload_result, dict):
            for key in ("a", "b"):
                levels = payload_result.get(key)
                if isinstance(levels, list) and limit < len(levels):
                    payload_result[key] = levels[:limit]
        return result

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


def test_signal_executor_resolves_stubbed_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = Settings(ai_enabled=True)
    monkeypatch.setattr(signal_executor_module, "get_settings", lambda: sentinel)

    stub_bot = type("BotStub", (), {"settings": "not-settings"})()
    executor = SignalExecutor(stub_bot)

    assert executor._resolve_settings() is sentinel


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


def test_signal_sizing_factor_uses_performance_metrics() -> None:
    base_summary = {
        "actionable": True,
        "mode": "buy",
        "symbol": "ETHUSDT",
        "probability": 0.68,
        "ev_bps": 25.0,
    }

    low_metrics = {
        "primary_watch": {
            "win_rate_pct": 38.0,
            "realized_bps_avg": -15.0,
            "median_hold_sec": 2.5 * 60.0 * 60.0,
        }
    }

    high_metrics = {
        "primary_watch": {
            "win_rate_pct": 70.0,
            "realized_bps_avg": 25.0,
            "median_hold_sec": 10.0 * 60.0,
        }
    }

    settings = Settings(ai_enabled=True)
    low_summary = copy.deepcopy(base_summary)
    low_summary.update(low_metrics)
    high_summary = copy.deepcopy(base_summary)
    high_summary.update(high_metrics)

    executor = SignalExecutor(StubBot(base_summary, settings))

    low_factor = executor._signal_sizing_factor(low_summary, settings)
    high_factor = executor._signal_sizing_factor(high_summary, settings)

    assert low_factor < high_factor
    assert low_factor < 0.55
    assert high_factor > 0.8


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
            "hold_seconds": 15000.0,
            "price": 0.95,
            "pnl_value": -0.5,
            "pnl_bps": -5.0,
            "quote_notional": 9.5,
        }
    }

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_open_positions",
        lambda self, settings, summary, **_: positions,
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
    assert metadata.get("hold_threshold_minutes") == pytest.approx(240.0)
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
        lambda self, settings, summary, **_: positions,
    )

    executor = SignalExecutor(bot)
    forced_summary, metadata = executor._maybe_force_exit(summary, settings)

    assert forced_summary is None
    assert metadata is None


def test_signal_executor_force_exit_triggers_on_trade_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": False, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        ai_max_hold_minutes=0.0,
        ai_min_exit_bps=None,
        ai_max_trade_loss_pct=1.0,
    )
    bot = StubBot(summary, settings)
    monkeypatch.setattr(bot, "trade_statistics", lambda limit=None: {})

    positions = {
        "ETHUSDT": {
            "qty": 2.0,
            "avg_cost": 100.0,
            "realized_pnl": 0.0,
            "hold_seconds": 120.0,
            "price": 87.5,
            "pnl_value": -25.0,
            "pnl_bps": -1250.0,
            "quote_notional": 175.0,
        }
    }

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_open_positions",
        lambda self, settings, summary, **_: positions,
    )

    executor = SignalExecutor(bot)
    forced_summary, metadata = executor._maybe_force_exit(
        summary,
        settings,
        portfolio_total_equity=1000.0,
    )

    assert forced_summary is not None
    assert forced_summary["symbol"] == "ETHUSDT"
    assert forced_summary["mode"] == "sell"
    assert metadata is not None
    assert metadata.get("loss_value") == pytest.approx(25.0)
    assert metadata.get("loss_percent") == pytest.approx(2.5)
    triggers = metadata.get("triggers") or []
    assert any(trigger.get("type") == "max_loss" for trigger in triggers)


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

    def fake_read_ledger(
        limit: int | None = None,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows = list(events)
        if return_meta:
            return rows, "evt-0", True
        return rows

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)

    def fake_inventory(
        *,
        events=None,
        settings=None,
        return_layers: bool = False,
        **_,
    ):
        inventory = {
            "BTCUSDT": {"position_qty": 2.0, "avg_cost": 100.0, "realized_pnl": 0.0}
        }
        if return_layers:
            return inventory, {}
        return inventory

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


def test_collect_open_positions_avoids_full_ledger_on_empty_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True)
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    read_calls = {"count": 0}

    def fake_read_ledger(*args, **kwargs):
        read_calls["count"] += 1
        return []

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)

    def fake_inventory(
        *,
        events=None,
        settings=None,
        return_layers: bool = False,
        **_,
    ):
        assert events is None
        if return_layers:
            return {}, {}
        return {}

    monkeypatch.setattr(
        signal_executor_module,
        "spot_inventory_and_pnl",
        fake_inventory,
    )

    positions = executor._collect_open_positions(settings, summary)
    assert positions == {}

    repeated = executor._collect_open_positions(settings, summary)
    assert repeated == {}

    assert read_calls["count"] <= 1


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


def test_signal_executor_uses_spot_fallback_when_unified_short(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _universe_payload(
        [
            {
                "symbol": "ETHUSDT",
                "quoteCoin": "USDT",
                "status": "Trading",
                "baseCoin": "ETH",
                "lotSizeFilter": {
                    "minOrderAmt": "5",
                    "minOrderQty": "0.00000001",
                    "qtyStep": "0.00000001",
                },
                "priceFilter": {"tickSize": "0.1"},
            }
        ]
    )
    wallet_payload = {
        "UNIFIED": {
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalEquity": "100",
                        "availableBalance": "100",
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": "100",
                                "availableBalance": "3",
                            }
                        ],
                    }
                ]
            }
        },
        "SPOT": {
            "result": {
                "list": [
                    {
                        "accountType": "SPOT",
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": "150",
                                "availableBalance": "150",
                            }
                        ],
                    }
                ]
            }
        },
    }

    api = SpotDummyAPI(payload, wallet_payload=wallet_payload)

    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=100.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
        ai_max_slippage_bps=0,
        ai_max_spread_bps=150.0,
    )
    bot = StubBot(summary, settings)

    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_resolve_wallet(self, *, require_success: bool):
        return api, (100.0, 100.0), 100.0, {}

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        fake_resolve_wallet,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status != "rejected"
    assert api.place_calls, "executor should submit order when spot funds are available"
    assert api.wallet_calls >= 2, "spot fallback should refresh balances for affordability guard"


def test_signal_executor_handles_non_usdt_wallet_balances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_price = 27_000.0
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    _BALANCE_CACHE.clear()

    wallet_payload = {
        "result": {
            "list": [
                {
                    "coin": [
                        {
                            "coin": "BTC",
                            "equity": "0.5",
                            "availableBalance": "0.4",
                            "availableToWithdraw": "0.3",
                            "markPrice": str(mark_price),
                        }
                    ]
                }
            ]
        }
    }

    api = StubAPI(total=0.0, available=0.0)
    monkeypatch.setattr(api, "wallet_balance", lambda *args, **kwargs: wallet_payload)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.context is not None
    assert result.status == "skipped"
    assert result.reason is not None
    assert "Недостаточно свободного капитала" in result.reason
    assert result.context.get("quote_wallet_cap") == pytest.approx(0.0)


def test_signal_executor_skips_buy_when_usdt_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_price = 25_000.0
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=5.0,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    wallet_payload = {
        "result": {
            "list": [
                {
                    "coin": [
                        {
                            "coin": "BTC",
                            "equity": "0.4",
                            "availableBalance": "0.4",
                            "availableToWithdraw": "0.4",
                            "markPrice": str(mark_price),
                        }
                    ]
                }
            ]
        }
    }

    api = StubAPI(total=0.0, available=0.0)
    monkeypatch.setattr(api, "wallet_balance", lambda *args, **kwargs: wallet_payload)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    monkeypatch.setattr(
        signal_executor_module,
        "_wallet_available_balances",
        lambda api_obj, account_type="UNIFIED", **_: {
            "BTC": Decimal("0.4"),
            "USDT": Decimal("0"),
        },
    )

    place_called = {"value": False}

    def fail_place(*args: object, **kwargs: object) -> dict[str, object]:
        place_called["value"] = True
        raise AssertionError("place_spot_market_with_tolerance should not be called")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fail_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None
    assert "Недостаточно свободного капитала" in result.reason
    assert place_called["value"] is False
    assert result.context is not None
    assert result.context.get("quote_wallet_cap") == pytest.approx(0.0)
    assert result.context.get("available_equity_quote_limited") == pytest.approx(0.0)


def test_signal_executor_skips_buy_when_usdt_wallet_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        dry_run_mainnet=False,
        dry_run_testnet=False,
        ai_risk_per_trade_pct=5.0,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=500.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda api_obj, symbol: {"min_order_amt": "5"},
    )
    monkeypatch.setattr(
        signal_executor_module,
        "_wallet_available_balances",
        lambda api_obj, account_type="UNIFIED", **_: {"BTC": Decimal("0.4")},
    )

    place_called = {"value": False}

    def fail_place(*args: object, **kwargs: object) -> dict[str, object]:
        place_called["value"] = True
        raise AssertionError("place_spot_market_with_tolerance should not be called")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fail_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None
    assert "Недостаточно свободного капитала" in result.reason
    assert place_called["value"] is False
    assert result.context is not None
    assert result.context.get("quote_wallet_cap") == pytest.approx(0.0)


def test_signal_executor_buy_uses_spot_usdt_when_unified_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=True,
        ai_risk_per_trade_pct=5.0,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=1000.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    wallet_calls: list[Optional[str]] = []

    def fake_wallet_balances(
        api_obj,
        account_type: str = "UNIFIED",
        *,
        required_asset: str | None = None,
    ) -> Mapping[str, Decimal]:
        wallet_calls.append(required_asset)
        if required_asset:
            return {
                "BTC": Decimal("0.5"),
                "USDT": Decimal("42"),
            }
        return {"BTC": Decimal("0.5")}

    monkeypatch.setattr(
        signal_executor_module,
        "_wallet_available_balances",
        fake_wallet_balances,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert wallet_calls and wallet_calls[0] == "USDT"
    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "BTCUSDT"
    assert result.order["side"] == "Buy"

    expected_quote_cap = 42.0
    expected_notional = expected_quote_cap - (expected_quote_cap * 0.02)

    assert result.context is not None
    assert result.context.get("quote_wallet_cap") == pytest.approx(expected_quote_cap)
    assert result.context.get("available_equity_quote_limited") == pytest.approx(
        expected_quote_cap
    )
    assert result.context.get("usable_after_reserve") == pytest.approx(
        expected_notional
    )
    assert result.order.get("notional_quote") == pytest.approx(
        expected_notional, rel=1e-3
    )


def test_signal_executor_scales_to_min_notional_when_capital_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=0.1,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=100.0, available=100.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "min-size"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 5.0


def test_signal_executor_relaxes_reserve_for_minimum_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=10.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=5.0, available=5.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda api_obj, symbol: {"min_order_amt": "5"},
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "reserve-min"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 5.0
    assert result.context is not None
    assert result.context.get("reserve_relaxed_for_min_notional") is True
    assert result.context.get("usable_after_reserve") == pytest.approx(4.5)
    assert result.order is not None
    assert result.order.get("notional_quote") == pytest.approx(5.0)


def test_compute_notional_allows_zero_reserve() -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, spot_cash_reserve_pct=0.0, ai_risk_per_trade_pct=5.0)
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    notional, usable_after_reserve, reserve_relaxed, _ = executor._compute_notional(
        settings,
        total_equity=100.0,
        available_equity=100.0,
    )

    assert usable_after_reserve == pytest.approx(100.0)
    assert reserve_relaxed is False
    assert notional == pytest.approx(5.0)


def test_signal_executor_skips_min_buy_when_quote_cap_cannot_cover_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=0.0,
        ai_max_slippage_bps=500,
    )
    bot = StubBot(summary, settings)

    total_equity = 1_000.0
    available_equity = 250.0
    quote_cap = 5.0

    api = StubAPI(total=total_equity, available=available_equity)

    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda api_obj, symbol: {"min_order_amt": "5"},
    )

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success: (
            api,
            (total_equity, available_equity),
            quote_cap,
            {},
        ),
    )

    def fake_compute(
        self,
        settings: Settings,
        total_equity_value: float,
        available_equity_value: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> tuple[float, float, bool, bool]:
        return 5.0, 4.5, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute,
    )

    def fail_place(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("place_spot_market_with_tolerance should not be called")

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fail_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None
    assert "Недостаточно свободного капитала" in result.reason
    assert result.context is not None
    assert result.context.get("min_notional") == pytest.approx(5.0)
    assert result.context.get("quote_wallet_cap") == pytest.approx(quote_cap)


def test_signal_executor_buy_uses_quote_cap_when_wallet_totals_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=True,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=0.0, available=0.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda api_obj, symbol: {"min_order_amt": "5"},
    )

    quote_cap = 5.0

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success: (api, (0.0, 0.0), quote_cap, {}),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order.get("side") == "Buy"
    assert result.order.get("symbol") == "BTCUSDT"
    assert result.order.get("notional_quote") == pytest.approx(5.0)
    assert result.context is not None
    assert result.context.get("quote_wallet_cap") == pytest.approx(quote_cap)
    assert result.context.get("quote_wallet_cap_substituted") is True
    assert result.context.get("usable_after_reserve") == pytest.approx(5.0)


def test_signal_executor_skips_buy_when_wallet_balance_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True, dry_run=False)
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=250.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fail_place(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("place_spot_market_with_tolerance should not be called")

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        fail_place,
    )

    def raise_balances(*args: object, **kwargs: object) -> Mapping[str, object]:
        raise RuntimeError("balance unavailable")

    monkeypatch.setattr(
        signal_executor_module,
        "_wallet_available_balances",
        raise_balances,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.order is None
    assert result.reason is not None
    assert "Недостаточно свободного капитала" in result.reason
    assert result.context is not None
    assert result.context.get("quote_wallet_cap") == pytest.approx(0.0)
    assert (
        result.context.get("quote_wallet_cap_error")
        == "wallet_balance_unavailable"
    )
    wallet_meta = result.context.get("wallet_meta")
    assert isinstance(wallet_meta, Mapping)
    assert wallet_meta.get("quote_wallet_cap_error") == "wallet_balance_unavailable"


def test_signal_executor_reserve_respects_available_equity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=0.0,
        spot_cash_reserve_pct=10.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=10_000.0, available=10.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "reserve-available"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert captured["unit"] == "quoteCoin"
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 9.0

    assert result.context is not None
    assert result.context["usable_after_reserve"] == pytest.approx(9.0)
    assert result.context["available_equity"] == pytest.approx(10.0)
    assert result.context["total_equity"] == pytest.approx(10_000.0)


def test_signal_executor_uses_available_equity_for_min_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=25.0,
        spot_cash_reserve_pct=10.0,
        spot_max_cap_per_trade_pct=5.0,
        ai_max_slippage_bps=400,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=100.0, available=5.6)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "min-buy"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 5.0
    assert pytest.approx(float(captured["max_quote"]), rel=1e-9) == 5.2

    assert result.context is not None
    assert result.context["available_equity"] == pytest.approx(5.6)
    assert result.context["usable_after_reserve"] == pytest.approx(5.04)


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

    reserve_base = min(500.0, 120.0)
    usable_after_reserve = 120.0 - reserve_base * 0.02
    multiplier, _, _, _ = signal_executor_module._resolve_slippage_tolerance(
        "Percent", settings.ai_max_slippage_bps / 100.0
    )
    expected_qty = Decimal(str(usable_after_reserve)) / Decimal(str(multiplier))

    assert float(captured["qty"]) == pytest.approx(float(expected_qty), rel=1e-6)
    assert captured["max_quote"] == pytest.approx(usable_after_reserve)


def test_signal_executor_applies_volatility_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT", "volatility_pct": 20.0}
    settings = Settings(
        ai_enabled=True,
        ai_risk_per_trade_pct=10.0,
        spot_vol_target_pct=5.0,
        spot_vol_min_scale=0.2,
    )
    bot = StubBot(summary, settings)

    total_equity = 1_000.0
    available_equity = 1_000.0

    api = StubAPI(total=total_equity, available=available_equity)

    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(signal_executor_module, "_instrument_limits", lambda api_obj, symbol: {})

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success: (api, (total_equity, available_equity), None, {}),
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_portfolio_quote_exposure",
        lambda self, settings, summary, **_: ({}, 0.0),
    )

    recorded: dict[str, float] = {}

    def fake_compute(
        self,
        settings: Settings,
        total_equity_value: float,
        available_equity_value: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> tuple[float, float, bool, bool]:
        recorded["sizing_factor"] = sizing_factor
        return 100.0, available_equity_value, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute,
    )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", lambda *args, **kwargs: {}
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert recorded["sizing_factor"] == pytest.approx(0.25, rel=1e-6)
    assert result.context is not None
    volatility_meta = result.context.get("risk_controls", {}).get("volatility")
    assert isinstance(volatility_meta, dict)
    assert volatility_meta.get("scale") == pytest.approx(0.25, rel=1e-6)
    assert result.status == "dry_run"


def test_signal_executor_caps_notional_by_symbol_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        ai_risk_per_trade_pct=50.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=100.0,
        spot_max_cap_per_symbol_pct=20.0,
        spot_max_portfolio_pct=90.0,
        ai_max_slippage_bps=0,
    )
    bot = StubBot(summary, settings)

    total_equity = 1_000.0
    available_equity = 1_000.0
    existing_symbol_exposure = 180.0
    other_exposure = 50.0

    api = StubAPI(total=total_equity, available=available_equity)

    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(signal_executor_module, "_instrument_limits", lambda api_obj, symbol: {})

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success: (api, (total_equity, available_equity), None, {}),
    )

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_portfolio_quote_exposure",
        lambda self, settings, summary, **_: (
            {"BTCUSDT": existing_symbol_exposure, "ETHUSDT": other_exposure},
            existing_symbol_exposure + other_exposure,
        ),
    )

    def fake_compute(
        self,
        settings: Settings,
        total_equity_value: float,
        available_equity_value: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> tuple[float, float, bool, bool]:
        return 150.0, available_equity_value, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute,
    )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", lambda *args, **kwargs: {}
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["notional_quote"] == pytest.approx(20.0)
    assert result.context is not None
    risk_controls = result.context.get("risk_controls")
    assert isinstance(risk_controls, dict)
    assert risk_controls["symbol_cap"]["available"] == pytest.approx(20.0)
    cap_info = risk_controls.get("cap_adjustment")
    assert isinstance(cap_info, dict)
    assert cap_info.get("applied") is True
    assert cap_info.get("final_notional") == pytest.approx(20.0)


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
        spot_cash_reserve_pct=200.0,
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
    assert result.context.get("sell_fallback_applied") is True
    assert pytest.approx(
        float(result.context.get("sell_fallback_notional_quote")),
        rel=1e-9,
    ) == 25.0
    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert pytest.approx(fallback.get("available_base"), rel=1e-9) == 0.001
    assert pytest.approx(fallback.get("quote_notional"), rel=1e-9) == 25.0
    assert pytest.approx(fallback.get("min_order_amt"), rel=1e-9) == 5.0


def test_signal_executor_sell_scales_on_insufficient_balance(
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
        ai_risk_per_trade_pct=100.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=400.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    available_base = Decimal("0.1")
    snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("1000"),
        balances={"BTC": available_base},
        limits={"base_coin": "BTC", "min_order_amt": Decimal("5")},
    )
    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        lambda api_obj, symbol, **_: snapshot,
    )

    best_bid = Decimal("900")
    attempts: list[float] = []

    def fake_place(
        api_obj,
        symbol: str,
        side: str,
        qty: object,
        unit: str,
        tol_type: str,
        tol_value: float,
        max_quote: object | None,
        settings: Settings,
    ) -> dict[str, object]:
        qty_float = float(qty)
        attempts.append(qty_float)
        if len(attempts) == 1:
            raise OrderValidationError(
                "Недостаточно базового актива для продажи.",
                code="insufficient_balance",
                details={
                    "available": str(available_base),
                    "required": "0.11111111",
                    "best_bid": str(best_bid),
                },
            )
        return {
            "retCode": 0,
            "result": {
                "orderId": "sell-clipped",
                "cumExecQty": str(available_base),
                "cumExecValue": str(best_bid * available_base),
                "avgPrice": str(best_bid),
            },
        }

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        fake_place,
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_ledger_rows_snapshot",
        lambda self, **_: ([], None),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_place_tp_ladder",
        lambda *args, **kwargs: ([], {}, []),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_maybe_notify_trade",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "private_snapshot",
        lambda: {},
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert attempts and len(attempts) == 2
    assert attempts[0] > attempts[1]
    assert attempts[0] == pytest.approx(
        float(snapshot.price * available_base), rel=1e-9
    )
    assert attempts[1] == pytest.approx(
        float(best_bid * available_base), rel=1e-6
    )

    assert result.status == "filled"
    assert result.context is not None
    adjustments = result.context.get("insufficient_balance_adjustments")
    assert isinstance(adjustments, list)
    assert adjustments
    last_adjustment = adjustments[-1]
    assert pytest.approx(last_adjustment["available_base"], rel=1e-9) == float(
        available_base
    )

    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert pytest.approx(fallback.get("wallet_available_base"), rel=1e-9) == float(
        available_base
    )

def test_signal_executor_sell_reads_spot_balance_when_unified_empty(
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
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=200.0,
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

    calls: list[str] = []

    unified_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"USDT": Decimal("100.0"), "BTC": Decimal("0")},
        limits={"min_order_amt": Decimal("5"), "base_coin": "BTC"},
    )
    spot_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"BTC": Decimal("0.002")},
        limits={"base_coin": "BTC"},
    )

    def fake_prepare(api_obj, symbol, **kwargs):
        account_type = (kwargs.get("account_type") or "UNIFIED").upper()
        calls.append(account_type)
        if account_type == "SPOT":
            return spot_snapshot
        return unified_snapshot

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        fake_prepare,
    )

    def fake_compute_notional(
        self,
        settings_obj,
        total_equity,
        available_equity,
        sizing_factor,
        *,
        min_notional,
        quote_balance_cap=None,
    ) -> Tuple[float, float, bool, bool]:
        return 0.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-spot"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert captured["side"] == "Sell"
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 50.0

    assert calls[0] == "UNIFIED"
    assert "SPOT" in calls

    assert result.context is not None
    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert fallback.get("balance_account_type") == "SPOT"
    assert pytest.approx(fallback.get("available_base"), rel=1e-9) == 0.002


def test_signal_executor_sell_combines_unified_and_spot_balances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "BTCUSDT",
        "price": 25000.0,
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=200.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    calls: list[str] = []

    unified_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"BTC": Decimal("0.0002")},
        limits={"min_order_amt": Decimal("0.001"), "base_coin": "BTC"},
    )
    spot_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"BTC": Decimal("0.004")},
        limits={"base_coin": "BTC"},
    )

    def fake_prepare(api_obj, symbol, **kwargs):
        account_type = (kwargs.get("account_type") or "UNIFIED").upper()
        calls.append(account_type)
        if account_type == "SPOT":
            return spot_snapshot
        return unified_snapshot

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        fake_prepare,
    )

    def fake_compute_notional(
        self,
        settings_obj,
        total_equity,
        available_equity,
        sizing_factor,
        *,
        min_notional,
        quote_balance_cap=None,
    ) -> Tuple[float, float, bool, bool]:
        return 50.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    monkeypatch.setattr(
        SignalExecutor,
        "_ledger_rows_snapshot",
        lambda self, **_: ([], None),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_place_tp_ladder",
        lambda *args, **kwargs: ([], {}, []),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_maybe_notify_trade",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "private_snapshot",
        lambda: {},
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-combined"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert captured["side"] == "Sell"
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 50.0

    assert calls[0] == "UNIFIED"
    assert "SPOT" in calls

    assert result.context is not None
    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert fallback.get("balance_account_type") == "UNIFIED"
    assert fallback.get("balance_fallback_account_type") == "SPOT"
    assert pytest.approx(fallback.get("unified_available_base"), rel=1e-9) == 0.0002
    assert pytest.approx(fallback.get("spot_available_base"), rel=1e-9) == 0.004
    assert pytest.approx(fallback.get("combined_available_base"), rel=1e-9) == 0.0042
    assert pytest.approx(
        fallback.get("expected_base_requirement"), rel=1e-9
    ) == 0.002


def test_signal_executor_sell_uses_fallback_notional_when_below_min(
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
        ai_risk_per_trade_pct=1.0,
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

    def fake_compute_notional(
        self,
        settings_obj,
        total_equity,
        available_equity,
        sizing_factor,
        *,
        min_notional,
        quote_balance_cap=None,
    ) -> Tuple[float, float, bool, bool]:
        return 1.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-fallback-min"}}

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
    assert result.context.get("sell_fallback_applied") is True
    assert pytest.approx(
        float(result.context.get("sell_fallback_notional_quote")),
        rel=1e-9,
    ) == 25.0
    assert pytest.approx(
        float(result.context.get("risk_limited_notional_quote")),
        rel=1e-9,
    ) == 1.0


def test_signal_executor_sell_caps_notional_to_wallet_holdings(
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
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=200.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("25000"),
        balances={"BTC": Decimal("0.002")},
        limits={"min_order_amt": Decimal("5"), "base_coin": "BTC"},
    )
    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        lambda api_obj, symbol, **_: snapshot,
    )

    def fake_compute_notional(
        self,
        settings_obj,
        total_equity,
        available_equity,
        sizing_factor,
        *,
        min_notional,
        quote_balance_cap=None,
    ) -> Tuple[float, float, bool, bool]:
        return 100.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    captured: dict[str, object] = {}

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"retCode": 0, "result": {"orderId": "sell-wallet-cap"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert captured
    assert captured["side"] == "Sell"
    assert pytest.approx(float(captured["qty"]), rel=1e-9) == 50.0

    assert result.context is not None
    assert result.context.get("sell_fallback_applied") is True
    assert pytest.approx(
        float(result.context.get("sell_fallback_notional_quote")),
        rel=1e-9,
    ) == 50.0
    assert pytest.approx(
        float(result.context.get("risk_limited_notional_quote")),
        rel=1e-9,
    ) == 100.0

    fallback = result.context.get("sell_fallback")
    assert isinstance(fallback, dict)
    assert fallback.get("adjustment_reason") == "wallet"
    assert pytest.approx(
        float(fallback.get("risk_limited_notional_quote")),
        rel=1e-9,
    ) == 100.0
    assert pytest.approx(float(fallback.get("applied_notional")), rel=1e-9) == 50.0
    assert pytest.approx(float(fallback.get("requested_notional_quote")), rel=1e-9) == 100.0


def test_signal_executor_sell_refreshes_price_for_insufficient_balance(
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
        ai_risk_per_trade_pct=100.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=400.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    available_base = Decimal("0.1")
    unified_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("1000"),
        balances={"BTC": available_base},
        limits={"base_coin": "BTC", "min_order_amt": Decimal("5")},
    )
    spot_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=None,
        balances={"BTC": Decimal("0")},
        limits=None,
    )
    refreshed_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("900"),
        balances=None,
        limits=None,
    )

    refresh_calls: list[dict[str, object]] = []

    def fake_snapshot(api_obj, symbol, **kwargs: object) -> SpotTradeSnapshot:
        if kwargs.get("force_refresh"):
            refresh_calls.append(dict(kwargs))
            return refreshed_snapshot
        if kwargs.get("account_type") == "SPOT":
            return spot_snapshot
        return unified_snapshot

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        fake_snapshot,
    )

    attempts: list[float] = []

    def fake_place(
        api_obj,
        symbol: str,
        side: str,
        qty: object,
        unit: str,
        tol_type: str,
        tol_value: float,
        max_quote: object | None,
        settings: Settings,
    ) -> dict[str, object]:
        qty_float = float(qty)
        attempts.append(qty_float)
        if len(attempts) == 1:
            raise OrderValidationError(
                "Недостаточно базового актива для продажи.",
                code="insufficient_balance",
                details={"best_bid": "900"},
            )
        return {
            "retCode": 0,
            "result": {
                "orderId": "sell-refresh",
                "cumExecQty": str(available_base),
                "cumExecValue": str(Decimal("900") * available_base),
                "avgPrice": "900",
            },
        }

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert attempts and len(attempts) == 2
    assert attempts[1] < attempts[0]
    assert pytest.approx(attempts[0], rel=1e-9) == 100.0
    assert pytest.approx(attempts[1], rel=1e-9) == 90.0
    assert refresh_calls, "expected refreshed snapshot request"

    assert result.status == "filled"
    assert result.context is not None
    adjustments = result.context.get("insufficient_balance_adjustments")
    assert isinstance(adjustments, list) and adjustments
    latest_adjustment = adjustments[-1]
    assert pytest.approx(latest_adjustment.get("required_base"), rel=1e-9) == pytest.approx(
        float(attempts[0]) / 900.0,
        rel=1e-9,
    )
    assert pytest.approx(latest_adjustment.get("price_snapshot"), rel=1e-9) == 900.0
    assert result.order is not None
    assert pytest.approx(float(result.order.get("notional_quote")), rel=1e-9) == 90.0


def test_signal_executor_sell_retries_with_wallet_context(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "BTCUSDT",
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=100.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=400.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    base_available = Decimal("0.1")
    initial_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("1000"),
        balances={"BTC": base_available},
        limits={"base_coin": "BTC", "min_order_amt": Decimal("5")},
    )
    refreshed_snapshot = SpotTradeSnapshot(
        symbol="BTCUSDT",
        price=Decimal("900"),
        balances=None,
        limits=None,
    )

    def fake_snapshot(api_obj, symbol, **kwargs: object) -> SpotTradeSnapshot:
        if kwargs.get("force_refresh"):
            if kwargs.get("include_balances"):
                return SpotTradeSnapshot(
                    symbol="BTCUSDT",
                    price=Decimal("900"),
                    balances={"BTC": base_available},
                    limits=None,
                )
            return refreshed_snapshot
        return initial_snapshot

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        fake_snapshot,
    )

    def fake_compute_notional(
        self,
        settings_obj,
        total_equity,
        available_equity,
        sizing_factor,
        *,
        min_notional,
        quote_balance_cap=None,
    ) -> Tuple[float, float, bool, bool]:
        return 100.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    attempts: list[float] = []
    placed_orders: list[dict[str, object]] = []

    def fake_place(api_obj, **kwargs: object) -> dict[str, object]:
        qty_float = float(kwargs.get("qty", 0))
        attempts.append(qty_float)
        if len(attempts) == 1:
            raise OrderValidationError(
                "Недостаточно базового актива для продажи.",
                code="insufficient_balance",
                details={"best_bid": "900"},
            )
        placed_orders.append(dict(kwargs))
        return {
            "retCode": 0,
            "result": {
                "orderId": "sell-wallet-context",
                "avgPrice": "900",
                "cumExecQty": str(base_available),
                "cumExecValue": str(Decimal("900") * base_available),
            },
        }

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert attempts and len(attempts) == 2
    assert pytest.approx(attempts[0], rel=1e-9) == 100.0
    assert pytest.approx(attempts[1], rel=1e-9) == 90.0
    assert placed_orders and float(placed_orders[-1]["qty"]) == pytest.approx(90.0, rel=1e-9)

    assert result.status == "filled"
    assert result.context is not None
    assert result.context.get("sell_fallback") is not None
    adjustments = result.context.get("insufficient_balance_adjustments")
    assert isinstance(adjustments, list) and adjustments
    latest_adjustment = adjustments[-1]
    assert pytest.approx(latest_adjustment.get("scaling"), rel=1e-9) == pytest.approx(
        attempts[1] / attempts[0],
        rel=1e-9,
    )
    assert result.order is not None
    assert pytest.approx(
        float(result.order.get("notional_quote")),
        rel=1e-9,
    ) == 90.0

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
            "execTime": now - 21600,
        }
    ]

    def fake_read_ledger(
        n: int | None = 5000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows = list(events)
        if return_meta:
            return rows, "evt-guard", True
        return rows

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
    assert forced.get("hold_minutes") is not None and forced["hold_minutes"] >= 240.0
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

    register_calls: list[dict[str, object]] = []
    original_register = signal_executor_module.ws_manager.register_tp_ladder_plan

    def register_spy(symbol: str, **kwargs: object) -> None:
        register_calls.append(
            {
                "symbol": symbol,
                "status": kwargs.get("status"),
                "signature": tuple(
                    tuple(pair) for pair in kwargs.get("signature") or ()
                ),
                "handshake": tuple(kwargs.get("handshake") or ()),
                "ladder": tuple(
                    tuple(rung) for rung in kwargs.get("ladder") or ()
                ),
            }
        )
        original_register(symbol, **kwargs)

    cancel_calls: list[str] = []

    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "register_tp_ladder_plan",
        register_spy,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "_cancel_existing_tp_orders",
        lambda api_obj, symbol: cancel_calls.append(symbol),
    )
    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 3
    first, second, stop = api.orders
    assert first["side"] == "Sell"
    assert second["side"] == "Sell"
    assert first["orderType"] == "Limit"
    assert second["orderType"] == "Limit"
    assert Decimal(first["qty"]) == Decimal("0.45")
    assert Decimal(second["qty"]) == Decimal("0.30")
    assert Decimal(first["price"]) == Decimal("100.7")
    assert Decimal(second["price"]) == Decimal("101.2")
    assert stop["orderFilter"] == "tpslOrder"
    assert stop["orderType"] == "Market"
    assert stop["side"] == "Sell"
    assert stop["qty"] == "0.75"
    assert Decimal(stop["triggerPrice"]) == Decimal("99.2")
    assert result.order is not None
    assert result.order.get("take_profit_orders")
    stop_orders = result.order.get("stop_loss_orders")
    assert stop_orders and len(stop_orders) == 1
    assert Decimal(stop_orders[0]["triggerPrice"]) == Decimal("99.2")
    assert result.context is not None
    execution_payload = result.context.get("execution", {})
    assert Decimal(execution_payload.get("avg_price")) == Decimal("100")
    stop_context = result.context.get("stop_loss_orders")
    assert stop_context and stop_context[0]["orderType"] == "Market"

    assert [call["status"] for call in register_calls] == ["pending", "active"]
    assert all(call["symbol"] == "BTCUSDT" for call in register_calls)
    assert register_calls[0]["signature"] == register_calls[1]["signature"]
    assert all(call["handshake"] for call in register_calls)
    assert cancel_calls == []

    signal_executor_module.ws_manager.clear_tp_ladder_plan("BTCUSDT")


def test_signal_executor_tp_ladder_uses_fallback_for_small_size(
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

    api = StubAPI(total=500.0, available=450.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.15"))

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.15",
            "cumExecValue": "15",
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
        signal_executor_module, "place_spot_market_with_tolerance", lambda *a, **k: response_payload
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "register_tp_ladder_plan",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "clear_tp_ladder_plan",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "realtime_private_rows",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "resolve_tp_handshake",
        lambda *a, **k: (),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(api.orders) == 2
    tp_order = api.orders[0]
    assert tp_order["orderType"] == "Limit"
    assert Decimal(tp_order["qty"]) == Decimal("0.1")
    assert any(order.get("orderFilter") == "tpslOrder" for order in api.orders)


def test_signal_executor_trails_stop_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_tp_ladder_bps="50",
        spot_tp_ladder_split_pct="100",
        spot_trailing_stop_activation_bps=10.0,
        spot_trailing_stop_distance_bps=20.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=1000.0, available=900.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    patch_tp_sources(monkeypatch, Decimal("0.5"))

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.50",
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

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", lambda *_, **__: response_payload
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "register_tp_ladder_plan",
        lambda *_, **__: None,
    )

    executor = SignalExecutor(bot)
    first = executor.execute_once()
    assert first.status == "filled"
    assert any(order.get("orderFilter") == "tpslOrder" for order in api.orders)

    bot._summary = {
        "actionable": False,
        "mode": "wait",
        "symbol": "BTCUSDT",
        "prices": {"BTCUSDT": 101.5},
    }

    second = executor.execute_once()
    assert second.status == "skipped"
    assert api.amendments, "trailing stop was not amended"
    amended = api.amendments[-1]
    new_trigger = Decimal(amended["triggerPrice"])
    expected_trigger = (Decimal("101.5") * (Decimal("1") - Decimal("0.002"))).quantize(
        Decimal("0.0001"), rounding=ROUND_DOWN
    )
    assert new_trigger == expected_trigger


def test_executor_and_ws_manager_share_tp_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    monkeypatch.setattr(signal_executor_module, "ws_manager", manager)
    monkeypatch.setattr(ws_manager_module, "manager", manager)

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

    patch_tp_sources(monkeypatch, Decimal("0.75"))

    monkeypatch.setattr(
        signal_executor_module,
        "_instrument_limits",
        lambda _api, _symbol: {
            "qty_step": Decimal("0.05"),
            "min_order_qty": Decimal("0.05"),
            "tick_size": Decimal("0.1"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("0"),
            "max_price": Decimal("0"),
        },
    )
    monkeypatch.setattr(
        ws_manager_module,
        "_instrument_limits",
        lambda _api, _symbol: {
            "qty_step": Decimal("0.05"),
            "min_order_qty": Decimal("0.05"),
            "tick_size": Decimal("0.1"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("0"),
            "max_price": Decimal("0"),
        },
    )

    execution_rows = [
        {
            "symbol": "BTCUSDT",
            "orderId": "primary",
            "execId": "fill-1",
            "execQty": "0.75",
        }
    ]

    def fake_realtime_rows(topic_keyword: str, snapshot=None):
        keyword = topic_keyword.lower()
        if "execution" in keyword:
            return execution_rows
        if "order" in keyword:
            return []
        return []

    monkeypatch.setattr(manager, "realtime_private_rows", fake_realtime_rows)

    class LadderAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def place_order(self, **kwargs: object) -> Mapping[str, object]:
            self.calls.append(kwargs)
            return {"orderId": f"tp-{len(self.calls)}"}

    api = LadderAPI()

    response_payload = {
        "retCode": 0,
        "result": {
            "orderId": "primary",
            "avgPrice": "100",
            "cumExecQty": "0.75",
            "cumExecValue": "75",
        },
    }

    executor = SignalExecutor(bot)
    placed_orders, _, stop_orders = executor._place_tp_ladder(
        api,
        settings,
        "BTCUSDT",
        "Buy",
        response_payload,
        ledger_rows=[],
        private_snapshot=None,
    )

    assert placed_orders
    assert stop_orders
    plan_state = manager._tp_ladder_plan.get("BTCUSDT") or {}
    stored_signature = plan_state.get("signature")
    stored_ladder = plan_state.get("ladder")
    assert isinstance(stored_signature, tuple)
    assert isinstance(stored_ladder, tuple)

    fill_row = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderId": "primary",
        "execId": "fill-1",
        "execQty": "0.75",
        "execPrice": "100",
    }

    monkeypatch.setattr(manager, "_reserved_sell_qty", lambda symbol: Decimal("0"))

    execute_payloads: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        manager,
        "_execute_tp_plan",
        lambda api_obj, symbol, payload, on_first_success=None: (
            execute_payloads.append(payload),
            True,
        )[1],
    )
    monkeypatch.setattr(
        manager,
        "_cancel_existing_tp_orders",
        lambda api_obj, symbol: None,
    )

    limits_cache = {
        "BTCUSDT": {
            "qty_step": Decimal("0.05"),
            "tick_size": Decimal("0.1"),
            "min_order_qty": Decimal("0.05"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("0"),
            "max_price": Decimal("0"),
        }
    }

    manager._regenerate_tp_ladder(
        fill_row,
        {"BTCUSDT": {"position_qty": Decimal("0.75"), "avg_cost": Decimal("100")}},
        config=[(Decimal("5"), Decimal("1"))],
        api=object(),
        limits_cache=limits_cache,
        settings=None,
    )

    assert execute_payloads
    payload_prices = [entry.get("price_text") for entry in execute_payloads[0]]
    payload_qtys = [entry.get("qty_text") for entry in execute_payloads[0]]
    expected_prices = [pair[0] for pair in stored_signature]
    expected_qtys = [pair[1] for pair in stored_signature]
    assert payload_prices == expected_prices
    assert payload_qtys == expected_qtys

    refreshed_state = manager._tp_ladder_plan.get("BTCUSDT") or {}
    assert refreshed_state.get("ladder") == stored_ladder


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
    assert len(api.orders) == 2
    order = api.orders[0]
    assert order["qty"] == "1.0"
    assert order["price"] == "100.3"
    stop_order = api.orders[1]
    assert stop_order["orderFilter"] == "tpslOrder"
    assert stop_order["qty"] == "1.0"
    assert result.order is not None
    ladder = result.order.get("take_profit_orders")
    assert ladder is not None
    assert ladder[0]["profit_bps"] == "5,9"
    assert ladder[0]["orderId"] == "stub-1"
    assert result.order["execution"]["sell_budget_base"] == "1.0"
    stop_info = result.order.get("stop_loss_orders")
    assert stop_info and stop_info[0]["orderType"] == "Market"


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


def test_collect_filled_base_total_passes_settings_to_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(testnet=False)
    bot = StubBot(summary, settings)

    executor = SignalExecutor(bot)

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_filled_base_from_private_ws",
        lambda self, symbol, **_: Decimal("0"),
    )

    captured: dict[str, object] = {}

    def fake_read_ledger(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        **kwargs: object,
    ) -> list[Mapping[str, object]]:
        captured["n"] = n
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)

    total = executor._collect_filled_base_total(
        "BTCUSDT",
        settings=settings,
        order_id="order-1",
        order_link_id=None,
        executed_base=Decimal("0"),
        ws_rows=[],
        ledger_rows=None,
    )

    assert total == Decimal("0")
    assert captured["n"] == 2000
    assert captured["settings"] is settings
    assert isinstance(captured["settings"], Settings)
    assert captured["settings"].testnet is False


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
    def empty_read_ledger(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows: list[Mapping[str, object]] = []
        if return_meta:
            return rows, None, True
        return rows

    def empty_inventory(
        *,
        events=None,
        settings=None,
        return_layers: bool = False,
        **_,
    ):
        if return_layers:
            return {}, {}
        return {}

    monkeypatch.setattr(signal_executor_module, "read_ledger", empty_read_ledger)
    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", empty_inventory)
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
    assert len(api.orders) == 3
    first, second, stop = api.orders
    assert Decimal(first["qty"]) == Decimal("0.45")
    assert Decimal(second["qty"]) == Decimal("0.30")
    assert stop["orderFilter"] == "tpslOrder"


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
    assert len(api.orders) == 2
    order = api.orders[0]
    assert Decimal(order["price"]) == Decimal("105")
    assert Decimal(order["qty"]) == Decimal("0.2")
    assert api.orders[1]["orderFilter"] == "tpslOrder"


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
    assert len(api.orders) == 4
    limit_orders = api.orders[:-1]
    stop_order = api.orders[-1]
    qtys = [Decimal(order["qty"]) for order in limit_orders]
    prices = [Decimal(order["price"]) for order in limit_orders]
    total_qty = Decimal("0.75")
    fractions = [Decimal("0.60"), Decimal("0.25"), Decimal("0.15")]
    step = Decimal("0.01")
    expected_qtys: list[Decimal] = []
    remaining = total_qty
    for idx, frac in enumerate(fractions):
        if idx == len(fractions) - 1:
            qty = remaining
        else:
            qty = (total_qty * frac).quantize(step, rounding=ROUND_DOWN)
            if qty > remaining:
                qty = remaining
        qty = qty.quantize(step, rounding=ROUND_DOWN)
        expected_qtys.append(qty)
        remaining -= qty
    if remaining > Decimal("0"):
        expected_qtys[-1] = (expected_qtys[-1] + remaining).quantize(
            step, rounding=ROUND_DOWN
        )
    assert qtys == expected_qtys
    assert prices == [Decimal("100.9"), Decimal("101.3"), Decimal("101.7")]
    assert stop_order["orderFilter"] == "tpslOrder"
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


def test_signal_executor_follows_up_partial_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
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
        SignalExecutor,
        "_ledger_rows_snapshot",
        lambda self, **_: ([], None),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_place_tp_ladder",
        lambda *args, **kwargs: ([], {}, []),
    )
    monkeypatch.setattr(
        SignalExecutor,
        "_maybe_notify_trade",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        signal_executor_module.ws_manager,
        "private_snapshot",
        lambda: {},
    )

    def fake_compute_notional(
        self,
        settings_obj: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> Tuple[float, float, bool, bool]:
        return 100.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    attempts: list[dict[str, object]] = []

    def _attempt_payload(executed_quote: str, leaves_base: str) -> dict[str, object]:
        return {
            "executed_base": str(Decimal(executed_quote) / Decimal("25000")),
            "executed_quote": executed_quote,
            "leaves_base": leaves_base,
        }

    responses = [
        {
            "result": {"cumExecQty": "0.002", "cumExecValue": "50"},
            "_local": {
                "order_audit": {
                    "quote_step": "0.01",
                    "min_order_amt": "5",
                    "qty_step": "0.00000001",
                },
                "attempts": [
                    _attempt_payload("50", "0.002"),
                ],
            },
        },
        {
            "result": {"cumExecQty": "0.002", "cumExecValue": "50"},
            "_local": {
                "order_audit": {
                    "quote_step": "0.01",
                    "min_order_amt": "5",
                    "qty_step": "0.00000001",
                },
                "attempts": [
                    _attempt_payload("50", "0"),
                ],
            },
        },
    ]

    def fake_place(api_obj, **kwargs):
        attempts.append(kwargs)
        response = responses[min(len(attempts) - 1, len(responses) - 1)]
        return copy.deepcopy(response)

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        fake_place,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    assert len(attempts) == 2
    assert result.order is not None
    partial_meta = result.order.get("partial_fill")
    assert isinstance(partial_meta, dict)
    assert partial_meta.get("status") == "complete"
    followups = partial_meta.get("followups")
    assert isinstance(followups, list) and followups
    assert followups[0].get("status") == "filled"
    assert followups[0].get("executed_quote") == "50"

    response_meta = result.response.get("_local") if isinstance(result.response, dict) else {}
    attempts_meta = response_meta.get("attempts") if isinstance(response_meta, dict) else None
    assert isinstance(attempts_meta, list)
    assert len(attempts_meta) == 2


def test_signal_executor_records_dust_after_partial_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "sell", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=100.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    api = StubAPI(total=500.0, available=400.0)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_maybe_notify_trade",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_place_tp_ladder",
        lambda *args, **kwargs: ([], {}, []),
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_maybe_flush_dust",
        lambda self, api_obj, settings_obj, now=None: None,
    )

    def fake_compute_notional(
        self,
        settings_obj: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> Tuple[float, float, bool, bool]:
        return 50.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_sell_notional_from_holdings",
        lambda self, api_obj, symbol, **kwargs: (50.0, {}, 5.0),
    )

    response_payload = {
        "result": {"orderId": "dust-stub"},
        "_local": {
            "order_audit": {
                "qty_step": "0.0001",
                "min_order_amt": "5",
                "price_payload": "100.0",
            }
        },
    }
    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: copy.deepcopy(response_payload),
    )

    partial_meta = {
        "status": "complete_below_minimum",
        "remaining_quote": "3.2",
        "min_order_amt": "5",
    }

    def fake_chase(self, **kwargs):
        return response_payload, Decimal("0.01"), Decimal("50"), partial_meta

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_chase_partial_fill",
        fake_chase,
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "filled"
    dust_entry = executor._dust_positions.get("BTCUSDT")
    assert dust_entry is not None
    assert executor._decimal_from(dust_entry.get("quote")) == Decimal("3.2")
    assert (
        executor._decimal_from(dust_entry.get("min_notional"))
        == Decimal("5")
    )


def test_signal_executor_liquidates_dust_via_ioc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "sell", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=False)
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    api = StubAPI(total=1000.0, available=900.0)

    def fake_snapshot(
        api_obj,
        symbol: str,
        *,
        include_limits: bool = True,
        include_price: bool = True,
        include_balances: bool = True,
        force_refresh: bool = False,
    ) -> SpotTradeSnapshot:
        if getattr(fake_snapshot, "calls", 0) == 0:
            fake_snapshot.calls = 1
            return SpotTradeSnapshot(
                symbol=symbol,
                price=Decimal("2000"),
                balances={"ETH": Decimal("0.003")},
                limits={
                    "min_order_amt": Decimal("5"),
                    "base_coin": "ETH",
                    "quote_coin": "USDT",
                },
            )
        return SpotTradeSnapshot(
            symbol=symbol,
            price=Decimal("2000"),
            balances={"ETH": Decimal("0")},
            limits={
                "min_order_amt": Decimal("5"),
                "base_coin": "ETH",
                "quote_coin": "USDT",
            },
        )

    fake_snapshot.calls = 0

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_trade_snapshot",
        fake_snapshot,
    )

    def fake_prepare(
        api_obj,
        symbol: str,
        side: str,
        qty: object,
        **kwargs: object,
    ) -> PreparedSpotMarketOrder:
        return PreparedSpotMarketOrder(
            payload={
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "qty": "0.003",
                "price": "2000",
                "timeInForce": "GTC",
                "orderFilter": "Order",
                "accountType": "UNIFIED",
            },
            audit={"min_order_amt": "5"},
        )

    monkeypatch.setattr(
        signal_executor_module,
        "prepare_spot_market_order",
        fake_prepare,
    )

    executor._register_dust(
        "ETHUSDT",
        quote=Decimal("6"),
        min_notional=Decimal("5"),
        price=Decimal("2000"),
        source="test",
    )

    executor._maybe_flush_dust(api, settings, now=time.time())

    assert api.orders
    order = api.orders[0]
    assert order["timeInForce"] == "IOC"
    assert order["side"] == "Sell"
    assert order["qty"] == "0.003"
    assert executor._dust_positions.get("ETHUSDT") is None


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
            [{"orderType": "Market", "triggerPrice": "98.0"}],
        )

    monkeypatch.setattr(SignalExecutor, "_place_tp_ladder", fake_tp)

    ledger_rows = [
        {"symbol": "BTCUSDT", "category": "spot", "side": "Buy", "execPrice": 100.0, "execQty": 0.2, "execFee": 0.0},
        {"symbol": "BTCUSDT", "category": "spot", "side": "Sell", "execPrice": 120.0, "execQty": 0.1, "execFee": 0.0},
    ]
    def ledger_snapshot(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows = list(ledger_rows)
        if return_meta:
            return rows, "ledger-last", True
        return rows

    monkeypatch.setattr(signal_executor_module, "read_ledger", ledger_snapshot)

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

    monkeypatch.setattr(
        SignalExecutor,
        "_place_tp_ladder",
        lambda *args, **kwargs: ([], exec_stats, []),
    )

    def empty_snapshot(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows: list[Mapping[str, object]] = []
        if return_meta:
            return rows, None, True
        return rows

    def empty_inventory(*, events=None, settings=None, return_layers: bool = False, **_):
        if return_layers:
            return {}, {}
        return {}

    monkeypatch.setattr(signal_executor_module, "read_ledger", empty_snapshot)
    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", empty_inventory)

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
    incremental_rows = [
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

    snapshots: list[tuple[list[Mapping[str, object]], str]] = [
        (before_rows, "before"),
        (incremental_rows, "after"),
    ]

    def fake_snapshot(
        self,
        limit: int = 2000,
        *,
        settings: Settings | None = None,
        last_exec_id: str | None = None,
    ) -> tuple[list[Mapping[str, object]], str]:
        if snapshots:
            rows, last_id = snapshots.pop(0)
            return [dict(entry) for entry in rows], last_id
        return [dict(entry) for entry in incremental_rows], "after"

    monkeypatch.setattr(SignalExecutor, "_ledger_rows_snapshot", fake_snapshot)

    helper_calls: list[tuple[int, int]] = []

    def fake_realized_delta(
        self,
        rows_before,
        rows_after,
        symbol,
        *,
        new_rows=None,
        **_kwargs,
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


def test_signal_executor_long_ledger_realized_pnl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "sell", "symbol": "BTCUSDT"}
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
            "orderId": "long-ledger-sell",
            "cumExecQty": "50",
            "cumExecValue": "1500",
            "avgPrice": "30",
        },
        "_local": {
            "order_audit": {"qty_step": "0.0001", "limit_price": "30"},
            "order_payload": {"qty": "50", "price": "30"},
        },
    }

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: response_payload,
    )

    total_events = 2100
    before_full: list[dict[str, object]] = []
    for idx in range(total_events):
        price = 10.0 if idx < 100 else 20.0
        before_full.append(
            {
                "symbol": "BTCUSDT",
                "category": "spot",
                "side": "Buy",
                "execPrice": price,
                "execQty": 1.0,
                "execFee": 0.0,
                "execId": f"buy-{idx}",
            }
        )

    truncated_before = before_full[-2000:]
    before_last_id = truncated_before[-1]["execId"]

    sell_row = {
        "symbol": "BTCUSDT",
        "category": "spot",
        "side": "Sell",
        "execPrice": "30",
        "execQty": "50",
        "execFee": "0",
        "execId": "sell-2100",
    }

    def fake_read_ledger(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        if last_exec_id == before_last_id:
            rows = [sell_row]
            marker_found = True
            last_seen = sell_row["execId"]
        elif last_exec_id is not None:
            rows = truncated_before[-(n or len(truncated_before)) :] + [sell_row]
            marker_found = False
            last_seen = sell_row["execId"]
        else:
            rows = truncated_before[-(n or len(truncated_before)) :]
            marker_found = True
            last_seen = before_last_id

        snapshot = [dict(entry) for entry in rows]
        if return_meta:
            return snapshot, last_seen, marker_found
        return snapshot

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)

    inventory_state, layer_state = spot_pnl_module.spot_inventory_and_pnl(
        events=before_full, return_layers=True
    )

    def fake_inventory(
        *,
        events=None,
        settings=None,
        return_layers: bool = False,
        **_,
    ):
        inventory_copy = copy.deepcopy(inventory_state)
        layers_copy = copy.deepcopy(layer_state)
        if return_layers:
            return inventory_copy, layers_copy
        return inventory_copy

    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", fake_inventory)

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
    assert "+523.81 USDT" in message

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

    inventory_state, layer_state = spot_pnl_module.spot_inventory_and_pnl(
        events=ledger_rows_before, return_layers=True
    )

    def fake_inventory(
        *,
        events=None,
        settings=None,
        return_layers: bool = False,
        **_,
    ):
        inventory_copy = copy.deepcopy(inventory_state)
        layers_copy = copy.deepcopy(layer_state)
        if return_layers:
            return inventory_copy, layers_copy
        return inventory_copy

    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", fake_inventory)

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
        **_kwargs,
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

def test_signal_executor_skips_on_wide_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
    )
    bot = StubBot(summary, settings)

    wide_orderbook = {
        "result": {
            "a": [["110.0", "50"]],
            "b": [["100.0", "50"]],
        }
    }

    api = StubAPI(total=1000.0, available=800.0, orderbook_payload=wide_orderbook)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )
    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: pytest.fail("liquidity guard should skip before order placement"),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None and "спред" in result.reason
    assert result.context is not None
    guard_meta = result.context.get("liquidity_guard")
    assert isinstance(guard_meta, dict)
    assert guard_meta.get("spread_bps") > guard_meta.get("max_spread_bps", 0.0)


def test_liquidity_guard_skips_on_recent_spread_violation() -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        ai_spread_compression_window_sec=5.0,
    )
    bot = StubBot(summary, settings)

    executor = SignalExecutor(bot)

    orderbook_top = {
        "best_ask": 100.0,
        "best_bid": 99.95,
        "best_ask_qty": 1.0,
        "best_bid_qty": 1.0,
        "spread_bps": 5.0,
        "ts": time.time() * 1000.0,
        "spread_window_stats": {
            "window_sec": 5.0,
            "observations": 5,
            "max_bps": 55.0,
            "min_bps": 5.0,
            "avg_bps": 20.0,
            "latest_bps": 5.0,
            "age_ms": 0.0,
        },
    }

    decision = executor._apply_liquidity_guard(
        "Buy",
        100.0,
        orderbook_top,
        settings=settings,
        price_hint=100.0,
    )

    assert decision is not None
    assert decision.get("decision") == "skip"
    reason = decision.get("reason")
    assert isinstance(reason, str) and "сжатия" in reason
    context = decision.get("context")
    assert isinstance(context, dict)
    stats = context.get("spread_window_stats")
    assert isinstance(stats, dict)
    assert stats.get("max_bps") == 55.0


def test_signal_executor_skips_on_thin_top_of_book(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        ai_top_depth_coverage=0.9,
        ai_top_depth_shortfall_usd=2.0,
        allow_partial_fills=False,
        twap_enabled=False,
    )
    bot = StubBot(summary, settings)

    thin_orderbook = {
        "result": {
            "a": [["100.0", "0.1"]],
            "b": [["99.9", "50"]],
        }
    }

    api = StubAPI(total=1000.0, available=800.0, orderbook_payload=thin_orderbook)
    monkeypatch.setattr(signal_executor_module, "get_api_client", lambda: api)
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_compute_notional(
        self,
        settings: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> tuple[float, float, bool, bool]:
        return 100.0, available_equity, False, False

    monkeypatch.setattr(
        SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )
    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        lambda *args, **kwargs: pytest.fail("liquidity guard should skip before order placement"),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "skipped"
    assert result.reason is not None
    assert "первом уровне" in result.reason or "дефицит" in result.reason
    assert result.context is not None
    guard_meta = result.context.get("liquidity_guard")
    assert isinstance(guard_meta, dict)
    assert guard_meta.get("coverage_ratio", 1.0) < guard_meta.get("coverage_threshold", 1.0)


def test_liquidity_guard_relaxes_when_partial_supported() -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ws_watchdog_enabled=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=0.0,
        ai_top_depth_coverage=0.8,
        ai_top_depth_shortfall_usd=10.0,
        allow_partial_fills=True,
        twap_enabled=True,
    )
    bot = StubBot(summary, settings)

    executor = SignalExecutor(bot)

    orderbook_top = {
        "best_ask": 100.0,
        "best_bid": 99.9,
        "best_ask_qty": 0.2,
        "best_bid_qty": 10.0,
        "spread_bps": 10.0,
        "ts": time.time() * 1000.0,
    }

    decision = executor._apply_liquidity_guard(
        "Buy",
        100.0,
        orderbook_top,
        settings=settings,
        price_hint=100.0,
    )

    assert decision is not None
    assert decision.get("decision") == "relaxed"
    context = decision.get("context")
    assert isinstance(context, dict)
    assert context.get("guard_relaxed") is True
    assert "coverage" in (context.get("liquidity_guard_reasons") or [])


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
    def empty_read(
        n: int | None = 2000,
        *,
        settings: object | None = None,
        network: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows: list[Mapping[str, object]] = []
        if return_meta:
            return rows, None, True
        return rows

    def empty_inventory(*, events=None, settings=None, return_layers: bool = False, **_):
        if return_layers:
            return {}, {}
        return {}

    monkeypatch.setattr(signal_executor_module, "read_ledger", empty_read)
    monkeypatch.setattr(signal_executor_module, "spot_inventory_and_pnl", empty_inventory)
    monkeypatch.setattr(
        SignalExecutor, "_place_tp_ladder", lambda *args, **kwargs: ([], None, [])
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


def test_price_limit_backoff_preserves_price_cap_without_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    attempt = {"value": 0}

    def failing_place(*_args, **_kwargs):
        attempt["value"] += 1
        base_details = {
            "requested_quote": "100.0",
            "available_quote": "42.0",
            "side": "buy",
            "price_limit_hit": True,
        }
        if attempt["value"] == 1:
            details = dict(base_details)
            details["price_cap"] = "123.45"
        else:
            details = dict(base_details)
        raise OrderValidationError(
            "Недостаточная глубина стакана для заданного объёма в котировочной валюте.",
            code="insufficient_liquidity",
            details=details,
        )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", failing_place
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
    executor._symbol_quarantine.pop("ETHUSDT", None)
    backoff_state = executor._price_limit_backoff.get("ETHUSDT")
    resume_time = None
    if isinstance(backoff_state, dict):
        ttl = backoff_state.get("quarantine_ttl")
        last_updated = backoff_state.get("last_updated")
        if isinstance(ttl, (int, float)) and isinstance(last_updated, (int, float)):
            resume_time = float(last_updated + ttl + 10.0)
    if resume_time is None:
        resume_time = base_time + signal_executor_module._PRICE_LIMIT_LIQUIDITY_TTL + 10.0
    time_state["value"] = resume_time
    second_result = executor.execute_once()

    assert first_result.status == "skipped"
    assert second_result.status == "skipped"
    assert attempt["value"] == 2

    backoff_state = executor._price_limit_backoff.get("ETHUSDT")
    assert isinstance(backoff_state, dict)
    assert backoff_state.get("price_cap") == pytest.approx(123.45)

    adjusted_notional, adjusted_slippage, adjustments = executor._apply_price_limit_backoff(
        "ETHUSDT",
        "Buy",
        notional_quote=100.0,
        slippage_pct=0.5,
        min_notional=5.0,
    )

    assert adjusted_notional <= 100.0
    assert adjusted_slippage >= 0.5
    assert isinstance(adjustments, dict)
    assert adjustments.get("price_cap") == pytest.approx(123.45)


def test_signal_executor_skips_price_limit_price_deviation(monkeypatch: pytest.MonkeyPatch) -> None:
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
        raise OrderValidationError(
            "Ожидаемая цена превышает допустимый предел для инструмента.",
            code="price_deviation",
            details={
                "side": "buy",
                "limit_price": "100.1",
                "price_cap": "100.05",
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
    assert call_counter["value"] == 1

    second_result = executor.execute_once()

    assert second_result.status == "skipped"
    assert call_counter["value"] == 1


def test_signal_executor_retries_price_limit_price_deviation_with_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    call_records: list[dict[str, object]] = []

    def fake_place(*_args, **kwargs):
        call_records.append(dict(kwargs))
        if len(call_records) == 1:
            raise OrderValidationError(
                "Ожидаемая цена превышает допустимый предел для инструмента.",
                code="price_deviation",
                details={
                    "side": "buy",
                    "limit_price": "0.332757",
                    "price_cap": "0.2883219",
                    "price_limit_hit": True,
                    "requested_quote": "150.0",
                    "available_quote": "45.0",
                },
            )
        return {"retCode": 0, "result": {"orderId": "stub-order"}}

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", fake_place
    )

    def fake_read_ledger(
        limit: int | None = None,
        *,
        settings: object | None = None,
        last_exec_id: str | None = None,
        return_meta: bool = False,
        **_: object,
    ):
        rows: list[dict[str, object]] = []
        if return_meta:
            return rows, None, {}
        return rows

    monkeypatch.setattr(signal_executor_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(
        SignalExecutor, "_place_tp_ladder", lambda *args, **kwargs: ([], {}, [])
    )
    monkeypatch.setattr(signal_executor_module.ws_manager, "private_snapshot", lambda: {})

    executor = SignalExecutor(bot)

    result = executor.execute_once()

    assert result.status == "filled"
    assert len(call_records) == 2

    first_qty = float(call_records[0]["qty"])
    retry_qty = float(call_records[1]["qty"])
    assert retry_qty < first_qty

    assert result.order is not None
    assert result.order.get("notional_quote") == pytest.approx(retry_qty)

    assert result.context is not None
    adjustments = result.context.get("price_limit_adjustments")
    assert isinstance(adjustments, list) and adjustments
    assert adjustments[0].get("notional_quote") == pytest.approx(retry_qty)

    backoff_meta = result.context.get("price_limit_backoff")
    assert isinstance(backoff_meta, dict)
    assert backoff_meta.get("retries") == 1
    assert "ETHUSDT" not in executor._symbol_quarantine


def test_signal_executor_handles_runtime_price_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def failing_place(*_args, **_kwargs):
        call_counter["value"] += 1
        raise RuntimeError(
            "Bybit error 170193: Buy order price cannot be higher than 0.2824137USDT"
        )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", failing_place
    )

    base_time = 1700000000.0
    time_state = {"value": base_time}

    def fake_now(self):
        current = time_state["value"]
        time_state["value"] += 30.0
        return current

    monkeypatch.setattr(SignalExecutor, "_current_time", fake_now)

    executor = SignalExecutor(bot)

    result = executor.execute_once()

    assert result.status == "skipped"
    assert call_counter["value"] == 1
    assert result.context is not None
    assert result.context.get("validation_code") == "price_deviation"
    details = result.context.get("validation_details")
    assert isinstance(details, dict)
    assert details.get("price_limit_hit") is True
    assert details.get("price_cap") == "0.2824137"

    backoff_state = executor._price_limit_backoff.get("ETHUSDT")
    assert isinstance(backoff_state, dict)
    assert backoff_state.get("retries") == 1
    assert executor._symbol_quarantine.get("ETHUSDT") is not None


def test_signal_executor_handles_runtime_price_floor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "sell", "symbol": "ETHUSDT"}
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

    def failing_place(*_args, **_kwargs):
        call_counter["value"] += 1
        raise RuntimeError(
            "Bybit error 170194: Sell order price cannot be lower than 99.5USDT"
        )

    monkeypatch.setattr(
        signal_executor_module, "place_spot_market_with_tolerance", failing_place
    )

    base_time = 1700000000.0
    time_state = {"value": base_time}

    def fake_now(self):
        current = time_state["value"]
        time_state["value"] += 30.0
        return current

    monkeypatch.setattr(SignalExecutor, "_current_time", fake_now)

    executor = SignalExecutor(bot)

    result = executor.execute_once()

    assert result.status == "skipped"
    assert call_counter["value"] == 1
    assert result.context is not None
    assert result.context.get("validation_code") == "price_deviation"
    details = result.context.get("validation_details")
    assert isinstance(details, dict)
    assert details.get("price_limit_hit") is True
    assert details.get("price_floor") == "99.5"

    backoff_state = result.context.get("price_limit_backoff")
    assert isinstance(backoff_state, dict)
    assert backoff_state.get("retries") == 1
    assert backoff_state.get("price_floor") == pytest.approx(99.5)

    executor_state = executor._price_limit_backoff.get("ETHUSDT")
    assert isinstance(executor_state, dict)
    assert executor_state.get("price_floor") == pytest.approx(99.5)
    assert executor._symbol_quarantine.get("ETHUSDT") is not None


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
    assert result.order["notional_quote"] == pytest.approx(6.69, rel=1e-3)


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


def test_fee_conversion_prevents_daily_loss_false_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    now_ms = int(now * 1000)
    events = [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "0.01",
            "feeCurrency": "BTC",
            "category": "spot",
            "execTime": now_ms,
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "0.01",
            "feeCurrency": "BTC",
            "category": "spot",
            "execTime": now_ms + 1000,
        },
    ]

    ledger_path = tmp_path / "executions.testnet.jsonl"
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    fifo_books = spot_fifo_module.spot_fifo_pnl(ledger_path)
    assert fifo_books
    book = fifo_books.get("BTCUSDT")
    assert book is not None
    assert book["realized_pnl"] == pytest.approx(-2.0)
    assert book["position_qty"] == pytest.approx(0.0)

    inventory, layer_state = spot_pnl_module.spot_inventory_and_pnl(
        events=events, return_layers=True
    )
    spot_state = inventory.get("BTCUSDT")
    assert spot_state is not None
    assert spot_state["realized_pnl"] == pytest.approx(-2.0)
    assert spot_state["position_qty"] == pytest.approx(0.0)
    layers_for_symbol = layer_state.get("BTCUSDT")
    assert layers_for_symbol is not None
    assert layers_for_symbol.get("position_qty") == pytest.approx(0.0)

    summary = pnl_module._build_daily_summary(events)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    day_bucket = summary.get(day_key, {})
    symbol_bucket = day_bucket.get("BTCUSDT", {})
    assert symbol_bucket.get("spot_fees") == pytest.approx(2.0)
    assert symbol_bucket.get("spot_net") == pytest.approx(-2.0)

    summary_path = tmp_path / "pnl_daily.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pnl_module, "_SUMMARY", summary_path)
    monkeypatch.setattr(pnl_module, "read_ledger", lambda *_, **__: events)
    pnl_module.invalidate_daily_pnl_cache()

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success=False: (None, (100.0, 100.0), None, {}),
    )

    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=1.0)
    bot = StubBot({"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}, settings)
    executor = SignalExecutor(bot)
    executor._current_time = lambda: now  # type: ignore[assignment]

    guard = executor._daily_loss_guard(settings)
    assert guard is not None
    message, context = guard
    assert context.get("guard") == "daily_loss_limit"
    assert "Дневной убыток" in message
    assert context.get("loss_value") == pytest.approx(2.0)
    assert context.get("loss_percent") == pytest.approx(2.0)


def test_daily_loss_guard_handles_missing_fee_currency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    now_ms = int(now * 1000)
    events = [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "0.003",
            "category": "spot",
            "execTime": now_ms,
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "0.003",
            "category": "spot",
            "execTime": now_ms + 1000,
        },
    ]

    ledger_path = tmp_path / "executions.testnet.jsonl"
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    fifo_books = spot_fifo_module.spot_fifo_pnl(ledger_path)
    book = fifo_books.get("BTCUSDT")
    assert book is not None
    assert book["realized_pnl"] == pytest.approx(-0.6)
    assert book["position_qty"] == pytest.approx(0.0)

    summary = pnl_module._build_daily_summary(events)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    day_bucket = summary.get(day_key, {})
    symbol_bucket = day_bucket.get("BTCUSDT", {})
    assert symbol_bucket.get("spot_fees") == pytest.approx(0.6)
    assert symbol_bucket.get("spot_net") == pytest.approx(-0.6)
    assert events[0].get("feeCurrency") == "BTC"
    assert events[1].get("feeCurrency") == "BTC"

    summary_path = tmp_path / "pnl_daily.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pnl_module, "_SUMMARY", summary_path)
    monkeypatch.setattr(pnl_module, "read_ledger", lambda *_, **__: events)
    pnl_module.invalidate_daily_pnl_cache()

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        lambda self, require_success=False: (None, (30.0, 30.0), None, {}),
    )

    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=1.0)
    bot = StubBot({"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}, settings)
    executor = SignalExecutor(bot)
    executor._current_time = lambda: now  # type: ignore[assignment]

    guard = executor._daily_loss_guard(settings)
    assert guard is not None
    message, context = guard
    assert context.get("guard") == "daily_loss_limit"
    assert "Дневной убыток" in message
    assert context.get("loss_value") == pytest.approx(0.6)
    assert context.get("loss_percent") == pytest.approx(2.0)


def test_signal_executor_daily_loss_treats_rebates_as_benefit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = time.time()
    now_ms = int(now * 1000)
    events = [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "-0.01",
            "feeCurrency": "BTC",
            "category": "spot",
            "execTime": now_ms,
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": "100",
            "execQty": "1",
            "execFee": "-0.01",
            "feeCurrency": "BTC",
            "category": "spot",
            "execTime": now_ms + 1000,
        },
    ]

    summary = pnl_module._build_daily_summary(events)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    day_bucket = summary.get(day_key, {})
    symbol_bucket = day_bucket.get("BTCUSDT", {})
    assert symbol_bucket.get("spot_pnl") == pytest.approx(0.0)
    assert symbol_bucket.get("spot_fees") == pytest.approx(-2.0)
    assert symbol_bucket.get("spot_net") == pytest.approx(2.0)

    summary_path = tmp_path / "pnl_daily.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pnl_module, "_SUMMARY", summary_path)
    monkeypatch.setattr(pnl_module, "read_ledger", lambda *_, **__: events)
    pnl_module.invalidate_daily_pnl_cache()

    monkeypatch.setattr(
        signal_executor_module, "daily_pnl", lambda **_: summary
    )

    settings = Settings(ai_enabled=True, ai_daily_loss_limit_pct=1.0)
    bot = StubBot({"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}, settings)
    executor = SignalExecutor(bot)
    executor._current_time = lambda: now  # type: ignore[assignment]

    guard = executor._daily_loss_guard(settings)
    assert guard is None


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


def test_portfolio_loss_guard_triggers_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        ai_portfolio_loss_limit_pct=2.0,
        ai_kill_switch_cooldown_min=15.0,
    )
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    positions = {
        "BTCUSDT": {
            "qty": 1.0,
            "avg_cost": 100.0,
            "realized_pnl": 0.0,
            "hold_seconds": 300.0,
            "price": 90.0,
            "pnl_value": -10.0,
            "pnl_bps": -1000.0,
            "quote_notional": 90.0,
        },
        "ETHUSDT": {
            "qty": 2.0,
            "avg_cost": 50.0,
            "realized_pnl": 0.0,
            "hold_seconds": 180.0,
            "price": 45.0,
            "pnl_value": -10.0,
            "pnl_bps": -1000.0,
            "quote_notional": 90.0,
        },
    }

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_collect_open_positions",
        lambda self, settings, summary, **_: positions,
    )

    captured: Dict[str, object] = {}

    def fake_set_pause(minutes: float, reason: str) -> float:
        captured["minutes"] = minutes
        captured["reason"] = reason
        return 1700000000.0

    monkeypatch.setattr(signal_executor_module, "activate_kill_switch", fake_set_pause)

    now = time.time()
    message, context = executor._portfolio_loss_guard(
        settings,
        summary,
        total_equity=800.0,
        current_time=now,
        summary_meta=(None, None),
        price_meta=(None, None),
    )

    assert message is not None and "Совокупный" in message
    assert context.get("guard") == "portfolio_loss_limit"
    assert context.get("loss_value") == pytest.approx(20.0)
    assert context.get("loss_percent") == pytest.approx(2.5)
    assert captured.get("minutes") == pytest.approx(15.0)
    assert context.get("kill_switch_until") == pytest.approx(1700000000.0)


def test_loss_streak_guard_triggers_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(
        ai_enabled=True,
        ai_kill_switch_loss_streak=3,
        ai_kill_switch_cooldown_min=30.0,
    )
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    snapshot = TradePerformanceSnapshot(
        results=(-1, -1, -1),
        win_streak=0,
        loss_streak=3,
        realised_sum=-15.0,
        average_pnl=-5.0,
        sample_count=3,
        last_exit_ts=1700.0,
        window_ms=86_400_000,
    )
    executor._performance_state = snapshot

    captured: Dict[str, object] = {}

    def fake_set_pause(minutes: float, reason: str) -> float:
        captured["minutes"] = minutes
        captured["reason"] = reason
        return 1700000500.0

    monkeypatch.setattr(signal_executor_module, "activate_kill_switch", fake_set_pause)

    guard = executor._loss_streak_guard(settings)
    assert guard is not None
    message, context = guard
    assert "Серия" in message
    assert context.get("guard") == "loss_streak_limit"
    assert context.get("loss_streak") == 3
    assert captured.get("minutes") == pytest.approx(30.0)
    assert context.get("kill_switch_until") == pytest.approx(1700000500.0)


def test_kill_switch_guard_reports_active_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "BTCUSDT"}
    settings = Settings(ai_enabled=True)
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    monkeypatch.setattr(
        signal_executor_module,
        "kill_switch_state",
        lambda: KillSwitchState(paused=True, until=1700.0, reason="manual"),
    )

    guard = executor._kill_switch_guard()
    assert guard is not None
    message, context = guard
    assert "Автоторговля" in message
    assert context.get("guard") == "kill_switch"
    assert context.get("reason") == "manual"
    assert context.get("until") == pytest.approx(1700.0)


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
    ) -> Tuple[Optional[object], Tuple[float, float], Optional[float], Dict[str, object]]:
        return None, (1000.0, 1000.0), None, {}

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


def test_automation_loop_sweeper_forces_retry() -> None:
    class _Executor:
        def __init__(self) -> None:
            self.calls = 0

        def current_signature(self) -> str:
            return "sig"

        def settings_marker(self) -> tuple[bool, bool, bool]:
            return (True, True, True)

        def execute_once(self) -> ExecutionResult:
            self.calls += 1
            return ExecutionResult(status="dry_run")

    executor = _Executor()
    sweeper_state = {"count": 0}

    def sweeper() -> bool:
        sweeper_state["count"] += 1
        return sweeper_state["count"] == 2

    loop = AutomationLoop(
        executor,
        poll_interval=0.0,
        success_cooldown=60.0,
        error_backoff=0.0,
        sweeper=sweeper,
    )

    first_delay = loop._tick()
    assert executor.calls == 1
    assert first_delay == 60.0

    second_delay = loop._tick()
    assert executor.calls == 2
    assert second_delay == 60.0
    assert sweeper_state["count"] >= 2


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


def test_automation_loop_retries_transient_status_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": False, "mode": "wait", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-skip")

    executor = SignalExecutor(bot)

    class FakeTime:
        def __init__(self) -> None:
            self.value = 0.0

        def monotonic(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    fake_time = FakeTime()
    monkeypatch.setattr(signal_executor_module, "time", fake_time)

    call_count = {"value": 0}
    statuses = [
        ExecutionResult(status="skipped", reason="not actionable"),
        ExecutionResult(status="dry_run"),
    ]

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        return statuses[min(call_count["value"] - 1, len(statuses) - 1)]

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(
        executor,
        poll_interval=3.5,
        success_cooldown=1.0,
        error_backoff=0.0,
    )

    first_delay = loop._tick()
    assert call_count["value"] == 1
    assert first_delay == pytest.approx(1.0)

    second_delay = loop._tick()
    assert call_count["value"] == 1
    assert second_delay == pytest.approx(1.0)

    fake_time.advance(1.0)

    third_delay = loop._tick()
    assert call_count["value"] == 2
    assert third_delay == pytest.approx(1.0)


def test_automation_loop_retries_disabled_status_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {"actionable": True, "mode": "buy", "symbol": "ETHUSDT"}
    settings = Settings(ai_enabled=True, dry_run=True)
    bot = StubBot(summary, settings, fingerprint="sig-disabled")

    executor = SignalExecutor(bot)

    class FakeTime:
        def __init__(self) -> None:
            self.value = 0.0

        def monotonic(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    fake_time = FakeTime()
    monkeypatch.setattr(signal_executor_module, "time", fake_time)

    call_count = {"value": 0}
    statuses = [
        ExecutionResult(status="disabled", reason="guard tripped"),
        ExecutionResult(status="dry_run"),
    ]

    def fake_execute_once() -> ExecutionResult:
        call_count["value"] += 1
        return statuses[min(call_count["value"] - 1, len(statuses) - 1)]

    executor.execute_once = fake_execute_once  # type: ignore[assignment]

    loop = AutomationLoop(
        executor,
        poll_interval=5.0,
        success_cooldown=3.0,
        error_backoff=0.0,
    )

    first_delay = loop._tick()
    assert call_count["value"] == 1
    assert first_delay == pytest.approx(3.0)

    second_delay = loop._tick()
    assert call_count["value"] == 1
    assert second_delay == pytest.approx(3.0)

    fake_time.advance(3.0)

    third_delay = loop._tick()
    assert call_count["value"] == 2
    assert third_delay == pytest.approx(3.0)


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
    assert second_delay == pytest.approx(1.5, rel=1e-3)

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


def _configure_sell_snapshot_environment(
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, object],
) -> dict[str, object]:
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=10.0,
        spot_cash_reserve_pct=0.0,
        spot_max_cap_per_trade_pct=0.0,
    )
    bot = StubBot(summary, settings)
    now = time.time()

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_current_time",
        lambda self: now,
    )
    monkeypatch.setattr(
        signal_executor_module,
        "resolve_trade_symbol",
        lambda symbol, api, allow_nearest=True: (symbol, {"reason": "exact"}),
    )

    def fake_resolve_wallet(self, *, require_success: bool):
        api = object()
        totals = (1000.0, 800.0)
        return api, totals, None, {}

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_resolve_wallet",
        fake_resolve_wallet,
    )

    def fake_compute_notional(
        self,
        settings_obj: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> Tuple[float, float, bool, bool]:
        return 100.0, available_equity, False, False

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_compute_notional",
        fake_compute_notional,
    )

    captured: dict[str, object] = {}

    def fake_sell_notional(
        self,
        api: object,
        symbol: str,
        *,
        summary: Optional[Mapping[str, object]] = None,
        expected_base_requirement: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, object]], Optional[float]]:
        captured["expected_base_requirement"] = expected_base_requirement
        return 100.0, {"price_snapshot": summary.get("price") if summary else None}, None

    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_sell_notional_from_holdings",
        fake_sell_notional,
    )

    def fake_place_order(api_obj: object, **payload: object) -> dict[str, object]:
        return {"retCode": 0, "result": {"orderId": "snapshot-check"}}

    monkeypatch.setattr(
        signal_executor_module,
        "place_spot_market_with_tolerance",
        fake_place_order,
    )
    monkeypatch.setattr(
        signal_executor_module.SignalExecutor,
        "_ledger_rows_snapshot",
        lambda self, **_: ([], None),
    )

    executor = SignalExecutor(bot)
    result = executor.execute_once()
    assert result.status == "filled"
    return captured


def test_signal_executor_sell_skips_stale_summary_price_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "ETHUSDT",
        "price": 25.0,
        "age_seconds": 12.0,
        "price_meta": {"age_seconds": 12.0},
    }

    captured = _configure_sell_snapshot_environment(monkeypatch, summary)
    assert captured.get("expected_base_requirement") is None


def test_signal_executor_sell_skips_stale_price_meta_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "ETHUSDT",
        "price": 40.0,
        "age_seconds": 0.5,
        "price_meta": {"age_seconds": 10.0},
    }

    captured = _configure_sell_snapshot_environment(monkeypatch, summary)
    assert captured.get("expected_base_requirement") is None


def test_signal_executor_sell_uses_fresh_summary_price_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = {
        "actionable": True,
        "mode": "sell",
        "symbol": "ETHUSDT",
        "price": 50.0,
        "age_seconds": 0.5,
        "price_meta": {"age_seconds": 0.1},
    }

    captured = _configure_sell_snapshot_environment(monkeypatch, summary)
    expected = captured.get("expected_base_requirement")
    assert expected is not None
    assert expected == pytest.approx(2.0)



def test_stop_loss_enforced_for_impulse(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "actionable": True,
        "mode": "buy",
        "symbol": "BTCUSDT",
        "watchlist": [
            {
                "symbol": "BTCUSDT",
                "impulse_signal": True,
                "impulse_strength": math.log(2.0),
                "volume_impulse": {"1h": math.log(2.0)},
            }
        ],
    }
    settings = Settings(
        ai_enabled=True,
        dry_run=False,
        ai_risk_per_trade_pct=1.0,
        spot_cash_reserve_pct=5.0,
        spot_stop_loss_bps=0.0,
        spot_trailing_stop_distance_bps=0.0,
        spot_trailing_stop_activation_bps=0.0,
        spot_impulse_stop_loss_bps=55.0,
    )
    bot = StubBot(summary, settings)
    executor = SignalExecutor(bot)

    api = StubAPI(total=1000.0, available=900.0)
    placed: list[dict[str, object]] = []

    def fake_place(self, **kwargs: object) -> dict[str, object]:
        placed.append(kwargs)
        return {"result": {"orderId": "sl-order"}}

    monkeypatch.setattr(api, "place_order", fake_place.__get__(api, StubAPI))

    stop_orders = executor._place_stop_loss_orders(
        api,
        settings,
        "BTCUSDT",
        "buy",
        avg_price=Decimal("100"),
        qty_step=Decimal("0.01"),
        price_step=Decimal("0.01"),
        sell_budget=Decimal("0.75"),
        min_qty=Decimal("0.01"),
        price_band_min=Decimal("0"),
        price_band_max=Decimal("0"),
        force_stop_loss=True,
        fallback_stop_loss_bps=50.0,
    )

    assert placed, "stop-loss order was not sent"
    payload = placed[0]
    assert payload["orderFilter"] == "tpslOrder"
    assert payload["triggerPrice"] == "99.50"
    assert stop_orders and stop_orders[0].get("impulseEnforced") is True
    assert executor._active_stop_orders["BTCUSDT"]["current_stop"] == Decimal("99.50")
