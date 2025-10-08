import copy
from typing import Optional

import pytest

import bybit_app.utils.signal_executor as signal_executor_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.signal_executor import (
    AutomationLoop,
    ExecutionResult,
    SignalExecutor,
)
from bybit_app.utils.spot_market import OrderValidationError


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
    assert first["qty"] == "0.45"
    assert second["qty"] == "0.3"
    assert first["price"] == "100.5"
    assert second["price"] == "101"
    assert result.order is not None
    assert result.order.get("take_profit_orders")
    assert result.context is not None
    assert result.context.get("execution", {}).get("avg_price") == "100"


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
    assert order["qty"] == "1"
    assert order["price"] == "100.1"
    assert result.order is not None
    ladder = result.order.get("take_profit_orders")
    assert ladder is not None
    assert ladder[0]["profit_bps"] == "5,9"
    assert ladder[0]["orderId"] == "stub-1"


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
    qtys = [order["qty"] for order in api.orders]
    prices = [order["price"] for order in api.orders]
    assert qtys == ["0.37", "0.22", "0.16"]
    assert prices == ["100.35", "100.7", "101.1"]
    assert result.order is not None
    execution = result.order.get("execution")
    assert execution is not None
    assert execution.get("executed_base") == "0.75"
    assert execution.get("avg_price") == "100"

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

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "disabled"
    assert result.reason is not None and "WebSocket" in result.reason
    assert api_called["value"] is False


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

