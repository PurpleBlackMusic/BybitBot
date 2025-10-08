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
        self.orders: list[dict] = []

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

