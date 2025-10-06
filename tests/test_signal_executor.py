import copy

import pytest

import bybit_app.utils.signal_executor as signal_executor_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.signal_executor import ExecutionResult, SignalExecutor


class StubBot:
    def __init__(self, summary: dict, settings: Settings | None = None) -> None:
        self._summary = summary
        self.settings = settings or Settings()

    def status_summary(self) -> dict:
        return copy.deepcopy(self._summary)


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

    executor = SignalExecutor(bot)
    result = executor.execute_once()

    assert result.status == "dry_run"
    assert result.order is not None
    assert result.order["symbol"] == "ETHUSDT"
    assert result.order["notional_quote"] < 20.0
    assert result.order["notional_quote"] == pytest.approx(6.62, rel=1e-3)

