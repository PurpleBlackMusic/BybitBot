from __future__ import annotations

import json
import time
from pathlib import Path

from bybit_app.utils.envs import Settings
from bybit_app.utils.guardian_bot import GuardianBot


def _make_bot(tmp_path: Path, settings: Settings | None = None) -> GuardianBot:
    return GuardianBot(data_dir=tmp_path, settings=settings or Settings())


def test_guardian_brief_waits_when_status_missing(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()
    assert brief.mode == "wait"
    assert "спокойный режим" in brief.headline.lower()
    assert brief.updated_text.startswith("Данные обновлены")
    assert brief.analysis


def test_guardian_brief_buy_signal(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.82,
        "ev_bps": 24.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "explanation": "Растёт интерес покупателей",
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(
        tmp_path,
        Settings(ai_symbols="ETHUSDT", ai_buy_threshold=0.6, ai_min_ev_bps=10.0),
    )
    brief = bot.generate_brief()
    assert brief.mode == "buy"
    assert "ETHUSDT" in brief.headline
    assert "покуп" in brief.action_text.lower()
    assert "без" in brief.caution.lower()
    assert "интерес" in brief.analysis.lower()


def test_guardian_plan_and_risk_summary(tmp_path: Path) -> None:
    settings = Settings(
        dry_run=False,
        spot_cash_reserve_pct=15.0,
        ai_risk_per_trade_pct=0.75,
        ai_daily_loss_limit_pct=2.5,
        ai_max_concurrent=2,
        spot_cash_only=False,
    )
    bot = _make_bot(tmp_path, settings)
    plan = bot.plan_steps()
    assert len(plan) == 3
    risk = bot.risk_summary()
    assert "15.0%" in risk
    assert "0.75%" in risk
    assert "заёмные" in risk


def test_guardian_profit_answer_uses_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "10000",
            "execQty": "1",
            "execFee": "0",
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": "12000",
            "execQty": "0.4",
            "execFee": "0",
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("сколько прибыли сейчас?")
    assert "480.00" in reply or "800.00" in reply
    assert "USDT" in reply


def test_guardian_plan_answer_in_chat(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    response = bot.answer("какой план действий?")
    assert response.startswith("План действий:")
    assert "1." in response


def test_guardian_market_story_returns_explanation(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.7,
        "ev_bps": 15.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "explanation": "Модель увидела повышенный объём покупок",
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()
    story = bot.market_story(brief)
    assert "повышенный объём" in story


def test_guardian_staleness_alert_when_old(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.5,
        "ev_bps": 5.0,
        "side": "wait",
        "last_tick_ts": time.time() - 3600,
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()
    alert = bot.staleness_alert(brief)
    assert alert is not None
    assert "15 минут" in alert
