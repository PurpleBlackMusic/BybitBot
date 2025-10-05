from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import bybit_app.utils.guardian_bot as guardian_bot_module
from bybit_app.utils import trade_control
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


def test_guardian_brief_prefers_status_symbol(tmp_path: Path) -> None:
    status = {
        "symbol": "SOLUSDT",
        "probability": 0.7,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path, Settings(ai_symbols=""))
    brief = bot.generate_brief()

    assert brief.symbol == "SOLUSDT"
    summary = bot.status_summary()
    assert summary["symbol"] == "SOLUSDT"


def test_guardian_brief_respects_explicit_mode_hint(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.72,
        "ev_bps": 22.0,
        "mode": "wait",
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    assert brief.mode == "wait"
    assert "спокойный режим" in brief.headline.lower()

    summary = bot.status_summary()
    assert summary["mode"] == "wait"
    assert summary["mode_hint"] == "wait"
    assert summary["mode_hint_source"] == "mode"
    assert any("Нет активного сигнала" in reason for reason in summary["actionable_reasons"])


def test_guardian_brief_handles_sell_mode_hint(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.4,
        "ev_bps": 12.0,
        "mode": "sell",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    assert brief.mode == "sell"
    assert "прод" in brief.action_text.lower()

    summary = bot.status_summary()
    assert summary["mode"] == "sell"
    assert summary["mode_hint"] == "sell"
    assert summary["mode_hint_source"] == "mode"
    assert any("уверенность" in reason.lower() for reason in summary["actionable_reasons"])


def test_guardian_brief_understands_russian_buy_hint(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.8,
        "ev_bps": 30.0,
        "mode": "Покупаем аккуратно",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    assert brief.mode == "buy"
    assert "покуп" in brief.action_text.lower()


def test_guardian_brief_wait_hint_from_russian_phrase(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.52,
        "ev_bps": 8.0,
        "mode": "ничего не делаем, ждём",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    assert brief.mode == "wait"
    assert "спокойный режим" in brief.headline.lower()


def test_guardian_settings_answer_highlights_thresholds(tmp_path: Path) -> None:
    settings = Settings(
        ai_enabled=True,
        ai_symbols="BTCUSDT, ETHUSDT",
        ai_buy_threshold=0.62,
        ai_sell_threshold=0.38,
        ai_min_ev_bps=15.0,
        ai_risk_per_trade_pct=0.45,
        spot_cash_reserve_pct=22.0,
        ai_daily_loss_limit_pct=4.5,
        ai_max_concurrent=2,
        ai_retrain_minutes=90,
        ws_watchdog_enabled=True,
        execution_watchdog_max_age_sec=600,
        dry_run=False,
        testnet=False,
        spot_cash_only=True,
    )

    bot = _make_bot(tmp_path, settings)
    response = bot.answer("Какие настройки сейчас активны?")

    assert "62.00%" in response
    assert "15.00" in response
    assert "капитала" in response
    assert "600 с" in response
    assert "ETHUSDT" in response and "BTCUSDT" in response


def test_guardian_settings_answer_explains_safe_mode(tmp_path: Path) -> None:
    settings = Settings(
        ai_enabled=False,
        dry_run=True,
        testnet=True,
        spot_cash_only=False,
        ai_risk_per_trade_pct=0.2,
        spot_cash_reserve_pct=10.0,
        ai_daily_loss_limit_pct=0.0,
    )

    bot = _make_bot(tmp_path, settings)
    response = bot.answer("Настройки бота?")

    assert "тестнете" in response.lower()
    assert "ai сигналы выключены" in response.lower()
    assert "плеч" in response.lower()


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


def test_guardian_answer_reports_portfolio_positions(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": "2000",
            "execQty": "0.5",
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "27000",
            "execQty": "0.01",
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("что в портфеле?")

    assert "В портфеле" in reply
    assert "ETHUSDT" in reply
    assert "BTCUSDT" in reply


def test_guardian_answer_exposure_summary(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "27000",
            "execQty": "0.01",
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": "1800",
            "execQty": "0.3",
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("какая сейчас экспозиция?")

    lowered = reply.lower()
    assert "в работе" in lowered
    assert "резерв" in lowered
    assert "btc" in lowered and "eth" in lowered
    assert "%" in reply


def test_guardian_answer_exposure_when_empty(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path, Settings(spot_cash_reserve_pct=25.0, ai_risk_per_trade_pct=1.2))
    reply = bot.answer("что с загрузкой капитала?")

    lowered = reply.lower()
    assert "капитал свободен" in lowered
    assert "25.0%" in reply
    assert "1.20%" in reply


def test_guardian_answer_watchlist_summary(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.5,
        "side": "buy",
        "last_tick_ts": time.time(),
        "watchlist": {
            "ETHUSDT": {"score": 0.7, "trend": "buy", "note": "объём растёт"},
            "XRPUSDT": {"score": 0.4},
            "DOGEUSDT": {"trend": "wait"},
        },
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    reply = bot.answer("что в вочлисте?")

    assert reply.startswith("Список наблюдения:")
    assert "ETHUSDT" in reply
    assert "оценка 0.70" in reply


def test_guardian_answer_health_summary(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.55,
        "ev_bps": 8.0,
        "side": "wait",
        "last_tick_ts": time.time() - 4000,
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    old_trade_ts = time.time() - 7200
    trade_event = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "execPrice": "26000",
        "execQty": "0.01",
        "execTime": old_trade_ts,
    }
    with ledger_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(trade_event) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("как здоровье данных?")

    assert "AI сигнал" in reply
    assert "Журнал исполнений" in reply
    assert "API" in reply


def test_guardian_answer_manual_control(tmp_path: Path) -> None:
    trade_control.clear_trade_commands(data_dir=tmp_path)
    trade_control.request_trade_start(
        symbol="BTCUSDT",
        mode="buy",
        probability_pct=62.5,
        ev_bps=14.0,
        note="оператор проверяет гипотезу",
        data_dir=tmp_path,
    )
    trade_control.request_trade_cancel(
        symbol="BTCUSDT",
        reason="фиксируем результат",
        data_dir=tmp_path,
    )

    bot = _make_bot(tmp_path)
    reply = bot.answer("что по ручному управлению?")

    lowered = reply.lower()
    assert "останов" in lowered
    assert "BTCUSDT" in reply
    assert "уверенность 62.5%" in lowered
    assert "оператор" in lowered


def test_guardian_answer_trade_history(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "27000",
            "execQty": "0.01",
            "execFee": "0.15",
            "execTime": now - 120,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "execPrice": "1850",
            "execQty": "0.2",
            "execFee": "0.05",
            "execTime": now - 30,
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("покажи последние сделки")

    assert "В журнале 2 сделок" in reply
    assert "BTCUSDT" in reply and "ETHUSDT" in reply
    assert "комиссия" in reply
    assert "последние" in reply.lower()


def test_guardian_answer_fees_without_history(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    reply = bot.answer("какие комиссии мы платим?")

    lowered = reply.lower()
    assert "комиссий" in lowered
    assert "журнал" in lowered


def test_guardian_answer_fee_activity_summary(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "27000",
            "execQty": "0.002",
            "execFee": "0.00010",
            "isMaker": True,
            "execTime": (now - timedelta(minutes=2)).isoformat(),
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": "28000",
            "execQty": "0.002",
            "execFee": "0.00009",
            "isMaker": False,
            "execTime": (now - timedelta(minutes=1)).isoformat(),
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": "1600",
            "execQty": "0.01",
            "execFee": "0.00005",
            "isMaker": True,
            "execTime": (now - timedelta(hours=2)).isoformat(),
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    bot = _make_bot(tmp_path)
    reply = bot.answer("дай сводку по комиссиям и maker/taker")

    assert "0.00024" in reply
    assert "66.7%" in reply
    assert "за 15 минут 2" in reply
    assert "за час 2" in reply
    assert "за сутки 3" in reply
    assert "BTCUSDT" in reply and "110.00" in reply


def test_guardian_signal_quality_answer(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.72,
        "ev_bps": 24.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(
        tmp_path,
        Settings(
            ai_buy_threshold=0.65,
            ai_sell_threshold=0.35,
            ai_min_ev_bps=12.0,
        ),
    )

    reply = bot.answer("расскажи про сигнал и пороги")

    assert "72.00%" in reply
    assert "24.00 б.п." in reply
    assert "65.00%" in reply and "35.00%" in reply
    assert "минимальная выгода 12.00 б.п." in reply


def test_guardian_signal_quality_answer_without_status(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    reply = bot.answer("что с качеством сигнала?")

    assert "Живой сигнал пока не загружен" in reply


def test_guardian_update_answer_reports_freshness(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.7,
        "ev_bps": 18.0,
        "last_tick_ts": time.time() - 120,
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path, Settings(ai_retrain_minutes=45))
    reply = bot.answer("когда обновлялся сигнал?")

    assert "AI сигнал обновился" in reply
    assert "Данные обновлены" in reply
    assert "Последняя отметка обновления" in reply
    assert "Текущий возраст сигнала" in reply
    assert "45 мин" in reply
    assert "Текущий символ" in reply


def test_guardian_update_answer_without_status(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    reply = bot.answer("почему статус давно не обновлялся?")

    assert "Свежие данные ещё не поступали" in reply
    assert "ai/status.json" in reply


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


def test_guardian_watchlist_and_scorecard(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.5,
        "watchlist": {
            "ETHUSDT": {"score": 0.7, "trend": "buy", "note": "объём растёт"},
            "XRPUSDT": 0.4,
            "DOGEUSDT": {"score": "n/a", "trend": "wait"},
        },
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    watchlist = bot.market_watchlist()
    assert [item["symbol"] for item in watchlist] == ["ETHUSDT", "XRPUSDT", "DOGEUSDT"]
    assert watchlist[0]["score"] == 0.7
    assert watchlist[1]["score"] == 0.4
    assert watchlist[2]["score"] is None

    scorecard = bot.signal_scorecard(brief)
    assert scorecard["symbol"] == "BTCUSDT"
    assert scorecard["probability_pct"] == 60.0
    assert scorecard["ev_bps"] == 12.5


def test_guardian_recent_trades(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"category": "spot", "symbol": "BTCUSDT", "side": "Buy", "execPrice": "27000", "execQty": "0.01", "execTime": time.time()},
        {"category": "spot", "symbol": "ETHUSDT", "side": "Sell", "execPrice": "1900", "execQty": "0.5", "execFee": "0.2", "execTime": time.time()},
        {"category": "linear", "symbol": "BTCUSDT", "side": "Sell", "execPrice": "28000", "execQty": "0.01"},
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    bot = _make_bot(tmp_path)
    trades = bot.recent_trades()
    assert len(trades) == 2
    assert all(trade["symbol"].endswith("USDT") for trade in trades)
    assert trades[0]["when"] != "—"


def test_guardian_status_summary_and_report(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.75,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "extra": "ignored",
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(
        ai_symbols="ETHUSDT",
        ai_buy_threshold=0.7,
        ai_sell_threshold=0.4,
        ai_min_ev_bps=10.0,
    )
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    assert summary["symbol"] == "ETHUSDT"
    assert summary["probability_pct"] == 75.0
    assert summary["actionable"] is True
    assert summary["actionable_reasons"] == []
    assert summary["thresholds"]["buy_probability_pct"] == 70.0
    assert summary["fallback_used"] is False
    assert summary["status_source"] == "live"
    assert summary["staleness"]["state"] == "fresh"
    assert "extra" in summary["raw_keys"]

    # ensure copies are returned
    summary["symbol"] = "CHANGED"
    assert bot.status_summary()["symbol"] == "ETHUSDT"

    report = bot.unified_report()
    assert report["status"]["symbol"] == "ETHUSDT"
    assert report["status"]["actionable"] is True
    assert report["status"]["status_source"] == "live"
    assert report["status"]["actionable_reasons"] == []
    assert report["status"]["staleness"]["state"] == "fresh"
    report["status"]["symbol"] = "MUTATED"
    assert bot.unified_report()["status"]["symbol"] == "ETHUSDT"


def test_guardian_status_recovers_from_partial_write(tmp_path: Path, monkeypatch) -> None:
    status_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    original_read_bytes = Path.read_bytes
    call_counter = {"count": 0}

    def flaky_read(self: Path, *args, **kwargs):
        if self == status_path and call_counter["count"] == 0:
            call_counter["count"] += 1
            return b"{\"symbol\": \"BTCUSDT\","
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", flaky_read)

    bot = _make_bot(tmp_path)
    summary = bot.status_summary()

    assert summary["status_source"] == "live"
    assert summary["fallback_used"] is False
    assert summary["symbol"] == "BTCUSDT"
    assert summary.get("status_error") is None
    assert call_counter["count"] == 1


def test_guardian_status_fallback_when_file_missing(tmp_path: Path) -> None:
    status_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.64,
        "ev_bps": 16.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    bot = _make_bot(tmp_path)
    live_summary = bot.status_summary()
    assert live_summary["status_source"] == "live"

    status_path.unlink()

    fallback_summary = bot.status_summary()
    assert fallback_summary["fallback_used"] is True
    assert fallback_summary["status_source"] == "cached"
    assert fallback_summary["symbol"] == "BTCUSDT"

    health = bot.data_health()
    assert health["ai_signal"]["ok"] is False
    assert "сохран" in health["ai_signal"]["message"].lower()


def test_guardian_status_fallback_forces_retry(tmp_path: Path, monkeypatch) -> None:
    status_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.7,
        "ev_bps": 20.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    bot = _make_bot(tmp_path)
    live_summary = bot.status_summary()
    assert live_summary["status_source"] == "live"

    initial_signature = bot._snapshot.status_signature

    status_path.write_text("{\"symbol\": \"BTCUSDT\"", encoding="utf-8")

    call_counter = {"count": 0}
    original_read_bytes = Path.read_bytes

    def counting_read(self: Path, *args, **kwargs):
        if self == status_path:
            call_counter["count"] += 1
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting_read)

    fallback_summary = bot.status_summary()

    assert fallback_summary["status_source"] == "cached"
    assert fallback_summary["fallback_used"] is True
    assert call_counter["count"] >= 1
    assert bot._snapshot.status_signature == initial_signature

    call_counter["count"] = 0

    cached_summary = bot.status_summary()

    assert cached_summary["fallback_used"] is True
    assert call_counter["count"] >= 1


def test_guardian_status_refreshes_when_mtime_static(tmp_path: Path) -> None:
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    first_status = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "explanation": "короткое описание",
    }
    status_path.write_text(json.dumps(first_status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    initial_summary = bot.status_summary()
    assert initial_summary["probability_pct"] == 60.0
    assert "короткое" in initial_summary["analysis"].lower()

    prev_stat = status_path.stat()

    updated_status = {
        "symbol": "BTCUSDT",
        "probability": 0.75,
        "ev_bps": 20.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "explanation": "длинное описание сигнала с дополнительными деталями",
    }
    status_path.write_text(json.dumps(updated_status), encoding="utf-8")
    os.utime(
        status_path,
        ns=(
            getattr(prev_stat, "st_atime_ns", int(prev_stat.st_atime * 1_000_000_000)),
            getattr(prev_stat, "st_mtime_ns", int(prev_stat.st_mtime * 1_000_000_000)),
        ),
    )

    refreshed_summary = bot.status_summary()
    assert refreshed_summary["probability_pct"] == 75.0
    assert refreshed_summary["ev_bps"] == 20.0
    assert "длинное" in refreshed_summary["analysis"].lower()


def test_guardian_status_refreshes_when_content_size_static(tmp_path: Path) -> None:
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    first_payload = (
        '{"symbol":"BTCUSDT","probability":0.61,"ev_bps":12,"side":"buy","last_tick_ts":1234567890.0,"explanation":"alpha"}'
    )
    second_payload = (
        '{"symbol":"BTCUSDT","probability":0.62,"ev_bps":12,"side":"buy","last_tick_ts":1234567891.0,"explanation":"bravo"}'
    )
    third_payload = (
        '{"symbol":"BTCUSDT","probability":0.63,"ev_bps":12,"side":"buy","last_tick_ts":1234567892.0,"explanation":"charl"}'
    )

    assert len(first_payload) == len(second_payload) == len(third_payload)

    status_path.write_text(first_payload, encoding="utf-8")

    bot = _make_bot(tmp_path)

    first_summary = bot.status_summary()
    assert first_summary["probability_pct"] == 61.0
    assert "alpha" in first_summary["analysis"].lower()

    prev_stat = status_path.stat()

    status_path.write_text(second_payload, encoding="utf-8")
    os.utime(
        status_path,
        ns=(
            getattr(prev_stat, "st_atime_ns", int(prev_stat.st_atime * 1_000_000_000)),
            getattr(prev_stat, "st_mtime_ns", int(prev_stat.st_mtime * 1_000_000_000)),
        ),
    )

    refreshed_summary = bot.status_summary()

    assert refreshed_summary["probability_pct"] == 62.0
    assert "bravo" in refreshed_summary["analysis"].lower()

    mid_stat = status_path.stat()
    assert getattr(prev_stat, "st_size", None) == getattr(mid_stat, "st_size", None)

    status_path.write_text(third_payload, encoding="utf-8")
    os.utime(
        status_path,
        ns=(
            getattr(mid_stat, "st_atime_ns", int(mid_stat.st_atime * 1_000_000_000)),
            getattr(mid_stat, "st_mtime_ns", int(mid_stat.st_mtime * 1_000_000_000)),
        ),
    )

    final_stat = status_path.stat()
    assert getattr(mid_stat, "st_size", None) == getattr(final_stat, "st_size", None)

    final_summary = bot.status_summary()

    assert final_summary["probability_pct"] == 63.0
    assert "charl" in final_summary["analysis"].lower()

def test_guardian_status_summary_marks_stale_signal(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.82,
        "ev_bps": 24.0,
        "side": "buy",
        "last_tick_ts": time.time() - 3600,
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path, Settings(ai_buy_threshold=0.6, ai_min_ev_bps=10.0))
    summary = bot.status_summary()

    assert summary["actionable"] is False
    assert summary["staleness"]["state"] == "stale"
    assert any("устар" in reason.lower() for reason in summary["actionable_reasons"])

    health = bot.data_health()
    assert health["ai_signal"]["ok"] is False
    assert "15 минут" in health["ai_signal"]["message"] or "статус" in health["ai_signal"]["message"].lower()


def test_guardian_trade_statistics(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    events = [
        {"symbol": "BTCUSDT", "side": "Buy", "execPrice": 27000, "execQty": 0.01, "execTime": now - 120},
        {"symbol": "BTCUSDT", "side": "Sell", "execPrice": 27300, "execQty": 0.01, "execTime": now - 60},
        {"symbol": "ETHUSDT", "side": "Buy", "execPrice": 1900, "execQty": 0.3, "execTime": now - 3600},
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    bot = _make_bot(tmp_path)
    stats = bot.trade_statistics()
    assert stats["trades"] == 3
    assert "BTCUSDT" in stats["symbols"]
    assert stats["gross_volume"] > 0
    assert stats["per_symbol"]


def test_guardian_unified_report_is_serialisable(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.62,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({"category": "spot", "symbol": "BTCUSDT", "execPrice": 20000, "execQty": 0.1})
        + "\n",
        encoding="utf-8",
    )

    bot = _make_bot(tmp_path)
    report = bot.unified_report()

    assert isinstance(report["brief"], dict)
    assert report["brief"]["symbol"] == "BTCUSDT"
    assert isinstance(bot.brief_payload(), dict)
    assert report["brief"]["mode"] in {"buy", "wait", "sell"}


def test_guardian_data_health(tmp_path: Path, monkeypatch) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.7,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execPrice": 27000,
                "execQty": 0.01,
                "execTime": time.time() - 120,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        guardian_bot_module,
        "bybit_realtime_status",
        lambda _settings: {
            "title": "RT",
            "ok": True,
            "message": "Биржа отвечает",
            "details": "stub",
        },
    )

    bot = _make_bot(tmp_path, Settings(api_key="k", api_secret="s"))
    bot.generate_brief()
    health = bot.data_health()

    assert health["ai_signal"]["ok"] is True
    assert "AI сигнал" in health["ai_signal"]["title"]
    assert health["executions"]["trades"] == 1
    assert health["executions"]["ok"] is True
    assert health["api_keys"]["ok"] is True
    assert health["realtime_trading"]["ok"] is True
    assert health["realtime_trading"]["title"] == "RT"


def test_guardian_unified_report(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.65,
        "ev_bps": 14.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "watchlist": {"ETHUSDT": {"score": 0.6, "trend": "buy"}},
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execPrice": 27000,
                "execQty": 0.01,
                "execTime": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bot = _make_bot(tmp_path, Settings(api_key="k", api_secret="s"))
    report = bot.unified_report()

    assert report["brief"]["symbol"] == "BTCUSDT"
    assert isinstance(report["portfolio"], dict)
    assert report["statistics"]["trades"] == 1
    assert report["health"]["api_keys"]["ok"] is True
    assert report["watchlist"]
    assert report["generated_at"] <= time.time()


def test_guardian_reuses_ledger_cache(tmp_path: Path) -> None:
    class CountingGuardianBot(GuardianBot):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.load_calls = 0
            self.portfolio_calls = 0
            self.recent_calls = 0

        def _load_ledger_events(self) -> List[Dict[str, object]]:  # type: ignore[override]
            self.load_calls += 1
            return super()._load_ledger_events()

        def _build_portfolio(self, spot_events: List[Dict[str, object]]) -> Dict[str, object]:  # type: ignore[override]
            self.portfolio_calls += 1
            return super()._build_portfolio(spot_events)

        def _build_recent_trades(
            self, spot_events: List[Dict[str, object]], limit: int = 50
        ) -> List[Dict[str, object]]:  # type: ignore[override]
            self.recent_calls += 1
            return super()._build_recent_trades(spot_events, limit)

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execPrice": 25000,
                "execQty": 0.01,
                "execTime": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bot = CountingGuardianBot(data_dir=tmp_path)
    first_report = bot.unified_report()
    assert first_report["statistics"]["trades"] == 1
    assert bot.load_calls == 1
    assert bot.portfolio_calls == 1
    assert bot.recent_calls == 1

    time.sleep(1.1)
    status_payload["probability"] = 0.4
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    second_report = bot.unified_report()
    assert second_report["brief"]["mode"] in {"wait", "sell"}
    assert bot.load_calls == 1
    assert bot.portfolio_calls == 1
    assert bot.recent_calls == 1

    reply = bot.answer("сколько сейчас прибыли?")
    assert "USDT" in reply
    assert bot.load_calls == 1
    assert bot.portfolio_calls == 1
    assert bot.recent_calls == 1


def test_guardian_manual_trade_state_uses_bot_directory(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)

    initial_state = bot.manual_trade_state()
    assert initial_state.active is False

    trade_control.clear_trade_commands(data_dir=bot.data_dir)
    trade_control.request_trade_start(symbol="BNBUSDT", mode="buy", data_dir=bot.data_dir)

    updated_state = bot.manual_trade_state()
    assert updated_state.active is True
    assert updated_state.last_start is not None
    assert updated_state.last_start["symbol"] == "BNBUSDT"

    trade_control.clear_trade_commands(data_dir=bot.data_dir)


def test_guardian_manual_trade_wrappers_round_trip(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)

    bot.manual_trade_clear()
    assert bot.manual_trade_history() == ()

    start_record = bot.manual_trade_start(
        symbol="ethusdt",
        mode="buy",
        probability_pct=55.5,
        ev_bps=120.0,
        source="pytest",
        note="начало теста",
        extra={"foo": "bar"},
    )
    assert start_record["symbol"] == "ETHUSDT"
    assert start_record["action"] == "start"

    history = bot.manual_trade_history()
    assert history
    assert history[-1]["action"] == "start"

    cancel_record = bot.manual_trade_cancel(
        symbol="ethusdt",
        reason="stop test",
        source="pytest",
        note="остановка",
    )
    assert cancel_record["action"] == "cancel"

    state = bot.manual_trade_state()
    assert state.active is False
    assert state.last_cancel is not None

    history = bot.manual_trade_history()
    assert len(history) >= 2
    assert history[-1]["action"] == "cancel"

    bot.manual_trade_clear()
    assert bot.manual_trade_history() == ()


def test_guardian_manual_summary_updates_and_isolated(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)

    bot.manual_trade_clear()
    base_summary = bot.manual_trade_summary()
    assert base_summary["status_label"] == "idle"
    assert base_summary["history_count"] == 0
    assert base_summary["history"] == []

    start_record = bot.manual_trade_start(
        symbol="adausdt",
        mode="buy",
        source="pytest",
    )
    assert start_record["action"] == "start"

    summary_after_start = bot.manual_trade_summary()
    assert summary_after_start["status_label"] == "active"
    assert summary_after_start["history_count"] == 1
    assert summary_after_start["last_action"]["action"] == "start"
    assert len(summary_after_start["history"]) == 1

    summary_clone = bot.manual_trade_summary()
    summary_clone["history"].append({"action": "fake"})
    refreshed_summary = bot.manual_trade_summary()
    assert refreshed_summary["history_count"] == 1
    assert len(refreshed_summary["history"]) == 1

    status_manual = bot.status_summary()["manual_control"]
    assert status_manual["status_label"] == "active"
    assert status_manual["history_count"] == 1

    state = bot.manual_trade_state()
    if state.last_action:
        state.last_action["action"] = "mutated"
    fresh_state = bot.manual_trade_state()
    if fresh_state.last_action:
        assert fresh_state.last_action["action"] == "start"

    history_tail = bot.manual_trade_history(limit=1)
    assert len(history_tail) == 1
    assert history_tail[0]["action"] == "start"

    cancel_record = bot.manual_trade_cancel(symbol="adausdt", reason="pytest stop")
    assert cancel_record["action"] == "cancel"

    summary_after_cancel = bot.manual_trade_summary()
    assert summary_after_cancel["status_label"] == "stopped"
    assert summary_after_cancel["history_count"] == 2
    assert summary_after_cancel["last_action"]["action"] == "cancel"
    assert summary_after_cancel["last_cancel"] is not None

    history_tail_after = bot.manual_trade_history(limit=1)
    assert len(history_tail_after) == 1
    assert history_tail_after[0]["action"] == "cancel"

    bot.manual_trade_clear()
    cleared_summary = bot.manual_trade_summary()
    assert cleared_summary["status_label"] == "idle"
    assert cleared_summary["history_count"] == 0
