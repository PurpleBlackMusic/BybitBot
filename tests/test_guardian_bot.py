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


def test_guardian_watchlist_and_scorecard(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 12.5,
        "watchlist": {
            "ETHUSDT": {"score": 0.7, "trend": "buy", "note": "объём растёт"},
            "XRPUSDT": 0.4,
        },
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)
    brief = bot.generate_brief()

    watchlist = bot.market_watchlist()
    assert any(item["symbol"] == "ETHUSDT" for item in watchlist)
    assert any(item["symbol"] == "XRPUSDT" for item in watchlist)

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


def test_guardian_data_health(tmp_path: Path) -> None:
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

    bot = _make_bot(tmp_path, Settings(api_key="k", api_secret="s"))
    bot.generate_brief()
    health = bot.data_health()

    assert health["ai_signal"]["ok"] is True
    assert "AI сигнал" in health["ai_signal"]["title"]
    assert health["executions"]["trades"] == 1
    assert health["executions"]["ok"] is True
    assert health["api_keys"]["ok"] is True
