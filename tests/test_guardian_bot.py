from __future__ import annotations

import copy
import enum
from collections import deque
from dataclasses import dataclass
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from types import MappingProxyType

import pytest

import bybit_app.utils.guardian_bot as guardian_bot_module
from bybit_app.utils.envs import Settings
from bybit_app.utils.guardian_bot import GuardianBot
from bybit_app.utils.live_signal import LiveSignalError


def _make_bot(
    tmp_path: Path,
    settings: Settings | None = None,
    *,
    live_only: bool = False,
) -> GuardianBot:
    settings = settings or Settings(ai_live_only=False)
    settings.ai_market_scan_enabled = False
    settings.ai_live_only = live_only
    return GuardianBot(data_dir=tmp_path, settings=settings)


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
        Settings(ai_live_only=False, ai_symbols="ETHUSDT", ai_buy_threshold=0.6, ai_min_ev_bps=10.0),
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

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_symbols=""))
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


def test_guardian_respects_custom_buy_threshold(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.42,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.4,
        ai_min_ev_bps=10.0,
        ai_enabled=True,
    )
    bot = _make_bot(tmp_path, settings)

    brief = bot.generate_brief()
    assert brief.mode == "buy"

    summary = bot.status_summary()
    assert summary["actionable"] is True
    assert summary["actionable_reasons"] == []


def test_guardian_summary_flags_low_confidence_under_defaults(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.48,
        "ev_bps": 22.0,
        "mode": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)

    summary = bot.status_summary()

    assert summary["mode"] == "buy"
    assert summary["actionable"] is False
    assert any("порог" in reason.lower() for reason in summary["actionable_reasons"])
    reason_text = " ".join(summary["actionable_reasons"])
    assert "48.00%" in reason_text
    assert "55.00%" in reason_text
    assert summary["thresholds"]["buy_probability_pct"] == 55.0
    assert summary["thresholds"]["sell_probability_pct"] == 45.0
    assert (
        summary["thresholds"]["effective_buy_probability_pct"]
        == summary["thresholds"]["buy_probability_pct"]
    )
    assert (
        summary["thresholds"]["effective_sell_probability_pct"]
        == summary["thresholds"]["sell_probability_pct"]
    )


def test_guardian_summary_reports_effective_thresholds(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.35,
        "ev_bps": 25.0,
        "side": "sell",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.4,
        ai_sell_threshold=0.8,
        ai_min_ev_bps=10.0,
    )

    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    thresholds = summary["thresholds"]

    assert thresholds["buy_probability_pct"] == 40.0
    assert thresholds["sell_probability_pct"] == 80.0
    assert thresholds["effective_buy_probability_pct"] == 40.0
    assert thresholds["effective_sell_probability_pct"] == 40.0

    response = bot.answer("какие пороги сигнала сейчас?")
    lowered = response.lower()
    assert "продажа" in lowered
    assert "40.00%" in response
    assert "80.00%" in response
    assert "однако для безопасности используется 40.00%" in lowered


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


def test_guardian_summary_highlights_ev_shortfall(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.68,
        "ev_bps": 7.5,
        "mode": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.6,
        ai_min_ev_bps=12.0,
    )
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()

    assert summary["actionable"] is False
    assert any("выгода" in reason.lower() for reason in summary["actionable_reasons"])
    reason_text = " ".join(summary["actionable_reasons"])
    assert "7.50 б.п." in reason_text
    assert "12.00 б.п." in reason_text


def test_guardian_summary_highlights_sell_threshold(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.42,
        "ev_bps": 18.0,
        "mode": "sell",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.6,
        ai_sell_threshold=0.5,
        ai_min_ev_bps=10.0,
    )
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()

    assert summary["mode"] == "sell"
    assert summary["actionable"] is False
    reason_text = " ".join(summary["actionable_reasons"])
    assert "42.00%" in reason_text
    assert "50.00%" in reason_text


def test_guardian_summary_mentions_disabled_ai(tmp_path: Path) -> None:
    status = {
        "symbol": "BTCUSDT",
        "probability": 0.78,
        "ev_bps": 18.0,
        "mode": "buy",
        "last_tick_ts": time.time(),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.6,
        ai_min_ev_bps=5.0,
        ai_enabled=False,
    )
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()

    assert summary["mode"] == "buy"
    assert summary["actionable"] is False
    lowered = [reason.lower() for reason in summary["actionable_reasons"]]
    assert any("выключены" in reason for reason in lowered) or any(
        "ручной режим" in reason for reason in lowered
    )


def test_guardian_operation_mode_auto(tmp_path: Path) -> None:
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "probability": 0.8,
                "ev_bps": 20.0,
                "mode": "buy",
                "last_tick_ts": time.time(),
            }
        ),
        encoding="utf-8",
    )

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_enabled=True, ai_symbols="BTCUSDT"))
    summary = bot.status_summary()

    assert summary["operation_mode"] == "auto"
    assert "manual_guidance" not in summary


def test_guardian_settings_answer_highlights_thresholds(tmp_path: Path) -> None:
    settings = Settings(ai_live_only=False, 
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
    settings = Settings(ai_live_only=False, 
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
    settings = Settings(ai_live_only=False, 
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


def test_guardian_answer_handles_non_string_questions(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)

    empty_prompt = bot.answer(None)
    assert "Спросите" in empty_prompt

    numeric_prompt = bot.answer(42)
    assert isinstance(numeric_prompt, str)
    assert numeric_prompt

    bytes_prompt = bot.answer(b" profit ")
    assert "прибыл" in bytes_prompt.lower()

    dict_prompt = bot.answer({"text": "Risk overview, please"})
    assert "риск" in dict_prompt.lower()

    structured_prompt = bot.answer(
        {
            "content": [
                {"type": "meta", "text": {"value": "ignored"}},
                {"type": "input_text", "text": {"value": "Что за план?"}},
            ]
        }
    )
    assert "шаг" in structured_prompt.lower() or "план" in structured_prompt.lower()

    chat_prompt = bot.answer(
        {
            "messages": [
                {"role": "system", "content": "Объясняй коротко"},
                {"role": "assistant", "content": "Последний ответ"},
                {"role": "user", "content": [{"type": "text", "text": "Риск какой?"}]},
            ]
        }
    )
    assert "риск" in chat_prompt.lower()

    fallback_prompt = bot.answer(
        {
            "messages": [{"role": "assistant", "content": "Отвечаю вместо пользователя"}],
            "question": "Когда было обновление?",
        }
    )
    assert "обнов" in fallback_prompt.lower()

    streaming_prompt = bot.answer(
        {
            "choices": [
                {"delta": {"role": "assistant"}},
                {"delta": {"content": "Profit insight"}},
            ]
        }
    )
    assert "прибыл" in streaming_prompt.lower()

    hybrid_role_prompt = bot.answer(
        {
            "messages": [
                {"role": {"type": "system", "value": "assistant"}, "content": "Прошлый ответ"},
                {
                    "role": {"kind": "User", "name": "Trader"},
                    "content": {"prompt": {"value": "Расскажи про прибыль"}},
                },
            ]
        }
    )
    assert "прибыл" in hybrid_role_prompt.lower()

    class Role(enum.Enum):
        USER = "User"
        ASSISTANT = "assistant"

    enum_role_prompt = bot.answer(
        {
            "messages": [
                {"role": Role.ASSISTANT, "content": "Предыдущий ответ"},
                {
                    "role": Role.USER,
                    "content": {"arguments": {"details": {"value": "Подскажи по рискам"}}},
                },
            ]
        }
    )
    assert "риск" in enum_role_prompt.lower()

    deeply_nested_prompt = bot.answer(
        {
            "payload": {
                "data": {
                    "messages": [
                        {"role": "assistant", "content": "Старый ответ"},
                        {
                            "role": "user",
                            "content": {
                                "details": {
                                    "args": [
                                        {
                                            "type": "input_text",
                                            "text": {"value": "Что с риском?"},
                                        }
                                    ]
                                }
                            },
                        },
                    ]
                }
            }
        }
    )
    assert "риск" in deeply_nested_prompt.lower()

    json_string_prompt = bot.answer(
        {
            "messages": [
                {"role": "user", "content": '{"prompt": {"text": "Поделись планом"}}'},
            ]
        }
    )
    assert "план" in json_string_prompt.lower()

    tool_call_prompt = bot.answer(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "relay_user_question",
                                "arguments": json.dumps({"question": "Когда последнее обновление?"}),
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert "обнов" in tool_call_prompt.lower()

    dict_content_prompt = bot.answer(
        {"messages": [{"role": "user", "content": {"text": "Расскажи про риск"}}]}
    )
    assert "риск" in dict_content_prompt.lower()

    @dataclass
    class DataclassMessage:
        role: object
        content: object

    dataclass_prompt = bot.answer(
        {
            "messages": [
                DataclassMessage(role="assistant", content="Исторический ответ"),
                DataclassMessage(
                    role="user",
                    content={"prompt": {"text": "Поделись рисками"}},
                ),
            ]
        }
    )
    assert "риск" in dataclass_prompt.lower()

    @dataclass
    class DataclassRole:
        kind: str

    dataclass_role_prompt = bot.answer(
        {
            "messages": [
                {
                    "role": DataclassRole(kind="assistant"),
                    "content": "Исторический ответ",
                },
                {
                    "role": DataclassRole(kind="user"),
                    "content": {"prompt": {"text": "Поделись обновлениями"}},
                },
            ]
        }
    )
    assert "обнов" in dataclass_role_prompt.lower()

    class ModelLike:
        def __init__(self, payload: object) -> None:
            self._payload = payload

        def model_dump(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self._payload

    model_prompt = bot.answer(
        ModelLike(
            {
                "messages": [
                    {"role": "assistant", "content": "Прошлый ответ"},
                    {
                        "role": "user",
                        "content": {"details": {"value": "Расскажи про прибыль"}},
                    },
                ]
            }
        )
    )
    assert "прибыл" in model_prompt.lower()

    class ObjectMessage:
        def __init__(self, role: object, content: object) -> None:
            self.role = role
            self.content = content

    class ObjectEnvelope:
        def __init__(self, messages: object) -> None:
            self.messages = messages

    object_prompt = bot.answer(
        ObjectEnvelope(
            [
                ObjectMessage(role="assistant", content="Исторический ответ"),
                ObjectMessage(
                    role="user",
                    content={"payload": {"text": "Поделись планом действий"}},
                ),
            ]
        )
    )
    assert "план" in object_prompt.lower()

    response_prompt = bot.answer(
        {
            "response": {
                "output": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "output_text", "text": "Поделись прибылью"},
                        ],
                    }
                ]
            }
        }
    )
    assert "прибыл" in response_prompt.lower()

    instructions_prompt = bot.answer({"instructions": {"text": "Какой риск сейчас?"}})
    assert "риск" in instructions_prompt.lower()

    sequence_prompt = bot.answer(deque([{"prompt": "Что по плану?"}]))
    assert "план" in sequence_prompt.lower()

    proxy_prompt = bot.answer(MappingProxyType({"question": "Когда было обновление?"}))
    assert "обнов" in proxy_prompt.lower()


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
    bot = _make_bot(tmp_path, Settings(ai_live_only=False, spot_cash_reserve_pct=25.0, ai_risk_per_trade_pct=1.2))
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
    assert "Автоматизация" in reply
    assert "Журнал исполнений" in reply
    assert "API" in reply


def test_guardian_data_health_highlights_disabled_ai(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_enabled=False))
    health = bot.data_health()

    automation = health["automation"]
    assert automation["ok"] is False
    assert "выключ" in automation["message"].lower()


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
        Settings(ai_live_only=False, 
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

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_retrain_minutes=45))
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
    assert watchlist[0]["actionable"] is True
    assert watchlist[0]["trend_hint"] == "buy"
    assert watchlist[0]["probability_pct"] == 70.0
    assert isinstance(watchlist[0]["edge_score"], float)

    scorecard = bot.signal_scorecard(brief)
    assert scorecard["symbol"] == "BTCUSDT"
    assert scorecard["probability_pct"] == 60.0
    assert scorecard["ev_bps"] == 12.5
    assert scorecard["buy_threshold"] == 55.0
    assert scorecard["sell_threshold"] == 45.0
    assert scorecard["configured_buy_threshold"] == 55.0
    assert scorecard["configured_sell_threshold"] == 45.0

    summary = bot.status_summary()
    assert summary["watchlist_total"] == 3
    assert summary["watchlist_actionable"] == 1
    highlights = summary["watchlist_highlights"]
    assert highlights[0]["symbol"] == "ETHUSDT"
    assert highlights[0]["primary"] is True
    assert highlights[0]["actionable"] is True
    assert summary["primary_watch"]["symbol"] == "ETHUSDT"


def test_guardian_brief_selects_best_watchlist_symbol(tmp_path: Path) -> None:
    status = {
        "analysis": "ETH идёт впереди по объёмам.",
        "watchlist": {
            "ETHUSDT": {"probability": 0.72, "ev_bps": 18.0, "trend": "buy"},
            "BTCUSDT": {"probability": 0.55, "ev_bps": 8.0, "trend": "buy"},
            "SOLUSDT": {"probability": 0.4, "ev_bps": 4.0, "trend": "wait"},
        },
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT,ETHUSDT",
        ai_buy_threshold=0.6,
        ai_min_ev_bps=10.0,
        ai_enabled=True,
    )
    bot = _make_bot(tmp_path, settings)

    brief = bot.generate_brief()
    assert brief.symbol == "ETHUSDT"
    assert brief.mode == "buy"
    assert "ETHUSDT" in brief.headline

    summary = bot.status_summary()
    assert summary["symbol"] == "ETHUSDT"
    assert summary["probability_pct"] == 72.0
    assert summary["ev_bps"] == 18.0
    assert summary["symbol_source"] == "watchlist"
    assert summary["probability_source"] == "watchlist"
    assert summary.get("mode_hint_source") == "watchlist"


def test_guardian_watchlist_skips_neutral_leaders(tmp_path: Path) -> None:
    status = {
        "watchlist": [
            {"symbol": "LTCUSDT", "score": 0.9, "probability": 0.9, "ev_bps": 14.0, "trend": "wait"},
            {"symbol": "ETHUSDT", "probability": 0.71, "ev_bps": 16.0, "trend": "buy"},
            {"symbol": "XRPUSDT", "probability": 0.68, "ev_bps": 11.0, "trend": "buy"},
        ]
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="BTCUSDT,ETHUSDT",
        ai_buy_threshold=0.6,
        ai_min_ev_bps=10.0,
        ai_enabled=True,
    )
    bot = _make_bot(tmp_path, settings)

    brief = bot.generate_brief()

    assert brief.symbol == "ETHUSDT"
    assert brief.mode == "buy"

    summary = bot.status_summary()
    assert summary["symbol"] == "ETHUSDT"
    assert summary["probability_pct"] == 71.0
    assert summary["ev_bps"] == 16.0
    assert summary["symbol_source"] == "watchlist"
    assert summary["probability_source"] == "watchlist"


def test_guardian_watchlist_enriches_actionable_entries(tmp_path: Path) -> None:
    status = {
        "watchlist": [
            {
                "symbol": "ADAUSDT",
                "probability": 0.66,
                "ev_bps": 15.0,
                "trend": "buy",
                "note": "спрос растёт",
            },
            {
                "symbol": "LTCUSDT",
                "probability": 0.44,
                "ev_bps": 12.0,
                "trend": "sell",
                "note": "давление продавцов",
            },
            {
                "symbol": "XRPUSDT",
                "probability": 0.52,
                "ev_bps": 6.0,
                "trend": "buy",
            },
        ]
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="ADAUSDT,LTCUSDT,XRPUSDT",
        ai_buy_threshold=0.6,
        ai_min_ev_bps=10.0,
        ai_sell_threshold=0.45,
        ai_enabled=True,
    )

    bot = _make_bot(tmp_path, settings)

    watchlist = bot.market_watchlist()
    assert [item["symbol"] for item in watchlist[:2]] == ["ADAUSDT", "LTCUSDT"]
    assert all(item["actionable"] for item in watchlist[:2])
    assert watchlist[0]["edge_score"] >= watchlist[1]["edge_score"]

    summary = bot.status_summary()
    assert summary["watchlist_total"] == 3
    assert summary["watchlist_actionable"] == 2
    highlights = summary["watchlist_highlights"]
    assert [item["symbol"] for item in highlights] == ["ADAUSDT", "LTCUSDT", "XRPUSDT"]
    assert highlights[0]["probability_pct"] == 66.0
    assert highlights[1]["trend"] == "sell"
    assert highlights[1]["probability_gap_pct"] == 1.0
    assert summary["primary_watch"]["symbol"] == "ADAUSDT"

    breakdown = summary["watchlist_breakdown"]
    counts = breakdown["counts"]
    assert counts["total"] == 3
    assert counts["actionable"] == 2
    assert counts["actionable_buys"] == 1
    assert counts["actionable_sells"] == 1
    assert counts["buys"] == 2
    assert counts["sells"] == 1
    assert counts["neutral"] == 0
    assert breakdown["dominant_trend"] == "buy"
    assert [item["symbol"] for item in breakdown["top_buys"]] == ["ADAUSDT", "XRPUSDT"]
    assert [item["symbol"] for item in breakdown["top_sells"]] == ["LTCUSDT"]
    assert breakdown["top_neutral"] == []
    assert [item["symbol"] for item in breakdown["actionable"]] == [
        "ADAUSDT",
        "LTCUSDT",
    ]
    metrics = breakdown["metrics"]
    actionable_metrics = metrics["actionable"]
    assert actionable_metrics["probability_avg_pct"] == pytest.approx(55.0)
    assert actionable_metrics["ev_avg_bps"] == pytest.approx(13.5)
    overall_metrics = metrics["overall"]
    assert overall_metrics["probability_avg_pct"] == pytest.approx(54.0)
    assert overall_metrics["ev_avg_bps"] == pytest.approx(11.0)

    digest = summary["watchlist_digest"]
    assert digest["headline"].startswith("3 инструментов в наблюдении")
    assert digest["counts"]["actionable"] == 2
    assert any("ADAUSDT" in detail for detail in digest["details"])
    assert digest["dominant_trend"] == "buy"
    assert any("Средние показатели активных идей" in detail for detail in digest["details"])
    assert digest["metrics"] == metrics

    watchlist_breakdown = bot.watchlist_breakdown()
    assert watchlist_breakdown["counts"]["actionable"] == 2
    assert watchlist_breakdown["dominant_trend"] == "buy"

    watchlist_digest = bot.watchlist_digest()
    assert watchlist_digest == digest


def test_guardian_watchlist_breakdown_cache_reuse_and_invalidation(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    entries = [
        {
            "symbol": "ADAUSDT",
            "trend": "buy",
            "actionable": True,
            "probability_pct": 66.0,
            "ev_bps": 15.0,
        },
        {
            "symbol": "LTCUSDT",
            "trend": "sell",
            "actionable": True,
            "probability_pct": 44.0,
            "ev_bps": 12.0,
        },
        {
            "symbol": "XRPUSDT",
            "trend": "buy",
            "actionable": False,
            "probability_pct": 52.0,
            "ev_bps": 6.0,
        },
    ]

    breakdown_first = bot._watchlist_breakdown(entries)
    signature_first = bot._watchlist_breakdown_cache_signature
    assert signature_first is not None
    assert bot._watchlist_breakdown_cache is not None

    breakdown_second = bot._watchlist_breakdown(copy.deepcopy(entries))
    assert breakdown_second == breakdown_first

    breakdown_second["counts"]["total"] = 0
    breakdown_third = bot._watchlist_breakdown(entries)
    assert breakdown_third["counts"]["total"] == breakdown_first["counts"]["total"]

    mutated_entries = copy.deepcopy(entries)
    mutated_entries[0]["trend"] = "sell"
    breakdown_mutated = bot._watchlist_breakdown(mutated_entries)
    assert breakdown_mutated["counts"]["buys"] != breakdown_first["counts"]["buys"]
    assert bot._watchlist_breakdown_cache_signature != signature_first


def test_guardian_watchlist_digest_cache_returns_copies(tmp_path: Path) -> None:
    bot = _make_bot(tmp_path)
    entries = [
        {
            "symbol": "ADAUSDT",
            "trend": "buy",
            "actionable": True,
            "probability_pct": 66.0,
            "ev_bps": 15.0,
        },
        {
            "symbol": "LTCUSDT",
            "trend": "sell",
            "actionable": True,
            "probability_pct": 44.0,
            "ev_bps": 12.0,
        },
        {
            "symbol": "XRPUSDT",
            "trend": "buy",
            "actionable": False,
            "probability_pct": 52.0,
            "ev_bps": 6.0,
        },
    ]

    breakdown = bot._watchlist_breakdown(entries)
    digest_first = bot._watchlist_digest(breakdown)
    signature_first = bot._digest_cache_signature
    assert signature_first is not None
    assert bot._digest_cache is not None

    digest_second = bot._watchlist_digest(copy.deepcopy(breakdown))
    assert digest_second == digest_first

    digest_second["counts"]["actionable"] = 0
    digest_third = bot._watchlist_digest(breakdown)
    assert digest_third["counts"]["actionable"] == digest_first["counts"]["actionable"]

    breakdown_mutated = copy.deepcopy(breakdown)
    breakdown_mutated["counts"]["actionable"] = 0
    digest_mutated = bot._watchlist_digest(breakdown_mutated)
    assert digest_mutated["counts"]["actionable"] == 0
    assert bot._digest_cache_signature != signature_first


def test_guardian_market_scan_extends_watchlist(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "ai" / "market_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_payload = {
        "ts": time.time(),
        "rows": [
            {
                "symbol": "SOLUSDT",
                "turnover24h": "6000000",
                "price24hPcnt": "2.4",
                "bestBidPrice": "20",
                "bestAskPrice": "20.02",
                "volume24h": "1200000",
            },
            {
                "symbol": "XRPUSDT",
                "turnover24h": "3500000",
                "price24hPcnt": "-1.6",
                "bestBidPrice": "0.5",
                "bestAskPrice": "0.501",
                "volume24h": "2200000",
            },
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_market_scan_enabled=True,
        ai_enabled=True,
        ai_min_turnover_usd=1_000_000.0,
        ai_min_ev_bps=40.0,
        ai_max_spread_bps=40.0,
        ai_symbols="",
        ai_max_concurrent=2,
    )

    bot = GuardianBot(data_dir=tmp_path, settings=settings)

    watchlist = bot.market_watchlist()
    symbols = [entry["symbol"] for entry in watchlist[:2]]
    assert symbols == ["SOLUSDT", "XRPUSDT"]
    assert any(entry.get("source") == "market_scanner" for entry in watchlist)

    summary = bot.status_summary()
    assert summary["watchlist_total"] >= 2
    assert summary["watchlist_highlights"][0]["symbol"] == "SOLUSDT"


def test_guardian_market_scan_can_override_status_symbol(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "ai" / "market_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_payload = {
        "ts": time.time(),
        "rows": [
            {
                "symbol": "SOLUSDT",
                "turnover24h": "7200000",
                "price24hPcnt": "2.8",
                "bestBidPrice": "20",
                "bestAskPrice": "20.02",
                "volume24h": "1800000",
            },
            {
                "symbol": "BTCUSDT",
                "turnover24h": "12000000",
                "price24hPcnt": "0.003",
                "bestBidPrice": "30000",
                "bestAskPrice": "30010",
                "volume24h": "4500",
            },
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    status_path = tmp_path / "ai" / "status.json"
    status_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.82,
        "ev_bps": 45.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_market_scan_enabled=True,
        ai_enabled=True,
        ai_min_turnover_usd=1_000_000.0,
        ai_min_ev_bps=30.0,
        ai_max_spread_bps=40.0,
        ai_symbols="BTCUSDT",
        ai_buy_threshold=0.6,
    )

    bot = GuardianBot(data_dir=tmp_path, settings=settings)

    summary = bot.status_summary()
    assert summary["symbol"] == "SOLUSDT"
    assert summary["symbol_source"] == "watchlist"
    assert summary["primary_watch"]["symbol"] == "SOLUSDT"


def test_guardian_market_scan_respects_lists(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "ai" / "market_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_payload = {
        "ts": time.time(),
        "rows": [
            {
                "symbol": "SOLUSDT",
                "turnover24h": "6500000",
                "price24hPcnt": "1.8",
                "bestBidPrice": "20.4",
                "bestAskPrice": "20.42",
                "volume24h": "920000",
            },
            {
                "symbol": "DOGEUSDT",
                "turnover24h": "500000",
                "price24hPcnt": "2.5",
                "bestBidPrice": "0.07",
                "bestAskPrice": "0.0702",
                "volume24h": "32000000",
            },
            {
                "symbol": "XRPUSDT",
                "turnover24h": "4200000",
                "price24hPcnt": "-1.2",
                "bestBidPrice": "0.5",
                "bestAskPrice": "0.5008",
                "volume24h": "18000000",
            },
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_market_scan_enabled=True,
        ai_enabled=True,
        ai_min_turnover_usd=1_000_000.0,
        ai_min_ev_bps=25.0,
        ai_max_spread_bps=50.0,
        ai_symbols="",
        ai_whitelist="DOGEUSDT",
        ai_blacklist="XRPUSDT",
        ai_max_concurrent=2,
    )

    bot = GuardianBot(data_dir=tmp_path, settings=settings)

    watchlist = bot.market_watchlist()
    symbols = [entry["symbol"] for entry in watchlist[:3]]
    assert "SOLUSDT" in symbols
    assert "DOGEUSDT" in symbols
    assert "XRPUSDT" not in symbols
    doge_entry = next(item for item in watchlist if item["symbol"] == "DOGEUSDT")
    assert doge_entry["actionable"] is True


def test_guardian_dynamic_symbols_prioritize_actionable(tmp_path: Path) -> None:
    status = {
        "watchlist": [
            {
                "symbol": "SOLUSDT",
                "probability": 0.72,
                "ev_bps": 20.0,
                "trend": "buy",
            },
            {
                "symbol": "XRPUSDT",
                "probability": 0.35,
                "ev_bps": 12.0,
                "trend": "sell",
            },
            {
                "symbol": "DOGEUSDT",
                "probability": 0.52,
                "ev_bps": 14.0,
                "trend": "buy",
            },
        ]
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_buy_threshold=0.55,
        ai_sell_threshold=0.45,
        ai_min_ev_bps=10.0,
        ai_max_concurrent=2,
        ai_enabled=True,
    )

    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    plan = summary["symbol_plan"]
    assert plan["actionable"] == ("SOLUSDT", "XRPUSDT")
    assert summary["candidate_symbols"] == ["SOLUSDT", "XRPUSDT"]

    candidates = summary["trade_candidates"]
    assert [item["symbol"] for item in candidates[:3]] == [
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    ]
    assert candidates[0]["actionable"] is True
    assert candidates[1]["trend"] == "sell"
    assert any("watchlist" in source for source in candidates[0]["sources"])

    method_candidates = bot.trade_candidates(limit=3)
    assert [item["symbol"] for item in method_candidates] == [
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    ]

    stats = plan["stats"]
    assert stats["actionable_count"] == 2
    assert stats["ready_count"] == 1

    actionable_summary = plan["actionable_summary"]
    assert actionable_summary["count"] == 2
    assert actionable_summary["top_probability"]["symbol"] == "SOLUSDT"
    assert actionable_summary["top_ev"]["symbol"] == "SOLUSDT"

    assert bot.dynamic_symbols() == ["SOLUSDT", "XRPUSDT"]
    assert bot.dynamic_symbols(limit=0) == [
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    ]
    assert bot.dynamic_symbols(limit=0) == [
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    ]
    assert bot.dynamic_symbols(limit=0, only_actionable=True) == [
        "SOLUSDT",
        "XRPUSDT",
    ]

    capacity = plan["capacity_summary"]
    assert capacity["limit"] == 2
    assert capacity["backlog"] == 0
    assert capacity["slot_pressure"] == pytest.approx(1.0)
    assert capacity["can_take_all_actionable"] is True
    assert capacity["needs_attention"] is False

    breakdown = plan["source_breakdown"]
    assert breakdown["watchlist"] == 3
    assert breakdown["actionable"] == 2
    assert breakdown["ready"] == 1
    assert breakdown["multi_source"] >= 3

    assert "diversification" not in plan["position_summary"]


def test_guardian_symbol_plan_prioritises_positions(tmp_path: Path) -> None:
    status = {
        "watchlist": [
            {
                "symbol": "SOLUSDT",
                "probability": 0.66,
                "ev_bps": 25.0,
                "trend": "buy",
                "actionable": True,
                "note": "сильный импульс",
            },
            {
                "symbol": "XRPUSDT",
                "probability": 0.51,
                "ev_bps": 8.0,
                "trend": "buy",
                "probability_ready": True,
            },
        ]
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "category": "spot",
            "symbol": "ADAUSDT",
            "side": "Buy",
            "execPrice": "0.40",
            "execQty": "100",
            "execFee": "0",
        },
        {
            "category": "spot",
            "symbol": "DOGEUSDT",
            "side": "Buy",
            "execPrice": "0.08",
            "execQty": "1000",
            "execFee": "0",
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    settings = Settings(ai_live_only=False, 
        ai_symbols="SOLUSDT",
        ai_max_concurrent=1,
        ai_enabled=True,
        ai_market_scan_enabled=False,
    )

    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    plan = summary["symbol_plan"]

    assert plan["positions"] == ("ADAUSDT", "DOGEUSDT")
    assert plan["positions_only"] == plan["positions"]
    stats = plan["stats"]
    assert stats["position_count"] == 2
    assert stats["open_slots"] == 0
    assert plan["position_summary"]["total_notional"] == pytest.approx(120.0, rel=1e-3)
    largest = plan["position_summary"]["largest"]
    assert largest["symbol"] == "DOGEUSDT"
    assert largest["exposure_pct"] == pytest.approx(66.67, rel=1e-3)
    assert summary["candidate_symbols"][:2] == ["ADAUSDT", "DOGEUSDT"]

    summary_candidates = summary["trade_candidates"]
    assert [item["symbol"] for item in summary_candidates[:3]] == [
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
    ]
    assert summary_candidates[0]["holding"] is True
    assert summary_candidates[2]["actionable"] is True
    assert summary_candidates[2]["probability_pct"] == pytest.approx(66.0, rel=1e-3)

    assert bot.dynamic_symbols() == ["ADAUSDT", "DOGEUSDT"]
    assert bot.dynamic_symbols(limit=0)[:3] == ["ADAUSDT", "DOGEUSDT", "SOLUSDT"]
    assert bot.dynamic_symbols(limit=0, only_actionable=True)[:3] == [
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
    ]

    method_candidates = bot.trade_candidates(limit=4)
    assert [item["symbol"] for item in method_candidates[:3]] == [
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
    ]
    assert method_candidates[0]["holding"] is True
    filtered_candidates = bot.trade_candidates(limit=0)
    assert [item["symbol"] for item in filtered_candidates[:4]] == [
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
        "XRPUSDT",
    ]

    details = plan["details"]
    ada = details["ADAUSDT"]
    doge = details["DOGEUSDT"]
    sol = details["SOLUSDT"]

    assert ada["holding"] is True
    assert doge["holding"] is True
    assert set(sol["sources"]) >= {"watchlist", "actionable"}
    assert sol["actionable"] is True
    assert ada["position_qty"] == pytest.approx(100.0)
    assert doge["position_qty"] == pytest.approx(1000.0)
    assert ada["exposure_pct"] == pytest.approx(33.33, rel=1e-3)
    assert doge["exposure_pct"] == pytest.approx(66.67, rel=1e-3)
    assert sol["probability_pct"] == pytest.approx(66.0, rel=1e-3)

    priority_order = [item["symbol"] for item in plan["priority_table"][:4]]
    assert priority_order[:3] == ["ADAUSDT", "DOGEUSDT", "SOLUSDT"]
    doge_entry = next(
        item for item in plan["priority_table"] if item["symbol"] == "DOGEUSDT"
    )
    assert "holding" in doge_entry["reasons"]

    capacity = plan["capacity_summary"]
    assert capacity["limit"] == 2
    assert capacity["backlog"] == 1
    assert capacity["slot_pressure"] == pytest.approx(1.0)
    assert capacity["needs_attention"] is True

    breakdown = plan["source_breakdown"]
    assert breakdown["holding"] == 2
    assert breakdown["positions_only"] == 2
    assert breakdown["watchlist"] == 2
    assert breakdown["multi_source"] >= 2

    diversification = plan["position_summary"]["diversification"]
    assert diversification["effective_positions"] == pytest.approx(1.8, rel=1e-3)
    assert diversification["largest_share_pct"] == pytest.approx(66.67, rel=1e-3)
    assert diversification["concentration_level"] == "high"


def test_guardian_symbol_plan_cache_updates_on_watchlist_change(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.72,
        "ev_bps": 22.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "watchlist": [
            {
                "symbol": "ETHUSDT",
                "probability": 0.72,
                "ev_bps": 22.0,
                "trend": "buy",
                "actionable": True,
                "note": "первичное наблюдение",
            }
        ],
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    bot = _make_bot(tmp_path)

    summary_initial = bot.status_summary()
    details_initial = summary_initial["symbol_plan"]["details"]["ETHUSDT"]
    assert details_initial["note"] == "первичное наблюдение"
    initial_signature = bot._plan_cache_signature
    assert initial_signature is not None

    status["watchlist"][0]["note"] = "обновлено наблюдение"
    time.sleep(0.01)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    summary_updated = bot.status_summary()
    details_updated = summary_updated["symbol_plan"]["details"]["ETHUSDT"]
    assert details_updated["note"] == "обновлено наблюдение"
    assert bot._plan_cache_signature != initial_signature


def test_guardian_symbol_plan_cache_updates_on_portfolio_change(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.7,
        "ev_bps": 18.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "watchlist": [
            {
                "symbol": "ETHUSDT",
                "probability": 0.7,
                "ev_bps": 18.0,
                "trend": "buy",
                "actionable": True,
            }
        ],
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    initial_events = [
        {
            "category": "spot",
            "symbol": "ADAUSDT",
            "side": "Buy",
            "execPrice": "0.50",
            "execQty": "100",
            "execFee": "0",
        }
    ]
    with ledger_path.open("w", encoding="utf-8") as fh:
        for event in initial_events:
            fh.write(json.dumps(event) + "\n")

    bot = _make_bot(tmp_path)

    summary_initial = bot.status_summary()
    plan_initial = summary_initial["symbol_plan"]
    assert plan_initial["positions"] == ("ADAUSDT",)
    initial_signature = bot._plan_cache_signature
    assert initial_signature is not None

    new_event = {
        "category": "spot",
        "symbol": "DOGEUSDT",
        "side": "Buy",
        "execPrice": "0.08",
        "execQty": "1000",
        "execFee": "0",
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(new_event) + "\n")

    os.utime(ledger_path, None)
    time.sleep(0.05)

    summary_updated = bot.status_summary()
    plan_updated = summary_updated["symbol_plan"]
    assert plan_updated["positions"] == ("ADAUSDT", "DOGEUSDT")
    assert bot._plan_cache_signature != initial_signature


def test_guardian_dynamic_symbols_fallbacks_to_status_symbol(tmp_path: Path) -> None:
    status = {
        "symbol": "ETHUSDT",
        "probability": 0.61,
        "ev_bps": 18.0,
        "last_tick_ts": time.time(),
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, ai_enabled=True, ai_symbols="ETHUSDT")
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    assert summary["candidate_symbols"] == ["ETHUSDT"]

    assert bot.dynamic_symbols() == ["ETHUSDT"]
    assert bot.dynamic_symbols(only_actionable=True) == ["ETHUSDT"]


def test_guardian_actionable_opportunities(tmp_path: Path) -> None:
    status = {
        "watchlist": [
            {
                "symbol": "SOLUSDT",
                "probability": 0.66,
                "ev_bps": 18.0,
                "trend": "buy",
                "note": "обновление максимума",
            },
            {
                "symbol": "XRPUSDT",
                "probability": 0.44,
                "ev_bps": 13.0,
                "trend": "sell",
            },
            {
                "symbol": "DOGEUSDT",
                "probability": 0.55,
                "ev_bps": 6.0,
                "trend": "wait",
            },
        ]
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    settings = Settings(ai_live_only=False, 
        ai_symbols="SOLUSDT,XRPUSDT,DOGEUSDT",
        ai_buy_threshold=0.6,
        ai_sell_threshold=0.45,
        ai_min_ev_bps=10.0,
        ai_enabled=True,
    )

    bot = _make_bot(tmp_path, settings)

    actionable = bot.actionable_opportunities()
    assert [item["symbol"] for item in actionable] == ["SOLUSDT", "XRPUSDT"]
    assert all(item["actionable"] for item in actionable)
    assert actionable[0]["note"] == "обновление максимума"

    limited = bot.actionable_opportunities(limit=1)
    assert [item["symbol"] for item in limited] == ["SOLUSDT"]

    with_neutral = bot.actionable_opportunities(include_neutral=True)
    assert [item["symbol"] for item in with_neutral] == [
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
    ]

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

    settings = Settings(ai_live_only=False, 
        ai_symbols="ETHUSDT",
        ai_buy_threshold=0.7,
        ai_sell_threshold=0.4,
        ai_min_ev_bps=10.0,
        ai_enabled=True,
    )
    bot = _make_bot(tmp_path, settings)

    summary = bot.status_summary()
    assert summary["symbol"] == "ETHUSDT"
    assert summary["probability_pct"] == 75.0
    assert summary["actionable"] is True
    assert summary["actionable_reasons"] == []
    assert summary["thresholds"]["buy_probability_pct"] == 70.0
    assert summary["fallback_used"] is False
    assert summary["status_source"] == "file"
    assert summary["staleness"]["state"] == "fresh"
    assert "extra" in summary["raw_keys"]

    # ensure copies are returned
    summary["symbol"] = "CHANGED"
    assert bot.status_summary()["symbol"] == "ETHUSDT"

    report = bot.unified_report()
    assert report["status"]["symbol"] == "ETHUSDT"
    assert report["status"]["actionable"] is True
    assert report["status"]["status_source"] == "file"
    assert report["status"]["actionable_reasons"] == []
    assert report["status"]["staleness"]["state"] == "fresh"
    report["status"]["symbol"] = "MUTATED"
    assert bot.unified_report()["status"]["symbol"] == "ETHUSDT"


def test_guardian_status_fingerprint_tracks_changes(tmp_path: Path) -> None:
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    first_status = {
        "symbol": "BTCUSDT",
        "probability": 0.6,
        "ev_bps": 15.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path.write_text(json.dumps(first_status), encoding="utf-8")

    bot = _make_bot(
        tmp_path,
        Settings(ai_live_only=False, ai_symbols="BTCUSDT", ai_enabled=True),
    )

    fingerprint_one = bot.status_fingerprint()
    assert fingerprint_one is not None
    assert isinstance(fingerprint_one, str)

    bot.status_summary()

    updated_status = dict(first_status)
    updated_status["probability"] = 0.8
    status_path.write_text(json.dumps(updated_status), encoding="utf-8")

    bot.refresh()
    fingerprint_two = bot.status_fingerprint()
    assert fingerprint_two is not None
    assert isinstance(fingerprint_two, str)
    assert fingerprint_two != fingerprint_one


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

    assert summary["status_source"] == "file"
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
    assert live_summary["status_source"] == "file"

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
    assert live_summary["status_source"] == "file"

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


def test_guardian_replaces_demo_with_live_signal(tmp_path: Path, monkeypatch) -> None:
    demo_status = {
        "ts": time.time() + 86400 * 5,
        "symbol": "BTCUSDT",
        "probability": 0.5,
        "ev_bps": 10.0,
        "side": "buy",
        "watchlist": [],
    }

    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(demo_status), encoding="utf-8")

    live_payload = {
        "symbol": "LTCUSDT",
        "probability": 0.68,
        "ev_bps": 42.0,
        "side": "buy",
        "last_tick_ts": time.time(),
        "watchlist": [
            {
                "symbol": "LTCUSDT",
                "trend": "buy",
                "probability": 0.68,
                "ev_bps": 42.0,
                "actionable": True,
            }
        ],
        "source": "live_scanner",
    }

    class DummyFetcher:
        def __init__(self, *_, **__):
            pass

        def fetch(self):
            return copy.deepcopy(live_payload)

    monkeypatch.setattr(guardian_bot_module, "LiveSignalFetcher", DummyFetcher)

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_enabled=True))
    summary = bot.status_summary()

    assert summary["symbol"] == "LTCUSDT"
    assert summary["status_source"] == "live"
    assert summary["fallback_used"] is False
    assert summary["watchlist_total"] == 1


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

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_buy_threshold=0.6, ai_min_ev_bps=10.0))
    summary = bot.status_summary()

    assert summary["actionable"] is False
    assert summary["staleness"]["state"] == "stale"
    assert any("устар" in reason.lower() for reason in summary["actionable_reasons"])

    health = bot.data_health()
    assert health["ai_signal"]["ok"] is False
    assert "15 минут" in health["ai_signal"]["message"] or "статус" in health["ai_signal"]["message"].lower()


def test_guardian_status_refreshes_stale_signal_with_live_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_status = {
        "symbol": "BTCUSDT",
        "probability": 0.9,
        "ev_bps": 35.0,
        "side": "buy",
        "last_tick_ts": time.time() - (guardian_bot_module.STALE_SIGNAL_SECONDS * 2),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(old_status), encoding="utf-8")

    fresh_payload = {
        "symbol": "BTCUSDT",
        "probability": 0.95,
        "ev_bps": 40.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }

    class DummyFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def fetch(self) -> Dict[str, object]:
            self.calls += 1
            return fresh_payload

    fetcher = DummyFetcher()

    def _get_live_fetcher(self: GuardianBot) -> DummyFetcher:
        return fetcher

    monkeypatch.setattr(GuardianBot, "_get_live_fetcher", _get_live_fetcher, raising=False)

    bot = _make_bot(
        tmp_path,
        Settings(ai_live_only=False, ai_enabled=True, ai_buy_threshold=0.6, ai_min_ev_bps=10.0),
    )
    summary = bot.status_summary()

    assert fetcher.calls == 1
    assert summary["status_source"] == "live"
    assert summary["staleness"]["state"] == "fresh"
    assert not any(
        "устар" in str(reason).lower() for reason in summary["actionable_reasons"]
    )


def test_guardian_live_only_prefers_live_feed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stale_status = {
        "symbol": "BTCUSDT",
        "probability": 0.9,
        "ev_bps": 40.0,
        "side": "buy",
        "last_tick_ts": time.time() - 60.0,
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(stale_status), encoding="utf-8")

    live_payload = {
        "symbol": "XRPUSDT",
        "probability": 0.7,
        "ev_bps": 28.0,
        "side": "sell",
        "last_tick_ts": time.time(),
        "analysis": "Live feed",  # ensure dict copy
    }

    class DummyFetcher:
        def __init__(self, *_, **__):
            pass

        def fetch(self) -> Dict[str, object]:
            return copy.deepcopy(live_payload)

    monkeypatch.setattr(guardian_bot_module, "LiveSignalFetcher", DummyFetcher)

    bot = _make_bot(tmp_path, live_only=True)
    summary = bot.status_summary()

    assert summary["symbol"] == "XRPUSDT"
    assert summary["status_source"] == "live"
    assert summary["fallback_used"] is False


def test_guardian_live_only_returns_empty_when_feed_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    demo_payload = {
        "symbol": "ETHUSDT",
        "probability": 0.85,
        "ev_bps": 35.0,
        "side": "buy",
        "last_tick_ts": time.time(),
    }
    status_path.write_text(json.dumps(demo_payload), encoding="utf-8")

    class EmptyFetcher:
        def __init__(self, *_, **__):
            pass

        def fetch(self) -> Dict[str, object]:
            return {}

    monkeypatch.setattr(guardian_bot_module, "LiveSignalFetcher", EmptyFetcher)

    bot = _make_bot(tmp_path, live_only=True)
    summary = bot.status_summary()

    assert summary["status_source"] == "missing"
    assert summary["symbol"] != "ETHUSDT"


def test_guardian_reports_live_fetch_error_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "symbol": "SOLUSDT",
        "probability": 0.58,
        "ev_bps": 12.0,
        "side": "buy",
        "last_tick_ts": time.time() - (guardian_bot_module.STALE_SIGNAL_SECONDS + 5),
    }
    status_path = tmp_path / "ai" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    class RaisingFetcher:
        def fetch(self) -> Dict[str, object]:
            raise LiveSignalError("scanner down")

    monkeypatch.setattr(guardian_bot_module, "LiveSignalFetcher", lambda *args, **kwargs: RaisingFetcher())

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, ai_enabled=True))
    summary = bot.status_summary()

    assert summary["status_source"] == "file"
    assert summary.get("status_error")
    assert "scanner down" in summary["status_error"].lower()
    assert summary["fallback_used"] is False
    assert summary["actionable"] is False
    assert summary.get("watchlist_total", 0) == 0


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

    monkeypatch.setattr(
        guardian_bot_module,
        "api_key_status",
        lambda _settings: {
            "title": "API",
            "ok": True,
            "message": "Ключ проверен",
            "details": {"network": "Testnet", "mode": "Live"},
        },
    )

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, api_key="k", api_secret="s"))
    bot.generate_brief()
    health = bot.data_health()

    assert health["ai_signal"]["ok"] is True
    assert "AI сигнал" in health["ai_signal"]["title"]
    automation = health["automation"]
    assert automation["title"] == "Автоматизация"
    assert automation["ok"] in (True, False)
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

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardian_bot_module,
        "api_key_status",
        lambda _settings: {
            "title": "API",
            "ok": True,
            "message": "Ключ проверен",
            "details": {"network": "Testnet", "mode": "Live"},
        },
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

    bot = _make_bot(tmp_path, Settings(ai_live_only=False, api_key="k", api_secret="s"))
    report = bot.unified_report()

    monkeypatch.undo()

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

    bot = CountingGuardianBot(data_dir=tmp_path, settings=Settings(ai_live_only=False, ai_market_scan_enabled=False))
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


