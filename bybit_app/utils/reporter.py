from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
from .paths import LOG_DIR
from .telegram_notify import send_telegram
from .envs import get_settings
from .log import read_tail

def _today_bounds():
    now = datetime.now(timezone.utc)
    # count "today" by UTC to avoid TZ drift
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return int(start.timestamp()*1000), int(now.timestamp()*1000)

def summarize_today() -> dict:
    """Parse log tail and compute quick counters for Simple Mode UI."""
    start_ms, now_ms = _today_bounds()
    lines = read_tail(3000)  # tail is enough for the day stats
    events = signals = orders = errors = 0
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = rec.get("ts", 0)
        if ts < start_ms:  # only today
            continue
        events += 1
        ev = rec.get("event","")
        if ev.startswith("ai.signal"):
            signals += 1
        if "order" in ev or ev.startswith("api.order"):
            orders += 1
        if ev.endswith(".error") or "error" in ev:
            errors += 1
    return {"events": events, "signals": signals, "orders": orders, "errors": errors}

def _format_summary_text(stats: dict) -> str:
    return (
        "Отчёт за сегодня\\n"
        f"Событий: {stats.get('events',0)}\\n"
        f"Сигналов: {stats.get('signals',0)}\\n"
        f"Заявок: {stats.get('orders',0)}\\n"
        f"Ошибок: {stats.get('errors',0)}"
    )

def send_daily_report() -> dict:
    s = get_settings()
    # allow silent disable when no telegram configured
    if not getattr(s, "telegram_token", None) or not getattr(s, "telegram_chat_id", None):
        return {"status": "disabled"}
    stats = summarize_today()
    text = _format_summary_text(stats)
    r = send_telegram(text)
    if r.get("ok"):
        return {"status": 200}
    return {"status": "error", "detail": r}

def send_test_message(text: str = "Тест: бот на связи ✅") -> dict:
    s = get_settings()
    if not getattr(s, "telegram_token", None) or not getattr(s, "telegram_chat_id", None):
        return {"status": "disabled"}
    r = send_telegram(text)
    if r.get("ok"):
        return {"status": 200}
    return {"status": "error", "detail": r}
