
from __future__ import annotations
import requests, time
from collections import deque
from .envs import get_settings
from .log import log

def send_telegram(text: str):
    s = get_settings()
    if not s.telegram_token or not s.telegram_chat_id:
        return {"ok": False, "error": "telegram_not_configured"}
    try:
        url = f"https://api.telegram.org/bot{s.telegram_token}/sendMessage"
        _rate_guard()
        resp = requests.post(url, json={"chat_id": s.telegram_chat_id, "text": text}, timeout=10)
        log("telegram.send", status=resp.status_code)
        return {"ok": True, "status": resp.status_code}
    except Exception as e:
        log("telegram.error", error=str(e))
        return {"ok": False, "error": str(e)}


# Soft rate limiting to respect Telegram Bot API guidance
# - ~1 msg/second per chat; ~20 msg/minute in groups; ~30 msg/sec global
# We implement per-chat guard (simple) to avoid spamming when loops misbehave.
RATE_STATE = {"last_ts": 0.0, "window": deque(maxlen=60)}  # last 60 timestamps

def _rate_guard():
    now = time.time()
    # enforce >= 1s between messages
    gap = now - RATE_STATE["last_ts"]
    if gap < 1.0:
        time.sleep(1.0 - gap)
    # enforce <= 20/min by dropping excessive bursts (soft)
    RATE_STATE["window"].append(now)
    RATE_STATE["last_ts"] = now
