
from __future__ import annotations
import requests, time
from collections import deque
from .envs import get_settings
from .log import log


def send_telegram(text: str):
    s = get_settings()
    if not s.telegram_token or not s.telegram_chat_id:
        return {"ok": False, "error": "telegram_not_configured"}

    url = f"https://api.telegram.org/bot{s.telegram_token}/sendMessage"
    payload = {"chat_id": s.telegram_chat_id, "text": text}

    _rate_guard()

    try:
        response = requests.post(url, json=payload, timeout=10)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        log("telegram.error", error=str(exc))
        return {"ok": False, "error": str(exc)}

    response_data: dict | None
    try:
        response_data = response.json()
    except ValueError:
        response_data = None

    success = response.ok and not (
        isinstance(response_data, dict) and response_data.get("ok") is False
    )

    if not success:
        description = ""
        if isinstance(response_data, dict):
            description = str(
                response_data.get("description")
                or response_data.get("error")
                or ""
            )
        if not description:
            description = response.text or response.reason
        log(
            "telegram.error",
            status=response.status_code,
            error=description,
        )
        return {
            "ok": False,
            "status": response.status_code,
            "error": description,
            "response": response_data,
        }

    log("telegram.send", status=response.status_code)
    return {"ok": True, "status": response.status_code, "response": response_data}


# Soft rate limiting to respect Telegram Bot API guidance
# - ~1 msg/second per chat; ~20 msg/minute in groups; ~30 msg/sec global
# We implement per-chat guard (simple) to avoid spamming when loops misbehave.
RATE_STATE = {"last_ts": 0.0, "window": deque(maxlen=60)}  # last 60 timestamps


def _rate_guard() -> None:
    now = time.time()

    gap = now - RATE_STATE["last_ts"]
    if gap < 1.0:
        time.sleep(1.0 - gap)
        now = time.time()

    window = RATE_STATE["window"]
    while window and now - window[0] > 60.0:
        window.popleft()

    if len(window) >= 20:
        wait_for = 60.0 - (now - window[0])
        if wait_for > 0:
            time.sleep(wait_for)
            now = time.time()
            while window and now - window[0] > 60.0:
                window.popleft()

    window.append(now)
    RATE_STATE["last_ts"] = now
