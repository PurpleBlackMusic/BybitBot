from __future__ import annotations

import os
from typing import Callable, TypeVar

import requests

from bybit_app.utils.ai.kill_switch import clear_pause as local_clear_pause
from bybit_app.utils.ai.kill_switch import set_pause as local_set_pause

T = TypeVar("T")

_BACKEND_URL = os.getenv("BYBITBOT_BACKEND_URL", "").strip()
_TIMEOUT = float(os.getenv("BYBITBOT_BACKEND_TIMEOUT", "5.0"))
_SESSION = requests.Session()


def _resolve_url(path: str) -> str:
    base = _BACKEND_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def backend_enabled() -> bool:
    return bool(_BACKEND_URL)


def call_backend(path: str, fallback: Callable[[], T]) -> T:
    if not backend_enabled():
        return fallback()
    try:
        response = _SESSION.get(_resolve_url(path), timeout=_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return fallback()
    except ValueError:
        return fallback()
    return payload  # type: ignore[return-value]


def pause_kill_switch(minutes: float | None, reason: str) -> bool:
    reason_text = (reason or "Paused via UI").strip() or "Paused via UI"
    if not backend_enabled():
        local_set_pause(minutes, reason_text)
        return True
    payload = {"minutes": minutes, "reason": reason_text}
    try:
        response = _SESSION.post(_resolve_url("/kill-switch/pause"), json=payload, timeout=_TIMEOUT)
        response.raise_for_status()
        return True
    except requests.RequestException:
        local_set_pause(minutes, reason_text)
        return False


def resume_kill_switch() -> bool:
    if not backend_enabled():
        local_clear_pause()
        return True
    try:
        response = _SESSION.post(_resolve_url("/kill-switch/resume"), timeout=_TIMEOUT)
        response.raise_for_status()
        return True
    except requests.RequestException:
        local_clear_pause()
        return False
