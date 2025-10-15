"""Simple file-based kill-switch for pausing automation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..paths import DATA_DIR

_STATE_PATH = DATA_DIR / "ai" / "kill_switch.json"


@dataclass(frozen=True)
class KillSwitchState:
    """Describes the current kill-switch status."""

    paused: bool
    until: Optional[float]
    reason: Optional[str]


def _ensure_parent() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_raw() -> dict[str, object]:
    if not _STATE_PATH.exists():
        return {}

    try:
        payload = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def _write_state(until: float, reason: str, *, now: Optional[float] = None) -> None:
    _ensure_parent()
    payload = {
        "until": float(until),
        "reason": str(reason) if reason else "",
        "created_at": float(now if now is not None else time.time()),
    }
    tmp_path = _STATE_PATH.with_suffix(_STATE_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(_STATE_PATH)


def get_state(*, now: Optional[float] = None) -> KillSwitchState:
    """Return the current kill-switch state.

    Expired pauses are cleared automatically to prevent stale state from keeping
    the automation disabled forever.
    """

    reference = float(now if now is not None else time.time())
    payload = _load_raw()
    try:
        until_value = float(payload.get("until", 0.0) or 0.0)
    except (TypeError, ValueError):
        until_value = 0.0

    if until_value <= reference:
        return KillSwitchState(paused=False, until=None, reason=None)

    reason_value = payload.get("reason")
    if isinstance(reason_value, str):
        reason_text = reason_value or None
    else:
        reason_text = None

    return KillSwitchState(paused=True, until=until_value, reason=reason_text)


def is_paused(*, now: Optional[float] = None) -> Tuple[bool, Optional[float], Optional[str]]:
    state = get_state(now=now)
    return state.paused, state.until, state.reason


def clear_pause() -> None:
    """Remove kill-switch state entirely."""

    try:
        _STATE_PATH.unlink()
    except FileNotFoundError:
        return


def set_pause(minutes: float, reason: str, *, now: Optional[float] = None) -> float:
    """Activate the kill-switch for ``minutes`` minutes.

    Returns the timestamp (epoch seconds) when the automation may resume.
    A non-positive ``minutes`` value clears the pause immediately.
    """

    reference = float(now if now is not None else time.time())
    try:
        duration = float(minutes)
    except (TypeError, ValueError):
        duration = 0.0

    if duration <= 0.0:
        clear_pause()
        return reference

    until = reference + (duration * 60.0)
    _write_state(until, reason, now=reference)
    return until


__all__ = ["KillSwitchState", "get_state", "is_paused", "set_pause", "clear_pause"]
