"""Background automation helpers for the simple mode page.

The original project shipped two divergent versions of this module.  The
"stable" build provided a fairly feature rich scheduler that could start the
AI runner on a timetable and dispatch daily Telegram reports.  Later the
"MAXOPT" build introduced a second, cut-down implementation that only wrote a
heartbeat file required by the simplified Streamlit page.  During the merge the
lightweight version replaced the original one in some archives which left the
application with broken automation: panic-stop flags set by the UI were never
respected, the heartbeat file was missing and the AI runner no longer started
automatically.

This module combines both behaviours.  It keeps the automation features while
also exposing the state helpers and heartbeat file expected by the UI.  The
code is intentionally defensive because the scheduler runs in a background
thread and should never crash the application.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Optional

from .envs import get_settings, update_settings
from .paths import DATA_DIR
from .reporter import send_daily_report
from .ai.live import AIRunner
from .log import log

# File layout
STATE_FILE = DATA_DIR / "simple_mode_state.json"
LEGACY_STATE_FILE = DATA_DIR / "automation.json"
HEARTBEAT_FILE = DATA_DIR / "simple_mode_heartbeat.json"

_stop_event = threading.Event()
_thread_lock = threading.Lock()
_thread: threading.Thread | None = None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _load_state_file() -> dict:
    """Return scheduler state saved by the simple mode page.

    We prefer the new location (``simple_mode_state.json``), but keep backward
    compatibility with the legacy ``automation.json`` used by the older build.
    """

    if STATE_FILE.exists():
        data = _read_json(STATE_FILE)
        if data:
            return data
    if LEGACY_STATE_FILE.exists():
        data = _read_json(LEGACY_STATE_FILE)
        if data:
            return data
    return {}


def _save_state_file(obj: dict) -> bool:
    """Persist automation state for UI controls and guards.

    The main file is ``simple_mode_state.json``; we also best-effort mirror the
    contents into ``automation.json`` so older scripts keep working.
    """

    saved = _write_json(STATE_FILE, obj)
    # Keep the legacy path in sync; ignore failures because the UI does not
    # depend on it anymore.
    _write_json(LEGACY_STATE_FILE, obj)
    return saved


def _write_heartbeat(ts: float | None = None) -> None:
    payload = {"ts": int((ts or time.time()))}
    _write_json(HEARTBEAT_FILE, payload)


def _parse_time_str(s: str) -> Optional[dt.time]:
    try:
        hh, mm = str(s).split(":")
        return dt.time(hour=int(hh), minute=int(mm))
    except Exception:
        return None


def _should_send_report(now: dt.datetime, last_sent_date: Optional[str], report_time: dt.time) -> bool:
    today = now.date().isoformat()
    if last_sent_date == today:
        return False
    return now.time() >= report_time


def _cleanup_stop_day(state: dict, today: str) -> bool:
    """Reset panic-stop flag when a new day starts.

    Returns ``True`` when the state was modified.
    """

    if not state.get("stop_day_locked"):
        return False
    locked_date = str(state.get("stop_day_date") or "")
    if locked_date and locked_date != today:
        state["stop_day_locked"] = False
        state.pop("stop_day_reason", None)
        state.pop("stop_day_date", None)
        return True
    return False


def _is_stop_day_active(state: dict, today: str) -> bool:
    if not state.get("stop_day_locked"):
        return False
    locked_date = str(state.get("stop_day_date") or today)
    return locked_date == today


def _loop() -> None:
    runner = AIRunner()
    while not _stop_event.is_set():
        loop_started = time.time()
        now = dt.datetime.now()
        today = now.date().isoformat()
        state = _load_state_file()
        state_changed = _cleanup_stop_day(state, today)
        _write_heartbeat(loop_started)

        try:
            settings = get_settings()

            # --- Daily report automation ---
            if getattr(settings, "daily_report_enabled", False):
                time_str = getattr(settings, "daily_report_time", "20:00")
                report_time = _parse_time_str(time_str) or dt.time(20, 0)
                last_date = state.get("last_report_date")
                if _should_send_report(now, last_date, report_time):
                    resp = send_daily_report()
                    state["last_report_date"] = today
                    state["last_report_resp"] = resp
                    state_changed = True
                    log("auto.report.sent", resp=resp)

            # --- Timed auto trading ---
            panic_active = _is_stop_day_active(state, today)
            auto_enabled = bool(getattr(settings, "auto_trade_enabled", False))
            if auto_enabled and not panic_active:
                start_time = _parse_time_str(getattr(settings, "auto_start_time", "09:00")) or dt.time(9, 0)
                stop_time = _parse_time_str(getattr(settings, "auto_stop_time", "21:00")) or dt.time(21, 0)
                now_time = now.time()
                if start_time <= now_time <= stop_time:
                    if not runner.running:
                        if getattr(settings, "auto_dry_run", True) != getattr(settings, "dry_run", True):
                            update_settings(dry_run=bool(getattr(settings, "auto_dry_run", True)))
                        runner.start()
                        log("auto.trade.started", t=str(now_time))
                else:
                    if runner.running:
                        runner.stop()
                        log("auto.trade.stopped", reason="window")
            else:
                if runner.running:
                    runner.stop()
                    reason = "stop_day" if panic_active else "auto_disabled"
                    log("auto.trade.stopped", reason=reason)

            if state_changed:
                _save_state_file(state)

        except Exception as exc:  # pragma: no cover - defensive logging
            log("auto.loop.error", err=str(exc))

        # Respect stop requests quickly while keeping CPU usage low.
        sleep_for = max(5.0, 30.0 - (time.time() - loop_started))
        if _stop_event.wait(timeout=sleep_for):
            break

    if runner.running:
        try:
            runner.stop()
        except Exception as exc:  # pragma: no cover - defensive logging
            log("auto.loop.stop.error", err=str(exc))


def start_background_loop() -> bool:
    """Idempotent background scheduler entry point."""

    global _thread
    with _thread_lock:
        if _thread and _thread.is_alive():
            return True
        _stop_event.clear()
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()
    return True


def stop_background_loop() -> bool:
    _stop_event.set()
    with _thread_lock:
        thread = _thread
        _thread = None
    if thread and thread.is_alive():
        thread.join(timeout=1.0)
    return True


__all__ = [
    "start_background_loop",
    "stop_background_loop",
    "_load_state_file",
    "_save_state_file",
]

