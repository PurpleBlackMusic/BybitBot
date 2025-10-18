"""Session state helpers and cached data loaders for the UI layer."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import time
from typing import Any

import streamlit as st

from bybit_app.utils.background import (
    get_guardian_state,
    get_preflight_snapshot,
    get_ws_snapshot,
)
from bybit_app.utils.envs import get_api_client

BASE_SESSION_STATE: dict[str, Any] = {
    "auto_refresh_enabled": True,
    "refresh_interval": 20,
    "refresh_idle_interval": 8,
    "refresh_idle_after": 45.0,
    "trade_symbol": "BTCUSDT",
    "trade_side": "Buy",
    "trade_notional": 100.0,
    "trade_tolerance_bps": 50.0,
    "trade_feedback": None,
    "logs_level": "INFO",
    "logs_limit": 400,
    "logs_query": "",
    "ui_theme": "dark",
    "signals_actionable_only": False,
    "signals_ready_only": False,
    "signals_hide_skipped": False,
    "signals_min_ev": 0.0,
    "signals_min_probability": 0.0,
    "pause_minutes": 60,
    "kill_reason": "Manual kill-switch",
    "kill_custom_minutes": 60,
    "kill_mode": "pause",
    "quick_trade_symbol": "BTCUSDT",
    "quick_trade_side": "Buy",
    "quick_trade_notional": 100.0,
    "quick_trade_tolerance_bps": 50.0,
    "quick_trade_feedback": None,
    "trade_auto_pause": False,
    "quick_trade_auto_pause": False,
}

_AUTO_REFRESH_HOLDS_KEY = "_auto_refresh_holds"
_COOLDOWN_UNTIL_KEY = "_auto_refresh_cooldown_until"
_COOLDOWN_REASON_KEY = "_auto_refresh_cooldown_reason"
_LAST_INTERACTION_KEY = "_last_interaction_ts"


def _as_mutable(state: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    if isinstance(state, MutableMapping):
        return state
    # ``st.session_state`` is ``MutableMapping``-like but may not register as such.
    # Fallback to attribute checks used by Streamlit proxies.
    for method_name in ("__setitem__", "__delitem__", "pop"):
        if not hasattr(state, method_name):
            return None
    return state  # type: ignore[return-value]


def ensure_keys(overrides: Mapping[str, Any] | None = None) -> None:
    """Populate ``st.session_state`` with default values for the dashboard."""

    defaults = dict(BASE_SESSION_STATE)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            defaults[key] = value

    state = st.session_state
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


@st.cache_resource(show_spinner=False)
def cached_api_client():
    """Return a cached Bybit API client instance."""

    return get_api_client()


def _ensure_mapping(payload: object) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if payload is None:
        return {}
    try:
        return dict(payload)  # type: ignore[arg-type]
    except Exception:
        return {}


@st.cache_data(ttl=10.0, show_spinner=False)
def cached_guardian_snapshot() -> dict[str, Any]:
    """Fetch the latest guardian snapshot with lightweight caching."""

    return _ensure_mapping(get_guardian_state())


@st.cache_data(ttl=10.0, show_spinner=False)
def cached_ws_snapshot() -> dict[str, Any]:
    """Fetch the latest websocket snapshot with lightweight caching."""

    return _ensure_mapping(get_ws_snapshot())


@st.cache_data(ttl=30.0, show_spinner=False)
def cached_preflight_snapshot() -> dict[str, Any]:
    """Return the latest automation/preflight snapshot."""

    return _ensure_mapping(get_preflight_snapshot())


def clear_data_caches() -> None:
    """Invalidate cached data loaders so the next rerun refreshes state."""

    cached_guardian_snapshot.clear()  # type: ignore[attr-defined]
    cached_ws_snapshot.clear()  # type: ignore[attr-defined]
    cached_preflight_snapshot.clear()  # type: ignore[attr-defined]


def set_auto_refresh_hold(key: str, reason: str | None) -> None:
    """Register a reason to temporarily pause automatic refreshes."""

    state = st.session_state
    reason_text = str(reason or "").strip() or "Автообновление приостановлено."
    holds = dict(state.get(_AUTO_REFRESH_HOLDS_KEY, {}))
    holds[key] = reason_text
    state[_AUTO_REFRESH_HOLDS_KEY] = holds


def clear_auto_refresh_hold(key: str) -> None:
    """Remove a previously registered auto-refresh pause."""

    state = st.session_state
    holds = dict(state.get(_AUTO_REFRESH_HOLDS_KEY, {}))
    if key in holds:
        holds.pop(key)
        if holds:
            state[_AUTO_REFRESH_HOLDS_KEY] = holds
        else:
            state.pop(_AUTO_REFRESH_HOLDS_KEY, None)


def set_auto_refresh_cooldown(reason: str, duration_seconds: float) -> None:
    """Pause auto refresh for ``duration_seconds`` after an interaction."""

    seconds = max(float(duration_seconds or 0.0), 0.0)
    if seconds <= 0:
        clear_auto_refresh_cooldown()
        return

    now = time.time()
    state = st.session_state
    state[_COOLDOWN_UNTIL_KEY] = now + seconds
    state[_COOLDOWN_REASON_KEY] = str(reason or "Недавно изменены настройки")


def clear_auto_refresh_cooldown() -> None:
    state = st.session_state
    state.pop(_COOLDOWN_UNTIL_KEY, None)
    state.pop(_COOLDOWN_REASON_KEY, None)


def get_auto_refresh_cooldown(
    state: Mapping[str, Any] | None = None,
) -> tuple[str, float] | None:
    """Return the active cooldown reason and remaining seconds, if any."""

    if state is None:
        state = st.session_state

    until = state.get(_COOLDOWN_UNTIL_KEY)
    try:
        expires_at = float(until)
    except (TypeError, ValueError):
        mutable = _as_mutable(state)
        if mutable is not None:
            mutable.pop(_COOLDOWN_UNTIL_KEY, None)
            mutable.pop(_COOLDOWN_REASON_KEY, None)
        return None

    remaining = expires_at - time.time()
    if remaining <= 0:
        mutable = _as_mutable(state)
        if mutable is not None:
            mutable.pop(_COOLDOWN_UNTIL_KEY, None)
            mutable.pop(_COOLDOWN_REASON_KEY, None)
        return None

    reason = state.get(_COOLDOWN_REASON_KEY, "Недавно изменены настройки")
    return str(reason), remaining


def note_user_interaction(reason: str, *, cooldown: float | None = None) -> None:
    """Register a user interaction for adaptive refresh heuristics."""

    state = st.session_state
    state[_LAST_INTERACTION_KEY] = time.time()
    if cooldown is not None:
        set_auto_refresh_cooldown(reason, cooldown)


def get_last_interaction_timestamp(state: Mapping[str, Any] | None = None) -> float | None:
    if state is None:
        state = st.session_state
    value = state.get(_LAST_INTERACTION_KEY)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def track_value_change(
    state: MutableMapping[str, Any],
    key: str,
    current_value: Any,
    *,
    reason: str,
    cooldown: float = 3.0,
) -> bool:
    """Record ``current_value`` and debounce auto refresh when it changes."""

    sentinel = f"_watch_{key}"
    if sentinel not in state:
        state[sentinel] = current_value
        return False

    previous = state.get(sentinel)
    if previous != current_value:
        state[sentinel] = current_value
        note_user_interaction(reason, cooldown=cooldown)
        return True

    return False


def get_auto_refresh_holds(state: Mapping[str, Any] | None = None) -> list[str]:
    """Return the list of active auto-refresh pause reasons."""

    if state is None:
        state = st.session_state
    holds = state.get(_AUTO_REFRESH_HOLDS_KEY, {})
    messages: list[str] = []
    if isinstance(holds, Mapping):
        for reason in holds.values():
            text = str(reason).strip()
            if text:
                messages.append(text)

    cooldown = get_auto_refresh_cooldown(state)
    if cooldown is not None:
        reason, remaining = cooldown
        if remaining >= 1:
            messages.append(f"{reason} (ещё {int(remaining):d} с)")
        else:
            messages.append(reason)

    return messages
