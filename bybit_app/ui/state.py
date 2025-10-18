"""Session state helpers and cached data loaders for the UI layer."""
from __future__ import annotations

from typing import Any, Mapping

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


def get_auto_refresh_holds(state: Mapping[str, Any] | None = None) -> list[str]:
    """Return the list of active auto-refresh pause reasons."""

    if state is None:
        state = st.session_state
    holds = state.get(_AUTO_REFRESH_HOLDS_KEY, {})
    if isinstance(holds, Mapping):
        return [str(reason) for reason in holds.values() if str(reason).strip()]
    return []
