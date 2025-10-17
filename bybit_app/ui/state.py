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
}


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
