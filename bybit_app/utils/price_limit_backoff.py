"""Price-limit backoff bookkeeping shared by the signal executor.

The :mod:`bybit_app.utils.signal_executor` module keeps a fair amount of
stateful logic for handling Bybit's price-limit rejections.  Moving the
supporting helpers here trims the executor while keeping the behaviour fully
tested and importable from other components when needed.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import math

from .signal_helpers import _normalise_slippage_percent, _safe_float

PRICE_LIMIT_LIQUIDITY_TTL = 150.0


def price_limit_quarantine_ttl_for_retries(retries: int) -> float:
    """Return the adaptive quarantine window for price-limit rejections."""

    safe_retries = max(int(retries), 1)
    exponent = safe_retries - 1
    multiplier = min(2.0**exponent, 4.0)
    ttl = PRICE_LIMIT_LIQUIDITY_TTL * multiplier
    return max(ttl, PRICE_LIMIT_LIQUIDITY_TTL)


def price_limit_backoff_expiry(now: float, ttl: float) -> float:
    """Calculate how long liquidity hints should remain valid."""

    buffer = max(ttl * 3.0, PRICE_LIMIT_LIQUIDITY_TTL * 2.0)
    return now + buffer


def purge_backoff_state(
    backoff_state: MutableMapping[str, Dict[str, object]], *, now: float
) -> None:
    """Drop expired entries from ``backoff_state`` in-place."""

    if not backoff_state:
        return
    for symbol, payload in list(backoff_state.items()):
        expires_at = payload.get("expires_at")
        try:
            expiry_value = float(expires_at) if expires_at is not None else None
        except (TypeError, ValueError):
            expiry_value = None
        if expiry_value is not None and expiry_value > now:
            continue
        backoff_state.pop(symbol, None)


def record_price_limit_hit(
    backoff_state: MutableMapping[str, Dict[str, object]],
    symbol: str,
    details: Optional[Mapping[str, object]],
    *,
    last_notional: float,
    last_slippage: float,
    now: float,
) -> Dict[str, object]:
    """Update ``backoff_state`` with the latest rejection details."""

    if not symbol:
        return {}

    purge_backoff_state(backoff_state, now=now)

    existing_state = backoff_state.get(symbol)
    if isinstance(existing_state, Mapping):
        state: Dict[str, object] = dict(existing_state)
    else:
        state = {}

    try:
        retries = int(state.get("retries", 0)) + 1
    except (TypeError, ValueError):
        retries = 1

    payload: Dict[str, object] = dict(state)
    payload.update(
        {
            "retries": retries,
            "last_notional": float(last_notional),
            "last_slippage": float(last_slippage),
            "last_updated": now,
        }
    )

    if details:
        for key in (
            "available_quote",
            "available_base",
            "requested_quote",
            "requested_base",
        ):
            value = _safe_float(details.get(key))
            if value is not None and math.isfinite(value):
                payload[key] = value
        for key in ("price_cap", "price_floor"):
            value = _safe_float(details.get(key))
            if value is not None and math.isfinite(value):
                payload[key] = value

    quarantine_ttl = price_limit_quarantine_ttl_for_retries(retries)
    payload["quarantine_ttl"] = quarantine_ttl
    payload["expires_at"] = price_limit_backoff_expiry(now, quarantine_ttl)

    backoff_state[symbol] = payload
    return payload


def apply_price_limit_backoff(
    backoff_state: MutableMapping[str, Dict[str, object]],
    symbol: str,
    side: str,
    notional_quote: float,
    slippage_pct: float,
    min_notional: float,
    *,
    now: float,
) -> Tuple[float, float, Optional[Dict[str, object]]]:
    """Apply liquidity hints for ``symbol`` to the supplied quote size."""

    if not symbol:
        return notional_quote, slippage_pct, None

    state = backoff_state.get(symbol)
    if not state:
        return notional_quote, slippage_pct, None

    adjustments: Dict[str, object] = {}
    try:
        retries = int(state.get("retries", 0))
    except (TypeError, ValueError):
        retries = 0
    adjustments["retries"] = retries

    requested_key = "requested_quote" if side == "Buy" else "requested_base"
    available_key = "available_quote" if side == "Buy" else "available_base"
    requested = _safe_float(state.get(requested_key))
    available = _safe_float(state.get(available_key))
    price_cap = _safe_float(state.get("price_cap"))
    price_floor = _safe_float(state.get("price_floor"))

    candidate_notional = notional_quote
    ratio: Optional[float] = None
    if requested is not None and requested > 0 and available is not None and available >= 0:
        ratio = max(min(available / requested, 1.0), 0.0)
    if available is not None and available > 0:
        if side == "Buy":
            candidate_notional = min(candidate_notional, available * 0.98)
            adjustments["available_quote"] = available
        else:
            price_hint = price_cap if price_cap and price_cap > 0 else price_floor
            if price_hint and price_hint > 0:
                candidate_notional = min(candidate_notional, available * price_hint * 0.98)
            adjustments["available_base"] = available
    if ratio is not None:
        candidate_notional = min(candidate_notional, notional_quote * max(ratio * 0.98, 0.0))

    adjusted_notional = max(min(candidate_notional, notional_quote), 0.0)
    if min_notional > 0 and adjusted_notional > 0 and adjusted_notional < min_notional:
        adjusted_notional = min_notional
    if not math.isclose(adjusted_notional, notional_quote, rel_tol=1e-9, abs_tol=1e-9):
        adjustments["notional_quote"] = adjusted_notional

    base_slippage = slippage_pct
    previous_slippage = _safe_float(state.get("last_slippage"))
    if previous_slippage is not None and previous_slippage > base_slippage:
        base_slippage = previous_slippage
    growth = 1.0 + min(max(retries, 0), 4) * 0.25
    expanded_slippage = _normalise_slippage_percent(base_slippage * growth)
    if expanded_slippage > slippage_pct:
        slippage_pct = expanded_slippage
        adjustments["slippage_percent"] = slippage_pct

    if price_cap:
        adjustments["price_cap"] = price_cap
    if price_floor:
        adjustments["price_floor"] = price_floor

    state["last_notional"] = adjusted_notional
    state["last_slippage"] = slippage_pct
    state["last_updated"] = now
    expires_at = state.get("quarantine_ttl")
    ttl = _safe_float(expires_at)
    if ttl is not None and ttl > 0:
        state["expires_at"] = price_limit_backoff_expiry(now, ttl)
    backoff_state[symbol] = state

    if not adjustments:
        return adjusted_notional, slippage_pct, None
    return adjusted_notional, slippage_pct, adjustments


def clear_price_limit_backoff(
    backoff_state: MutableMapping[str, Dict[str, object]], symbol: str
) -> None:
    """Remove ``symbol`` from ``backoff_state``."""

    if not symbol:
        return
    backoff_state.pop(symbol, None)

