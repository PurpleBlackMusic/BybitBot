"""Utilities for normalising AI trading thresholds."""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` to ``float``."""

    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalise_min_ev_bps(raw_value: Any, *, default_bps: float) -> float:
    """Coerce ``ai_min_ev_bps`` style inputs to a non-negative basis-point value.

    The configuration historically mixed three different units:

    ``basis points``
        ``20`` → 20 bps → 0.20%

    ``percent``
        ``"0.2%"`` or ``0.2`` → 0.2% → 20 bps

    ``ratio``
        ``0.002`` → 0.20% → 20 bps

    ``Bybit`` UI users frequently supply percentages while programmatic clients
    prefer basis points.  To avoid silently requiring manual conversions the
    helper recognises common spellings and normalises everything to basis
    points.  Values below zero are treated as "no minimum" and collapse to 0.
    """

    # Handle string suffixes explicitly first to avoid expensive heuristics.
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if not text:
            return max(default_bps, 0.0)
        if text.endswith("bps"):
            candidate = _safe_float(text[:-3])
            if candidate is not None:
                return max(candidate, 0.0)
        if text.endswith("bp"):
            candidate = _safe_float(text[:-2])
            if candidate is not None:
                return max(candidate, 0.0)
        if text.endswith("%"):
            candidate = _safe_float(text[:-1])
            if candidate is not None:
                return max(candidate * 100.0, 0.0)

    numeric = _safe_float(raw_value)
    if numeric is None:
        return max(default_bps, 0.0)

    if numeric <= 0:
        return 0.0

    # Extremely small magnitudes are usually ratios (e.g. 0.002 → 20 bps).
    if numeric < 0.02:
        return max(numeric * 10_000.0, 0.0)

    # Values within ``(0.02, 1.0)`` are likely user-supplied percentages.
    if numeric < 1.0:
        return max(numeric * 100.0, 0.0)

    return max(numeric, 0.0)


def min_change_from_ev_bps(min_ev_bps: float, *, floor: float) -> float:
    """Translate an EV requirement to a price change filter with a floor."""

    if min_ev_bps <= 0.0:
        return floor
    return max(min_ev_bps / 10_000.0, floor)


def resolve_min_ev_from_settings(settings: Any, *, default_bps: float) -> float:
    """Return the normalised min-EV basis points from ``settings``."""

    if settings is None:
        return max(default_bps, 0.0)
    raw_value = getattr(settings, "ai_min_ev_bps", default_bps)
    return normalise_min_ev_bps(raw_value, default_bps=default_bps)

