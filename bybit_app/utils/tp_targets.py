"""Helpers for calculating take-profit targets."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

_DEFAULT_FEE_GUARD_BPS = Decimal("20")
_BPS_FACTOR = Decimal("10000")
_ONE = Decimal("1")


def _to_decimal(value: Any, *, default: Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def resolve_fee_guard_fraction(settings: Any, default_bps: Decimal = _DEFAULT_FEE_GUARD_BPS) -> Decimal:
    """Return the TP fee guard as a Decimal fraction."""

    raw_bps = default_bps
    if settings is not None:
        candidate = getattr(settings, "spot_tp_fee_guard_bps", None)
        if candidate is not None:
            raw_bps = _to_decimal(candidate, default=default_bps)
    if raw_bps < 0:
        raw_bps = Decimal("0")
    return raw_bps / _BPS_FACTOR


def target_multiplier(profit_fraction: Decimal, fee_guard_fraction: Decimal) -> Decimal:
    """Calculate the multiplier for a TP price with a fee guard applied."""

    if profit_fraction < Decimal("0"):
        profit_fraction = Decimal("0")
    if fee_guard_fraction < Decimal("0"):
        fee_guard_fraction = Decimal("0")
    return _ONE + profit_fraction + fee_guard_fraction
