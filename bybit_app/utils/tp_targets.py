"""Helpers for calculating take-profit targets."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .fees import resolve_fee_guard_bps as _resolve_fee_guard_bps

_DEFAULT_FEE_GUARD_BPS = Decimal("20")
_BPS_FACTOR = Decimal("10000")
_ONE = Decimal("1")


def resolve_fee_guard_bps(
    settings: Any,
    *,
    symbol: str | None = None,
    api: Any | None = None,
    default_bps: Decimal = _DEFAULT_FEE_GUARD_BPS,
) -> Decimal:
    """Return the TP fee guard expressed in basis points."""

    guard = _resolve_fee_guard_bps(
        settings,
        symbol,
        category="spot",
        api=api,
        default_bps=default_bps,
    )
    if guard < 0:
        return Decimal("0")
    return guard


def resolve_fee_guard_fraction(
    settings: Any,
    *,
    symbol: str | None = None,
    api: Any | None = None,
    default_bps: Decimal = _DEFAULT_FEE_GUARD_BPS,
) -> Decimal:
    """Return the TP fee guard as a Decimal fraction."""

    guard_bps = resolve_fee_guard_bps(
        settings,
        symbol=symbol,
        api=api,
        default_bps=default_bps,
    )
    if guard_bps < 0:
        guard_bps = Decimal("0")
    return guard_bps / _BPS_FACTOR


def target_multiplier(profit_fraction: Decimal, fee_guard_fraction: Decimal) -> Decimal:
    """Calculate the multiplier for a TP price with a fee guard applied."""

    if profit_fraction < Decimal("0"):
        profit_fraction = Decimal("0")
    if fee_guard_fraction < Decimal("0"):
        fee_guard_fraction = Decimal("0")
    return _ONE + profit_fraction + fee_guard_fraction
