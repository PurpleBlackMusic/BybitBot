"""Helpers for resolving account fee tiers and guard rails."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Tuple

from .envs import get_api_client, get_settings
from .log import log


@dataclass(frozen=True)
class FeeRate:
    """Container for maker/taker fees expressed as Decimal fractions."""

    maker: Decimal
    taker: Decimal


_FEE_RATE_TTL_SECONDS = 900.0
_FEE_RATE_CACHE: dict[Tuple[str, str, bool], tuple[float, FeeRate]] = {}


def clear_fee_rate_cache() -> None:
    """Purge the memoised fee rate entries (primarily for tests)."""

    _FEE_RATE_CACHE.clear()


def _normalise_symbol(symbol: object | None) -> str:
    if isinstance(symbol, str):
        return symbol.strip().upper()
    if symbol is None:
        return ""
    return str(symbol).strip().upper()


def _to_decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _extract_fee_rate_entry(entry: Mapping[str, object]) -> Optional[FeeRate]:
    maker = _to_decimal(entry.get("makerFeeRate") or entry.get("maker_fee_rate"))
    taker = _to_decimal(entry.get("takerFeeRate") or entry.get("taker_fee_rate"))
    if maker is None and taker is None:
        return None
    return FeeRate(maker=maker or Decimal("0"), taker=taker or Decimal("0"))


def _parse_fee_rate_response(payload: Mapping[str, object], symbol: str) -> Optional[FeeRate]:
    """Extract a fee rate payload from a Bybit ``fee-rate`` response."""

    target = _normalise_symbol(symbol)

    def _iter_rows(container: Mapping[str, object]) -> list[Mapping[str, object]]:
        rows: list[Mapping[str, object]] = []
        raw = container.get("list") or container.get("rows")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping):
                    rows.append(item)
        return rows

    rows: list[Mapping[str, object]] = []
    if isinstance(payload.get("result"), Mapping):
        rows.extend(_iter_rows(payload["result"]))
    elif isinstance(payload.get("list"), list):  # pragma: no branch - alternate schema
        for item in payload.get("list", []):
            if isinstance(item, Mapping):
                rows.append(item)

    candidate: Optional[FeeRate] = None
    for row in rows:
        row_symbol = _normalise_symbol(row.get("symbol"))
        rate = _extract_fee_rate_entry(row)
        if rate is None:
            continue
        if target and row_symbol and row_symbol != target:
            if candidate is None:
                candidate = rate
            continue
        return rate

    return candidate


def _resolve_context_settings(settings: Any | None) -> Any | None:
    if settings is not None:
        return settings
    try:
        return get_settings()
    except Exception:
        return None


def resolve_fee_rate(
    settings: Any | None,
    symbol: str | None,
    *,
    category: str = "spot",
    api: Any | None = None,
    ttl: float = _FEE_RATE_TTL_SECONDS,
) -> Optional[FeeRate]:
    """Return the cached fee rate for the provided instrument if available."""

    resolved_settings = _resolve_context_settings(settings)
    testnet = bool(getattr(resolved_settings, "testnet", False))
    key = (category.lower(), _normalise_symbol(symbol), testnet)

    now = time.time()
    cached = _FEE_RATE_CACHE.get(key)
    if cached and now - cached[0] <= ttl:
        return cached[1]

    if api is None:
        try:
            api = get_api_client()
        except Exception as exc:  # pragma: no cover - defensive guard
            log("fees.api.unavailable", err=str(exc))
            if cached:
                return cached[1]
            return None

    if api is None or not hasattr(api, "fee_rate"):
        return cached[1] if cached else None

    try:
        response = api.fee_rate(category=category, symbol=symbol)
    except Exception as exc:  # pragma: no cover - defensive guard
        log(
            "fees.fetch.error",
            err=str(exc),
            category=category,
            symbol=_normalise_symbol(symbol),
        )
        return cached[1] if cached else None

    if not isinstance(response, Mapping):
        return cached[1] if cached else None

    rate = _parse_fee_rate_response(response, _normalise_symbol(symbol))
    if rate is None:
        return cached[1] if cached else None

    _FEE_RATE_CACHE[key] = (now, rate)
    return rate


def resolve_fee_guard_bps(
    settings: Any | None,
    symbol: str | None,
    *,
    category: str = "spot",
    api: Any | None = None,
    default_bps: Decimal = Decimal("20"),
) -> Decimal:
    """Return the combined entry+exit fee guard in basis points."""

    sentinel = default_bps if isinstance(default_bps, Decimal) else Decimal(str(default_bps))
    raw_override = getattr(settings, "spot_tp_fee_guard_bps", None) if settings is not None else None
    override = _to_decimal(raw_override)

    if override is not None and override > 0 and override != sentinel:
        return override

    rate = resolve_fee_rate(settings, symbol, category=category, api=api)
    if rate is not None:
        taker = abs(rate.taker)
        maker = abs(rate.maker)
        combined = taker * Decimal("2")
        if combined <= 0 and maker > 0:
            combined = maker * Decimal("2")
        guard_bps = combined * Decimal("10000")
        if guard_bps > 0:
            return guard_bps

    if override is not None and override > 0:
        return override

    return sentinel
