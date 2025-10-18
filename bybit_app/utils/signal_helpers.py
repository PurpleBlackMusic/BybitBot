"""Shared helper utilities for the signal execution layer.

The :mod:`bybit_app.utils.signal_executor` module grew organically and started
hosting a mixture of low level helpers and the main execution state machine.
Extracting generic helpers into this module keeps the executor focused on the
high level orchestration logic while still exposing the tiny utilities that are
reused across multiple code paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
import math
import re
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List

from .bybit_errors import parse_bybit_error_message
from .precision import format_to_step, quantize_to_step
from .pnl import execution_fee_in_quote

_DEFAULT_DECIMAL_STEP = Decimal("0.00000001")

# Bybit restricts the percent tolerance that can be supplied alongside market
# orders.  The range is kept here so that both the executor and potential future
# consumers share a single source of truth.
_PERCENT_TOLERANCE_MIN = 0.05
_PERCENT_TOLERANCE_MAX = 5.0

# Certain Bybit error codes should not trigger aggressive retry logic when a
# take-profit ladder is being submitted.  Keeping the allow list close to the
# helper that parses the error keeps the intent explicit.
_TP_LADDER_SKIP_CODES = {"170194", "170131"}


def _decimal_from(value: object, default: Decimal = Decimal("0")) -> Decimal:
    """Return a :class:`Decimal` instance for ``value`` or ``default``."""

    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    """Convert ``value`` to a finite float where possible."""

    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(candidate):
        return candidate
    return None


def _parse_decimal_sequence(raw: object) -> list[Decimal]:
    """Return a sequence of positive decimals extracted from ``raw``."""

    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [token.strip() for token in re.split(r"[;,]", raw) if token.strip()]
    elif isinstance(raw, Sequence):
        tokens = list(raw)
    else:
        tokens = [raw]
    values: list[Decimal] = []
    for token in tokens:
        candidate = token
        if isinstance(candidate, str):
            candidate = candidate.strip()
        dec = _decimal_from(candidate)
        if dec > 0:
            values.append(dec)
    return values


def _infer_price_step(audit: Mapping[str, object] | None) -> Decimal:
    """Infer the price step used when placing an order."""

    candidates: list[str] = []
    if isinstance(audit, Mapping):
        for key in ("price_payload", "limit_price"):
            raw = audit.get(key)
            if raw is None:
                continue
            if isinstance(raw, str) and raw.strip():
                candidates.append(raw.strip())
                break
            candidates.append(str(raw))
    for text in candidates:
        try:
            value = Decimal(text)
        except (InvalidOperation, ValueError):
            continue
        exponent = value.normalize().as_tuple().exponent
        if exponent < 0:
            return Decimal("1").scaleb(exponent)
    return _DEFAULT_DECIMAL_STEP


def _round_to_step(value: Decimal, step: Decimal, *, rounding: str) -> Decimal:
    """Round ``value`` to the provided ``step`` using ``rounding`` mode."""

    return quantize_to_step(value, step, rounding=rounding)


def _format_decimal_step(value: Decimal, step: Decimal) -> str:
    """Format ``value`` following the precision described by ``step``."""

    return format_to_step(value, step, rounding=ROUND_DOWN)


def _format_price_step(value: Decimal, step: Decimal) -> str:
    """Format a price using the rounding rules dictated by ``step``."""

    return format_to_step(value, step, rounding=ROUND_UP)


def _clamp_price_to_band(
    price: Decimal,
    *,
    price_step: Decimal,
    band_min: Decimal,
    band_max: Decimal,
) -> Decimal:
    """Clamp ``price`` to ``[band_min, band_max]`` while respecting ``price_step``."""

    adjusted = price
    if band_min > 0 and adjusted < band_min:
        adjusted = band_min
    if band_max > 0 and adjusted > band_max:
        adjusted = band_max
    adjusted = _round_to_step(adjusted, price_step, rounding=ROUND_UP)
    if band_min > 0 and adjusted < band_min:
        adjusted = _round_to_step(band_min, price_step, rounding=ROUND_UP)
    if band_max > 0 and adjusted > band_max:
        adjusted = _round_to_step(band_max, price_step, rounding=ROUND_DOWN)
    return adjusted


def _coerce_timestamp(value: object) -> Optional[float]:
    """Normalise ``value`` to a UNIX timestamp expressed in seconds."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            ts = float(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    elif isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    else:
        return None

    if ts > 1e18:
        ts /= 1e9
    elif ts > 1e12:
        ts /= 1e3
    return ts


def _build_position_layers(
    events: Iterable[Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    """Reconstruct spot position layers from raw execution events."""

    states: Dict[str, Dict[str, object]] = {}
    processed: List[Tuple[float, int, Mapping[str, object], Optional[float]]] = []

    for idx, raw_event in enumerate(events or []):
        if not isinstance(raw_event, Mapping):
            continue

        category = str(raw_event.get("category") or "spot").lower()
        if category != "spot":
            continue

        price = _safe_float(raw_event.get("execPrice"))
        qty = _safe_float(raw_event.get("execQty"))
        fee = execution_fee_in_quote(raw_event, price=price)
        side = str(raw_event.get("side") or "").lower()
        symbol_value = raw_event.get("symbol") or raw_event.get("ticker")
        symbol = str(symbol_value or "").strip().upper()

        if not symbol or price is None or qty is None or qty <= 0 or price <= 0:
            continue

        timestamp = None
        for key in ("execTime", "execTimeNs", "transactTime", "ts", "created_at"):
            timestamp = _coerce_timestamp(raw_event.get(key))
            if timestamp is not None:
                break

        sort_key = timestamp if timestamp is not None else float(idx)
        processed.append((sort_key, idx, raw_event, timestamp))

    processed.sort(key=lambda item: (item[0], item[1]))

    for _, _, event, actual_ts in processed:
        price = _safe_float(event.get("execPrice")) or 0.0
        qty = _safe_float(event.get("execQty")) or 0.0
        fee = execution_fee_in_quote(event, price=price)
        side = str(event.get("side") or "").lower()
        symbol_value = event.get("symbol") or event.get("ticker")
        symbol = str(symbol_value or "").strip().upper()

        if not symbol or price <= 0 or qty <= 0:
            continue

        state = states.setdefault(symbol, {"layers": deque(), "position_qty": 0.0})
        layers = state["layers"]
        if not isinstance(layers, deque):
            state["layers"] = layers = deque()

        if side == "buy":
            effective_cost = (price * qty + fee) / qty
            layers.append({"qty": float(qty), "price": float(effective_cost), "ts": actual_ts})
            state["position_qty"] = float(state.get("position_qty", 0.0) + qty)
            continue

        if side != "sell":
            continue

        remain = float(qty)
        while remain > 1e-12 and layers:
            layer = layers[0]
            layer_qty = float(layer.get("qty") or 0.0)
            take = min(layer_qty, remain)
            layer_qty -= take
            remain -= take
            state["position_qty"] = float(max(0.0, state.get("position_qty", 0.0) - take))
            if layer_qty <= 1e-12:
                layers.popleft()
            else:
                layer["qty"] = layer_qty

    final_states: Dict[str, Dict[str, object]] = {}
    for symbol, state in states.items():
        layers = state.get("layers")
        if isinstance(layers, deque):
            final_layers = [dict(layer) for layer in layers]
        else:
            final_layers = []
        final_states[symbol] = {
            "position_qty": float(state.get("position_qty", 0.0)),
            "layers": final_layers,
        }

    return final_states


def _lookup_price_in_mapping(value: object, symbol: str) -> Optional[float]:
    """Search ``value`` for a price associated with ``symbol``."""

    if value is None:
        return None

    upper_symbol = symbol.upper()
    lower_symbol = upper_symbol.lower()
    base_symbol = upper_symbol[:-4] if upper_symbol.endswith("USDT") else upper_symbol
    candidates = [upper_symbol, lower_symbol, base_symbol, base_symbol.lower()]

    if isinstance(value, Mapping):
        for key in candidates:
            if not key:
                continue
            price = _safe_float(value.get(key))
            if price is not None:
                return price

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            item_symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
            if item_symbol != upper_symbol:
                continue
            for field in (
                "price",
                "last_price",
                "lastPrice",
                "mark_price",
                "markPrice",
                "close",
                "close_price",
                "closePrice",
            ):
                price = _safe_float(item.get(field))
                if price is not None:
                    return price
    return None


def _extract_market_price(
    summary: Mapping[str, object],
    symbol: str,
) -> Optional[float]:
    """Return the best market price candidate for ``symbol`` from ``summary``."""

    if not isinstance(summary, Mapping):
        return None

    upper_symbol = symbol.upper()
    summary_symbol = summary.get("symbol")
    if isinstance(summary_symbol, str) and summary_symbol.strip().upper() == upper_symbol:
        for field in ("price", "last_price", "lastPrice", "mark_price", "markPrice"):
            direct_value = summary.get(field)
            if isinstance(direct_value, Mapping):
                price = _lookup_price_in_mapping(direct_value, upper_symbol)
            else:
                price = _safe_float(direct_value)
            if price is not None:
                return price

    for key in (
        "prices",
        "price_map",
        "priceMap",
        "mark_prices",
        "markPrices",
        "last_prices",
        "lastPrices",
    ):
        price = _lookup_price_in_mapping(summary.get(key), upper_symbol)
        if price is not None:
            return price

    plan = summary.get("symbol_plan")
    if isinstance(plan, Mapping):
        for field in ("positions", "priority_table", "priorityTable", "combined"):
            price = _lookup_price_in_mapping(plan.get(field), upper_symbol)
            if price is not None:
                return price

    price = _lookup_price_in_mapping(summary.get("trade_candidates"), upper_symbol)
    if price is not None:
        return price

    price = _lookup_price_in_mapping(summary.get("watchlist"), upper_symbol)
    if price is not None:
        return price

    return None


def _extract_execution_totals(
    response: Mapping[str, object] | None,
) -> tuple[Decimal, Decimal]:
    """Compute executed base/quote totals from Bybit order responses."""

    executed_base = Decimal("0")
    executed_quote = Decimal("0")

    payloads: list[Mapping[str, object]] = []
    if isinstance(response, Mapping):
        payloads.append(response)
        result = response.get("result")
        if isinstance(result, Mapping):
            payloads.append(result)
        elif isinstance(result, Sequence) and result:
            first = result[0]
            if isinstance(first, Mapping):
                payloads.append(first)

    for payload in payloads:
        qty = _decimal_from(payload.get("cumExecQty"))
        if qty <= 0:
            qty = _decimal_from(payload.get("cumExecQtyForCloud"))
        quote = _decimal_from(payload.get("cumExecValue"))
        if qty > 0:
            executed_base = max(executed_base, qty)
        if quote <= 0 and qty > 0:
            avg_price = _decimal_from(payload.get("avgPrice"))
            if avg_price <= 0:
                avg_price = _decimal_from(payload.get("orderPrice"))
            if avg_price > 0:
                quote = qty * avg_price
        if quote > 0:
            executed_quote = max(executed_quote, quote)

    if (executed_base <= 0 or executed_quote <= 0) and isinstance(response, Mapping):
        local = response.get("_local")
        attempts = None
        if isinstance(local, Mapping):
            attempts = local.get("attempts")
        if isinstance(attempts, Sequence):
            base_total = Decimal("0")
            quote_total = Decimal("0")
            for entry in attempts:
                if not isinstance(entry, Mapping):
                    continue
                base_total += _decimal_from(entry.get("executed_base"))
                quote_total += _decimal_from(entry.get("executed_quote"))
            if base_total > 0:
                executed_base = max(executed_base, base_total)
            if quote_total > 0:
                executed_quote = max(executed_quote, quote_total)

    return executed_base, executed_quote


def _format_decimal_for_meta(value: Decimal) -> str:
    """Render a decimal into the most compact string representation."""

    quantised = value
    if value == value.to_integral():
        quantised = value
    else:
        quantised = value.normalize()
    return format(quantised, "f")


def _partial_attempts(response: Mapping[str, object] | None) -> list[dict[str, object]]:
    """Return copies of stored partial-attempt metadata."""

    if not isinstance(response, Mapping):
        return []
    local = response.get("_local") if isinstance(response, Mapping) else None
    attempts = None
    if isinstance(local, Mapping):
        attempts = local.get("attempts")
    if not isinstance(attempts, Sequence):
        return []
    extracted: list[dict[str, object]] = []
    for entry in attempts:
        if isinstance(entry, Mapping):
            extracted.append(dict(entry))
    return extracted


def _store_partial_attempts(
    response: Mapping[str, object] | None,
    attempts: Sequence[Mapping[str, object]],
) -> None:
    """Persist partial-attempt metadata on ``response`` for later inspection."""

    if not isinstance(response, MutableMapping):  # type: ignore[arg-type]
        return
    local = response.get("_local") if isinstance(response.get("_local"), Mapping) else None
    if not isinstance(local, MutableMapping):  # type: ignore[arg-type]
        local = {}
        response["_local"] = local
    local["attempts"] = list(attempts)


@dataclass(frozen=True)
class _LadderStep:
    """Describe a single rung within the take-profit ladder."""

    profit_bps: Decimal
    size_fraction: Decimal

    @property
    def profit_fraction(self) -> Decimal:
        """Return the step profit expressed as a fraction for convenience."""

        return self.profit_bps / Decimal("10000")


def _normalise_slippage_percent(value: float) -> float:
    """Clamp a raw percent tolerance to Bybit's allowed range."""

    if value <= 0.0:
        return 0.0
    return max(_PERCENT_TOLERANCE_MIN, min(value, _PERCENT_TOLERANCE_MAX))


def _safe_symbol(value: object) -> Optional[str]:
    """Return a normalised symbol string or ``None`` when the value is invalid."""

    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _safe_float(value: object) -> Optional[float]:
    """Safely coerce ``value`` to ``float`` when possible."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _format_bybit_error(exc: Exception) -> str:
    """Return a readable message from a raw Bybit API exception."""

    text = str(exc)
    parsed = parse_bybit_error_message(text)
    if parsed:
        code, message = parsed
        return f"Bybit отказал ({code}): {message}"
    return f"Не удалось отправить ордер: {text}"


def _extract_bybit_error_code(exc: Exception) -> Optional[str]:
    """Extract the Bybit error code from an exception when possible."""

    parsed = parse_bybit_error_message(str(exc))
    if parsed:
        code, _ = parsed
        return code
    return None


__all__ = [
    "_LadderStep",
    "_TP_LADDER_SKIP_CODES",
    "_clamp_price_to_band",
    "_decimal_from",
    "_decimal_to_float",
    "_extract_bybit_error_code",
    "_extract_execution_totals",
    "_format_bybit_error",
    "_format_decimal_for_meta",
    "_format_decimal_step",
    "_format_price_step",
    "_infer_price_step",
    "_normalise_slippage_percent",
    "_parse_decimal_sequence",
    "_partial_attempts",
    "_round_to_step",
    "_safe_float",
    "_safe_symbol",
    "_store_partial_attempts",
]
