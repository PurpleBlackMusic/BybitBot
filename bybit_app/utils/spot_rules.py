from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Mapping, Sequence

from . import validators
from .bybit_api import BybitAPI
from .precision import quantize_to_step


DecimalLike = Decimal | float | int | str


class SpotInstrumentNotFound(RuntimeError):
    """Raised when the spot instrument metadata cannot be located."""


def _to_decimal(value: DecimalLike) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _round_up_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    multiplier = (value / step).to_integral_value(rounding=ROUND_UP)
    if multiplier * step < value:
        multiplier += 1
    return multiplier * step


def _extract_instrument(response: Mapping[str, object] | None, symbol: str) -> Mapping[str, object]:
    if not isinstance(response, Mapping):
        raise SpotInstrumentNotFound(f"Не удалось получить данные об инструменте {symbol}")

    result = response.get("result")
    entries: Sequence[object]
    if isinstance(result, Mapping):
        entries = result.get("list") or []  # type: ignore[assignment]
    elif isinstance(result, Sequence):  # pragma: no cover - defensive branch
        entries = result
    else:
        entries = []

    if not entries:
        raise SpotInstrumentNotFound(f"Инструмент {symbol} не найден в ответе биржи")

    key = symbol.upper()
    for entry in entries:
        if isinstance(entry, Mapping) and str(entry.get("symbol") or "").upper() == key:
            return entry

    fallback = entries[0]
    if isinstance(fallback, Mapping):
        return fallback

    raise SpotInstrumentNotFound(f"Инструмент {symbol} не найден в ответе биржи")


def load_spot_instrument(api: BybitAPI, symbol: str) -> Mapping[str, object]:
    """Return raw spot instrument metadata for ``symbol``."""

    response = api.instruments_info(category="spot", symbol=symbol.upper())
    return _extract_instrument(response, symbol)


def quantize_spot_order(
    *,
    instrument: Mapping[str, object],
    price: DecimalLike,
    qty: DecimalLike,
    side: str,
) -> validators.SpotValidationResult:
    """Quantise ``price``/``qty`` using exchange rules and enforce minimums."""

    price_candidate: DecimalLike = price
    qty_candidate: DecimalLike = qty
    side_normalised = (side or "").capitalize()
    rounding_mode = ROUND_UP if side_normalised == "Buy" else ROUND_DOWN

    for _ in range(6):
        validated = validators.validate_spot_rules(
            instrument=instrument,
            price=price_candidate,
            qty=qty_candidate,
            side=side_normalised,
        )
        if validated.ok:
            price_candidate = validated.price
            qty_candidate = validated.qty

        adjusted = False
        price_candidate = validated.price
        qty_candidate = validated.qty

        if validated.tick_size > 0:
            quantized_price = quantize_price_only(
                price_candidate,
                tick_size=validated.tick_size,
                rounding=rounding_mode,
            )
            if quantized_price != price_candidate:
                price_candidate = quantized_price
                adjusted = True

        if validated.qty_step > 0:
            quantized_qty = quantize_to_step(
                qty_candidate,
                validated.qty_step,
                rounding=rounding_mode,
            )
            if quantized_qty != qty_candidate:
                qty_candidate = quantized_qty
                adjusted = True

        if validated.min_qty > 0 and any(reason.startswith("qty") for reason in validated.reasons):
            target_qty = validated.min_qty
            target_qty = _round_up_to_step(target_qty, validated.qty_step)
            if qty_candidate < target_qty:
                qty_candidate = target_qty
                adjusted = True

        if (
            validated.min_notional > 0
            and validated.price > 0
            and any(reason.startswith("notional") for reason in validated.reasons)
        ):
            target_qty = validated.min_notional / validated.price
            target_qty = _round_up_to_step(target_qty, validated.qty_step)
            if qty_candidate < target_qty:
                qty_candidate = target_qty
                adjusted = True

        if not adjusted and validated.ok:
            return validated

    return validated


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def quantize_price_only(
    value: DecimalLike,
    *,
    tick_size: Decimal,
    rounding=ROUND_DOWN,
) -> Decimal:
    decimal_value = _to_decimal(value)
    if tick_size <= 0:
        return decimal_value
    if rounding == ROUND_UP:
        return _round_up_to_step(decimal_value, tick_size)
    return quantize_to_step(decimal_value, tick_size, rounding=rounding)
