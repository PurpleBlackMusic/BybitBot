from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Union

getcontext().prec = 28

NumberLike = Union[Decimal, float, int, str]


def _to_decimal(value: NumberLike) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def decimal_step_places(step: NumberLike) -> int:
    step_decimal = _to_decimal(step)
    if step_decimal <= 0:
        return 0
    exponent = step_decimal.normalize().as_tuple().exponent
    return -exponent if exponent < 0 else 0


def quantize_to_step(value: NumberLike, step: NumberLike, rounding=ROUND_DOWN) -> Decimal:
    value_decimal = _to_decimal(value)
    step_decimal = _to_decimal(step)
    if step_decimal <= 0:
        return value_decimal
    multiplier = (value_decimal / step_decimal).to_integral_value(rounding=rounding)
    return multiplier * step_decimal


def format_to_step(value: NumberLike, step: NumberLike, rounding=ROUND_DOWN) -> str:
    step_decimal = _to_decimal(step)
    if step_decimal <= 0:
        quantized = _to_decimal(value)
        places = decimal_step_places(quantized)
    else:
        quantized = quantize_to_step(value, step_decimal, rounding=rounding)
        places = decimal_step_places(step_decimal)

    if quantized == 0:
        return f"{Decimal('0'):.{places}f}" if places > 0 else "0"

    if places > 0:
        return f"{quantized:.{places}f}"

    if quantized == quantized.to_integral_value():
        quantized = quantized.quantize(Decimal("1"), rounding=rounding)
    else:
        quantized = quantized.normalize()

    return format(quantized, "f")


def quantize_price(px: NumberLike, tick: NumberLike) -> str:
    tick_decimal = _to_decimal(tick)
    if tick_decimal <= 0:
        return format(_to_decimal(px), "f")
    return format_to_step(px, tick_decimal, rounding=ROUND_DOWN)


def quantize_qty(qty: NumberLike, step: NumberLike) -> str:
    step_decimal = _to_decimal(step)
    if step_decimal <= 0:
        return format(_to_decimal(qty), "f")
    return format_to_step(qty, step_decimal, rounding=ROUND_DOWN)


def ceil_qty_to_min_notional(
    qty: NumberLike,
    px: NumberLike,
    min_notional: NumberLike,
    step: NumberLike,
) -> str:
    qty_decimal = _to_decimal(qty)
    price_decimal = _to_decimal(px)
    min_notional_decimal = _to_decimal(min_notional)
    step_decimal = _to_decimal(step if step else Decimal("0"))

    if price_decimal <= 0:
        return format(qty_decimal, "f")

    def _ceil_to_step(val: Decimal) -> Decimal:
        if step_decimal <= 0:
            return val
        multiplier = (val / step_decimal).to_integral_value(rounding=ROUND_DOWN)
        if multiplier * step_decimal < val:
            multiplier += 1
        return multiplier * step_decimal

    target = qty_decimal
    notional = qty_decimal * price_decimal
    if notional < min_notional_decimal:
        add = (min_notional_decimal - notional) / price_decimal
        target = qty_decimal + add

    target = _ceil_to_step(target)
    return format_to_step(target, step_decimal, rounding=ROUND_DOWN)
