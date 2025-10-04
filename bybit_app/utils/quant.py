
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN

D = Decimal


def _to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return D(str(value))


def floor_to_step(x, step) -> Decimal:
    """Floor ``x`` to the nearest ``step`` using exchange precision rules."""

    x_dec = _to_decimal(x)
    step_dec = _to_decimal(step)
    if step_dec <= 0:
        return x_dec
    multiplier = (x_dec / step_dec).to_integral_value(rounding=ROUND_DOWN)
    return multiplier * step_dec


def clamp_qty(qty, step, epsilon_steps=0, max_prec: int | None = None):
    """Clamp a quantity to the exchange step size and optional precision."""

    step_dec = _to_decimal(step)
    qty_dec = _to_decimal(qty)

    if step_dec > 0:
        clamped = floor_to_step(qty_dec, step_dec)
    else:
        clamped = qty_dec

    if epsilon_steps and step_dec > 0:
        clamped -= step_dec * _to_decimal(epsilon_steps)

    if max_prec is not None:
        quant = D(10) ** -int(max_prec)
        clamped = clamped.quantize(quant, rounding=ROUND_DOWN)

    if clamped < 0:
        clamped = D("0")
    return clamped


def gte_min_notional(qty, price, min_amt):
    qty_dec = _to_decimal(qty)
    price_dec = _to_decimal(price)
    min_dec = _to_decimal(min_amt)
    if qty_dec <= 0 or price_dec <= 0:
        return False
    return qty_dec * price_dec >= min_dec
