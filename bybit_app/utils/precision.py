
from __future__ import annotations
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP
getcontext().prec = 28

def quantize_price(px: float | str, tick: float | str) -> str:
    px = Decimal(str(px)); tick = Decimal(str(tick))
    if tick <= 0: return str(px)
    q = (px / tick).to_integral_value(rounding=ROUND_DOWN) * tick
    return format(q, 'f')

def quantize_qty(qty: float | str, step: float | str) -> str:
    qty = Decimal(str(qty)); step = Decimal(str(step))
    if step <= 0: return str(qty)
    q = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
    return format(q, 'f')

def ceil_qty_to_min_notional(qty: float | str, px: float | str, min_notional: float | str, step: float | str) -> str:
    qty = Decimal(str(qty)); px = Decimal(str(px)); mn = Decimal(str(min_notional)); step = Decimal(str(step if step else '0'))
    if px <= 0: return format(qty, 'f')
    notional = qty * px
    if notional >= mn: 
        return format(qty, 'f')
    # ceil qty to reach min notional
    add = (mn - notional) / px
    qty2 = qty + add
    if step and step>0:
        # round UP to step grid
        k = (qty2 / step).to_integral_value(rounding=ROUND_HALF_UP)
        if k*step < qty2: k = k + 1
        qty2 = k * step
    return format(qty2, 'f')
