
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN

D = Decimal

def floor_to_step(x, step) -> Decimal:
    x = D(str(x))
    step = D(str(step))
    return (x // step) * step

def clamp_qty(qty, step, epsilon_steps=0, max_prec: int | None = None):
    q = floor_to_step(qty, step)
    if epsilon_steps:
        q -= D(step) * D(epsilon_steps)
    if max_prec is not None:
        q = q.quantize(D(10) ** -int(max_prec), rounding=ROUND_DOWN)
    if q < 0:
        q = D("0")
    return q

def gte_min_notional(qty, price, min_amt):
    return D(str(qty)) * D(str(price)) >= D(str(min_amt))
