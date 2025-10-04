
from __future__ import annotations
def quantize_price(price: float, tick: float) -> float:
    if tick<=0: return price
    return round(price / tick) * tick

def quantize_qty(qty: float, step: float) -> float:
    if step<=0: return qty
    return round(qty / step) * step

def validate_spot_rules(instr: dict, price: float, qty: float) -> dict:
    """Проверяет minNotional, minQty, шаги tick/qtyStep. Возвращает {ok, price_q, qty_q, reasons[]}"""
    res = {"ok": True, "price_q": price, "qty_q": qty, "reasons": []}
    pf = (instr.get("priceFilter") or {})
    lf = (instr.get("lotSizeFilter") or {})
    tick = float(pf.get("tickSize") or 0.0)
    step = float(lf.get("qtyStep") or 0.0)
    min_not = float(lf.get("minOrderAmt") or lf.get("minNotional") or 0.0)
    min_qty = float(lf.get("minOrderQty") or 0.0)
    # quantize
    p_q = quantize_price(price, tick) if tick else price
    q_q = quantize_qty(qty, step) if step else qty
    notional = p_q * q_q
    if min_not and notional < min_not: 
        res["ok"] = False; res["reasons"].append(f"notional {notional:.8f} < minNotional {min_not}")
    if min_qty and q_q < min_qty:
        res["ok"] = False; res["reasons"].append(f"qty {q_q:.8f} < minQty {min_qty}")
    res["price_q"] = p_q; res["qty_q"] = q_q
    return res


from decimal import Decimal
from .precision import quantize_price, quantize_qty, ceil_qty_to_min_notional
from .log import log

MIN_ORDER_VALUE_USDT = Decimal('5')

def _validate_spot_rules_flex(*args, **kwargs):
    """
    Flexible adapter:
    - Old style: validate_spot_rules(instrument, price=, qty=)
    - New style: validate_spot_rules(category, symbol, side, price, qty, instrument)
    Returns dict {'price': str, 'qty': str, 'min_notional_applied': str}
    """
    instrument = None
    price = kwargs.get("price")
    qty = kwargs.get("qty")
    if len(args) == 1 and isinstance(args[0], dict):
        instrument = args[0]
    elif len(args) >= 6:
        # category, symbol, side, price, qty, instrument
        price = args[3]; qty = args[4]; instrument = args[5]
    else:
        # try kw-only
        instrument = kwargs.get("instrument", instrument)
    if instrument is None:
        raise ValueError("instrument is required for validation")
    tick = Decimal(str((instrument.get("priceFilter") or {}).get("tickSize") or instrument.get("tickSize") or "0.00000001"))
    step = Decimal(str((instrument.get("lotSizeFilter") or {}).get("qtyStep") or instrument.get("lotSize") or "0.00000001"))
    min_qty = Decimal(str((instrument.get("lotSizeFilter") or {}).get("minQty") or instrument.get("minQty") or "0"))
    min_notional = Decimal(str((instrument.get("lotSizeFilter") or {}).get("minNotional") or instrument.get("minNotional") or "0"))
    # quantize
    px_q = quantize_price(price, tick)
    qty_q = quantize_qty(qty, step)
    mn = max(min_notional, MIN_ORDER_VALUE_USDT)
    qty_q = ceil_qty_to_min_notional(qty_q, px_q, str(mn), step)
    if Decimal(qty_q) < min_qty:
        qty_q = quantize_qty(min_qty, step)
    return {"price": px_q, "qty": qty_q, "min_notional_applied": str(mn)}

# expose as main name
validate_spot_rules = _validate_spot_rules_flex