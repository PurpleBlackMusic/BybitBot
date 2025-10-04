
from __future__ import annotations
from typing import Literal, Tuple, Dict, Any
import math

def estimate_vwap_from_orderbook(ob: dict, side: Literal["Buy","Sell"], qty_base: float | None = None, notional_quote: float | None = None) -> dict:
    """Считает VWAP и импакт (в б.п.) из snapshot orderbook (Bybit /v5/market/orderbook).
    ob: ответ целиком; берём ob['result']['a'/'b'] списки [price, qty].
    Если задан qty_base — используем его; иначе считаем по notional_quote.
    Возвращает: {vwap, filled_qty, notional, impact_bps, best, mid}
    """
    res = (ob.get("result") or {})
    bids = res.get("b") or []
    asks = res.get("a") or []
    if not bids or not asks:
        return {"vwap": None, "filled_qty": 0.0, "notional": 0.0, "impact_bps": None, "best": None, "mid": None}
    best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
    mid = (best_bid + best_ask)/2.0
    levels = asks if side=="Buy" else bids
    # traverse book
    need_qty = qty_base
    need_notional = notional_quote
    filled_qty = 0.0
    spent = 0.0
    for px_str, qty_str in levels:
        px = float(px_str); q = float(qty_str)
        if need_qty is not None:
            take = min(q, max(0.0, need_qty - filled_qty))
            spent += take * px
            filled_qty += take
            if filled_qty >= need_qty - 1e-12:
                break
        else:
            # spend notional on adverse side
            can_take = min(q, max(0.0, (need_notional - spent)/px))
            spent += can_take * px
            filled_qty += can_take
            if spent >= need_notional - 1e-8:
                break
    if filled_qty <= 0 or spent <= 0:
        return {"vwap": None, "filled_qty": 0.0, "notional": 0.0, "impact_bps": None, "best": None, "mid": mid}
    vwap = spent / filled_qty
    # impact vs mid
    if side=="Buy":
        impact = (vwap / mid - 1.0)*10000.0
    else:
        impact = (1.0 - vwap / mid)*10000.0
    return {"vwap": vwap, "filled_qty": filled_qty, "notional": spent, "impact_bps": impact, "best": {"bid": best_bid, "ask": best_ask}, "mid": mid}
