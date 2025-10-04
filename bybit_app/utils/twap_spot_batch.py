
from __future__ import annotations
from .helpers import ensure_link_id
import time
from .bybit_api import BybitAPI
from .log import log

def _bps(x): return float(x)/10000.0

def twap_spot_batch(api: BybitAPI, symbol: str, side: str, total_qty: float, slices: int = 5, aggressiveness_bps: float = 2.0):
    """SPOT TWAP через /v5/order/create-batch (до 10 ордеров/запрос для спота).
    Делаем список IOC-лимиток вокруг лучшей цены. Возвращает ответ API.
    """
    ob = api.orderbook(category="spot", symbol=symbol, limit=5)
    bids = ((ob.get("result") or {}).get("b") or [])
    asks = ((ob.get("result") or {}).get("a") or [])
    if not bids or not asks:
        return {"error": "empty orderbook"}
    best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
    req = []
    q_child = max(total_qty / max(1, int(slices)), 0.0)
    for i in range(int(slices)):
        if side.lower()=="buy": px = best_ask*(1+_bps(aggressiveness_bps))
        else: px = best_bid*(1-_bps(aggressiveness_bps))
        req.append({
            "symbol": symbol, "side": side, "orderType": "Limit", "qty": f"{q_child:.10f}", "price": f"{px:.10f}",
            "timeInForce": "IOC", "orderLinkId": f"TWAPB-{int(time.time()*1000)}-{i}"
        })
    r = api.batch_place(category="spot", orders=req)
    log("twap.batch", symbol=symbol, side=side, slices=slices, resp=r)
    return r
