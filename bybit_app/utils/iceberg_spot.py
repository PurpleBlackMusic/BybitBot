
from __future__ import annotations
from .helpers import ensure_link_id
import time, math
from .bybit_api import BybitAPI
from .log import log

def iceberg_spot(api: BybitAPI, symbol: str, side: str, total_qty: float, child_qty: float | None = None, splits: int | None = None, mode: str = "better", offset_bps: float = 1.0, tif: str = "PostOnly", price_limit: float | None = None, sleep_ms: int = 300):
    """Client-side ICEBERG для спота.
    mode: 'fast' (Bid1/Ask1), 'better' (отступ в bps от лучшей), 'fixed' (по фиксированной цене limit=price_limit).
    Если задан child_qty — используем его, иначе splits (равные части).
    Важное: официальный API не даёт прямого параметра "iceberg" для спота — реализуем на клиенте.
    """
    side = side.capitalize()
    assert side in ("Buy","Sell")
    if not child_qty and not splits: raise ValueError("Укажите child_qty или splits")
    if child_qty and splits: raise ValueError("Либо child_qty, либо splits")
    if not splits: splits = max(1, int(math.ceil(total_qty/child_qty)))
    q_left = float(total_qty)

    def best_px():
        ob = api.orderbook(category="spot", symbol=symbol, limit=1)
        a = ((ob.get("result") or {}).get("a") or [[None]])[0][0]
        b = ((ob.get("result") or {}).get("b") or [[None]])[0][0]
        ask = float(a) if a else None; bid = float(b) if b else None
        if mode=="fast":
            return (ask if side=="Buy" else bid)
        elif mode=="better":
            base = (ask if side=="Buy" else bid)
            if base is None: return None
            return base*(1+ (offset_bps/10000.0) if side=="Buy" else 1- (offset_bps/10000.0))
        elif mode=="fixed":
            return price_limit
        else:
            return None

    i=0; results=[]
    while q_left > 1e-12 and i < splits:
        q_child = min(q_left, child_qty if child_qty else total_qty/splits)
        px = best_px()
        if px is None: 
            log("iceberg.skip.noob", symbol=symbol); break
        if price_limit is not None:
            if side=="Buy" and px>price_limit: 
                log("iceberg.stop.limit", reason="px above limit", px=px, limit=price_limit); break
            if side=="Sell" and px<price_limit:
                log("iceberg.stop.limit", reason="px below limit", px=px, limit=price_limit); break
        link = f"ICB-{symbol}-{int(time.time()*1000)}-{i}"
        body = {"category":"spot","symbol":symbol,"side":side,"orderType":"Limit","qty": f"{q_child:.10f}","price": f"{px:.10f}","timeInForce": tif, "orderLinkId": link}
        r = api._req("POST","/v5/order/create", json=body)
        results.append(r)
        log("iceberg.child", i=i, body=body, resp=r)
        i+=1; q_left -= q_child
        time.sleep(sleep_ms/1000.0)
    return {"children": i, "responses": results}
