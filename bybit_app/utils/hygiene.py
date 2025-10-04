
from __future__ import annotations
from .helpers import ensure_link_id
import time
from .bybit_api import BybitAPI
from .log import log

def cancel_stale_orders(api: BybitAPI, category: str = "spot", symbol: str | None = None, older_than_sec: int = 900, batch_size: int = 10):
    r = api._req("GET", "/v5/order/realtime", signed=True, params={"category": category, "openOnly": 1, "symbol": symbol} if symbol else {"category": category, "openOnly": 1})
    rows = (r.get("result") or {}).get("list") or []
    now_ms = int(time.time()*1000)
    to_cancel = []
    for it in rows:
        ctime = int(it.get("createdTime") or it.get("updatedTime") or 0)
        if now_ms - ctime >= older_than_sec*1000:
            link = it.get("orderLinkId")
            oid = it.get("orderId")
            to_cancel.append({"category": category, "symbol": it.get("symbol"), "orderId": oid, "orderLinkId": link})
    out = {"total": len(to_cancel), "batches": []}
    for i in range(0, len(to_cancel), batch_size):
        chunk = to_cancel[i:i+batch_size]
        req = [{"category": category, "symbol": it["symbol"], "orderId": it["orderId"], "orderLinkId": it["orderLinkId"]} for it in chunk]
        resp = api._req("POST", "/v5/order/cancel-batch", json={"category": category, "request": req})
        out["batches"].append(resp)
        log("order.hygiene.cancel_batch", count=len(chunk), resp=resp)
    return out
