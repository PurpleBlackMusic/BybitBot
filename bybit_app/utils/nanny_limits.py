
from __future__ import annotations
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings

def spot_order_counters(api: BybitAPI):
    r = api._safe_req("GET", "/v5/order/realtime", params={"category": "spot"}, signed=True)
    rows = (r.get("result") or {}).get("list") or []
    total_open = len(rows)
    tp_sl = sum(1 for it in rows if (it.get("isTpsl") in (True, "True", "true", 1)))
    cond = sum(1 for it in rows if (it.get("orderType") or "").lower() in ("market","limit") and (it.get("triggerPrice") is not None))
    return {"total_open": total_open, "tp_sl": tp_sl, "conditional": cond}
