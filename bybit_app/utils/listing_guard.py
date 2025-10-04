
from __future__ import annotations
import time
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings

def is_recently_listed(symbol: str, minutes: int = 5) -> bool:
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    r = api._safe_req("GET", "/v5/market/instruments-info", params={"category":"spot","symbol": symbol})
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return False
    it = rows[0]
    # try multiple fields that appear in various docs; fallback to no-guard if absent
    ts = int(it.get("launchTime") or it.get("createdTime") or 0)
    if ts <= 0: 
        return False
    return (int(time.time()*1000) - ts) < minutes*60*1000
