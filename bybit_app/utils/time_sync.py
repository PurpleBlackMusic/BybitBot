
from __future__ import annotations
import time
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings
from .log import log

def check_time_drift_seconds() -> float:
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    try:
        r = api._req("GET", "/v5/market/time")
        server_ms = int(((r or {}).get("result") or {}).get("timeNano",0))//1_000_000
        local_ms = int(time.time()*1000)
        return (local_ms - server_ms)/1000.0
    except Exception as e:
        log("time.drift.error", err=str(e))
        return 0.0
