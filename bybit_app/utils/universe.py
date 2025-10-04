
from __future__ import annotations
import json
import time
from typing import List, Dict
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings, update_settings
from .paths import DATA_DIR

UNIVERSE_FILE = DATA_DIR / "config" / "universe.json"

def build_universe(api: BybitAPI, size: int = 8, min_turnover: float = 2_000_000.0, max_spread_bps: float = 20.0) -> list[str]:
    r = api._req("GET", "/v5/market/tickers", params={"category":"spot"})
    rows = (r.get("result") or {}).get("list") or []
    # fields: symbol, lastPrice, volume24h, turnover24h, highPrice24h, lowPrice24h, price24hPcnt, bestBidPrice, bestAskPrice, 
    scored = []
    for it in rows:
        sym = it.get("symbol","")
        if not sym.endswith("USDT"): continue
        try:
            turnover = float(it.get("turnover24h") or 0.0)
            bid = float(it.get("bestBidPrice") or 0.0); ask = float(it.get("bestAskPrice") or 0.0)
            spr_bps = ((ask - bid)/ask*10000.0) if ask>0 else 1e9
            if turnover >= float(min_turnover) and spr_bps <= float(max_spread_bps):
                scored.append((turnover, sym))
        except Exception:
            continue
    scored.sort(reverse=True)
    top = [s for _,s in scored[:int(size)]]
    # save
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(json.dumps({"ts": int(time.time()*1000), "symbols": top}, ensure_ascii=False, indent=2), encoding="utf-8")
    return top

def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists(): return []
    try:
        d = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
        return d.get("symbols") or []
    except Exception:
        return []

def apply_universe_to_settings(symbols: list[str]):
    s = get_settings()
    update_settings(ai_symbols=",".join(symbols))


def liquidity_score(turnover24h: float, spread_bps: float) -> float:
    return float(turnover24h) / max(1.0, (spread_bps + 1.0))

def build_universe_scored(api: BybitAPI, size: int = 8, min_turnover: float = 2_000_000.0, max_spread_bps: float = 25.0, whitelist: list[str] | None = None, blacklist: list[str] | None = None) -> list[tuple[str,float]]:
    r = api._safe_req("GET", "/v5/market/tickers", params={"category":"spot"})
    rows = (r.get("result") or {}).get("list") or []
    scored = []
    wset = set([x.upper() for x in (whitelist or [])])
    bset = set([x.upper() for x in (blacklist or [])])
    for it in rows:
        sym = (it.get("symbol") or "").upper()
        if not sym.endswith("USDT"): continue
        if sym in bset: continue
        try:
            turnover = float(it.get("turnover24h") or 0.0)
            bid = float(it.get("bestBidPrice") or 0.0); ask = float(it.get("bestAskPrice") or 0.0)
            spr_bps = ((ask - bid)/ask*10000.0) if ask>0 else 1e9
            if turnover >= float(min_turnover) and spr_bps <= float(max_spread_bps):
                score = liquidity_score(turnover, spr_bps)
                scored.append((sym, score))
        except Exception:
            continue
    # force include whitelist
    for s in wset:
        if s not in [x for x,_ in scored]:
            scored.append((s, float("inf")))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:int(size)]

def auto_rotate_universe(api: BybitAPI, size: int, min_turnover: float, max_spread_bps: float, whitelist: list[str], blacklist: list[str]):
    from .cache_kv import TTLKV
    kv = TTLKV(DATA_DIR / "config" / "universe_kv.json")
    last = kv.get("last_rotate_ts", ttl_sec=None, default=0) or 0
    if time.time() - float(last) < 22*3600:  # не чаще раза в ~сутки
        return None
    top = build_universe_scored(api, size=size, min_turnover=min_turnover, max_spread_bps=max_spread_bps, whitelist=whitelist, blacklist=blacklist)
    syms = [s for s,_ in top]
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(json.dumps({"ts": int(time.time()*1000), "symbols": syms}, ensure_ascii=False, indent=2), encoding="utf-8")
    kv.set("last_rotate_ts", time.time())
    return syms
