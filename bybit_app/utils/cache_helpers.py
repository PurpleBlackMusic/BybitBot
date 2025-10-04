
from __future__ import annotations
import streamlit as st
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings

@st.cache_data(ttl=30)
def cached_tickers(category: str = "spot", symbol: str | None = None):
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    return api.tickers(category=category, symbol=symbol)

@st.cache_data(ttl=300)
def cached_instruments(category: str = "spot", symbol: str | None = None):
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    return api.instruments_info(category=category, symbol=symbol)


@st.cache_data(ttl=900)
def cached_fee_rate(category: str = "spot", symbol: str | None = None, baseCoin: str | None = None):
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    try:
        return api.fee_rate(category=category, symbol=symbol, baseCoin=baseCoin)
    except Exception as e:
        return {"error": str(e)}


import json, time
from .paths import DATA_DIR

def refresh_instruments_cache(symbols_csv: str, max_age_sec: int = 6*3600):
    cache = DATA_DIR / "instruments_cache.json"
    if cache.exists():
        try:
            meta = json.loads(cache.read_text(encoding="utf-8"))
            ts = float(meta.get("_ts", 0))
            if time.time() - ts < max_age_sec:
                return meta
        except Exception:
            pass
    s = get_settings()
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    out = {"_ts": time.time()}
    for sym in [x.strip().upper() for x in (symbols_csv or '').split(',') if x.strip()]:
        try:
            out[sym] = api.instruments_info(category="spot", symbol=sym).get("result")
        except Exception as e:
            out[sym] = {"error": str(e)}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
