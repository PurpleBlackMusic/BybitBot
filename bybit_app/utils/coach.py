
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time

_OB_CACHE = {}
_TICKERS_CACHE = None
_TICKERS_CACHE_TS = 0.0
def _memo_orderbook(api, category, symbol, ttl=60):
    key = (category, symbol)
    now = time.time()
    v = _OB_CACHE.get(key)
    if v and now - v[0] < ttl:
        return v[1]
    ob = api.orderbook(category=category, symbol=symbol, limit=1)
    _OB_CACHE[key] = (now, ob)
    return ob

def _get_tickers(api, category):
    global _TICKERS_CACHE, _TICKERS_CACHE_TS
    now = time.time()
    if _TICKERS_CACHE and now - _TICKERS_CACHE_TS < 30:
        return _TICKERS_CACHE
    res = api.tickers(category=category) or {}
    _TICKERS_CACHE = res
    _TICKERS_CACHE_TS = now
    return res


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _apply_lists(symbols, s):
    wl = set([x.strip().upper() for x in str(getattr(s, 'ai_symbols_whitelist', '') or '').split(',') if x.strip()])
    bl = set([x.strip().upper() for x in str(getattr(s, 'ai_symbols_blacklist', '') or '').split(',') if x.strip()])
    arr = [sym for sym in symbols if (not wl or sym in wl) and (sym not in bl)]
    return arr if arr else symbols


def market_health(api, category: str = "spot") -> Dict[str, Any]:
    """Return a very simple 'traffic light' for today's trading.
    Uses turnover and spreads of top tickers to say: green / yellow / red.
    """
    res = api.tickers(category=category) or {}
    items = (res.get("result") or {}).get("list") or []
    # filter by reasonable pairs (USDT)
    filt = [it for it in items if isinstance(it, dict) and str(it.get("symbol","")).endswith("USDT")]
    # compute simple liquidity score
    scores: List[Tuple[str, float, float]] = []  # (symbol, turnover24h, spread_pct)
    for it in filt:
        sym = it.get("symbol")
        # we don't have direct spread in tickers; sample via orderbook when needed
        ob = None
        try:
            ob = _memo_orderbook(api, category, sym, ttl=60)
        except Exception:
            ob = {}
        bids = ((ob.get("result") or {}).get("b") or [[None, None]])
        asks = ((ob.get("result") or {}).get("a") or [[None, None]])
        try:
            bid = float(bids[0][0] or 0)
            ask = float(asks[0][0] or 0)
            spread = (ask - bid) / ask if ask > 0 else 1.0
        except Exception:
            spread = 1.0
        turn = _safe_float(it.get("turnover24h"))
        scores.append((sym, turn, spread*10000.0))  # bps

    # apply filters from settings (preset)
    from .envs import get_settings
    s = get_settings()
    max_spread = float(getattr(s, 'ai_max_spread_bps', 25.0))
    min_turn = float(getattr(s, 'ai_min_turnover_usd', 2_000_000.0))
    scores = [t for t in scores if t[2] <= max_spread and t[1] >= min_turn]
    scores.sort(key=lambda t: (t[1], -t[2]), reverse=True)  # high turnover, low spread
    top = scores[:10]

    # traffic light
    if not top:
        light = "red"
        reason = "Нет ликвидных пар по данным биржи."
    else:
        avg_turn = sum(t[1] for t in top) / max(len(top),1)
        avg_spread_bps = sum(t[2] for t in top) / max(len(top),1)
        # heuristics
        if avg_turn > 5e7 and avg_spread_bps < 5:   # 50m USDT and <5 bps
            light = "green"
            reason = "Высокая ликвидность и узкий спред."
        elif avg_turn > 1e7 and avg_spread_bps < 12:
            light = "yellow"
            reason = "Ликвидность средняя, спред умеренный."
        else:
            light = "red"
            reason = "Слабая ликвидность или широкий спред."
    return {"light": light, "reason": reason, "top": [{"symbol": s, "turnover24h": t, "spread_bps": round(sp,2)} for s,t,sp in top]}

def pick_symbols(api, n: int = 3, category: str = "spot") -> List[str]:
    info = market_health(api, category=category)
    syms = [x["symbol"] for x in info.get("top", [])]
    return syms[:n] if syms else ["BTCUSDT","ETHUSDT","SOLUSDT"][:n]

def estimate_training_minutes(symbols_count: int, interval: str, horizon_bars: int) -> int:
    """Rough ETA in minutes. We avoid heavy ML here; just a user-facing estimate."""
    base = 1
    per_symbol = 0.4
    per_bar = 0.05
    k_interval = {"1":1.0, "3":0.9, "5":0.8, "15":0.7, "30":0.6, "60":0.5, "240":0.4, "D":0.3}.get(str(interval), 0.7)
    eta = base + symbols_count*per_symbol + horizon_bars*per_bar*k_interval
    return int(max(1, round(eta)))

def build_autopilot_settings(s, api) -> Dict[str, Any]:
    """Return a dict for update_settings() with safe defaults for a beginner."""
    syms = pick_symbols(api, n=3, category="spot")
    interval = "15"
    risk_pct = 0.25
    settings = dict(
        ai_enabled=True,
        ai_category="spot",
        ai_symbols=",".join(syms),
        ai_interval=interval,
        ai_horizon_bars=12,
        ai_buy_threshold=0.55,
        ai_sell_threshold=0.45,
        ai_risk_per_trade_pct=risk_pct,
        ai_fee_bps=7.0,
        ai_slippage_bps=10.0,
        ai_retrain_minutes=60,
        ai_min_ev_bps=0.5,
        spot_cash_only=True,
        spot_cash_reserve_pct=10.0,
        spot_max_cap_per_trade_pct=5.0,
        spot_max_cap_per_symbol_pct=25.0,
        spot_limit_tif="PostOnly",
        spot_server_tpsl=True,
        spot_tpsl_tp_order_type="Market",
        spot_tpsl_sl_order_type="Market",
        twap_enabled=True,
        twap_slices=5,
        twap_child_secs=8,
        twap_aggressiveness_bps=2.0,
        ai_daily_loss_limit_pct=1.0,
        ai_max_concurrent=1,
    )
    eta = estimate_training_minutes(len(syms), interval, settings["ai_horizon_bars"])
    return {"settings": settings, "eta_minutes": eta, "symbols": syms}
