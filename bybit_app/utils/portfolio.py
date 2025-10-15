
from __future__ import annotations
import time, math, json
import numpy as np
import pandas as pd
from .bybit_api import BybitAPI, BybitCreds
from .envs import get_settings
from .paths import DATA_DIR
from .log import log
from .ohlcv import normalise_ohlcv_frame

RISK_DIR = DATA_DIR / "risk"
RISK_DIR.mkdir(parents=True, exist_ok=True)

def fetch_klines_df(api: BybitAPI, symbol: str, interval: str = "60", lookback_hours: int = 24*30) -> pd.DataFrame:
    end = int(time.time()*1000)
    start = end - lookback_hours*60*60*1000
    r = api.kline(category="spot", symbol=symbol, interval=interval, start=start, end=end, limit=1000)
    rows = (r.get("result") or {}).get("list") or []
    # bybit returns reverse chronological
    rows = list(reversed(rows))
    if not rows:
        return pd.DataFrame(columns=["start","open","high","low","close","volume"])
    df = pd.DataFrame(
        rows,
        columns=["start","open","high","low","close","volume","turnover"][: len(rows[0])],
    )
    numeric_cols = [col for col in ["open", "high", "low", "close", "volume", "turnover"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = normalise_ohlcv_frame(df, timestamp_col="start")
    return df

def compute_returns(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    return df["close"].pct_change().dropna()

def corr_matrix(api: BybitAPI, symbols: list[str], interval: str = "60", lookback_hours: int = 24*30) -> pd.DataFrame:
    rets = {}
    for sym in symbols:
        df = fetch_klines_df(api, sym, interval=interval, lookback_hours=lookback_hours)
        r = compute_returns(df)
        if not r.empty: rets[sym] = r
    if not rets:
        return pd.DataFrame()
    # align by timestamps (inner join)
    R = pd.DataFrame(rets).dropna()
    return R.corr()

def save_corr(df_corr: pd.DataFrame):
    p = RISK_DIR / "corr.json"
    p.write_text(df_corr.to_json(orient="split"), encoding="utf-8")

def load_corr() -> pd.DataFrame:
    p = RISK_DIR / "corr.json"
    if not p.exists(): return pd.DataFrame()
    return pd.read_json(p.read_text(encoding="utf-8"), orient="split")

def vol_estimate(api: BybitAPI, symbol: str, interval: str = "60", lookback_hours: int = 24*14) -> float:
    df = fetch_klines_df(api, symbol, interval=interval, lookback_hours=lookback_hours)
    r = compute_returns(df)
    if r.empty: return 0.0
    # дневная вола из часовой: sqrt(24)*std_hourly
    std_hour = float(r.std())
    return std_hour * (24.0**0.5)

def group_members(df_corr: pd.DataFrame, threshold: float = 0.8) -> dict[str, list[str]]:
    """Формируем группы: для каждого символа — список сильно скоррелированных (>= threshold)."""
    groups = {}
    if df_corr.empty: return groups
    syms = list(df_corr.columns)
    for s in syms:
        peers = [p for p in syms if p != s and df_corr.loc[s,p] >= threshold]
        groups[s] = sorted(set([s] + peers))
    return groups

def estimate_portfolio_allocation(api: BybitAPI, symbols: list[str], prices: dict[str,float], equity_usdt: float, settings) -> dict[str, float]:
    """Возвращает max notional по символу с учётом vol-target и капитальных лимитов группы/портфеля.
    На вход подаём текущие цены и общий капитал (equity_usdt).
    """
    # параметры
    max_port = float(getattr(settings, "portfolio_max_usdt_pct", 70.0))/100.0
    group_th = float(getattr(settings, "group_corr_threshold", 0.8))
    group_cap = float(getattr(settings, "group_max_cap_pct", 40.0))/100.0
    vol_target_bps = float(getattr(settings, "vol_target_bps", 150.0))/10000.0  # дневная вола позиции

    # корреляции
    C = load_corr()
    if C.empty:
        C = corr_matrix(api, symbols, interval="60", lookback_hours=24*30)
        save_corr(C)
    groups = group_members(C, threshold=group_th)

    # волатильность по символам (дневная)
    vols = {s: max(1e-6, vol_estimate(api, s, interval="60", lookback_hours=24*14)) for s in symbols}

    # базовый vol-target sizing: notional ~ target_vol / symbol_vol
    raw = {s: (vol_target_bps / max(vols[s], 1e-6)) * equity_usdt for s in symbols}

    # нормируем, чтобы суммарно не превышать max_port * equity
    total_cap = sum(raw.values())
    cap_port = equity_usdt * max_port
    scale = 1.0 if total_cap <= cap_port else (cap_port / total_cap if total_cap>0 else 1.0)
    alloc = {s: raw[s]*scale for s in symbols}

    # ограничим на группы: любая группа не более group_cap * equity
    # подсчёт групп по представителю (минимальное имя)
    used = {s: 0.0 for s in symbols}
    group_caps = {}
    for s in symbols:
        g = tuple(groups.get(s, [s]))
        gkey = ",".join(sorted(g))
        if gkey not in group_caps:
            group_caps[gkey] = group_cap * equity_usdt
        # ничего не делаем здесь; проверка будет в live перед размещением

    return {"alloc": alloc, "vols": vols, "corr": C.to_dict(), "groups": groups, "group_caps": group_caps, "cap_port": cap_port}


def effective_number_of_bets(C: pd.DataFrame) -> float:
    """ENB по eigenvalues корреляционной матрицы: (sum λ)^2 / sum λ^2. Для корр. матрицы sum λ = n."""
    if C is None or C.empty: return 0.0
    vals = np.linalg.eigvalsh(C.values)
    s1 = float(np.sum(vals))
    s2 = float(np.sum(vals*vals))
    if s2 <= 0: return 0.0
    return (s1*s1)/s2
