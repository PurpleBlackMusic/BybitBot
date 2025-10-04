
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from ..paths import DATA_DIR

DEC = DATA_DIR / "pnl" / "decisions.jsonl"
LED = DATA_DIR / "pnl" / "executions.jsonl"

def _read_jsonl(p: Path):
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def tune_buy_threshold(symbol: str, rr: float = 1.8, fee_bps: float = 7.0, window=200):
    """Грубая подстройка buy_threshold по последним решениям (symbol).
    Берём последние N решений и смотрим, какова доля удачных после фактовых движений (приближённо: close[t+h] > close[t]).
    Здесь используем поля decisions: p_up (если есть) или proxied через ev; пока используем p_up, если был сохранён.
    Возвращает: best_threshold, table (для UI).
    """
    decs = [d for d in _read_jsonl(DEC) if d.get("symbol")==symbol.upper()]
    if not decs: return None, []
    # Используем p_up из decision, если писали; если нет — пропустим (в текущей версии нет p_up => сделаем устойчивую заглушку)
    ps = []
    for d in decs[-window:]:
        p = d.get("p_up")
        if p is None:
            # аппроксимация из EV и RR: EV = p*RR - (1-p) - cost -> p ≈ (EV + 1 + cost)/(RR+1)
            ev = d.get("ev_bps_pred")
            cost = (d.get("fee_bps") or 0)/10000.0
            if ev is None:
                continue
            p = min(0.9, max(0.5, (ev/10000.0 + 1.0 + cost)/(rr+1.0)))
        ps.append(float(p))
    if not ps: return None, []
    ps = np.array(ps)
    grid = np.linspace(0.52, 0.75, 12)
    best_th, best_score = 0.55, -1e9
    rows = []
    for th in grid:
        # surrogate EV score: trades when p>=th -> mean( p*RR - (1-p) )
        mask = ps >= th
        if mask.sum()==0:
            score = -1e9
        else:
            score = float((ps[mask]*rr - (1-ps[mask])).mean())
        rows.append({"threshold": float(th), "score": float(score), "n_trades": int(mask.sum())})
        if score > best_score:
            best_score, best_th = score, th
    return float(best_th), rows
