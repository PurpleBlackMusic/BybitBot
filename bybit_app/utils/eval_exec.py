
from __future__ import annotations
from pathlib import Path
import json, time, statistics as stats
from .paths import DATA_DIR

LEDGER = DATA_DIR / "pnl" / "executions.jsonl"
DEC_FILE = DATA_DIR / "pnl" / "decisions.jsonl"

def _read_jsonl(p: Path):
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def realized_impact_report(window_sec: int = 1800):
    decs = _read_jsonl(DEC_FILE)
    exes = _read_jsonl(LEDGER)
    out = {}
    now = time.time()*1000
    for d in decs:
        sym = d.get("symbol"); side = (d.get("side") or "").lower()
        mid = d.get("decision_mid"); ts = d.get("ts")
        if not sym or not mid or not ts: continue
        # collect executions within +- window
        fills = [e for e in exes if e.get("symbol")==sym and (e.get("side") or "").lower()==side and abs((e.get("execTime") or e.get("ts") or 0) - ts) <= window_sec*1000]
        if not fills: continue
        qty = sum(float(e.get("execQty") or 0) for e in fills)
        if qty <= 0: continue
        vwap = sum(float((e.get("execQty") or 0))*float((e.get("execPrice") or 0)) for e in fills)/qty
        # bps impact vs decision mid
        if side == "buy":
            imp = (vwap / mid - 1.0)*10000.0
        else:
            imp = (1.0 - vwap / mid)*10000.0
        rec = out.setdefault(sym, {"impacts": [], "n_trades": 0})
        rec["impacts"].append(imp)
        rec["n_trades"] += 1
    # aggregate
    report = {}
    for sym, rec in out.items():
        imps = rec["impacts"]
        p75 = stats.quantiles(imps, n=4)[2] if len(imps)>=4 else max(imps) if imps else None
        report[sym] = {
            "n_trades": rec["n_trades"],
            "avg_impact_bps": sum(imps)/len(imps) if imps else None,
            "p75_impact_bps": p75,
            "suggest_limit_bps": max(5.0, p75*1.1) if p75 else None
        }
    return report
