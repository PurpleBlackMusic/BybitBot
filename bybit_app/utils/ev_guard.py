
from __future__ import annotations
import json, time
from pathlib import Path
from .paths import DATA_DIR
from .pnl import _ledger_path_for
DECISIONS = DATA_DIR / "pnl" / "decisions.jsonl"
GUARD = DATA_DIR / "ai" / "ev_guard.json"

def _read_jsonl(p: Path):
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def realized_bps_last_sells(window: int = 10):
    rows = _read_jsonl(_ledger_path_for())
    sells = [r for r in rows if (r.get('category') or 'spot').lower()=='spot' and (r.get('side') or '').lower()=='sell']
    # группируем по символам и берём последние window исполнений sell на каждом символе
    out = {}
    for s in sells[-window*5:]:
        sym = s.get('symbol'); 
        if not sym: continue
        lst = out.setdefault(sym, [])
        # bps от entry avg (приближение): нужна средняя стоимость из инвентаря; если нет — используем vwap последних buy
        # упростим: если есть поле "avgEntry" в событии (в будущем), пока считаем bps=(execPrice - decisionMid)/decisionMid*10000
        mid = None
        # найдём последнее решение до этого времени
        decs = [d for d in _read_jsonl(DECISIONS) if d.get('symbol')==sym and (d.get('ts') or 0) <= (s.get('execTime') or s.get('ts') or 0)]
        if decs:
            mid = decs[-1].get('decision_mid')
        try:
            px = float(s.get('execPrice') or 0.0)
            if mid: bps = (px/mid - 1.0)*10000.0
            else: bps = float(s.get('execPnlBp') or 0.0)
        except Exception:
            continue
        lst.append(bps)
    # усредняем
    agg = {k: (sum(v)/len(v) if v else None) for k,v in out.items()}
    return agg

def check_and_update_pause(threshold_avg_bps: float = 0.0, min_trades: int = 5, pause_minutes: int = 30):
    from .ai.kill_switch import set_pause
    agg = realized_bps_last_sells()
    # если средний ex-post bps < threshold по любому активному символу — пауза
    bad = {k:v for k,v in agg.items() if v is not None and v < threshold_avg_bps}
    if bad:
        reason = f"expost_ev_avg<{threshold_avg_bps}bps: {bad}"
        until = set_pause(pause_minutes, reason)
        GUARD.write_text(json.dumps({"ts": int(time.time()*1000), "reason": reason, "until": until}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"paused": True, "reason": reason, "until": until}
    return {"paused": False, "agg": agg}
