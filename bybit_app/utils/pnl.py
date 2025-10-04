
from __future__ import annotations
from .helpers import ensure_link_id
from pathlib import Path
import json, time, threading
from .paths import DATA_DIR
from .log import log

LEDGER = DATA_DIR / "pnl" / "executions.jsonl"
_SUMMARY = DATA_DIR / "pnl" / "pnl_daily.json"
_LOCK = threading.Lock()

def add_execution(ev: dict):
    """Сохраняем fill (частичный/полный) из топика execution. Ожидаем поля: symbol, side, orderLinkId, execPrice, execQty, execFee, execTime."""
    rec = {
        "ts": int(time.time()*1000),
        "symbol": ev.get("symbol"),
        "side": ev.get("side"),
        "orderId": ev.get("orderId"),
        "orderLinkId": ev.get("orderLinkId"),
        "execPrice": _f(ev.get("execPrice")),
        "execQty": _f(ev.get("execQty")),
        "execFee": _f(ev.get("execFee")),
        "execTime": ev.get("execTime") or ev.get("transactionTime") or ev.get("ts"),
        "category": ev.get("category") or ev.get("orderCategory") or "spot"
    }
    with _LOCK:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("pnl.exec.add", symbol=rec["symbol"], qty=rec["execQty"], price=rec["execPrice"], fee=rec["execFee"], link=rec["orderLinkId"])

def _f(x):
    try: return float(x)
    except: return None

def read_ledger(n: int = 5000):
    if not LEDGER.exists():
        return []
    with LEDGER.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[-n:]

def daily_pnl():
    """Грубая агрегация PnL по группам OCO на основе fills.
    Для spot считаем: + (sell fills) - (buy fills) - fees.
    Для futures линейно не считаем (оставим на потом) — суммируем signed qty*price и fee как черновик.
    """
    rows = read_ledger(100000)
    by_day = {}
    for r in rows:
        day = time.strftime("%Y-%m-%d", time.gmtime((r.get("execTime") or r.get("ts") or 0)/1000 if r.get("execTime") else time.time()))
        sym = r.get("symbol","?")
        side = (r.get("side") or "").lower()
        px = r.get("execPrice") or 0.0
        qty = r.get("execQty") or 0.0
        fee = r.get("execFee") or 0.0
        cat = (r.get("category") or "spot").lower()
        if day not in by_day: by_day[day] = {}
        if sym not in by_day[day]: by_day[day][sym] = {"spot_pnl":0.0,"fees":0.0,"notional_buy":0.0,"notional_sell":0.0}
        if cat == "spot":
            if side == "buy":
                by_day[day][sym]["spot_pnl"] -= px*qty
                by_day[day][sym]["notional_buy"] += px*qty
            elif side == "sell":
                by_day[day][sym]["spot_pnl"] += px*qty
                by_day[day][sym]["notional_sell"] += px*qty
            by_day[day][sym]["fees"] += abs(fee or 0.0)
        else:
            # для фьючей пока просто аккумулируем нотации и комиссию (точный PnL требует позиций, оставим в v7b)
            if side == "buy":
                by_day[day][sym]["notional_buy"] += px*qty
            else:
                by_day[day][sym]["notional_sell"] += px*qty
            by_day[day][sym]["fees"] += abs(fee or 0.0)
    # сохранить сводку
    _SUMMARY.write_text(json.dumps(by_day, ensure_ascii=False, indent=2), encoding="utf-8")
    return by_day
