
from __future__ import annotations
import json
from pathlib import Path
from .paths import DATA_DIR

LEDGER = DATA_DIR / "pnl" / "executions.jsonl"

def spot_fifo_pnl(ledger_path: Path | None = None):
    """FIFO учёт по каждой монете.

    Parameters
    ----------
    ledger_path: Path | None
        Позволяет указать альтернативный путь к журналу исполнений (упрощает тестирование).

    Returns
    -------
    dict
        ``{symbol: {realized_pnl, position_qty, layers:[(qty, cost)]}}``
    """
    ledger = ledger_path or LEDGER
    out = {}
    if not ledger.exists():
        return out
    with ledger.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ev = json.loads(line)
            if (ev.get("category") or "spot").lower() != "spot":
                continue
            sym = ev.get("symbol"); side = (ev.get("side") or "").lower()
            px = float(ev.get("execPrice") or 0.0); qty = float(ev.get("execQty") or 0.0); fee = abs(float(ev.get("execFee") or 0.0))
            if not sym or qty<=0 or px<=0: continue
            if sym not in out: out[sym] = {"realized_pnl":0.0, "position_qty":0.0, "layers":[]}
            book = out[sym]
            if side=="buy":
                # комиссия в цену
                eff_cost = (px*qty + fee)/qty
                book["layers"].append([qty, eff_cost])
                book["position_qty"] += qty
            elif side=="sell":
                remain = qty
                proceeds = px*qty - fee
                # списываем FIFO
                cost_total = 0.0; used = 0.0
                while remain>1e-12 and book["layers"]:
                    lqty, lcost = book["layers"][0]
                    take = min(lqty, remain)
                    cost_total += take*lcost
                    used += take
                    lqty -= take; remain -= take
                    if lqty<=1e-12: book["layers"].pop(0)
                    else: book["layers"][0] = [lqty, lcost]
                book["position_qty"] -= used
                book["realized_pnl"] += proceeds - cost_total
    return out
