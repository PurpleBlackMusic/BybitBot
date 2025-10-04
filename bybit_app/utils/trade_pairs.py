
from __future__ import annotations
import json, time
from pathlib import Path
from typing import List, Dict, Any
from .paths import DATA_DIR

DEC = DATA_DIR / "pnl" / "decisions.jsonl"
LED = DATA_DIR / "pnl" / "executions.jsonl"
TRD = DATA_DIR / "pnl" / "trades.jsonl"

def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def _write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pair_trades(window_ms: int = 7*24*3600*1000) -> list[dict]:
    """Пары сделок (spot): entry=покупка (Buy), exit=продажа (Sell).
    Линкуем через orderLinkId при наличии, иначе по времени и символу.
    Выход: список трейдов с метриками: r_mult, bps_realized, hold_sec, fees.
    """
    decs = _read_jsonl(DEC)
    exes = _read_jsonl(LED)
    # Сортируем по времени
    decs.sort(key=lambda d: d.get("ts", 0))
    exes.sort(key=lambda e: e.get("execTime") or e.get("ts") or 0)

    trades = []
    # по символам ведём состояние позиции (qty>0 значит открыта)
    state = {}

    def push_trade(sym, entry_ts, exit_ts, entry_vwap, exit_vwap, qty, fees, sl=None, rr=None, mid=None):
        if qty <= 0: return
        # bps vs entry
        bps = (exit_vwap/entry_vwap - 1.0)*10000.0
        r_mult = None
        if sl and rr:
            # расстояние до SL в процентах от entry
            risk = abs((entry_vwap - sl)/entry_vwap)
            if risk > 1e-9:
                r_mult = (exit_vwap - entry_vwap)/ (entry_vwap * risk)
        trades.append({
            "symbol": sym, "entry_ts": entry_ts, "exit_ts": exit_ts, "hold_sec": int(max(0, (exit_ts-entry_ts)/1000)),
            "qty": qty, "entry_vwap": entry_vwap, "exit_vwap": exit_vwap,
            "fees": fees, "bps_realized": bps, "r_mult": r_mult, "decision_mid": mid
        })

    # индекс решений по символу
    dec_idx = {}
    for d in decs:
        sym = d.get("symbol"); 
        if not sym: continue
        (dec_idx.setdefault(sym, [])).append(d)

    # агрегируем исполнения в «сделки» Buy/Sell (VWAP по направлению)
    # для простоты — строим очередь открытых входов
    from collections import deque, defaultdict
    buckets_buy = defaultdict(list)
    buckets_sell = defaultdict(list)

    for e in exes:
        if (e.get("category") or "spot").lower() != "spot": continue
        sym = e.get("symbol"); side = (e.get("side") or "").lower()
        ts = int(e.get("execTime") or e.get("ts") or 0)
        px = float(e.get("execPrice") or 0.0); q = float(e.get("execQty") or 0.0); fee = float(abs(e.get("execFee") or 0.0))
        if px<=0 or q<=0: continue
        bucket = buckets_buy if side=="buy" else buckets_sell
        bucket[sym].append({"ts": ts, "px": px, "q": q, "fee": fee, "link": e.get("orderLinkId")})

    # теперь делаем паринг: для каждого sell сопоставляем ближайший не‑закрытый buy (FIFO по времени)
    for sym in sorted(set(list(buckets_buy.keys()) + list(buckets_sell.keys()))):
        buys = sorted(buckets_buy.get(sym, []), key=lambda x: x["ts"])
        sells = sorted(buckets_sell.get(sym, []), key=lambda x: x["ts"])
        buy_q = 0.0; buy_notional = 0.0; buy_fee = 0.0; buy_ts = None; buy_mid = None; buy_sl=None; buy_rr=None
        # возьмём последнее решение до buy_ts как контекст
        i = 0; j = 0
        while i < len(buys) or j < len(sells):
            if i < len(buys) and (j>=len(sells) or buys[i]["ts"] <= sells[j]["ts"]):
                b = buys[i]; i += 1
                # если нет открытой позиции — открываем
                if buy_q <= 1e-12:
                    buy_q = b["q"]; buy_notional = b["q"]*b["px"]; buy_fee = b["fee"]; buy_ts = b["ts"]
                    # найдем решение, ближайшее по времени до buy_ts
                    prior_decs = [d for d in dec_idx.get(sym, []) if (d.get("ts") or 0) <= buy_ts]
                    if prior_decs:
                        d = prior_decs[-1]
                        buy_mid = d.get("decision_mid"); buy_sl = d.get("sl"); buy_rr = d.get("rr")
                else:
                    # накапливаем
                    buy_q += b["q"]; buy_notional += b["q"]*b["px"]; buy_fee += b["fee"]
            else:
                if buy_q <= 1e-12:
                    # продажа без открытой позиции — пропускаем
                    j += 1; continue
                s = sells[j]; j += 1
                # закрываем не больше buy_q
                sell_q = min(s["q"], buy_q)
                # считаем vwap по текущей порции
                entry_vwap = buy_notional / buy_q if buy_q>0 else None
                exit_vwap = s["px"]
                fees = buy_fee + s["fee"]
                push_trade(sym, buy_ts, s["ts"], entry_vwap, exit_vwap, sell_q, fees, sl=buy_sl, rr=buy_rr, mid=buy_mid)
                # уменьшаем открытую позицию
                buy_q -= sell_q
                buy_notional = entry_vwap * buy_q
                # пропорционально уменьшим fee
                if buy_q>1e-12:
                    buy_fee *= buy_q / (buy_q + sell_q)
                else:
                    buy_fee = 0.0; buy_ts=None; buy_mid=None; buy_sl=None; buy_rr=None

    # записываем
    _write_jsonl(TRD, trades)
    return trades
