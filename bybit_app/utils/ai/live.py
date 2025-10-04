
from __future__ import annotations

import threading
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ..envs import get_api_client, get_settings
from ..helpers import ensure_link_id
from ..paths import DATA_DIR
try:
    from ..log import log
except Exception:
    def log(*a, **k):
        pass

@dataclass
class RunnerState:
    running: bool = False
    last_tick_ts: float = 0.0
    symbol: str = "BTCUSDT"
    side: str = ""
    qty: float = 0.0
    last_error: Optional[str] = None
    last_order: Optional[Dict[str, Any]] = None

class AIRunner:
    """Background AI trading loop (spot). Very lightweight and safe by default."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.s = get_settings()
        self._lock = threading.Lock()
        self.state = RunnerState()
        self.api = get_api_client()
        self._thr: Optional[threading.Thread] = None

    # ---- status helpers ----
    def _write_status(self):
        try:
            obj = {
                "running": self.state.running,
                "last_tick_ts": self.state.last_tick_ts,
                "symbol": self.state.symbol,
                "side": self.state.side,
                "qty": self.state.qty,
                "last_error": self.state.last_error,
            }
            (DATA_DIR / "ai").mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "ai" / "status.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---- public API ----
    def start(self) -> bool:
        with self._lock:
            if self.state.running:
                return True
            self.state.running = True
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
        try:
            log("ai.runner.start")
        except Exception:
            pass
        return True

    def stop(self) -> bool:
        with self._lock:
            self.state.running = False
        try:
            log("ai.runner.stop")
        except Exception:
            pass
        return True

    # ---- loop ----
    def _loop(self):
        while True:
            with self._lock:
                run = self.state.running
            if not run:
                break
            try:
                self._tick_once()
            except Exception as e:
                with self._lock:
                    self.state.last_error = str(e)
                try:
                    log("ai.error", err=str(e))
                except Exception:
                    pass
            self._write_status()
            try:
                log("ai.heartbeat", running=True, ts=int(time.time()*1000))
            except Exception:
                pass
            time.sleep(5.0)

    def _tick_once(self):
        """One AI iteration: pick liquid symbol, build trivial signal, maybe place order (DRY_RUN respected)."""
        self.state.last_tick_ts = time.time()
        s = self.s  # snapshot
        sym_list = [x.strip().upper() for x in (s.ai_symbols or "BTCUSDT,ETHUSDT").split(",") if x.strip()]

        # choose most liquid among list
        uni = self.api.tickers(category="spot")
        items = (uni.get("result") or {}).get("list") or []
        best_sym, best_turnover = None, -1.0
        for it in items:
            sym = it.get("symbol")
            if sym not in sym_list:
                continue
            try:
                turn = float(it.get("turnover24h") or 0.0)
            except Exception:
                turn = 0.0
            if turn > best_turnover:
                best_sym, best_turnover = sym, turn

        if best_sym is None:
            try:
                log("ai.universe.empty", want=sym_list, got=len(items))
            except Exception:
                pass
            return

        sym = best_sym
        try:
            log("ai.universe.pick", symbol=sym, turnover24h=float(best_turnover))
        except Exception:
            pass

        # very naive: best bid and decide random-like by micro-imbalance (placeholder for real model)
        ob = self.api.orderbook(category="spot", symbol=sym, limit=1)
        bids = ((ob.get("result") or {}).get("b") or [[None, None]])
        asks = ((ob.get("result") or {}).get("a") or [[None, None]])
        try:
            best_bid = float(bids[0][0] or 0)
            best_ask = float(asks[0][0] or 0)
        except Exception:
            best_bid = best_ask = 0.0

        side = "buy" if best_ask > 0 and (best_ask - best_bid) / best_ask < 0.0015 else "sell"
        try:
            log("ai.signal", symbol=sym, side=side, bid=best_bid, ask=best_ask)
        except Exception:
            pass

        # position sizing (simplified): risk % of USDT balance
        wal = self.api.wallet_balance(accountType="UNIFIED")
        lst = ((wal.get("result") or {}).get("list") or [])
        ava = 0.0
        if lst:
            coins = (lst[0].get("coin") or [])
            for c in coins:
                if (c.get("coin") or "").upper() == "USDT":
                    try:
                        ava = float(c.get("availableToWithdraw") or c.get("walletBalance") or 0.0)
                    except Exception:
                        ava = 0.0
                    break
        price = best_ask if side == "buy" else best_bid or (best_ask or 0.0)
        risk = max(float(s.ai_risk_per_trade_pct or 0.25), 0.01) / 100.0
        qty = 0.0
        if price > 0:
            qty = max(round((ava * risk) / price, 6), 0.0)

        # filters: if qty tiny — skip
        if qty <= 0.0:
            try:
                log("ai.no_trade", reason="qty<=0", ava=ava, price=price)
            except Exception:
                pass
            return

        # DRY_RUN respected: place real order only if dry_run=False
        if getattr(s, "dry_run", True):
            try:
                log("ai.no_trade", reason="DRY_RUN", qty=qty, side=side)
            except Exception:
                pass
            with self._lock:
                self.state.symbol = sym
                self.state.side = side
                self.state.qty = qty
            return

        # limit order at best bid/ask
        p_adj = price
        q_adj = qty
        order = dict(category="spot", symbol=sym, side=side.upper(), orderType="Limit",
                     price=f"{p_adj}", qty=f"{q_adj}", timeInForce="GTC",
                     orderFilter="Order", orderLinkId=ensure_link_id(f"AIR-{int(time.time())}"))
        try:
            log("ai.order.place", **order)
        except Exception:
            pass
        r = self.api.place_order(**order)
        with self._lock:
            self.state.symbol = sym
            self.state.side = side
            self.state.qty = q_adj
            self.state.last_order = r
        try:
            log("ai.order.result", resp=r)
        except Exception:
            pass
