from __future__ import annotations

import threading
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from ..envs import get_api_client, get_settings
from ..helpers import ensure_link_id
from ..paths import DATA_DIR
from .engine import AIPipeline
from .features import make_features
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
    probability: float = 0.0
    ev_bps: float = 0.0
    last_error: Optional[str] = None
    last_order: Optional[Dict[str, Any]] = None

class AIRunner:
    """Background AI trading loop (spot). Very lightweight and safe by default."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.pipe = AIPipeline(self.data_dir)
        self.s = get_settings()
        self._lock = threading.Lock()
        self.state = RunnerState()
        self.api = get_api_client()
        self._thr: Optional[threading.Thread] = None
        self._model_cache: Dict[str, Dict[str, Any]] = {}

    # ---- status helpers ----
    def _set_state(self, **updates):
        with self._lock:
            for k, v in updates.items():
                setattr(self.state, k, v)

    def _write_status(self):
        try:
            obj = {
                "running": self.state.running,
                "last_tick_ts": self.state.last_tick_ts,
                "symbol": self.state.symbol,
                "side": self.state.side,
                "qty": self.state.qty,
                "probability": self.state.probability,
                "ev_bps": self.state.ev_bps,
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
    def _model_path(self, category: str, symbol: str, interval: str) -> Path:
        return self.data_dir / f"model_{category}_{symbol}_{interval}.json"

    def _load_model(self, category: str, symbol: str, interval: str):
        key = f"{category}:{symbol}:{interval}"
        path = self._model_path(category, symbol, interval)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._model_cache.pop(key, None)
            return None, None
        cached = self._model_cache.get(key)
        if cached and cached.get("mtime") == mtime:
            return cached["model"], cached["meta"]
        loaded = self.pipe.load_model(path)
        if not loaded:
            self._model_cache.pop(key, None)
            return None, None
        model, meta = loaded
        self._model_cache[key] = {"model": model, "meta": meta, "mtime": mtime}
        return model, meta

    def _loop(self):
        while True:
            with self._lock:
                run = self.state.running
            if not run:
                break
            try:
                self._tick_once()
            except Exception as e:
                self._set_state(last_error=str(e))
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
        """One AI iteration: pick liquid symbol, score AI signal and place order if EV is attractive."""
        now = time.time()
        self._set_state(last_tick_ts=now)
        s = get_settings(force_reload=True)
        self.s = s
        category = (s.ai_category or "spot").lower()
        interval = str(s.ai_interval or "5")
        sym_list = [x.strip().upper() for x in (s.ai_symbols or "BTCUSDT,ETHUSDT").split(",") if x.strip()]
        if not sym_list:
            try:
                log("ai.universe.empty", reason="no_symbols")
            except Exception:
                pass
            self._set_state(symbol="", side="", qty=0.0, probability=0.0, ev_bps=0.0)
            return

        uni = self.api.tickers(category=category)
        items = (uni.get("result") or {}).get("list") or []
        best_sym, best_turnover = None, -1.0
        for it in items:
            sym = (it.get("symbol") or "").upper()
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
            self._set_state(symbol="", side="", qty=0.0, probability=0.0, ev_bps=0.0)
            return

        sym = best_sym
        try:
            log("ai.universe.pick", symbol=sym, turnover24h=float(best_turnover))
        except Exception:
            pass

        model, meta = self._load_model(category, sym, interval)
        if not model or not meta:
            try:
                log("ai.model.missing", symbol=sym, category=category, interval=interval)
            except Exception:
                pass
            self._set_state(symbol=sym, side="", qty=0.0, probability=0.0, ev_bps=0.0)
            return

        kl = self.pipe.fetch_klines(self.api, category, sym, interval, limit=500)
        feats_df = make_features(kl)
        feature_cols = meta.get("feature_names") or []
        missing = [c for c in feature_cols if c not in feats_df.columns]
        if missing or feats_df.empty:
            try:
                log("ai.features.missing", symbol=sym, missing=missing)
            except Exception:
                pass
            self._set_state(symbol=sym, side="", qty=0.0, probability=0.0, ev_bps=0.0)
            return

        latest = feats_df.iloc[-1][feature_cols].values.reshape(1, -1)
        prob_up = float(model.predict_proba(latest)[0])
        prob_down = 1.0 - prob_up
        pos_bps = float(meta.get("pos_ret_bps", 0.0) or 0.0)
        neg_bps = float(meta.get("neg_ret_bps", 0.0) or 0.0)
        fee_bps = float(getattr(s, "ai_fee_bps", meta.get("trading_cost_bps", 0.0)) or 0.0)
        slip_bps = float(getattr(s, "ai_slippage_bps", 0.0) or 0.0)
        total_cost = fee_bps + slip_bps
        ev_long = prob_up * pos_bps + prob_down * neg_bps - total_cost
        ev_short = prob_down * abs(neg_bps) - prob_up * max(pos_bps, 0.0) - total_cost

        buy_th = float(s.ai_buy_threshold or 0.55)
        sell_th = float(s.ai_sell_threshold or 0.45)
        min_ev = float(s.ai_min_ev_bps or 0.0)

        try:
            log(
                "ai.signal.score",
                symbol=sym,
                prob_up=prob_up,
                prob_down=prob_down,
                ev_long=ev_long,
                ev_short=ev_short,
                buy_th=buy_th,
                sell_th=sell_th,
                min_ev=min_ev,
            )
        except Exception:
            pass

        chosen_side = ""
        chosen_prob = prob_up
        chosen_ev = ev_long
        if category == "spot":
            if prob_up >= buy_th and ev_long >= min_ev:
                chosen_side = "buy"
            else:
                self._set_state(symbol=sym, side="", qty=0.0, probability=prob_up, ev_bps=ev_long, last_error=None)
                try:
                    log("ai.no_trade", reason="thresholds", symbol=sym)
                except Exception:
                    pass
                return
        else:
            if prob_up >= buy_th and ev_long >= min_ev:
                chosen_side = "buy"
            elif prob_down >= sell_th and ev_short >= min_ev:
                chosen_side = "sell"
                chosen_prob = prob_down
                chosen_ev = ev_short
            else:
                self._set_state(symbol=sym, side="", qty=0.0, probability=prob_up, ev_bps=max(ev_long, ev_short), last_error=None)
                try:
                    log("ai.no_trade", reason="thresholds", symbol=sym)
                except Exception:
                    pass
                return

        ob = self.api.orderbook(category=category, symbol=sym, limit=1)
        bids = ((ob.get("result") or {}).get("b") or [[None, None]])
        asks = ((ob.get("result") or {}).get("a") or [[None, None]])
        try:
            best_bid = float(bids[0][0] or 0)
            best_ask = float(asks[0][0] or 0)
        except Exception:
            best_bid = best_ask = 0.0

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
        price = best_ask if chosen_side == "buy" else (best_bid or best_ask)
        risk = max(float(s.ai_risk_per_trade_pct or 0.25), 0.01) / 100.0
        reserve_pct = max(float(getattr(s, "spot_cash_reserve_pct", 10.0) or 0.0), 0.0)
        deployable = max(ava * (1.0 - reserve_pct / 100.0), 0.0)
        cap_limit_pct = max(float(getattr(s, "spot_max_cap_per_trade_pct", 0.0) or 0.0), 0.0)
        trade_capital = deployable * risk
        if cap_limit_pct > 0:
            trade_capital = min(trade_capital, ava * cap_limit_pct / 100.0)
        qty = 0.0
        if price > 0:
            qty = max(round(trade_capital / price, 6), 0.0)

        if qty <= 0.0:
            try:
                log("ai.no_trade", reason="qty<=0", ava=ava, price=price)
            except Exception:
                pass
            self._set_state(symbol=sym, side="", qty=0.0, probability=chosen_prob, ev_bps=chosen_ev, last_error=None)
            return

        if getattr(s, "dry_run", True):
            try:
                log("ai.no_trade", reason="DRY_RUN", qty=qty, side=chosen_side, prob=chosen_prob, ev_bps=chosen_ev)
            except Exception:
                pass
            self._set_state(symbol=sym, side=chosen_side, qty=qty, probability=chosen_prob, ev_bps=chosen_ev, last_order=None, last_error=None)
            return

        p_adj = price
        q_adj = qty
        order = dict(
            category=category,
            symbol=sym,
            side=chosen_side.upper(),
            orderType="Limit",
            price=f"{p_adj}",
            qty=f"{q_adj}",
            timeInForce="GTC",
            orderFilter="Order",
            orderLinkId=ensure_link_id(f"AIR-{int(time.time())}"),
        )
        try:
            log("ai.order.place", **order)
        except Exception:
            pass
        r = self.api.place_order(**order)
        self._set_state(symbol=sym, side=chosen_side, qty=q_adj, probability=chosen_prob, ev_bps=chosen_ev, last_order=r, last_error=None)
        try:
            log("ai.order.result", resp=r)
        except Exception:
            pass
