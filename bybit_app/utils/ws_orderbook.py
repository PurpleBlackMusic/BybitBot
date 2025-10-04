
from __future__ import annotations
import time, threading, json
from typing import Dict, List, Tuple, Optional
from .bybit_api import BybitAPI
from .log import log

class LiveOrderbook:
    """Простой агрегатор. Если WebSocket недоступен, используем REST-снэпшот.

    WS часть оформлена как заготовка, чтобы не плодить зависимости. При отсутствии WS — просто не запустится.

    VWAP рассчитываем детерминированно на N уровнях.
    """
    def __init__(self, api: BybitAPI, category: str = "spot"):
        self.api = api
        self.category = category
        self.books: dict[str, dict] = {}  # symbol -> {"ts":ms,"b":[(px,qty),],"a":[]}
        self.lock = threading.Lock()
        self.ws_thread = None
        self._stop = False

    def start_ws(self):
        """Заготовка: пользователь может подключить websockets и запустить подписку orderbook.1/50/200/1000.

        API док: /v5/public/orderbook, частоты: Spot L1 10ms, L50 20ms, L200 200ms, L1000 300ms (2025‑08‑14)."""
        try:
            import websocket  # noqa: F401
        except Exception:
            log("ws.orderbook.disabled", reason="no websocket package")
            return False
        # Оставляем как заглушку: WS-логика зависит от инфраструктуры проекта
        return False

    def snapshot_rest(self, symbol: str, limit: int = 50):
        r = self.api._safe_req("GET", "/v5/market/orderbook", params={"category": self.category, "symbol": symbol, "limit": int(limit)})
        ob = (r.get("result") or {})
        b = [(float(x[0]), float(x[1])) for x in (ob.get("b") or [])]
        a = [(float(x[0]), float(x[1])) for x in (ob.get("a") or [])]
        with self.lock:
            self.books[symbol] = {"ts": int(time.time()*1000), "b": b, "a": a}
        return self.books[symbol]

    def get_book(self, symbol: str, max_age_ms: int = 1000, limit: int = 50):
        with self.lock:
            rec = self.books.get(symbol)
        if not rec or (int(time.time()*1000) - int(rec.get("ts",0)) > max_age_ms):
            rec = self.snapshot_rest(symbol, limit=limit)
        return rec

    @staticmethod
    def vwap(side: str, qty: float, levels: List[Tuple[float, float]]) -> Optional[float]:
        need = float(qty); acc = 0.0; cost = 0.0
        for px, av in levels:
            take = min(need, av)
            cost += take * px
            acc += take
            need -= take
            if need <= 1e-12: break
        if acc <= 0: return None
        return cost / acc

    def vwap_for(self, symbol: str, side: str, qty: float, limit: int = 200) -> dict:
        book = self.get_book(symbol, limit=limit)
        if not book: return {"error": "no book"}
        if side.lower() == "buy":
            # Покупка проталкивает ask
            vwap = self.vwap("buy", qty, [(px, av) for px, av in book["a"][:limit]])
            best = book["a"][0][0] if book["a"] else None
        else:
            vwap = self.vwap("sell", qty, [(px, av) for px, av in book["b"][:limit]])
            best = book["b"][0][0] if book["b"] else None
        return {"best": best, "vwap": vwap, "levels": limit, "ts": book.get("ts")}
