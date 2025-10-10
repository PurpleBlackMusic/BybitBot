
from __future__ import annotations
import json, time, threading
from typing import Dict, List, Tuple
from .log import log
from .envs import get_settings

class WSOrderbookV5:
    """V5 Public WS orderbook aggregator for Spot (levels 1/50/200/1000).
    Processes snapshot + delta per official rules. Fallback to REST must be handled by caller.
    """
    def __init__(self, url: str = "wss://stream.bybit.com/v5/public/spot", levels: int = 200):
        self.url = url
        self.levels = levels
        self._book: Dict[str, Dict[str, List[Tuple[float,float]]]] = {}  # {sym: {'b':[(px,qty)], 'a':[]}, 'ts': ms}
        self._lock = threading.Lock()
        self._stop = False
        self._ws = None
        self._thread = None

    def start(self, symbols: list[str]):
        try:
            import websocket
        except Exception as e:
            log("ws.orderbook.disabled", reason="no websocket-client", err=str(e))
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop = False
        topics = [f"orderbook.{self.levels}.{s}" for s in symbols]
        def run():
            import websocket, ssl
            try:
                settings = get_settings()
            except Exception:
                settings = None
            verify_ssl = True
            if settings is not None:
                verify_ssl = bool(getattr(settings, "verify_ssl", True))
            cert_reqs = ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE
            sslopt = {"cert_reqs": cert_reqs}
            attempt = 0
            backoff = 1.0
            max_backoff = 60.0
            while not self._stop:
                attempt += 1
                try:
                    ws = websocket.WebSocketApp(
                        self.url,
                        on_open=lambda w: self._on_open(w, topics),
                        on_message=self._on_msg,
                        on_error=lambda w, e: log("ws.orderbook.error", err=str(e)),
                        on_close=lambda w, c, m: log("ws.orderbook.close", code=c, msg=m),
                    )
                    self._ws = ws
                    ws.run_forever(sslopt=sslopt)
                except Exception as e:
                    log("ws.orderbook.run_err", err=str(e))
                finally:
                    self._ws = None
                if self._stop:
                    break
                sleep_for = min(backoff, max_backoff)
                log("ws.orderbook.retry", attempt=attempt, sleep=sleep_for)
                time.sleep(sleep_for)
                backoff = min(backoff * 2, max_backoff)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop = True
        try:
            if self._ws: self._ws.close()
        except Exception:
            pass

    def _on_open(self, ws, topics):
        sub = {"op": "subscribe", "args": topics}
        try:
            ws.send(json.dumps(sub))
        except Exception as e:
            log("ws.orderbook.send_err", err=str(e))

    def _apply_delta(self, side: str, cur: List[Tuple[float,float]], delta: List[List[str]]):
        # delta rows: [price, qty] as strings; qty=0 means remove level
        # keep book sorted: bids desc, asks asc
        book = {px: qty for px, qty in cur}
        for px_s, qty_s in delta:
            px = float(px_s); qty = float(qty_s)
            if qty <= 0:
                if px in book: del book[px]
            else:
                book[px] = qty
        # rebuild sorted
        if side == 'b':
            arr = sorted(book.items(), key=lambda x: x[0], reverse=True)
        else:
            arr = sorted(book.items(), key=lambda x: x[0])
        return arr[: self.levels]

    def _on_msg(self, ws, msg):
        try:
            j = json.loads(msg)
        except Exception:
            return
        if j.get("op") == "subscribe":
            return
        topic = j.get("topic") or ""
        if not topic.startswith("orderbook."):
            return
        data = j.get("data") or {}
        ts = int(data.get("ts", int(time.time()*1000)))
        with self._lock:
            sym = data.get("s") or topic.split('.')[-1]
            entry = self._book.get(sym) or {"b": [], "a": [], "ts": ts}
            if data.get("type") == "snapshot":
                # full snapshot arrays 'b' and 'a'
                entry["b"] = [(float(x[0]), float(x[1])) for x in data.get("b", [])][: self.levels]
                entry["a"] = [(float(x[0]), float(x[1])) for x in data.get("a", [])][: self.levels]
                entry["ts"] = ts
            elif data.get("type") == "delta":
                entry["b"] = self._apply_delta('b', entry.get("b", []), data.get("b", []))
                entry["a"] = self._apply_delta('a', entry.get("a", []), data.get("a", []))
                entry["ts"] = ts
            self._book[sym] = entry

    def get(self, symbol: str):
        with self._lock:
            return self._book.get(symbol)
