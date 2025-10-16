
from __future__ import annotations

import json
import threading
import time
import ssl
from collections import deque
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .bybit_api import BybitAPI
from .log import log


class LiveOrderbook:
    """Лёгкий агрегатор стакана с поддержкой WebSocket L1."""

    _WS_LEVEL = 1

    def __init__(self, api: BybitAPI, category: str = "spot"):
        self.api = api
        self.category = category
        self.books: dict[str, dict] = {}  # symbol -> {"ts":ms,"b":[(px,qty),],"a":[]}
        self._tops: dict[str, dict[str, float]] = {}
        self._spread_history: dict[str, Deque[Tuple[float, float]]] = {}
        self.lock = threading.Lock()
        self.ws_thread: Optional[threading.Thread] = None
        self._stop = False
        self._ws = None
        self._ws_connected = False
        self._ws_symbols: set[str] = set()
        self._ws_pending: set[str] = set()
        self._ws_lock = threading.Lock()

    # ------------------------------------------------------------------
    # WebSocket handling
    def start_ws(self, symbols: Iterable[str] | None = None) -> bool:
        """Ensure the WS loop is running and subscribed to provided symbols."""

        try:  # pragma: no cover - dependency check
            import websocket  # noqa: F401
        except Exception:
            log("ws.orderbook.disabled", reason="no websocket package")
            return False

        new_symbols = {
            self._normalise_symbol(symbol)
            for symbol in (symbols or [])
            if self._normalise_symbol(symbol)
        }

        if new_symbols:
            with self._ws_lock:
                self._ws_symbols.update(new_symbols)
            self._queue_subscriptions(new_symbols)

        if self.ws_thread and self.ws_thread.is_alive():
            return True

        with self._ws_lock:
            if self.ws_thread and self.ws_thread.is_alive():
                return True
            self._stop = False
            self.ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
            self.ws_thread.start()
        return True

    def stop_ws(self) -> None:
        with self._ws_lock:
            self._stop = True
            ws = self._ws
        try:
            if ws is not None:
                ws.close()  # type: ignore[call-arg]
        except Exception:
            pass

    # Internal helpers --------------------------------------------------
    def _run_ws_loop(self) -> None:
        try:
            import websocket
        except Exception as exc:  # pragma: no cover - defensive guard
            log("ws.orderbook.init_error", err=str(exc))
            return

        backoff = 1.0
        max_backoff = 30.0

        while True:
            with self._ws_lock:
                if self._stop:
                    break

            url = self._resolve_ws_url()

            try:
                ws_app = websocket.WebSocketApp(
                    url,
                    on_open=self._handle_ws_open,
                    on_message=self._handle_ws_message,
                    on_error=self._handle_ws_error,
                    on_close=self._handle_ws_close,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                log("ws.orderbook.app_error", err=str(exc))
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)
                continue

            with self._ws_lock:
                self._ws = ws_app

            try:
                ws_app.run_forever(
                    sslopt=self._ssl_opts(),
                    ping_interval=20,
                    ping_timeout=10,
                )
            except Exception as exc:  # pragma: no cover - network/ssl guard
                log("ws.orderbook.run_error", err=str(exc))
            finally:
                with self._ws_lock:
                    self._ws = None
                    self._ws_connected = False

            with self._ws_lock:
                if self._stop:
                    break

            sleep_for = min(backoff, max_backoff)
            log("ws.orderbook.retry", sleep=sleep_for)
            time.sleep(sleep_for)
            backoff = min(backoff * 2, max_backoff)

    def _handle_ws_open(self, ws) -> None:
        with self._ws_lock:
            self._ws_connected = True
            pending = set(self._ws_symbols)
            pending.update(self._ws_pending)
            self._ws_pending = set()
        if pending:
            self._send_subscription(ws, pending)

    def _handle_ws_close(self, ws, code, msg) -> None:  # pragma: no cover - network callback
        with self._ws_lock:
            self._ws_connected = False
        log("ws.orderbook.close", code=code, msg=msg)

    def _handle_ws_error(self, ws, error) -> None:  # pragma: no cover - network callback
        log("ws.orderbook.error", err=str(error))

    def _handle_ws_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        op = payload.get("op")
        if op == "ping":
            try:
                ws.send(json.dumps({"op": "pong", "req_id": payload.get("req_id")}))
            except Exception:
                pass
            return
        if op in {"pong", "subscribe", "unsubscribe"}:
            return

        topic = payload.get("topic")
        if not isinstance(topic, str) or "orderbook" not in topic:
            return

        symbol = self._extract_symbol(topic)
        if not symbol:
            return

        entry = self._extract_top_entry(payload)
        if not entry:
            return

        self._update_top(symbol, entry, source="ws")

    def _queue_subscriptions(self, symbols: Iterable[str]) -> None:
        cleaned = {self._normalise_symbol(sym) for sym in symbols if self._normalise_symbol(sym)}
        if not cleaned:
            return
        with self._ws_lock:
            self._ws_pending.update(cleaned)
        self._flush_pending_subscriptions()

    def _flush_pending_subscriptions(self) -> None:
        with self._ws_lock:
            if not self._ws_pending or not self._ws_connected or self._ws is None:
                return
            symbols = sorted(self._ws_pending)
            self._ws_pending = set()
            ws = self._ws
        self._send_subscription(ws, symbols)

    def _send_subscription(self, ws, symbols: Iterable[str]) -> None:
        topics = [self._topic_for_symbol(sym) for sym in symbols if sym]
        if not topics:
            return
        try:
            ws.send(json.dumps({"op": "subscribe", "args": topics}))
        except Exception as exc:  # pragma: no cover - network callback
            log("ws.orderbook.subscribe_error", err=str(exc))
            with self._ws_lock:
                self._ws_pending.update({self._extract_symbol(topic) for topic in topics})

    def _resolve_ws_url(self) -> str:
        testnet = bool(getattr(getattr(self.api, "creds", None), "testnet", False))
        if self.category.lower() == "spot":
            return (
                "wss://stream-testnet.bybit.com/v5/public/spot"
                if testnet
                else "wss://stream.bybit.com/v5/public/spot"
            )
        # Default to spot public stream if category unsupported
        return (
            "wss://stream-testnet.bybit.com/v5/public/spot"
            if testnet
            else "wss://stream.bybit.com/v5/public/spot"
        )

    def _ssl_opts(self) -> dict:
        verify = bool(getattr(self.api, "verify_ssl", True))
        cert_reqs = ssl.CERT_REQUIRED if verify else ssl.CERT_NONE
        return {"cert_reqs": cert_reqs}

    @staticmethod
    def _normalise_symbol(symbol: str | None) -> str:
        if not symbol:
            return ""
        try:
            text = str(symbol).strip().upper()
        except Exception:
            return ""
        return text

    @classmethod
    def _topic_for_symbol(cls, symbol: str) -> str:
        return f"orderbook.{cls._WS_LEVEL}.{symbol}"

    @staticmethod
    def _extract_symbol(topic: str) -> str:
        parts = topic.split(".")
        if len(parts) >= 3:
            return parts[-1].strip().upper()
        return topic.strip().upper()

    def _extract_top_entry(self, payload: Dict[str, object]) -> Optional[Dict[str, float]]:
        data = payload.get("data")
        entry: Optional[dict] = None
        if isinstance(data, Sequence):
            for item in data:
                if isinstance(item, dict):
                    entry = item
                    break
        elif isinstance(data, dict):
            entry = data

        if not isinstance(entry, dict):
            return None

        ts = self._coerce_int(entry.get("ts"))
        if ts is None:
            ts = self._coerce_int(payload.get("ts"))
        if ts is None:
            ts = int(time.time() * 1000)

        best_bid, best_bid_qty = self._extract_side(entry, ["bid1Price", "bidPrice", "bestBidPrice"], ["bid1Qty", "bidQty", "bestBidQty"], "b")
        best_ask, best_ask_qty = self._extract_side(entry, ["ask1Price", "askPrice", "bestAskPrice"], ["ask1Qty", "askQty", "bestAskQty"], "a")

        result: Dict[str, float] = {"ts": float(ts)}
        if best_bid is not None:
            result["best_bid"] = best_bid
        if best_bid_qty is not None:
            result["best_bid_qty"] = best_bid_qty
        if best_ask is not None:
            result["best_ask"] = best_ask
        if best_ask_qty is not None:
            result["best_ask_qty"] = best_ask_qty

        return result if len(result) > 1 else None

    def _extract_side(
        self,
        entry: dict,
        price_keys: Sequence[str],
        qty_keys: Sequence[str],
        ladder_key: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        price: Optional[float] = None
        qty: Optional[float] = None

        for key in price_keys:
            price = self._coerce_float(entry.get(key))
            if price is not None and price > 0:
                break
        for key in qty_keys:
            qty = self._coerce_float(entry.get(key))
            if qty is not None and qty > 0:
                break

        ladder = entry.get(ladder_key)
        if (price is None or qty is None) and isinstance(ladder, Sequence):
            for level in ladder:
                if not isinstance(level, Sequence) or len(level) < 2:
                    continue
                ladder_price = self._coerce_float(level[0])
                ladder_qty = self._coerce_float(level[1])
                if ladder_price is None or ladder_qty is None:
                    continue
                if ladder_price > 0 and ladder_qty > 0:
                    price = ladder_price if price is None else price
                    qty = ladder_qty if qty is None else qty
                    break

        if price is not None and price <= 0:
            price = None
        if qty is not None and qty <= 0:
            qty = None

        return price, qty

    def _update_top(self, symbol: str, entry: Dict[str, float], *, source: str) -> None:
        payload: Dict[str, float] = {"ts": float(entry.get("ts", int(time.time() * 1000)))}
        for key in ("best_bid", "best_bid_qty", "best_ask", "best_ask_qty"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and value > 0:
                payload[key] = float(value)
        payload["source"] = source

        with self.lock:
            existing = self._tops.get(symbol)
            if existing and existing.get("ts", 0) > payload.get("ts", 0):
                return
            self._tops[symbol] = payload
            self._record_spread_history_locked(symbol, payload)

    _SPREAD_HISTORY_MAX_MS = 60_000.0

    def _record_spread_history_locked(self, symbol: str, payload: Dict[str, float]) -> None:
        ts = float(payload.get("ts", time.time() * 1000))
        ask = payload.get("best_ask")
        bid = payload.get("best_bid")
        if not ask or not bid or ask <= 0:
            return
        spread_bps = max(((ask - bid) / ask) * 10_000.0, 0.0)
        history = self._spread_history.setdefault(symbol, deque())
        history.append((ts, spread_bps))
        cutoff = ts - self._SPREAD_HISTORY_MAX_MS
        while history and history[0][0] < cutoff:
            history.popleft()

    def spread_window_stats(self, symbol: str, window_sec: float) -> Optional[Dict[str, float]]:
        if window_sec <= 0:
            return None

        cleaned_symbol = self._normalise_symbol(symbol)
        if not cleaned_symbol:
            return None

        now_ms = time.time() * 1000.0
        cutoff = now_ms - max(window_sec, 0.0) * 1000.0
        cutoff_history = now_ms - self._SPREAD_HISTORY_MAX_MS

        with self.lock:
            history = self._spread_history.get(cleaned_symbol)
            if not history:
                return None
            while history and history[0][0] < cutoff_history:
                history.popleft()
            recent_values: list[float] = [spread for ts, spread in history if ts >= cutoff]
            latest_ts = history[-1][0] if history else 0.0
            latest_bps = history[-1][1] if history else 0.0

        if not recent_values:
            return {
                "window_sec": float(window_sec),
                "observations": 0,
                "max_bps": 0.0,
                "min_bps": 0.0,
                "avg_bps": 0.0,
                "latest_bps": float(latest_bps),
                "age_ms": max(now_ms - latest_ts, 0.0),
            }

        observations = len(recent_values)
        max_bps = max(recent_values)
        min_bps = min(recent_values)
        avg_bps = sum(recent_values) / len(recent_values)

        return {
            "window_sec": float(window_sec),
            "observations": observations,
            "max_bps": float(max_bps),
            "min_bps": float(min_bps),
            "avg_bps": float(avg_bps),
            "latest_bps": float(latest_bps),
            "age_ms": max(now_ms - latest_ts, 0.0),
        }

    def record_top(self, symbol: str, top: Mapping[str, float], *, source: str = "external") -> None:
        cleaned_symbol = self._normalise_symbol(symbol)
        if not cleaned_symbol:
            return

        entry: Dict[str, float] = {"ts": float(top.get("ts", time.time() * 1000.0))}
        for key in ("best_bid", "best_bid_qty", "best_ask", "best_ask_qty"):
            value = top.get(key)
            if isinstance(value, (int, float)) and value > 0:
                entry[key] = float(value)

        if len(entry) > 1:
            self._update_top(cleaned_symbol, entry, source=source)

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return result

    @staticmethod
    def _coerce_int(value: object) -> Optional[int]:
        try:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip():
                return int(float(value))
        except (TypeError, ValueError):
            return None
        return None

    # ------------------------------------------------------------------
    # REST snapshot utilities
    def snapshot_rest(self, symbol: str, limit: int = 50):
        cleaned_symbol = self._normalise_symbol(symbol)
        r = self.api._safe_req(
            "GET",
            "/v5/market/orderbook",
            params={"category": self.category, "symbol": cleaned_symbol, "limit": int(limit)},
        )
        ob = (r.get("result") or {})
        b = [(float(x[0]), float(x[1])) for x in (ob.get("b") or [])]
        a = [(float(x[0]), float(x[1])) for x in (ob.get("a") or [])]
        ts = int(time.time() * 1000)
        with self.lock:
            self.books[cleaned_symbol] = {"ts": ts, "b": b, "a": a}
        top_entry: Dict[str, float] = {"ts": float(ts)}
        if b:
            top_entry["best_bid"] = float(b[0][0])
            top_entry["best_bid_qty"] = float(b[0][1])
        if a:
            top_entry["best_ask"] = float(a[0][0])
            top_entry["best_ask_qty"] = float(a[0][1])
        if len(top_entry) > 1:
            self._update_top(cleaned_symbol, top_entry, source="rest")
        return self.books[cleaned_symbol]

    def get_book(self, symbol: str, max_age_ms: int = 1000, limit: int = 50):
        cleaned_symbol = self._normalise_symbol(symbol)
        with self.lock:
            rec = self.books.get(cleaned_symbol)
        if not rec or (int(time.time() * 1000) - int(rec.get("ts", 0)) > max_age_ms):
            rec = self.snapshot_rest(cleaned_symbol, limit=limit)
        return rec

    def get_top_ws(self, symbol: str, max_age_ms: int = 1200) -> Optional[Dict[str, float]]:
        cleaned_symbol = self._normalise_symbol(symbol)
        if not cleaned_symbol:
            return None
        with self.lock:
            entry = self._tops.get(cleaned_symbol)
        if not entry:
            return None
        ts = self._coerce_float(entry.get("ts"))
        if ts is None:
            return None
        age_ms = int(time.time() * 1000 - ts)
        if max_age_ms > 0 and age_ms > max_age_ms:
            return None

        snapshot: Dict[str, float] = {"ts": ts, "source": entry.get("source", "ws")}
        best_bid = entry.get("best_bid")
        best_bid_qty = entry.get("best_bid_qty")
        best_ask = entry.get("best_ask")
        best_ask_qty = entry.get("best_ask_qty")

        if isinstance(best_bid, (int, float)):
            snapshot["best_bid"] = float(best_bid)
        if isinstance(best_bid_qty, (int, float)):
            snapshot["best_bid_qty"] = float(best_bid_qty)
        if isinstance(best_ask, (int, float)):
            snapshot["best_ask"] = float(best_ask)
        if isinstance(best_ask_qty, (int, float)):
            snapshot["best_ask_qty"] = float(best_ask_qty)

        ask_val = snapshot.get("best_ask")
        bid_val = snapshot.get("best_bid")
        if ask_val and bid_val and ask_val > 0:
            snapshot["spread_bps"] = max(((ask_val - bid_val) / ask_val) * 10_000.0, 0.0)

        return snapshot if len(snapshot) > 1 else None

    # ------------------------------------------------------------------
    # VWAP calculation
    @staticmethod
    def vwap(side: str, qty: float, levels: List[Tuple[float, float]]) -> Optional[float]:
        need = float(qty)
        acc = 0.0
        cost = 0.0
        for px, av in levels:
            take = min(need, av)
            cost += take * px
            acc += take
            need -= take
            if need <= 1e-12:
                break
        if acc <= 0:
            return None
        return cost / acc

    def vwap_for(self, symbol: str, side: str, qty: float, limit: int = 200) -> dict:
        book = self.get_book(symbol, limit=limit)
        if not book:
            return {"error": "no book"}
        if side.lower() == "buy":
            # Покупка проталкивает ask
            vwap = self.vwap("buy", qty, [(px, av) for px, av in book["a"][:limit]])
            best = book["a"][0][0] if book["a"] else None
        else:
            vwap = self.vwap("sell", qty, [(px, av) for px, av in book["b"][:limit]])
            best = book["b"][0][0] if book["b"] else None
        return {"best": best, "vwap": vwap, "levels": limit, "ts": book.get("ts")}
