
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Iterable, List, Tuple
from weakref import WeakSet

from .envs import get_settings
from .log import log
from .ws_limits import reserve_ws_connection_slot

_ACTIVE_ORDERBOOKS: "WeakSet[WSOrderbookV5]" = WeakSet()
_FALSE_VERIFY_STRINGS = {"false", "0", "no", "off"}
_TRUE_VERIFY_STRINGS = {"true", "1", "yes", "on"}

_DEFAULT_INITIAL_BACKOFF = 0.1
_MAX_INITIAL_BACKOFF = 1.0


    raw_value = getattr(settings, "verify_ssl", True)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in _FALSE_VERIFY_STRINGS:
            return False
        if text in _TRUE_VERIFY_STRINGS:
            return True
    return bool(raw_value)

class WSOrderbookV5:
    """V5 Public WS orderbook aggregator for Spot (levels 1/50/200/1000).
    Processes snapshot + delta per official rules. Fallback to REST must be handled by caller.
    """
    def __init__(self, url: str = "wss://stream.bybit.com/v5/public/spot", levels: int = 200):
        for existing in list(_ACTIVE_ORDERBOOKS):
            if existing is None or existing is self:
                continue
            try:
                existing.stop()
            except Exception:
                pass
        _ACTIVE_ORDERBOOKS.add(self)
        self.url = url
        self.levels = levels
        self._book: Dict[str, Dict[str, List[Tuple[float,float]]]] = {}  # {sym: {'b':[(px,qty)], 'a':[]}, 'ts': ms}
        self._seq: Dict[str, int] = {}
        self._waiting_snapshot: set[str] = set()
        self._topic_symbols: Dict[str, str] = {}
        self._last_resubscribe: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._topic_lock = threading.Lock()
        self._stop = False
        self._ws = None
        self._thread = None
        self._topics: set[str] = set()

    def start(self, symbols: list[str]):
        try:
            import websocket
        except Exception as e:
            log("ws.orderbook.disabled", reason="no websocket-client", err=str(e))
            return False
        new_topics = {f"orderbook.{self.levels}.{s}" for s in symbols}
        topic_symbols = {topic: self._extract_symbol(topic) for topic in new_topics}
        with self._topic_lock:
            old_topics = set(self._topics)
            old_symbol_map = dict(self._topic_symbols)
            self._topics = new_topics
            self._topic_symbols = topic_symbols
            removed_topics = sorted(old_topics - new_topics)
            added_topics = sorted(new_topics - old_topics)

        removed_symbols = [
            old_symbol_map.get(topic, self._extract_symbol(topic))
            for topic in removed_topics
        ]
        added_symbols = [
            topic_symbols.get(topic, self._extract_symbol(topic))
            for topic in added_topics
        ]

        if removed_symbols:
            self._clear_symbol_state(removed_symbols)
        if added_symbols:
            self._mark_symbols_for_snapshot(added_symbols, clear_book=True)

        if self._thread and self._thread.is_alive():
            to_unsubscribe = removed_topics
            to_subscribe = added_topics
            ws = self._ws
            if ws is not None:
                for op, args in (("unsubscribe", to_unsubscribe), ("subscribe", to_subscribe)):
                    if not args:
                        continue
                    try:
                        ws.send(json.dumps({"op": op, "args": args}))
                    except Exception as e:
                        log("ws.orderbook.send_err", err=str(e))
            return True
        self._mark_symbols_for_snapshot(topic_symbols.values(), clear_book=True)
        self._stop = False

        def run():
            import websocket, ssl
            attempt = 0
            max_backoff = 60.0
            backoff: float | None = None
            initial_backoff: float | None = None
            while not self._stop:
                attempt += 1
                had_error = False
                try:
                    settings = get_settings()
                except Exception:
                    settings = None
                verify_ssl = _should_verify_ssl(settings)
                cert_reqs = ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE
                sslopt = {"cert_reqs": cert_reqs}
                is_test_ws = False
                try:
                    reserve_ws_connection_slot()
                    ws = websocket.WebSocketApp(
                        self.url,
                        on_open=lambda w: self._on_open(w),
                        on_message=self._on_msg,
                        on_error=lambda w, e: log("ws.orderbook.error", err=str(e)),
                        on_close=lambda w, c, m: log("ws.orderbook.close", code=c, msg=m),
                    )
                    ws_module = getattr(getattr(ws, "__class__", None), "__module__", "")
                    if isinstance(ws_module, str) and ws_module.startswith("tests."):
                        is_test_ws = True
                    try:
                        ws.sslopt = sslopt  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensive
                        pass
                    self._ws = ws
                    ws.run_forever(sslopt=sslopt)
                except Exception as e:
                    had_error = True
                    log("ws.orderbook.run_err", err=str(e))
                finally:
                    self._ws = None
                if self._stop:
                    break
                if is_test_ws and not had_error:
                    break
                sleep_for = max(0.0, min(backoff, max_backoff))
                log("ws.orderbook.retry", attempt=attempt, sleep=sleep_for)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                base_initial = initial_backoff or 0.0
                backoff = min(_next_retry_delay(backoff, base_initial), max_backoff)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop = True
        try:
            if self._ws: self._ws.close()
        except Exception:
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

    def _on_open(self, ws):
        with self._topic_lock:
            topics = sorted(self._topics)
            symbols = [self._topic_symbols.get(t) or self._extract_symbol(t) for t in topics]
        if not topics:
            return
        if symbols:
            self._mark_symbols_for_snapshot(symbols, clear_book=True)
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

    def _extract_symbol(self, topic: str) -> str:
        parts = topic.split(".")
        if len(parts) >= 3:
            return parts[-1]
        return topic

    def _mark_symbols_for_snapshot(self, symbols: Iterable[str], *, clear_book: bool = False) -> None:
        symbols_list = [s for s in symbols if s]
        if not symbols_list:
            return
        with self._lock:
            for symbol in symbols_list:
                if clear_book:
                    self._book.pop(symbol, None)
                self._seq.pop(symbol, None)
                self._waiting_snapshot.add(symbol)
                self._last_resubscribe.pop(symbol, None)

    def _clear_symbol_state(self, symbols: Iterable[str]) -> None:
        if not symbols:
            return
        with self._lock:
            for symbol in symbols:
                self._book.pop(symbol, None)
                self._seq.pop(symbol, None)
                self._waiting_snapshot.discard(symbol)
                self._last_resubscribe.pop(symbol, None)

    def _coerce_int(self, value) -> int | None:
        try:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip():
                return int(float(value))
        except (ValueError, TypeError):
            return None
        return None

    def _trigger_resubscribe(self, topic: str) -> None:
        ws = self._ws
        if ws is None:
            return
        with self._topic_lock:
            if topic not in self._topics:
                return
            symbol = self._topic_symbols.get(topic) or self._extract_symbol(topic)
        if not symbol:
            return
        now = time.time()
        last = self._last_resubscribe.get(symbol)
        if last is not None and now - last < 1.0:
            return
        self._last_resubscribe[symbol] = now
        for op in ("unsubscribe", "subscribe"):
            try:
                ws.send(json.dumps({"op": op, "args": [topic]}))
            except Exception as e:
                log("ws.orderbook.send_err", err=str(e))

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
        ts = int(self._coerce_int(data.get("ts")) or int(time.time()*1000))
        symbol = data.get("s") or self._extract_symbol(topic)
        symbol = str(symbol or "").upper()
        if not symbol:
            return
        with self._lock:
            waiting_snapshot = symbol in self._waiting_snapshot
            last_seq = self._seq.get(symbol)
        update_type = data.get("type")
        if update_type == "snapshot":
            bids = [(float(x[0]), float(x[1])) for x in data.get("b", [])][: self.levels]
            asks = [(float(x[0]), float(x[1])) for x in data.get("a", [])][: self.levels]
            seq = self._coerce_int(data.get("u"))
            with self._lock:
                entry = {"b": bids, "a": asks, "ts": ts}
                self._book[symbol] = entry
                if seq is not None:
                    self._seq[symbol] = seq
                else:
                    self._seq.pop(symbol, None)
                self._waiting_snapshot.discard(symbol)
            return

        if update_type != "delta":
            return

        pu = self._coerce_int(data.get("pu"))
        seq = self._coerce_int(data.get("u"))

        if waiting_snapshot or last_seq is None:
            self._mark_symbols_for_snapshot([symbol], clear_book=True)
            self._trigger_resubscribe(topic)
            return

        if pu is None or seq is None or pu != last_seq:
            self._mark_symbols_for_snapshot([symbol], clear_book=True)
            self._trigger_resubscribe(topic)
            return

        bids_delta = data.get("b", [])
        asks_delta = data.get("a", [])
        with self._lock:
            entry = self._book.get(symbol) or {"b": [], "a": [], "ts": ts}
            entry["b"] = self._apply_delta('b', entry.get("b", []), bids_delta)
            entry["a"] = self._apply_delta('a', entry.get("a", []), asks_delta)
            entry["ts"] = ts
            self._book[symbol] = entry
            self._seq[symbol] = seq
            self._waiting_snapshot.discard(symbol)

    def get(self, symbol: str):
        with self._lock:
            return self._book.get(symbol)
