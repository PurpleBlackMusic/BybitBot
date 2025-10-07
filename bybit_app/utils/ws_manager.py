from __future__ import annotations

import json
import threading
import time
import ssl
import random
from typing import Iterable, Optional

import websocket  # websocket-client

from copy import deepcopy

from .envs import get_settings
from .paths import DATA_DIR
from .store import JLStore
from .log import log
from .ws_private_v5 import WSPrivateV5


class WSManager:
    """Минимальный, но стабильный менеджер WS с авто‑пингом и переподключением."""

    def __init__(self):
        self.s = get_settings()

        # public
        self._pub_running: bool = False
        self._pub_thread: Optional[threading.Thread] = None
        self._pub_ws: Optional[websocket.WebSocketApp] = None
        self._pub_subs: tuple[str, ...] = tuple()
        self._pub_ping_thread: Optional[threading.Thread] = None

        # private
        self._priv: Optional[WSPrivateV5] = None
        self._priv_url: Optional[str] = None

        # stores
        self.pub_store = JLStore(DATA_DIR / "ws" / "public.jsonl", max_lines=2000)
        self.priv_store = JLStore(DATA_DIR / "ws" / "private.jsonl", max_lines=2000)

        # heartbeats
        self.last_beat: float = 0.0
        self.last_public_beat: float = 0.0
        self.last_private_beat: float = 0.0

        # cached payloads for UI/diagnostics
        self._state_lock = threading.Lock()
        self._last_order_update: Optional[dict] = None
        self._last_execution: Optional[dict] = None

    # ----------------------- Public -----------------------
    def _refresh_settings(self) -> None:
        """Reload settings so WS endpoints respect latest configuration."""
        try:
            self.s = get_settings(force_reload=True)
        except Exception as e:  # pragma: no cover - defensive, rare
            log("ws.settings.refresh.error", err=str(e))

    def _handle_private_beat(self) -> None:
        now = time.time()
        self.last_beat = now
        self.last_private_beat = now

    def _public_url(self) -> str:
        self._refresh_settings()
        return (
            "wss://stream-testnet.bybit.com/v5/public/spot"
            if self.s.testnet
            else "wss://stream.bybit.com/v5/public/spot"
        )

    def start_public(self, subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        subs = tuple(subs)
        self._pub_subs = subs

        # Если уже запущен — просто убедимся в подписке (если соединение активно)
        if self._pub_running and self._pub_ws is not None:
            if self._is_socket_connected(self._pub_ws):
                try:
                    for t in subs:
                        self._pub_ws.send(json.dumps({"op": "subscribe", "args": [t]}))
                except Exception as e:
                    log("ws.public.resub.error", err=str(e))
            return True

        url = self._public_url()
        self._pub_running = True

        def on_open(ws):
            log("ws.public.open", url=url)
            # стартуем лёгкий пинг‑луп (каждые ~20с)
            def _ping_loop():
                try:
                    while self._pub_running and self._pub_ws is ws:
                        try:
                            req = {"op": "ping", "req_id": str(int(time.time() * 1000))}
                            ws.send(json.dumps(req))
                        except Exception as e:
                            log("ws.public.ping.error", err=str(e))
                            break
                        time.sleep(20)
                except Exception as e:
                    log("ws.public.ping.exit", err=str(e))

            try:
                if not (self._pub_ping_thread and self._pub_ping_thread.is_alive()):
                    self._pub_ping_thread = threading.Thread(target=_ping_loop, daemon=True)
                    self._pub_ping_thread.start()
            except Exception as e:
                log("ws.public.ping.start.error", err=str(e))

            # первичная подписка
            try:
                current_subs = tuple(self._pub_subs)
                if current_subs:
                    req = {"op": "subscribe", "args": list(current_subs)}
                    ws.send(json.dumps(req))
            except Exception as e:
                log("ws.public.sub.error", err=str(e))

        def on_message(ws, message: str):
            now = time.time()
            self.last_beat = now
            self.last_public_beat = now
            try:
                obj = json.loads(message)
            except Exception:
                obj = {"raw": message}
            self.pub_store.append(obj)
            if isinstance(obj, dict) and obj.get("op") == "pong":
                log("ws.public.pong")

        def on_error(ws, error):
            log("ws.public.error", err=str(error))

        def on_close(ws, code, msg):
            log("ws.public.close", code=code, msg=msg)

        def run():
            backoff = 1.0
            while self._pub_running:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                self._pub_ws = ws
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                if not self._pub_running:
                    break
                # экспоненциальный backoff (cap 60s) + джиттер
                sleep_for = min(backoff, 60.0) + random.uniform(0, 0.5)
                log("ws.public.reconnect.wait", seconds=round(sleep_for, 2))
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 60.0)

                # попытка восстановить подписки при следующем on_open

        self._pub_thread = threading.Thread(target=run, daemon=True)
        self._pub_thread.start()
        return True

    def stop_public(self):
        self._pub_running = False
        try:
            if self._pub_ws:
                self._pub_ws.close()
        except Exception:
            pass
        self._pub_ws = None

    # ----------------------- Private ----------------------
    def _private_url(self) -> str:
        self._refresh_settings()
        return (
            "wss://stream-testnet.bybit.com/v5/private"
            if self.s.testnet
            else "wss://stream.bybit.com/v5/private"
        )

    def start_private(self) -> bool:
        try:
            url = self._private_url()

            if self._priv is not None and self._priv_url != url:
                try:
                    self._priv.stop()
                except Exception as e:  # pragma: no cover - defensive, rare
                    log("ws.private.stop.error", err=str(e))
                finally:
                    self._priv = None

            if self._priv is None:
                def _on_private_msg(message):
                    self.priv_store.append(message)
                    self._handle_private_beat()
                    self._process_private_payload(message)

                self._priv = WSPrivateV5(
                    url=url,
                    on_msg=_on_private_msg,
                )
                self._priv_url = url
            if getattr(self._priv, "is_running", None) and self._priv.is_running():
                return True
            started = self._priv.start()
            if not started:
                self._priv = None
                self._priv_url = None
            return bool(started)
        except Exception as e:
            log("ws.private.start.error", err=str(e))
            self._priv = None
            self._priv_url = None
            return False

    def _is_socket_connected(self, ws: Optional[websocket.WebSocketApp]) -> bool:
        """Return True if the given websocket has an active socket."""

        if ws is None:
            return False

        sock = getattr(ws, "sock", None)
        if sock is None:
            return False

        return bool(getattr(sock, "connected", False))

    def start(self, subs: Iterable[str] | None = None, include_private: bool = True) -> bool:
        """Start public and (optionally) private WebSocket channels."""
        subs_to_use: Iterable[str]
        if subs is None:
            subs_to_use = self._pub_subs or ("tickers.BTCUSDT",)
        else:
            subs_to_use = subs

        ok_public = self.start_public(subs_to_use)
        ok_private = True
        if include_private:
            ok_private = self.start_private()
        return bool(ok_public and ok_private)

    def stop_private(self):
        try:
            if self._priv:
                self._priv.stop()
        except Exception:
            pass
        finally:
            self._priv = None
            self._priv_url = None

    # ----------------------- Utils ------------------------
    def stop_all(self):
        self.stop_public()
        self.stop_private()

    def status(self):
        last_beat = self.last_beat if self.last_beat else None
        now = time.time()
        pub_last = self.last_public_beat or 0.0
        priv_last = self.last_private_beat or 0.0

        pub_age = max(0.0, now - pub_last) if pub_last else None
        priv_age = max(0.0, now - priv_last) if priv_last else None

        private_ws = getattr(self._priv, "_ws", None)

        public_ws = self._pub_ws
        public_socket_connected = False
        if public_ws is not None:
            sock = getattr(public_ws, "sock", None)
            if sock is not None:
                public_socket_connected = bool(getattr(sock, "connected", False))

        public_thread_alive = bool(
            self._pub_thread and getattr(self._pub_thread, "is_alive", lambda: False)()
        )

        public_running = bool(
            self._pub_running
            and (
                (public_thread_alive and public_ws)
                or public_socket_connected
            )
        )
        if not public_running and pub_age is not None and pub_age < 60.0:
            public_running = True

        priv = self._priv
        private_running = False
        if priv is not None:
            is_running = getattr(priv, "is_running", None)
            if callable(is_running):
                try:
                    private_running = bool(is_running())
                except Exception:  # pragma: no cover - defensive
                    private_running = False
            if not private_running:
                priv_thread = getattr(priv, "_thread", None)
                if priv_thread is not None:
                    is_alive = getattr(priv_thread, "is_alive", None)
                    if callable(is_alive):
                        try:
                            private_running = bool(is_alive())
                        except Exception:  # pragma: no cover - defensive
                            private_running = False
                if not private_running and private_ws is not None:
                    private_running = True
        if not private_running and priv_age is not None and priv_age < 90.0:
            private_running = True

        return {
            "public": {
                "running": public_running,
                "subscriptions": list(self._pub_subs),
                "last_beat": self.last_public_beat or None,
                "age_seconds": pub_age,
            },
            "private": {
                "running": private_running,
                "connected": bool(private_ws),
                "last_beat": self.last_private_beat or None,
                "age_seconds": priv_age,
            },
        }

    def _process_private_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return

        topic = str(payload.get("topic") or payload.get("topicName") or "").lower()
        if not topic:
            return

        data = payload.get("data")
        if isinstance(data, dict):
            rows = [data]
        elif isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        else:
            rows = []

        if not rows:
            return

        if "execution" in topic:
            try:
                from .pnl import add_execution
            except Exception as exc:  # pragma: no cover - import errors rare
                log("ws.private.execution.import_error", err=str(exc))
                return

            for row in rows:
                try:
                    add_execution(row)
                except Exception as exc:  # pragma: no cover - ledger write failures
                    log("ws.private.execution.persist.error", err=str(exc))
                else:
                    self._record_execution(row)
        elif "order" in topic:
            for row in rows:
                self._record_order_update(row)

    def _record_order_update(self, row: dict) -> None:
        if not isinstance(row, dict):
            return

        update = {
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "status": row.get("orderStatus") or row.get("status"),
            "orderLinkId": row.get("orderLinkId") or row.get("orderId"),
            "cancelType": row.get("cancelType") or row.get("cancelTypeV2"),
            "rejectReason": row.get("rejectReason")
            or row.get("triggerRejectReason")
            or row.get("rejectReasonV2"),
            "updatedTime": row.get("updatedTime")
            or row.get("updateTime")
            or row.get("transactTime"),
            "category": row.get("category"),
        }

        for key in ("orderLinkId", "cancelType", "rejectReason", "updatedTime"):
            value = update.get(key)
            if value is not None:
                update[key] = str(value)

        update["raw"] = deepcopy(row)

        with self._state_lock:
            self._last_order_update = update

    def _record_execution(self, row: dict) -> None:
        if not isinstance(row, dict):
            return

        update = {
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "orderLinkId": row.get("orderLinkId") or row.get("orderId"),
            "execType": row.get("execType"),
            "orderStatus": row.get("orderStatus") or row.get("status"),
            "execQty": row.get("execQty"),
            "execPrice": row.get("execPrice"),
            "execTime": row.get("execTime")
            or row.get("tradeTime")
            or row.get("transactionTime"),
            "category": row.get("category"),
        }

        for key in ("orderLinkId", "execQty", "execPrice", "execTime"):
            value = update.get(key)
            if value is not None:
                update[key] = str(value)

        update["raw"] = deepcopy(row)

        with self._state_lock:
            self._last_execution = update

    def latest_order_update(self) -> Optional[dict]:
        with self._state_lock:
            if self._last_order_update is None:
                return None
            return deepcopy(self._last_order_update)

    def latest_execution(self) -> Optional[dict]:
        with self._state_lock:
            if self._last_execution is None:
                return None
            return deepcopy(self._last_execution)


manager = WSManager()


# --- Backward‑compat helpers for legacy pages ---
def start(subs: Iterable[str] | None = None, include_private: bool = True) -> bool:
    """Module-level proxy mirroring :meth:`WSManager.start`."""
    return manager.start(subs=subs, include_private=include_private)


def stop(include_private: bool = True) -> bool:
    """Stop WebSocket channels started via :func:`start`."""
    manager.stop_public()
    if include_private:
        manager.stop_private()
    return True
