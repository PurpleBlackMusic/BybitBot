
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from typing import Any, Callable

from .envs import get_settings
from .log import log


class WSPrivateV5:
    def __init__(
        self,
        url: str = "wss://stream.bybit.com/v5/private",
        on_msg: Callable[[dict], None] | None = None,
    ):
        self.url = url
        self.on_msg = on_msg or (lambda m: None)
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop = False

    def _sign(self, ts: int, recv_window: int, key: str, secret: str) -> str:
        to_sign = f"{ts}{key}{recv_window}"
        return hmac.new(secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    def is_running(self) -> bool:
        """Return True if the internal websocket loop is active."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self, topics: list[str] | None = None) -> bool:
        if self.is_running():
            return True

        settings = get_settings()
        api_key = getattr(settings, "api_key", "") or ""
        api_secret = getattr(settings, "api_secret", "") or ""
        if not api_key or not api_secret:
            log("ws.private.disabled", reason="missing credentials")
            return False

        try:
            import websocket  # type: ignore
        except Exception as e:
            log("ws.private.disabled", reason="no websocket-client", err=str(e))
            return False

        recv_window = getattr(settings, "recv_window_ms", 5000) or 5000
        try:
            recv_window_int = int(recv_window)
        except Exception:
            recv_window_int = 5000

        ts = int(time.time() * 1000)
        sign = self._sign(ts, recv_window_int, api_key, api_secret)
        auth = {"op": "auth", "args": [api_key, str(ts), str(recv_window_int), sign]}
        subs = {"op": "subscribe", "args": topics or ["order", "execution"]}

        def run() -> None:
            import ssl
            import websocket  # type: ignore

            def handle_open(ws) -> None:
                try:
                    ws.send(json.dumps(auth))
                    time.sleep(0.2)
                    ws.send(json.dumps(subs))
                except Exception as exc:
                    log("ws.private.open.error", err=str(exc))
                    try:
                        ws.close()
                    except Exception:
                        pass

            def handle_message(ws, message: str) -> None:
                payload: Any
                try:
                    payload = json.loads(message)
                except Exception as exc:
                    log("ws.private.message.decode_error", err=str(exc))
                    payload = {"raw": message}

                try:
                    self.on_msg(payload)
                except Exception as exc:
                    log("ws.private.callback.error", err=str(exc))

            def handle_error(ws, error) -> None:
                log("ws.private.error", err=str(error))

            def handle_close(ws, code, msg) -> None:
                log("ws.private.close", code=code, msg=msg, requested=self._stop)

            ws = websocket.WebSocketApp(
                self.url,
                on_open=handle_open,
                on_message=handle_message,
                on_error=handle_error,
                on_close=handle_close,
            )

            self._ws = ws
            try:
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            finally:
                self._ws = None
                self._thread = None

        self._stop = False
        thread = threading.Thread(target=run, daemon=True)
        self._thread = thread
        thread.start()
        return True

    def stop(self) -> None:
        self._stop = True
        ws = self._ws
        thread = self._thread
        try:
            if ws:
                ws.close()
        except Exception:
            pass
        if thread and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        self._ws = None
        self._thread = None
