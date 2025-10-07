
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
        *,
        reconnect: bool = True,
    ):
        self.url = url
        self.on_msg = on_msg or (lambda m: None)
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop = False
        self._topics: tuple[str, ...] = ("order", "execution")
        self._authenticated = False
        self._ws_lock = threading.Lock()
        self._reconnect = reconnect

    def _sign(self, expires_ms: int, key: str, secret: str) -> str:
        """Return a Bybit-compatible signature for private WebSocket auth."""

        payload = f"GET/realtime{expires_ms}"
        return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _auth_args(self, key: str, secret: str, *, leeway: float = 10.0) -> tuple[str, str, str]:
        """Prepare authentication arguments for the `auth` request.

        Bybit expects the "expires" timestamp (in ms) to be slightly in the future.
        We add a configurable leeway (defaults to 10 seconds) to reduce the risk of
        clock skew or scheduling delays causing an immediate "Params Error".
        """

        expires_ms = int((time.time() + max(leeway, 1.0)) * 1000)
        signature = self._sign(expires_ms, key, secret)
        return key, str(expires_ms), signature

    def is_running(self) -> bool:
        """Return True if the internal websocket loop is active."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self, topics: list[str] | None = None) -> bool:
        if self.is_running():
            if topics:
                self._topics = tuple(topics)
                ws = self._ws
                if ws is not None:
                    try:
                        ws.send(json.dumps({"op": "subscribe", "args": topics}))
                    except Exception as exc:
                        log("ws.private.resub.error", err=str(exc))
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

        auth_args = self._auth_args(api_key, api_secret)
        auth = {"op": "auth", "args": list(auth_args)}
        self._topics = tuple(topics or ["order", "execution"])

        def run() -> None:
            import ssl
            import websocket  # type: ignore

            backoff = 1.0

            def handle_open(ws) -> None:
                try:
                    ws.send(json.dumps(auth))
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

                if isinstance(payload, dict):
                    if payload.get("op") == "auth":
                        success = bool(payload.get("success"))
                        if success:
                            self._authenticated = True
                            try:
                                ws.send(json.dumps({"op": "subscribe", "args": list(self._topics)}))
                            except Exception as exc:
                                log("ws.private.sub.error", err=str(exc))
                            else:
                                log("ws.private.auth.ok")
                        else:
                            log("ws.private.auth.error", msg=payload.get("ret_msg"))
                    elif payload.get("op") == "subscribe" and not payload.get("success", True):
                        log("ws.private.subscribe.error", msg=payload.get("ret_msg"))

                try:
                    self.on_msg(payload)
                except Exception as exc:
                    log("ws.private.callback.error", err=str(exc))

            def handle_error(ws, error) -> None:
                log("ws.private.error", err=str(error))

            def handle_close(ws, code, msg) -> None:
                log("ws.private.close", code=code, msg=msg, requested=self._stop)
                self._authenticated = False

            while not self._stop:
                ws = websocket.WebSocketApp(
                    self.url,
                    on_open=handle_open,
                    on_message=handle_message,
                    on_error=handle_error,
                    on_close=handle_close,
                )
                with self._ws_lock:
                    self._ws = ws
                try:
                    ws.run_forever(
                        sslopt={"cert_reqs": ssl.CERT_NONE},
                        ping_interval=20,
                        ping_timeout=10,
                    )
                finally:
                    with self._ws_lock:
                        self._ws = None
                    self._authenticated = False
                if self._stop or not self._reconnect:
                    break
                sleep_for = min(backoff, 60.0)
                log("ws.private.reconnect.wait", seconds=round(sleep_for, 2))
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 60.0)
                retry_args = self._auth_args(api_key, api_secret)
                auth.update({"args": list(retry_args)})

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
