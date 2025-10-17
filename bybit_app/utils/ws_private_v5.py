
from __future__ import annotations

import hashlib
import hmac
import json
import random
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from .bybit_api import API_MAIN, API_TEST
from .envs import get_settings
from .settings_loader import call_get_settings
from .log import log
from .time_sync import invalidate_synced_clock, synced_timestamp_ms


DEFAULT_TOPICS: tuple[str, ...] = (
    "order.spot",
    "execution.spot",
    "wallet",
    "position",
)


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
        self._topics: tuple[str, ...] = DEFAULT_TOPICS
        self._topic_lookup: dict[str, str] = {topic.lower(): topic for topic in self._topics}
        self._topic_keys: set[str] = set(self._topic_lookup)
        self._active_topics: set[str] = set()
        self._pending_topics: set[str] = set()
        self._topics_lock = threading.Lock()
        self._authenticated = False
        self._ws_lock = threading.Lock()
        self._reconnect = reconnect

    def _emit_to_callback(self, payload: Mapping[str, Any]) -> None:
        try:
            self.on_msg(dict(payload))
        except Exception as exc:
            log("ws.private.callback.error", err=str(exc))

    def _sign(self, expires_ms: int, key: str, secret: str) -> str:
        """Return a Bybit-compatible signature for private WebSocket auth."""

        payload = f"GET/realtime{expires_ms}"
        return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _rest_base(self) -> str:
        url = (self.url or "").lower()
        if "testnet" in url:
            return API_TEST
        return API_MAIN

    def _auth_args(self, key: str, secret: str, *, leeway: float = 10.0) -> tuple[str, str, str]:
        """Prepare authentication arguments for the `auth` request.

        Bybit expects the "expires" timestamp (in ms) to be slightly in the future.
        We add a configurable leeway (defaults to 10 seconds) to reduce the risk of
        clock skew or scheduling delays causing an immediate "Params Error".
        """

        settings = None
        try:
            settings = get_settings()
        except Exception:
            settings = None

        verify_ssl = True
        if settings is not None:
            verify_ssl = bool(getattr(settings, "verify_ssl", True))

        base_url = self._rest_base()
        try:
            now_ms = synced_timestamp_ms(
                base_url,
                timeout=5.0,
                verify=verify_ssl,
            )
        except Exception:  # pragma: no cover - defensive guard
            now_ms = int(time.time() * 1000)

        expires_ms = now_ms + int(max(leeway, 1.0) * 1000)
        signature = self._sign(expires_ms, key, secret)
        return key, str(expires_ms), signature

    def is_running(self) -> bool:
        """Return True if the internal websocket loop is active."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    def _normalise_topics(self, topics: list[str] | None) -> tuple[str, ...]:
        """Combine default topics with user supplied ones and drop duplicates."""

        combined: list[str] = list(DEFAULT_TOPICS)
        if topics:
            combined.extend(topics)

        seen: set[str] = set()
        normalised: list[str] = []
        for topic in combined:
            if topic is None:
                continue
            text = str(topic).strip()
            if not text:
                continue
            key = text.lower()
            if key == "order":
                text = "order.spot"
                key = text.lower()
            elif key == "execution":
                text = "execution.spot"
                key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalised.append(text)
        return tuple(normalised)

    def _extract_topic_keys(self, topics: Iterable[str]) -> set[str]:
        keys: set[str] = set()
        for topic in topics:
            if topic is None:
                continue
            text = str(topic).strip().lower()
            if not text:
                continue
            keys.add(text)
        return keys

    def _set_topics(self, topics: list[str] | None) -> list[str]:
        normalised = self._normalise_topics(topics)
        lookup = {topic.lower(): topic for topic in normalised}
        new_keys = set(lookup)
        with self._topics_lock:
            old_lookup = getattr(self, "_topic_lookup", {})
            old_keys = set(old_lookup)
            removed_keys = old_keys - new_keys
            removed_topics = [old_lookup[key] for key in removed_keys if key in old_lookup]
            self._topics = normalised
            self._topic_lookup = lookup
            self._topic_keys = new_keys
            if removed_keys:
                self._pending_topics.difference_update(removed_keys)
                self._active_topics.difference_update(removed_keys)
        return removed_topics

    def _unsubscribe_topics(self, ws, topics: Iterable[str]) -> None:
        args = [topic for topic in topics if topic]
        if not args:
            return
        try:
            ws.send(json.dumps({"op": "unsubscribe", "args": args}))
        except Exception as exc:
            log("ws.private.unsub.error", err=str(exc))

    def _handle_subscription_ack(self, keys: set[str], *, success: bool) -> None:
        if not keys:
            return
        with self._topics_lock:
            self._pending_topics.difference_update(keys)
            if success:
                self._active_topics.update(key for key in keys if key in self._topic_keys)
            else:
                self._active_topics.difference_update(keys)

    def _handle_unsubscribe_ack(self, keys: set[str], *, success: bool) -> None:
        if not keys:
            return
        with self._topics_lock:
            self._pending_topics.difference_update(keys)
            if success:
                self._active_topics.difference_update(keys)

    def _subscribe_missing(self, ws) -> None:
        with self._topics_lock:
            pending = self._pending_topics.copy()
            active = self._active_topics.copy()
            missing: list[str] = []
            keys_to_send: list[str] = []
            for topic in self._topics:
                if not topic:
                    continue
                key = topic.lower()
                if not key or key in active or key in pending or key in keys_to_send:
                    continue
                missing.append(topic)
                keys_to_send.append(key)
        if not missing:
            return
        try:
            ws.send(json.dumps({"op": "subscribe", "args": missing}))
        except Exception as exc:
            log("ws.private.sub.error", err=str(exc))
        else:
            with self._topics_lock:
                self._pending_topics.update(keys_to_send)

    def start(self, topics: list[str] | None = None) -> bool:
        if self.is_running():
            if topics is not None:
                removed_topics = self._set_topics(topics)
                ws = self._ws
                if ws is not None and self._is_socket_connected(ws):
                    if removed_topics:
                        self._unsubscribe_topics(ws, removed_topics)
                    try:
                        self._subscribe_missing(ws)
                    except Exception as exc:
                        log("ws.private.resub.error", err=str(exc))
            return True

        settings = call_get_settings(get_settings, force_reload=True)
        api_key = getattr(settings, "api_key", "") or ""
        api_secret = getattr(settings, "api_secret", "") or ""
        if not api_key or not api_secret:
            log("ws.private.disabled", reason="missing credentials")
            return False

        verify_ssl = bool(getattr(settings, "verify_ssl", True))

        try:
            import websocket  # type: ignore
        except Exception as e:
            log("ws.private.disabled", reason="no websocket-client", err=str(e))
            return False

        base_url = self._rest_base()
        auth_args = self._auth_args(api_key, api_secret)
        auth = {"op": "auth", "args": list(auth_args)}
        self._set_topics(topics)
        with self._topics_lock:
            self._active_topics.clear()
            self._pending_topics.clear()

        def run() -> None:
            import ssl
            import websocket  # type: ignore

            backoff = 1.0
            ping_thread: threading.Thread | None = None
            ping_stop: threading.Event | None = None

            def stop_ping_loop() -> None:
                nonlocal ping_thread, ping_stop
                if ping_stop is not None:
                    ping_stop.set()
                thread = ping_thread
                if thread and thread.is_alive():
                    try:
                        thread.join(timeout=1.0)
                    except Exception:
                        pass
                ping_thread = None
                ping_stop = None

            def start_ping_loop(ws) -> None:
                nonlocal ping_thread, ping_stop
                stop_ping_loop()
                ping_stop = threading.Event()

                def _ping_loop() -> None:
                    while not self._stop and not ping_stop.is_set():
                        if not self._is_socket_connected(ws):
                            break
                        try:
                            req_id = str(int(time.time() * 1000))
                            payload = {"op": "ping", "req_id": req_id}
                            ws.send(json.dumps(payload))
                            control_event = {
                                "op": "ping",
                                "source": "control",
                                "req_id": req_id,
                                "sent_at": time.time(),
                                "sent_monotonic": time.monotonic(),
                            }
                            self._emit_to_callback(control_event)
                        except Exception as exc:
                            log("ws.private.ping.error", err=str(exc))
                            break
                        interval = random.uniform(19.0, 21.0)
                        if ping_stop.wait(interval):
                            break

                ping_thread = threading.Thread(target=_ping_loop, daemon=True)
                ping_thread.start()

            def handle_open(ws) -> None:
                try:
                    ws.send(json.dumps(auth))
                except Exception as exc:
                    log("ws.private.open.error", err=str(exc))
                    try:
                        ws.close()
                    except Exception:
                        pass
                else:
                    start_ping_loop(ws)

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
                            self._subscribe_missing(ws)
                            log("ws.private.auth.ok")
                        else:
                            ret_code = (
                                payload.get("ret_code")
                                or payload.get("code")
                                or payload.get("retCode")
                            )
                            log(
                                "ws.private.auth.error",
                                msg=payload.get("ret_msg"),
                                code=ret_code,
                            )
                            if ret_code in (10002, "10002"):
                                invalidate_synced_clock(base_url)
                                retry_args = self._auth_args(api_key, api_secret)
                                auth.update({"args": list(retry_args)})
                                try:
                                    ws.send(json.dumps(auth))
                                except Exception as exc:
                                    log("ws.private.auth.retry.error", err=str(exc))
                    elif payload.get("op") == "subscribe":
                        success = payload.get("success", True)
                        args = payload.get("args")
                        if isinstance(args, list):
                            keys = self._extract_topic_keys(args)
                        else:
                            topic = payload.get("topic")
                            keys = self._extract_topic_keys([topic] if topic else [])
                        self._handle_subscription_ack(keys, success=bool(success))
                        if not success:
                            log("ws.private.subscribe.error", msg=payload.get("ret_msg"))
                    elif payload.get("op") == "unsubscribe":
                        success = payload.get("success", True)
                        args = payload.get("args")
                        if isinstance(args, list):
                            keys = self._extract_topic_keys(args)
                        else:
                            topic = payload.get("topic")
                            keys = self._extract_topic_keys([topic] if topic else [])
                        self._handle_unsubscribe_ack(keys, success=bool(success))
                        if not success:
                            log("ws.private.unsubscribe.error", msg=payload.get("ret_msg"))
                    elif payload.get("op") == "ping":
                        response = {
                            "op": "pong",
                            "req_id": payload.get("req_id")
                            or str(int(time.time() * 1000)),
                        }
                        try:
                            ws.send(json.dumps(response))
                        except Exception as exc:
                            log("ws.private.pong.error", err=str(exc))
                    elif payload.get("op") == "pong":
                        payload.setdefault("source", "message")
                        payload.setdefault("received_at", time.time())
                        payload.setdefault("received_monotonic", time.monotonic())

                self._emit_to_callback(payload)

            def handle_error(ws, error) -> None:
                log("ws.private.error", err=str(error))

            def handle_pong(ws, message) -> None:
                text: str | None
                if isinstance(message, (bytes, bytearray)):
                    try:
                        text = message.decode("utf-8")
                    except Exception:
                        text = None
                else:
                    text = str(message) if message else None
                payload: dict[str, Any] = {
                    "op": "pong",
                    "source": "control",
                    "received_at": time.time(),
                    "received_monotonic": time.monotonic(),
                }
                if text:
                    payload["raw"] = text
                self._emit_to_callback(payload)

            def handle_close(ws, code, msg) -> None:
                log("ws.private.close", code=code, msg=msg, requested=self._stop)
                self._authenticated = False
                with self._topics_lock:
                    self._active_topics.clear()
                    self._pending_topics.clear()
                stop_ping_loop()

            while not self._stop:
                ws = websocket.WebSocketApp(
                    self.url,
                    on_open=handle_open,
                    on_message=handle_message,
                    on_error=handle_error,
                    on_close=handle_close,
                    on_pong=handle_pong,
                )
                with self._ws_lock:
                    self._ws = ws
                try:
                    cert_reqs = ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE
                    sslopt = {"cert_reqs": cert_reqs}
                    ws.run_forever(
                        sslopt=sslopt,
                        ping_interval=0,
                        ping_timeout=None,
                    )
                finally:
                    with self._ws_lock:
                        self._ws = None
                    self._authenticated = False
                    stop_ping_loop()
                if self._stop or not self._reconnect:
                    break
                sleep_base = min(backoff, 60.0)
                jitter = random.uniform(0.0, sleep_base * 0.3 if sleep_base > 0 else 0.5)
                sleep_for = sleep_base + jitter
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

    def _is_socket_connected(self, ws: Any) -> bool:
        """Return True if the websocket has an active socket."""

        if ws is None:
            return False

        sock = getattr(ws, "sock", None)
        if sock is None:
            return False

        return bool(getattr(sock, "connected", False))

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
