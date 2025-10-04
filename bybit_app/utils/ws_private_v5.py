
from __future__ import annotations
import time, hmac, hashlib, json, threading
from typing import Callable, Dict, Any
from .envs import get_settings
from .log import log

class WSPrivateV5:
    def __init__(self, url: str = "wss://stream.bybit.com/v5/private", on_msg: Callable[[dict], None] | None = None):
        self.url = url
        self.on_msg = on_msg or (lambda m: None)
        self._ws = None
        self._thread = None
        self._stop = False

    def _sign(self, ts: int, recv_window: int, key: str, secret: str):
        to_sign = str(ts) + key + str(recv_window)
        return hmac.new(bytes(secret, 'utf-8'), bytes(to_sign, 'utf-8'), hashlib.sha256).hexdigest()

    def start(self, topics: list[str] = None):
        try:
            import websocket
        except Exception as e:
            log("ws.private.disabled", reason="no websocket-client", err=str(e))
            return False
        s = get_settings()
        api_key = s.api_key; api_secret = s.api_secret; recv = int(getattr(s, 'recv_window_ms', 5000) or 5000)
        ts = int(time.time()*1000)
        sign = self._sign(ts, recv, api_key, api_secret)
        auth = {"op":"auth","args":[api_key, str(ts), str(recv), sign]}
        subs = {"op":"subscribe","args": topics or ["order", "execution"]}
        def run():
            import websocket, ssl
            ws = websocket.WebSocketApp(self.url,
                on_open=lambda w: (w.send(json.dumps(auth)), time.sleep(0.2), w.send(json.dumps(subs))),
                on_message=lambda w, m: self.on_msg(json.loads(m)),
                on_error=lambda w, e: log("ws.private.error", err=str(e)),
                on_close=lambda w, c, m: log("ws.private.close", code=c, msg=m))
            self._ws = ws
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop = True
        try:
            if self._ws: self._ws.close()
        except Exception:
            pass
