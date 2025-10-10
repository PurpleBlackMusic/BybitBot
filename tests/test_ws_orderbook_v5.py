from __future__ import annotations

from types import SimpleNamespace
import sys
import ssl

import pytest


from bybit_app.utils.ws_orderbook_v5 import WSOrderbookV5


class _DummyWebSocketApp:
    instances: list["_DummyWebSocketApp"] = []

    def __init__(self, url, **_kwargs):
        self.url = url
        self.sslopt = None
        _DummyWebSocketApp.instances.append(self)

    def run_forever(self, sslopt=None):
        self.sslopt = sslopt or {}

    def close(self):  # pragma: no cover - defensive
        pass


@pytest.mark.parametrize("verify_flag, expected_cert", [(True, ssl.CERT_REQUIRED), (False, ssl.CERT_NONE)])
def test_ws_orderbook_v5_respects_verify_ssl(monkeypatch, verify_flag, expected_cert):
    dummy_module = SimpleNamespace(WebSocketApp=_DummyWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", dummy_module)

    from bybit_app.utils import ws_orderbook_v5

    monkeypatch.setattr(ws_orderbook_v5, "get_settings", lambda: SimpleNamespace(verify_ssl=verify_flag))

    _DummyWebSocketApp.instances.clear()
    ob = WSOrderbookV5()

    assert ob.start(["BTCUSDT"]) is True

    thread = ob._thread
    assert thread is not None
    thread.join(timeout=1)

    assert _DummyWebSocketApp.instances, "WebSocketApp was not instantiated"
    ws_instance = _DummyWebSocketApp.instances[-1]
    assert ws_instance.sslopt is not None
    assert ws_instance.sslopt.get("cert_reqs") == expected_cert

