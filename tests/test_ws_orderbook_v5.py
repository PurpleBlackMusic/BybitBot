from __future__ import annotations

from types import SimpleNamespace
import sys
import ssl

import pytest
import threading


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


def test_ws_orderbook_v5_reconnects_until_stopped(monkeypatch):
    class _FailingWebSocketApp:
        instances: list["_FailingWebSocketApp"] = []
        run_calls = 0

        def __init__(self, url, **_kwargs):
            self.url = url
            self.closed = False
            _FailingWebSocketApp.instances.append(self)

        def run_forever(self, sslopt=None):
            _FailingWebSocketApp.run_calls += 1
            if _FailingWebSocketApp.run_calls == 1:
                first_attempt.set()
                raise RuntimeError("boom")
            second_attempt.set()
            stop_event.wait(timeout=1)

        def close(self):
            self.closed = True
            stop_event.set()

    dummy_module = SimpleNamespace(WebSocketApp=_FailingWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", dummy_module)

    from bybit_app.utils import ws_orderbook_v5

    first_attempt = threading.Event()
    second_attempt = threading.Event()
    stop_event = threading.Event()
    sleep_calls: list[float] = []

    def fake_sleep(duration: float):
        sleep_calls.append(duration)

    monkeypatch.setattr(ws_orderbook_v5.time, "sleep", fake_sleep)

    _FailingWebSocketApp.instances.clear()
    _FailingWebSocketApp.run_calls = 0

    ob = WSOrderbookV5()

    assert ob.start(["BTCUSDT"]) is True

    assert first_attempt.wait(timeout=1), "first attempt not triggered"
    assert second_attempt.wait(timeout=1), "second attempt not triggered"

    assert _FailingWebSocketApp.run_calls >= 2
    assert len(_FailingWebSocketApp.instances) >= 2

    ob.stop()

    thread = ob._thread
    assert thread is not None
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert ob._ws is None
    assert stop_event.is_set()
    assert sleep_calls, "backoff sleep was not invoked"

