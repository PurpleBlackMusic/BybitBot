from __future__ import annotations

from types import SimpleNamespace
import sys
import ssl
import json

import pytest
import threading
import time


from bybit_app.utils import ws_orderbook_v5 as ws_orderbook_v5_module
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


class _RecordingEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.wait_calls: list[float | None] = []

    def wait(self, timeout=None):  # type: ignore[override]
        self.wait_calls.append(timeout)
        return super().wait(timeout=timeout)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("False", False),
        ("no", False),
        ("yes", True),
        (None, True),
    ],
)
def test_should_verify_ssl_normalises_inputs(raw_value, expected):
    settings = SimpleNamespace(verify_ssl=raw_value) if raw_value is not None else None
    assert ws_orderbook_v5_module._should_verify_ssl(settings) is expected


@pytest.mark.parametrize(
    "verify_flag, expected_cert",
    [
        (True, ssl.CERT_REQUIRED),
        (False, ssl.CERT_NONE),
        ("false", ssl.CERT_NONE),
        ("yes", ssl.CERT_REQUIRED),
        (0, ssl.CERT_NONE),
    ],
)
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
    recording_stop_event = _RecordingEvent()

    _FailingWebSocketApp.instances.clear()
    _FailingWebSocketApp.run_calls = 0

    ob = WSOrderbookV5(stop_event_factory=lambda: recording_stop_event)

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
    assert recording_stop_event.wait_calls, "backoff sleep was not invoked"
    assert recording_stop_event.wait_calls[0] is not None
    assert recording_stop_event.wait_calls[0] < 1.0


def test_ws_orderbook_v5_updates_topics_without_restarting(monkeypatch):
    ready_event = threading.Event()
    stop_event = threading.Event()
    sent_payloads: list[str] = []

    class _TrackingWebSocketApp:
        instances: list["_TrackingWebSocketApp"] = []

        def __init__(self, url, **kwargs):
            self.url = url
            self.on_open = kwargs.get("on_open")
            self.on_message = kwargs.get("on_message")
            self.on_error = kwargs.get("on_error")
            self.on_close = kwargs.get("on_close")
            self.sent: list[str] = []
            _TrackingWebSocketApp.instances.append(self)

        def run_forever(self, sslopt=None):
            self.sslopt = sslopt or {}
            if self.on_open:
                self.on_open(self)
            ready_event.set()
            stop_event.wait()

        def send(self, payload: str):
            self.sent.append(payload)
            sent_payloads.append(payload)

        def close(self):
            stop_event.set()
            if self.on_close:
                self.on_close(self, None, None)

    dummy_module = SimpleNamespace(WebSocketApp=_TrackingWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", dummy_module)

    _TrackingWebSocketApp.instances.clear()
    sent_payloads.clear()

    ob = WSOrderbookV5()

    assert ob.start(["BTCUSDT"]) is True

    assert ready_event.wait(timeout=1), "websocket thread did not call on_open"
    first_thread = ob._thread
    assert first_thread is not None and first_thread.is_alive()
    assert len(_TrackingWebSocketApp.instances) == 1

    initial_payloads = [json.loads(msg) for msg in sent_payloads]
    assert initial_payloads and initial_payloads[-1]["op"] == "subscribe"
    assert initial_payloads[-1]["args"] == [f"orderbook.{ob.levels}.BTCUSDT"]

    sent_payloads.clear()

    assert ob.start(["ETHUSDT"]) is True
    assert ob._thread is first_thread
    assert len(_TrackingWebSocketApp.instances) == 1

    updated_payloads = [json.loads(msg) for msg in sent_payloads]
    assert [p["op"] for p in updated_payloads] == ["unsubscribe", "subscribe"]
    assert updated_payloads[0]["args"] == [f"orderbook.{ob.levels}.BTCUSDT"]
    assert updated_payloads[1]["args"] == [f"orderbook.{ob.levels}.ETHUSDT"]

    ob.stop()
    assert stop_event.wait(timeout=1)
    first_thread.join(timeout=1)
    assert not first_thread.is_alive()


def test_ws_orderbook_v5_enforces_sequence_and_resubscribes(monkeypatch):
    ob = WSOrderbookV5()
    topic = f"orderbook.{ob.levels}.BTCUSDT"
    sent_payloads: list[dict] = []

    class _DummyWS:
        def send(self, payload: str):
            sent_payloads.append(json.loads(payload))

    ob._ws = _DummyWS()
    with ob._topic_lock:
        ob._topics = {topic}
        ob._topic_symbols = {topic: "BTCUSDT"}

    ob._mark_symbols_for_snapshot(["BTCUSDT"], clear_book=True)

    snapshot = {
        "topic": topic,
        "data": {
            "type": "snapshot",
            "ts": 1000,
            "s": "BTCUSDT",
            "u": 10,
            "b": [["100", "1"]],
            "a": [["101", "2"]],
        },
    }
    ob._on_msg(None, json.dumps(snapshot))

    book = ob.get("BTCUSDT")
    assert book is not None
    assert book["b"] == [(100.0, 1.0)]
    assert book["a"] == [(101.0, 2.0)]

    delta_ok = {
        "topic": topic,
        "data": {
            "type": "delta",
            "ts": 1100,
            "s": "BTCUSDT",
            "pu": 10,
            "u": 11,
            "b": [["100", "3"]],
            "a": [],
        },
    }
    ob._on_msg(None, json.dumps(delta_ok))

    book_after_delta = ob.get("BTCUSDT")
    assert book_after_delta is not None
    assert book_after_delta["b"] == [(100.0, 3.0)]

    sent_payloads.clear()
    delta_out_of_order = {
        "topic": topic,
        "data": {
            "type": "delta",
            "ts": 1200,
            "s": "BTCUSDT",
            "pu": 5,
            "u": 6,
            "b": [["99", "1"]],
            "a": [],
        },
    }
    ob._on_msg(None, json.dumps(delta_out_of_order))

    assert [p["op"] for p in sent_payloads] == ["unsubscribe", "subscribe"]

    # После рассинхронизации предыдущая книга сбрасывается и ждёт snapshot.
    assert ob.get("BTCUSDT") is None
    assert "BTCUSDT" in ob._waiting_snapshot

    # Следующий delta до snapshot будет проигнорирован, но спустя время повторится resubscribe.
    sent_payloads.clear()
    ob._last_resubscribe["BTCUSDT"] = time.time() - 2
    ob._on_msg(None, json.dumps(delta_ok))
    assert [p["op"] for p in sent_payloads] == ["unsubscribe", "subscribe"]

