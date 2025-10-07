from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from bybit_app.utils import ws_private_v5
from bybit_app.utils.ws_private_v5 import WSPrivateV5, DEFAULT_TOPICS


class _ImmediateThread:
    """Thread stub that runs the target synchronously for deterministic tests."""

    def __init__(self, target, daemon: bool = False):  # pragma: no cover - trivial
        self._target = target
        self._is_alive = False

    def start(self) -> None:  # pragma: no cover - trivial
        self._is_alive = True
        try:
            self._target()
        finally:
            self._is_alive = False

    def is_alive(self) -> bool:  # pragma: no cover - trivial
        return self._is_alive

    def join(self, timeout: float | None = None) -> None:  # pragma: no cover - trivial
        return None


def _install_thread_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_private_v5.threading, "Thread", _ImmediateThread)


def _install_websocket_stub(monkeypatch: pytest.MonkeyPatch, messages: list[str]) -> list[str]:
    sent: list[str] = []

    class FakeWebSocketApp:
        def __init__(self, url: str, on_open=None, on_message=None, on_error=None, on_close=None):
            self.url = url
            self._on_open = on_open
            self._on_message = on_message
            self._on_close = on_close

        def send(self, payload: str) -> None:
            sent.append(payload)

        def close(self) -> None:  # pragma: no cover - parity with real interface
            sent.append("__closed__")

        def run_forever(self, sslopt: dict[str, Any] | None = None, **kwargs) -> None:
            if self._on_open:
                self._on_open(self)
            if self._on_message:
                for message in list(messages):
                    self._on_message(self, message)
            if self._on_close:
                self._on_close(self, 1000, "normal")

    fake_module = SimpleNamespace(WebSocketApp=FakeWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", fake_module)
    monkeypatch.setattr(ws_private_v5, "websocket", fake_module, raising=False)
    return sent


def test_ws_private_v5_validates_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ws_private_v5, "log", lambda event, **payload: events.append((event, payload)))
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda: SimpleNamespace(api_key="", api_secret="", recv_window_ms=5000),
    )

    client = WSPrivateV5()
    assert client.start() is False
    assert events == [("ws.private.disabled", {"reason": "missing credentials"})]


def test_ws_private_v5_handles_decode_and_callback_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda: SimpleNamespace(api_key="abc", api_secret="def", recv_window_ms=5000),
    )
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ws_private_v5, "log", lambda event, **payload: events.append((event, payload)))
    _install_thread_stub(monkeypatch)

    messages = ["{", json.dumps({"foo": "bar"})]
    sent = _install_websocket_stub(monkeypatch, messages)

    received: list[Any] = []

    def on_msg(payload: dict[str, Any]) -> None:
        received.append(payload)
        if payload.get("foo") == "bar":
            raise RuntimeError("boom")

    client = WSPrivateV5(on_msg=on_msg, reconnect=False)
    assert client.start(topics=["order"]) is True

    auth_msgs = [json.loads(msg) for msg in sent if json.loads(msg).get("op") == "auth"]
    assert auth_msgs, "auth message was not sent"
    for auth_msg in auth_msgs:
        args = auth_msg.get("args")
        assert isinstance(args, list)
        assert len(args) == 3
    assert {tuple(msg.keys()) for msg in received if isinstance(msg, dict)} >= {("raw",), ("foo",)}
    assert {evt for evt, _ in events} >= {"ws.private.message.decode_error", "ws.private.callback.error", "ws.private.close"}
    assert client._ws is None
    assert client._thread is None


def test_ws_private_v5_subscribes_to_default_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda: SimpleNamespace(api_key="abc", api_secret="def", recv_window_ms=5000),
    )
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ws_private_v5, "log", lambda event, **payload: events.append((event, payload)))
    _install_thread_stub(monkeypatch)

    messages = [json.dumps({"op": "auth", "success": True})]
    sent = _install_websocket_stub(monkeypatch, messages)

    client = WSPrivateV5(reconnect=False)
    assert client.start() is True

    subscribe_msgs = [json.loads(msg) for msg in sent if json.loads(msg).get("op") == "subscribe"]
    assert subscribe_msgs, "subscribe message was not sent"
    assert subscribe_msgs[-1]["args"] == list(DEFAULT_TOPICS)
    assert ("ws.private.auth.ok", {}) in events


def test_ws_private_v5_merges_custom_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda: SimpleNamespace(api_key="abc", api_secret="def", recv_window_ms=5000),
    )
    _install_thread_stub(monkeypatch)

    messages = [json.dumps({"op": "auth", "success": True})]
    sent = _install_websocket_stub(monkeypatch, messages)

    client = WSPrivateV5(reconnect=False)
    assert client.start(topics=["position", "order", "wallet", "position"]) is True

    subscribe_msgs = [json.loads(msg) for msg in sent if json.loads(msg).get("op") == "subscribe"]
    assert subscribe_msgs
    assert subscribe_msgs[-1]["args"] == list(DEFAULT_TOPICS)


def test_ws_private_v5_stop_closes_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    client = WSPrivateV5()

    class DummyWS:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class DummyThread:
        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:  # pragma: no cover - parity
            return None

    dummy_ws = DummyWS()
    client._ws = dummy_ws
    client._thread = DummyThread()  # type: ignore[assignment]

    client.stop()
    assert dummy_ws.closed is True
    assert client._ws is None
    assert client._thread is None

