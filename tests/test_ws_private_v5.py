from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
import ssl

import pytest

from bybit_app.utils import ws_private_v5
from bybit_app.utils.ws_private_v5 import WSPrivateV5, DEFAULT_TOPICS


@pytest.fixture(autouse=True)
def _patch_time_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "synced_timestamp_ms",
        lambda *args, **kwargs: 1_700_000_000_000,
    )


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
        def __init__(
            self,
            url: str,
            on_open=None,
            on_message=None,
            on_error=None,
            on_close=None,
            on_pong=None,
        ):
            self.url = url
            self._on_open = on_open
            self._on_message = on_message
            self._on_close = on_close
            self._on_pong = on_pong

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
            if self._on_pong:
                self._on_pong(self, None)
            if self._on_close:
                self._on_close(self, 1000, "normal")

    fake_module = SimpleNamespace(WebSocketApp=FakeWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", fake_module)
    monkeypatch.setattr(ws_private_v5, "websocket", fake_module, raising=False)
    return sent


def test_ws_private_v5_does_not_duplicate_subscribe_requests() -> None:
    client = WSPrivateV5()

    class DummyWS:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    ws = DummyWS()

    client._subscribe_missing(ws)
    assert len(ws.sent) == 1

    # No additional subscribe should be emitted while the first request is pending.
    client._subscribe_missing(ws)
    assert len(ws.sent) == 1

    payload = ws.sent[0]
    assert payload.get("op") == "subscribe"
    args = payload.get("args")
    assert isinstance(args, list)
    keys = {str(topic).lower() for topic in args}
    client._handle_subscription_ack(keys, success=True)

    # Once the subscription is acknowledged, subsequent calls should be no-ops.
    client._subscribe_missing(ws)
    assert len(ws.sent) == 1


def test_ws_private_v5_unsubscribes_removed_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    client = WSPrivateV5()

    class AliveThread:
        def is_alive(self) -> bool:
            return True

    class DummyWS:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    dummy_ws = DummyWS()
    client._ws = dummy_ws
    client._thread = AliveThread()
    monkeypatch.setattr(client, "_is_socket_connected", lambda ws: True)

    client._set_topics(["custom.one"])
    with client._topics_lock:
        client._active_topics.update(topic.lower() for topic in client._topics)

    assert client.start(topics=["other.topic"]) is True

    ops = [payload.get("op") for payload in dummy_ws.sent]
    assert ops.count("unsubscribe") == 1
    assert ops.count("subscribe") == 1

    unsub_args = next(payload.get("args") for payload in dummy_ws.sent if payload.get("op") == "unsubscribe")
    assert unsub_args == ["custom.one"]

    sub_args = next(payload.get("args") for payload in dummy_ws.sent if payload.get("op") == "subscribe")
    assert "other.topic" in sub_args


def test_ws_private_v5_validates_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ws_private_v5, "log", lambda event, **payload: events.append((event, payload)))
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda force_reload=False: SimpleNamespace(
            api_key="",
            api_secret="",
            recv_window_ms=15000,
        ),
    )

    client = WSPrivateV5()
    assert client.start() is False
    assert events == [("ws.private.disabled", {"reason": "missing credentials"})]


def test_ws_private_v5_start_supports_keywordless_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(api_key="abc", api_secret="def", verify_ssl=True)
    monkeypatch.setattr(ws_private_v5, "get_settings", lambda: settings)

    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ws_private_v5, "log", lambda event, **payload: events.append((event, payload)))

    _install_thread_stub(monkeypatch)
    messages = [json.dumps({"op": "auth", "success": True})]
    sent = _install_websocket_stub(monkeypatch, messages)

    client = WSPrivateV5(reconnect=False)
    assert client.start() is True

    assert all(event != "ws.private.disabled" for event, _ in events)
    assert any(json.loads(msg).get("op") == "auth" for msg in sent)


def test_ws_private_v5_handles_decode_and_callback_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda force_reload=False: SimpleNamespace(
            api_key="abc",
            api_secret="def",
            recv_window_ms=15000,
        ),
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
        lambda force_reload=False: SimpleNamespace(
            api_key="abc",
            api_secret="def",
            recv_window_ms=15000,
        ),
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
        lambda force_reload=False: SimpleNamespace(
            api_key="abc",
            api_secret="def",
            recv_window_ms=15000,
        ),
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


def test_ws_private_v5_emits_heartbeat_on_pong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ws_private_v5,
        "get_settings",
        lambda force_reload=False: SimpleNamespace(
            api_key="abc",
            api_secret="def",
            recv_window_ms=15000,
        ),
    )
    _install_thread_stub(monkeypatch)

    messages = [json.dumps({"op": "auth", "success": True})]
    _install_websocket_stub(monkeypatch, messages)

    captured: list[dict[str, Any]] = []

    def on_msg(payload: dict[str, Any]) -> None:
        captured.append(payload)

    client = WSPrivateV5(on_msg=on_msg, reconnect=False)
    assert client.start() is True

    assert any(payload.get("op") == "pong" for payload in captured)


@pytest.mark.parametrize(
    "verify_flag, expected_cert",
    (
        (True, ssl.CERT_REQUIRED),
        (False, ssl.CERT_NONE),
    ),
)
def test_ws_private_v5_respects_verify_ssl(
    monkeypatch: pytest.MonkeyPatch, verify_flag: bool, expected_cert
) -> None:
    settings = SimpleNamespace(api_key="key", api_secret="secret", verify_ssl=verify_flag)
    monkeypatch.setattr(ws_private_v5, "get_settings", lambda *args, **kwargs: settings)

    captured: dict[str, Any] = {}

    class DummyWebSocketApp:
        def __init__(
            self,
            url: str,
            on_open=None,
            on_message=None,
            on_error=None,
            on_close=None,
            on_pong=None,
        ):
            self._on_open = on_open
            self._on_close = on_close

        def send(self, payload: str) -> None:  # pragma: no cover - for parity
            captured.setdefault("sent", []).append(payload)

        def run_forever(self, **kwargs) -> None:
            captured["sslopt"] = kwargs.get("sslopt")
            if self._on_open:
                self._on_open(self)
            if self._on_close:
                self._on_close(self, 1000, "normal")

        def close(self) -> None:  # pragma: no cover - parity with real interface
            captured["closed"] = True

    class ImmediateThread:
        def __init__(self, target, daemon: bool = False):
            self._target = target
            self.daemon = daemon
            self._is_alive = False

        def start(self) -> None:
            name = getattr(self._target, "__name__", "")
            if name == "_ping_loop":
                return
            self._is_alive = True
            try:
                self._target()
            finally:
                self._is_alive = False

        def is_alive(self) -> bool:
            return self._is_alive

        def join(self, timeout: float | None = None) -> None:  # pragma: no cover - parity
            return None

    fake_websocket_module = SimpleNamespace(WebSocketApp=DummyWebSocketApp)
    monkeypatch.setitem(sys.modules, "websocket", fake_websocket_module)
    monkeypatch.setattr(ws_private_v5, "websocket", fake_websocket_module, raising=False)
    monkeypatch.setattr(ws_private_v5.threading, "Thread", ImmediateThread)

    client = WSPrivateV5(reconnect=False)
    assert client.start() is True

    sslopt = captured.get("sslopt")
    assert isinstance(sslopt, dict)
    assert sslopt.get("cert_reqs") == expected_cert

