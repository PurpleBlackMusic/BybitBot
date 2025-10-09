from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Iterable

import pytest

from bybit_app.utils import ws_manager as ws_manager_module
import bybit_app.utils.pnl as pnl_module
from bybit_app.utils.ws_manager import WSManager
from bybit_app.utils.ws_private_v5 import WSPrivateV5, DEFAULT_TOPICS


def test_ws_manager_status_reports_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: True)
    manager._pub_ws = object()
    manager._pub_subs = ("tickers.BTCUSDT", "tickers.ETHUSDT")

    class DummyPrivate:
        _thread = SimpleNamespace(is_alive=lambda: True)

        def is_running(self) -> bool:
            return True

    manager._priv = DummyPrivate()
    manager.last_beat = 1_000_000.0
    manager.last_public_beat = 1_000_000.0
    manager.last_private_beat = 1_000_000.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 1_000_012.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["public"]["subscriptions"] == ["tickers.BTCUSDT", "tickers.ETHUSDT"]
    assert status["public"]["last_beat"] == manager.last_public_beat
    assert status["public"]["age_seconds"] == pytest.approx(12.0, abs=0.5)
    assert status["private"]["running"] is True
    assert status["private"]["last_beat"] == manager.last_private_beat
    assert status["private"]["age_seconds"] == pytest.approx(12.0, abs=0.5)


def test_ws_manager_status_without_heartbeat() -> None:
    manager = WSManager()
    status = manager.status()
    assert status["public"]["last_beat"] is None
    assert status["public"]["age_seconds"] is None
    assert status["public"]["subscriptions"] == []
    assert status["private"]["connected"] is False


def test_ws_manager_status_detects_inactive_channels() -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: False)
    manager._pub_ws = object()

    class DummyPrivate:
        def is_running(self) -> bool:
            return False

    manager._priv = DummyPrivate()

    status = manager.status()
    assert status["public"]["running"] is False
    assert status["private"]["running"] is False


def test_ws_manager_status_falls_back_to_private_ws_state() -> None:
    manager = WSManager()

    class DummyPrivate:
        _thread = SimpleNamespace(is_alive=lambda: False)
        _ws = object()

        def is_running(self) -> bool:
            return False

    manager._priv = DummyPrivate()

    status = manager.status()
    assert status["private"]["running"] is True


def test_ws_manager_status_uses_recent_beats(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.last_public_beat = 10_000.0
    manager.last_private_beat = 10_030.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 10_050.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["private"]["running"] is True


def test_ws_manager_autostart_respects_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    settings = SimpleNamespace(
        ws_autostart=True,
        ws_watchdog_max_age_sec=90,
        testnet=True,
        api_key="key",
        api_secret="secret",
    )

    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    manager.s = settings

    monkeypatch.setattr(
        manager,
        "status",
        lambda: {"public": {"running": False}, "private": {"running": False}},
    )

    calls: dict[str, object] = {}

    def fake_start_public(subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        calls["public"] = tuple(subs)
        return True

    def fake_start_private() -> bool:
        calls["private"] = True
        return True

    monkeypatch.setattr(manager, "start_public", fake_start_public)
    monkeypatch.setattr(manager, "start_private", fake_start_private)

    started_public, started_private = manager.autostart()

    assert started_public is True
    assert started_private is True
    assert calls["public"] == ("tickers.BTCUSDT",)
    assert calls["private"] is True


def test_ws_manager_autostart_returns_false_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    settings = SimpleNamespace(ws_autostart=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    manager.s = settings

    result = manager.autostart()
    assert result == (False, False)


def test_ws_manager_status_detects_connected_socket() -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: False)

    class DummySock:
        connected = True

    manager._pub_ws = SimpleNamespace(sock=DummySock())

    status = manager.status()
    assert status["public"]["running"] is True

def test_ws_manager_refreshes_settings_before_resolving_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(testnet=True)

    refreshed_public = SimpleNamespace(testnet=False)

    def fake_get_settings_public(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return refreshed_public

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings_public)

    pub_url = manager._public_url()
    assert pub_url.endswith("stream.bybit.com/v5/public/spot")
    assert manager.s is refreshed_public

    refreshed_private = SimpleNamespace(testnet=True)

    def fake_get_settings_private(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return refreshed_private

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings_private)

    priv_url = manager._private_url()
    assert priv_url.endswith("stream-testnet.bybit.com/v5/private")
    assert manager.s is refreshed_private


def test_start_private_uses_correct_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    captured: dict[str, object] = {}

    class DummyPrivate:
        def __init__(self, url: str, on_msg):
            captured["url"] = url
            captured["callback"] = on_msg

        def start(self) -> bool:
            captured["started"] = True
            return True

        def stop(self) -> None:  # pragma: no cover - parity with real class
            captured["stopped"] = True

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)
    calls: list[object] = []
    exec_calls: list[object] = []
    monkeypatch.setattr(manager.priv_store, "append", lambda payload: calls.append(payload))
    monkeypatch.setattr(pnl_module, "add_execution", lambda payload: exec_calls.append(payload))

    assert manager.start_private() is True
    assert captured["started"] is True
    callback = captured["callback"]
    assert callable(callback)

    sample_payload = {"topic": "execution", "data": [{"execQty": "1.0"}]}
    callback(sample_payload)

    assert calls == [sample_payload]
    assert manager.last_beat > 0
    assert exec_calls == [{"execQty": "1.0"}]
    execution = manager.latest_execution()
    assert execution is not None
    assert execution["execQty"] == "1.0"


def test_ws_manager_captures_order_update() -> None:
    manager = WSManager()
    payload = {
        "topic": "order",
        "data": [
            {
                "symbol": "BTCUSDT",
                "orderStatus": "Cancelled",
                "orderLinkId": "test-123",
                "cancelType": "INSUFFICIENT_BALANCE",
                "rejectReason": "INSUFFICIENT_BALANCE",
                "updatedTime": "1700000000000",
            }
        ],
    }

    manager._process_private_payload(payload)
    update = manager.latest_order_update()
    assert update is not None
    assert update["cancelType"] == "INSUFFICIENT_BALANCE"
    assert update["rejectReason"] == "INSUFFICIENT_BALANCE"
    assert update["updatedTime"] == "1700000000000"
    assert update["raw"]["orderStatus"] == "Cancelled"


def test_ws_manager_captures_execution_update() -> None:
    manager = WSManager()
    payload = {
        "topic": "execution",
        "data": [
            {
                "symbol": "ETHUSDT",
                "execQty": "0.5",
                "execPrice": "2000",
                "orderLinkId": "abc",
                "execTime": "1690000000000",
            }
        ],
    }

    manager._process_private_payload(payload)
    execution = manager.latest_execution()
    assert execution is not None
    assert execution["execQty"] == "0.5"
    assert execution["execPrice"] == "2000"
    assert execution["orderLinkId"] == "abc"
    assert execution["raw"]["symbol"] == "ETHUSDT"


def test_start_private_does_not_restart_running_client(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class DummyPrivate:
        def __init__(self, url: str, on_msg):  # pragma: no cover - used via manager
            self.url = url
            self.on_msg = on_msg
            self.started = 0
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def start(self) -> bool:
            self.started += 1
            self._running = True
            return True

        def stop(self) -> None:  # pragma: no cover - parity with real class
            self._running = False

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)

    assert manager.start_private() is True
    priv = manager._priv
    assert isinstance(priv, DummyPrivate)
    assert priv.started == 1

    assert manager.start_private() is True
    assert priv.started == 1  # second call should be a no-op


def test_start_private_handles_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class FailingPrivate:
        def __init__(self, url: str, on_msg):  # pragma: no cover - used via manager
            self.url = url
            self.on_msg = on_msg

        def start(self) -> bool:
            return False

        def stop(self) -> None:  # pragma: no cover - parity with real class
            pass

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", FailingPrivate)

    assert manager.start_private() is False
    assert manager._priv is None
    assert manager._priv_url is None


def test_start_private_restarts_when_environment_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    settings = SimpleNamespace(testnet=True)

    def fake_get_settings(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return settings

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings)

    events: list[str] = []

    class DummyPrivate:
        def __init__(self, url: str, on_msg):
            events.append(f"init:{url}")
            self.url = url
            self.on_msg = on_msg
            self.started = 0
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def start(self) -> bool:
            self.started += 1
            self._running = True
            return True

        def stop(self) -> None:
            events.append(f"stop:{self.url}")
            self._running = False

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)

    assert manager.start_private() is True
    assert events == ["init:wss://stream-testnet.bybit.com/v5/private"]

    priv_first = manager._priv
    assert isinstance(priv_first, DummyPrivate)
    assert priv_first.started == 1

    settings.testnet = False

    assert manager.start_private() is True
    assert events == [
        "init:wss://stream-testnet.bybit.com/v5/private",
        "stop:wss://stream-testnet.bybit.com/v5/private",
        "init:wss://stream.bybit.com/v5/private",
    ]

    priv_second = manager._priv
    assert isinstance(priv_second, DummyPrivate)
    assert priv_second is not priv_first
    assert priv_second.started == 1
    assert manager._priv_url == "wss://stream.bybit.com/v5/private"


def test_start_public_resubscribes_only_when_socket_connected() -> None:
    manager = WSManager()

    class DummyWS:
        def __init__(self, connected: bool) -> None:
            self.sock = SimpleNamespace(connected=connected)
            self.sent: list[dict] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    manager._pub_running = True

    ws_disconnected = DummyWS(False)
    manager._pub_ws = ws_disconnected
    assert manager.start_public(("tickers.ETHUSDT",)) is True
    assert ws_disconnected.sent == []

    ws_connected = DummyWS(True)
    manager._pub_ws = ws_connected
    assert manager.start_public(("tickers.XRPUSDT",)) is True
    assert ws_connected.sent == [
        {"op": "subscribe", "args": ["tickers.XRPUSDT"]}
    ]


def test_ws_private_v5_resubscribe_requires_connected_socket() -> None:
    ws_client = WSPrivateV5(reconnect=False)
    ws_client._thread = SimpleNamespace(is_alive=lambda: True)

    class DummyWS:
        def __init__(self, connected: bool) -> None:
            self.sock = SimpleNamespace(connected=connected)
            self.sent: list[dict] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    ws_client._ws = DummyWS(False)
    assert ws_client.start(["position"])
    assert ws_client._ws.sent == []

    ws_client._ws = DummyWS(True)
    assert ws_client.start(["wallet"])
    assert ws_client._ws.sent == [
        {"op": "subscribe", "args": list(DEFAULT_TOPICS)}
    ]
