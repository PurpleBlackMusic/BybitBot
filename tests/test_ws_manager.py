from __future__ import annotations

import pytest

from bybit_app.utils import ws_manager as ws_manager_module
from bybit_app.utils.ws_manager import WSManager


def test_ws_manager_status_reports_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager._pub_ws = object()
    manager._pub_subs = ("tickers.BTCUSDT", "tickers.ETHUSDT")
    manager._priv = object()
    manager.last_beat = 1_000_000.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 1_000_012.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["public"]["subscriptions"] == ["tickers.BTCUSDT", "tickers.ETHUSDT"]
    assert status["public"]["last_beat"] == manager.last_beat
    assert status["public"]["age_seconds"] == pytest.approx(12.0, abs=0.5)
    assert status["private"]["running"] is True
    assert status["private"]["age_seconds"] == pytest.approx(12.0, abs=0.5)


def test_ws_manager_status_without_heartbeat() -> None:
    manager = WSManager()
    status = manager.status()
    assert status["public"]["last_beat"] is None
    assert status["public"]["age_seconds"] is None
    assert status["public"]["subscriptions"] == []
    assert status["private"]["connected"] is False


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
    monkeypatch.setattr(manager.priv_store, "append", lambda payload: calls.append(payload))

    assert manager.start_private() is True
    assert captured["started"] is True
    callback = captured["callback"]
    assert callable(callback)

    sample_payload = {"foo": "bar"}
    callback(sample_payload)

    assert calls == [sample_payload]
    assert manager.last_beat > 0


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
