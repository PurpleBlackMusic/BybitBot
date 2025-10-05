from __future__ import annotations

import pytest

from bybit_app.utils import time_sync


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"time": "1700000000000"}, 1_700_000_000.0),
        ({"serverTime": 1_680_000_000.0}, 1_680_000_000.0),
        ({"result": {"timeNano": "1690000000000000000"}}, 1_690_000_000.0),
    ),
)
def test_extract_server_epoch_parses_common_formats(payload, expected) -> None:
    assert time_sync.extract_server_epoch(payload) == pytest.approx(expected)


def test_extract_server_epoch_returns_none_for_invalid() -> None:
    assert time_sync.extract_server_epoch({"result": {"foo": "bar"}}) is None


class _DummyAPI:
    def __init__(self, payload):
        self._payload = payload

    def server_time(self):
        return self._payload


def _install_api(monkeypatch: pytest.MonkeyPatch, payload) -> None:
    monkeypatch.setattr(time_sync, "get_api_client", lambda: _DummyAPI(payload))


def test_check_time_drift_seconds_returns_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_700_000_000.5
    server_epoch = now - 2.25
    payload = {"result": {"timeNano": str(int(server_epoch * 1_000_000_000))}}

    _install_api(monkeypatch, payload)
    monkeypatch.setattr(time_sync.time, "time", lambda: now)
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(time_sync, "log", lambda event, **payload: events.append((event, payload)))

    drift = time_sync.check_time_drift_seconds()

    assert drift == pytest.approx(2.25)
    assert events == []


def test_check_time_drift_seconds_logs_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_700_000_100.0
    payload = {"result": {"unexpected": "field"}}

    _install_api(monkeypatch, payload)
    monkeypatch.setattr(time_sync.time, "time", lambda: now)
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(time_sync, "log", lambda event, **payload: events.append((event, payload)))

    assert time_sync.check_time_drift_seconds() == 0.0
    assert events == [("time.drift.error", {"err": "invalid payload", "payload": payload})]


def test_check_time_drift_seconds_handles_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomAPI:
        def server_time(self):  # pragma: no cover - exercised via time_sync
            raise RuntimeError("boom")

    monkeypatch.setattr(time_sync, "get_api_client", lambda: BoomAPI())
    monkeypatch.setattr(time_sync.time, "time", lambda: 1_700_000_000.0)
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(time_sync, "log", lambda event, **payload: events.append((event, payload)))

    assert time_sync.check_time_drift_seconds() == 0.0
    assert events == [("time.drift.error", {"err": "boom"})]
