from __future__ import annotations

import pytest

from datetime import timezone

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


def test_extract_server_datetime_returns_aware_dt() -> None:
    payload = {"serverTime": 1_680_000_000.0}
    result = time_sync.extract_server_datetime(payload)

    assert result is not None
    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == timezone.utc.utcoffset(result)
    assert result.timestamp() == pytest.approx(1_680_000_000.0)


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


class _DummyResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _DummySession:
    def __init__(self, response: _DummyResponse):
        self._response = response
        self.closed = False
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, *, timeout: float, verify: bool) -> _DummyResponse:
        self.requests.append((url, {"timeout": timeout, "verify": verify}))
        return self._response

    def close(self) -> None:
        self.closed = True


def _install_time(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 0.0, 0.1, 0.1])
    monkeypatch.setattr(time_sync.time, "time", lambda: next(times))


def test_synced_clock_refresh_closes_local_session(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"time": 1_000.0}
    response = _DummyResponse(payload)
    dummy_session = _DummySession(response)
    monkeypatch.setattr(time_sync, "create_session", lambda: dummy_session)
    monkeypatch.setattr(time_sync, "extract_server_epoch", lambda payload: 123.0)
    _install_time(monkeypatch)

    clock = time_sync._SyncedClock()
    clock._refresh("https://example.com", session=None, timeout=5.0, verify=True)

    assert dummy_session.closed is True


def test_synced_clock_refresh_keeps_external_session(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"time": 1_000.0}
    response = _DummyResponse(payload)
    external_session = _DummySession(response)

    def boom_session():  # pragma: no cover - ensures factory is unused
        raise AssertionError("requests.Session should not be called")

    monkeypatch.setattr(time_sync, "create_session", boom_session)
    monkeypatch.setattr(time_sync, "extract_server_epoch", lambda payload: 123.0)
    _install_time(monkeypatch)

    clock = time_sync._SyncedClock()
    clock._refresh("https://example.com", session=external_session, timeout=5.0, verify=False)

    assert external_session.closed is False
