from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from bybit_app.utils import background as background_module
from bybit_app.utils.background import BackgroundServices
from bybit_app.utils.signal_executor import ExecutionResult


class StubExecutor:
    def __init__(self) -> None:
        self._marker = (True, True, True)

    def current_signature(self) -> str:
        return "sig"

    def settings_marker(self) -> tuple[bool, bool, bool]:
        return self._marker

    def execute_once(self) -> ExecutionResult:
        return ExecutionResult(status="skipped")


class StubLoop:
    def __init__(self, executor, *, on_cycle, **_: object) -> None:
        self.executor = executor
        self.on_cycle = on_cycle
        self.run_calls = 0

    def run(self, stop_event=None) -> None:
        self.run_calls += 1
        if self.on_cycle is not None:
            self.on_cycle(ExecutionResult(status="skipped"), "sig", (True, True, True))
        if stop_event is not None:
            stop_event.set()


def _make_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    automation_stale_after: float = 0.5,
    ws_stub: SimpleNamespace | None = None,
) -> BackgroundServices:
    bot_factory = lambda: SimpleNamespace(settings=SimpleNamespace())
    executor_factory = lambda bot: StubExecutor()
    loop_factory = lambda executor, on_cycle: StubLoop(executor, on_cycle=on_cycle)

    if ws_stub is None:
        ws_stub = SimpleNamespace(
            start=lambda *args, **kwargs: True,
            status=lambda: {
                "public": {"running": True, "age_seconds": 0.0},
                "private": {"running": True, "age_seconds": 0.0},
            },
            stop_all=lambda: None,
            latest_order_update=lambda: {},
            latest_execution=lambda: {},
        )
    monkeypatch.setattr(background_module, "ws_manager", ws_stub)

    return BackgroundServices(
        bot_factory=bot_factory,
        executor_factory=executor_factory,
        loop_factory=loop_factory,
        public_stale_after=0.25,
        private_stale_after=0.25,
        automation_stale_after=automation_stale_after,
    )


def test_ws_restart_triggered_by_staleness(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_start(*_: object) -> bool:
        calls.append("start")
        return True

    stops: list[str] = []

    def fake_stop() -> None:
        stops.append("stop")

    statuses: list[dict] = [
        {
            "public": {"running": True, "age_seconds": 0.1},
            "private": {"running": True, "age_seconds": 0.1},
        }
    ]

    def fake_status() -> dict:
        return statuses[-1]

    ws_stub = SimpleNamespace(
        start=fake_start,
        status=fake_status,
        stop_all=fake_stop,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
    )

    svc = _make_service(monkeypatch, ws_stub=ws_stub)

    assert svc.ensure_ws_started() is True
    assert calls == ["start"]
    assert stops == []

    statuses.append(
        {
            "public": {"running": True, "age_seconds": 5.0},
            "private": {"running": True, "age_seconds": 0.2},
        }
    )

    assert svc.ensure_ws_started() is True
    assert calls == ["start", "start"]
    assert stops == ["stop"]

    snapshot = svc.ws_snapshot()
    assert snapshot["public_stale"] is True
    assert snapshot["restart_count"] == 2


def test_restart_ws_forces_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    start_calls: list[int] = []
    stop_calls: list[int] = []

    ws_stub = SimpleNamespace(
        start=lambda: (start_calls.append(1) or True),
        status=lambda: {
            "public": {"running": True, "age_seconds": 0.0},
            "private": {"running": True, "age_seconds": 0.0},
        },
        stop_all=lambda: stop_calls.append(1),
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
    )

    svc = _make_service(monkeypatch, ws_stub=ws_stub)
    svc.ensure_ws_started()
    assert start_calls == [1]
    assert stop_calls == []

    assert svc.restart_ws() is True
    assert stop_calls == [1]
    assert start_calls == [1, 1]


def test_automation_loop_restarts_on_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(monkeypatch, automation_stale_after=0.1)
    svc.ensure_automation_loop()
    assert svc._automation_restart_count == 1

    original_thread = svc._automation_thread
    if original_thread is not None and hasattr(original_thread, "join"):
        original_thread.join(timeout=1)

    fake_event = background_module.threading.Event()
    svc._automation_thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout=None: None)
    svc._automation_stop_event = fake_event
    svc._automation_last_cycle = time.time() - 1.0

    svc.ensure_automation_loop()
    assert fake_event.is_set() is True
    assert svc._automation_restart_count == 2


def test_automation_snapshot_marks_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(monkeypatch, automation_stale_after=0.2)
    svc._automation_last_cycle = time.time() - 1.0
    snapshot = svc.automation_snapshot()
    assert snapshot["stale"] is True
    assert snapshot["restart_count"] == 0


def test_ensure_started_waits_for_private_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    status_calls = 0

    def fake_status() -> dict:
        nonlocal status_calls
        status_calls += 1
        if status_calls < 3:
            return {
                "public": {"running": True, "age_seconds": 0.0},
                "private": {"running": False, "connected": False, "age_seconds": 120.0},
            }
        return {
            "public": {"running": True, "age_seconds": 0.0},
            "private": {
                "running": True,
                "connected": True,
                "age_seconds": 0.0,
                "last_beat": time.time(),
            },
        }

    ws_stub = SimpleNamespace(
        start=lambda *_, **__: True,
        status=fake_status,
        stop_all=lambda: None,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
    )

    svc = _make_service(monkeypatch, ws_stub=ws_stub)
    monkeypatch.setattr(background_module.time, "sleep", lambda _: None)

    svc.ensure_started()

    assert status_calls >= 3
    assert svc._automation_restart_count == 1

    thread = svc._automation_thread
    if thread is not None and hasattr(thread, "join"):
        thread.join(timeout=1)


def test_ensure_started_skips_automation_without_private(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(monkeypatch)
    monkeypatch.setattr(svc, "_await_private_ready", lambda timeout=None: False)

    svc.ensure_started()

    assert svc._automation_restart_count == 0
