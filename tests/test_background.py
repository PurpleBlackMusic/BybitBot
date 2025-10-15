from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Callable

import pytest

from bybit_app.utils import background as background_module
from bybit_app.utils import hygiene as hygiene_module
from bybit_app.utils.background import BackgroundServices
from bybit_app.utils.signal_executor import ExecutionResult
from bybit_app.utils.ws_events import (
    event_queue_stats,
    fetch_events,
    publish_event,
    reset_event_queue,
)


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
    loop_factory: Callable[[StubExecutor, Callable[..., None]], StubLoop] | None = None,
) -> BackgroundServices:
    if loop_factory is None:
        loop_factory = lambda executor, on_cycle: StubLoop(executor, on_cycle=on_cycle)

    bot_factory = lambda: SimpleNamespace(settings=SimpleNamespace())
    executor_factory = lambda bot: StubExecutor()

    if ws_stub is None:
        ws_stub = SimpleNamespace(
            start=lambda *args, **kwargs: True,
            start_private=lambda *args, **kwargs: True,
            status=lambda: {
                "public": {"running": True, "age_seconds": 0.0},
                "private": {"running": True, "age_seconds": 0.0},
            },
            stop_all=lambda: None,
            stop_private=lambda: None,
            force_public_fallback=lambda *args, **kwargs: False,
            latest_order_update=lambda: {},
            latest_execution=lambda: {},
            private_events=lambda *, since=None, limit=None: fetch_events(
                scope="private", since=since, limit=limit
            ),
            private_event_stats=lambda: event_queue_stats(),
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

    fallback_calls: list[tuple[str, dict[str, object]]] = []

    def fake_force_public_fallback(reason: str, **payload: object) -> bool:
        fallback_calls.append((reason, dict(payload)))
        return True

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
        start_private=lambda: True,
        status=fake_status,
        stop_all=fake_stop,
        stop_private=lambda: None,
        force_public_fallback=fake_force_public_fallback,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
        private_events=lambda *args, **kwargs: [],
        private_event_stats=lambda: {"latest_id": 0, "size": 0, "dropped": 0},
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
    assert len(fallback_calls) == 1
    fallback_reason, fallback_payload = fallback_calls[0]
    assert fallback_reason == "public_channel_stale"
    assert fallback_payload["threshold"] == 0.25
    assert fallback_payload["age"] == pytest.approx(5.0, rel=0, abs=1e-6)

    snapshot = svc.ws_snapshot()
    assert snapshot["public_stale"] is True
    assert snapshot["restart_count"] == 2


def test_order_sweeper_cancels_old_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(monkeypatch)

    class _FakeTime:
        def __init__(self, value: float) -> None:
            self.value = value

        def time(self) -> float:
            return self.value

    fake_time = _FakeTime(10_000.0)
    monkeypatch.setattr(background_module, "time", fake_time)
    monkeypatch.setattr(hygiene_module, "time", fake_time)

    class _SweeperApi:
        def __init__(self) -> None:
            self.cancelled: list[list[dict[str, object]]] = []

        def open_orders(self, *, category: str, symbol: str | None = None, openOnly: int = 1):
            del category, openOnly
            stale_created = int((fake_time.value - 1_200.0) * 1000)
            fresh_created = int((fake_time.value - 100.0) * 1000)
            return {
                "result": {
                    "list": [
                        {
                            "symbol": symbol or "BTCUSDT",
                            "orderId": "1",
                            "orderLinkId": "AI-TP-BTC-1",
                            "orderType": "Limit",
                            "timeInForce": "GTC",
                            "createdTime": str(stale_created),
                        },
                        {
                            "symbol": symbol or "BTCUSDT",
                            "orderId": "2",
                            "orderLinkId": "MANUAL-ORDER",
                            "orderType": "Limit",
                            "timeInForce": "GTC",
                            "createdTime": str(stale_created),
                        },
                        {
                            "symbol": symbol or "BTCUSDT",
                            "orderId": "3",
                            "orderLinkId": "AI-TP-BTC-2",
                            "orderType": "Limit",
                            "timeInForce": "GTC",
                            "createdTime": str(fresh_created),
                        },
                    ]
                }
            }

        def cancel_batch(self, *, category: str, request: list[dict[str, object]]):
            del category
            self.cancelled.append(request)
            return {"retCode": 0}

    api = _SweeperApi()
    monkeypatch.setattr(background_module, "get_api_client", lambda: api)
    monkeypatch.setattr(background_module, "log", lambda *_, **__: None)

    assert svc._maybe_sweep_orders() is True
    assert api.cancelled
    cancelled_ids = {entry.get("orderId") for entry in api.cancelled[0]}
    assert cancelled_ids == {"1"}

    snapshot = svc.automation_snapshot()
    sweeper_info = snapshot.get("sweeper")
    assert sweeper_info
    assert sweeper_info["cancelled"] == 1
    assert sweeper_info["batches"] == 1


def test_private_ws_stale_triggers_targeted_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_calls: list[str] = []
    private_start_calls: list[str] = []
    private_stops: list[str] = []

    def fake_start() -> bool:
        start_calls.append("start")
        return True

    def fake_start_private() -> bool:
        private_start_calls.append("start_private")
        return True

    def fake_stop_private() -> None:
        private_stops.append("stop_private")

    statuses: list[dict[str, object]] = [
        {
            "public": {"running": True, "age_seconds": 0.0},
            "private": {"running": True, "age_seconds": 0.0},
        }
    ]

    def fake_status() -> dict[str, object]:
        return statuses[-1]

    fallback_calls: list[int] = []

    def fake_force_public_fallback(*_: object, **__: object) -> bool:
        fallback_calls.append(1)
        return True

    ws_stub = SimpleNamespace(
        start=fake_start,
        start_private=fake_start_private,
        status=fake_status,
        stop_all=lambda: None,
        stop_private=fake_stop_private,
        force_public_fallback=fake_force_public_fallback,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
    )

    svc = _make_service(monkeypatch, ws_stub=ws_stub)

    assert svc.ensure_ws_started() is True
    assert start_calls == ["start"]
    assert private_start_calls == []

    statuses.append(
        {
            "public": {"running": True, "age_seconds": 0.0},
            "private": {"running": True, "age_seconds": 5.0},
        }
    )

    assert svc.ensure_ws_started() is True
    assert private_stops == ["stop_private"]
    assert private_start_calls == ["start_private"]
    assert start_calls == ["start"]
    assert fallback_calls == []
    assert svc._ws_restart_count == 1


def test_restart_ws_forces_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    start_calls: list[int] = []
    stop_calls: list[int] = []

    ws_stub = SimpleNamespace(
        start=lambda: (start_calls.append(1) or True),
        start_private=lambda: True,
        status=lambda: {
            "public": {"running": True, "age_seconds": 0.0},
            "private": {"running": True, "age_seconds": 0.0},
        },
        stop_all=lambda: stop_calls.append(1),
        stop_private=lambda: None,
        force_public_fallback=lambda *_, **__: False,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
        private_events=lambda *args, **kwargs: [],
        private_event_stats=lambda: {"latest_id": 0, "size": 0, "dropped": 0},
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

    class FakeThread:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self._alive = False

    fake_event = background_module.threading.Event()
    svc._automation_thread = FakeThread()
    svc._automation_stop_event = fake_event
    svc._automation_last_cycle = time.time() - 1.0

    svc.ensure_automation_loop()
    assert fake_event.is_set() is True
    assert svc._automation_restart_count == 2


def test_automation_force_waits_for_previous_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running_event = threading.Event()
    waiting_event = threading.Event()
    release_event = threading.Event()

    class BlockingLoop:
        def __init__(self, executor, *, on_cycle, **_: object) -> None:
            self.executor = executor
            self.on_cycle = on_cycle
            self.run_calls = 0

        def run(self, stop_event=None) -> None:
            self.run_calls += 1
            running_event.set()
            if stop_event is not None and self.run_calls == 1:
                while not stop_event.is_set():
                    time.sleep(0.01)
                waiting_event.set()
                release_event.wait(timeout=1.0)
            if self.on_cycle is not None:
                self.on_cycle(ExecutionResult(status="skipped"), "sig", (True, True, True))
            if stop_event is not None:
                stop_event.set()

    svc = _make_service(
        monkeypatch,
        loop_factory=lambda executor, on_cycle: BlockingLoop(executor, on_cycle=on_cycle),
    )

    svc.ensure_automation_loop()
    initial_thread = svc._automation_thread
    assert initial_thread is not None
    assert running_event.wait(timeout=1.0)
    initial_executor = svc._automation_executor
    assert initial_executor is not None

    restart_done = threading.Event()

    def trigger_restart() -> None:
        svc.ensure_automation_loop(force=True)
        restart_done.set()

    restart_thread = threading.Thread(target=trigger_restart)
    restart_thread.start()

    assert waiting_event.wait(timeout=1.0)
    time.sleep(0.05)

    assert svc._automation_restart_count == 1
    assert svc._automation_thread is initial_thread

    running_event.clear()
    release_event.set()
    restart_thread.join(timeout=1.0)
    assert restart_done.is_set() is True

    new_thread = svc._automation_thread
    assert new_thread is not None and new_thread is not initial_thread
    assert svc._automation_restart_count == 2
    assert svc._automation_executor is initial_executor
    assert running_event.wait(timeout=1.0)

    stop_event = svc._automation_stop_event
    if stop_event is not None:
        stop_event.set()
    new_thread.join(timeout=1.0)

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
        start_private=lambda: True,
        status=fake_status,
        stop_all=lambda: None,
        stop_private=lambda: None,
        force_public_fallback=lambda *_, **__: False,
        latest_order_update=lambda: {},
        latest_execution=lambda: {},
        private_events=lambda *args, **kwargs: [],
        private_event_stats=lambda: {"latest_id": 0, "size": 0, "dropped": 0},
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


def test_ws_events_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service(monkeypatch)
    reset_event_queue()
    publish_event(scope="private", topic="orders", payload={"id": 1})
    publish_event(scope="private", topic="executions", payload={"id": 2})

    events_payload = svc.ws_events()
    assert isinstance(events_payload, dict)
    assert events_payload["events"]
    assert events_payload["events"][-1]["topic"] == "executions"
    assert isinstance(events_payload["stats"], dict)
