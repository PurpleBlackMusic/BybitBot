from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from bybit_app.utils import telegram_notify


class _FakeResponse:
    def __init__(self) -> None:
        self.ok = True
        self.status_code = 200

    def json(self) -> dict[str, object]:
        return {"ok": True}


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        telegram_notify,
        "get_settings",
        lambda: SimpleNamespace(telegram_token="token", telegram_chat_id="chat"),
    )


def test_dispatcher_enqueue_is_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    sleep_calls: list[int] = []

    def fake_rate_guard() -> None:
        sleep_calls.append(threading.get_ident())
        time.sleep(0.05)

    delivered: list[str] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        delivered.append(json.get("text", ""))
        return _FakeResponse()

    dispatcher = telegram_notify.TelegramDispatcher(
        http_post=fake_post,
        rate_guard=fake_rate_guard,
    )

    start = time.perf_counter()
    dispatcher.enqueue_message("one")
    dispatcher.enqueue_message("two")
    elapsed = time.perf_counter() - start

    # The caller should not wait for rate limiting sleeps executed in the worker thread.
    assert elapsed < 0.04

    dispatcher.shutdown(timeout=1.0)

    assert delivered == ["one", "two"]

    main_thread_id = threading.get_ident()
    assert sleep_calls and all(call != main_thread_id for call in sleep_calls)


def test_dispatcher_respects_rate_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    state = {"last": None}
    send_times: list[float] = []

    def fake_rate_guard() -> None:
        now = time.perf_counter()
        last = state["last"]
        if last is not None:
            wait = 0.02 - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.perf_counter()
        state["last"] = now

    def fake_post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        send_times.append(time.perf_counter())
        return _FakeResponse()

    dispatcher = telegram_notify.TelegramDispatcher(
        http_post=fake_post,
        rate_guard=fake_rate_guard,
    )

    dispatcher.enqueue_message("first")
    dispatcher.enqueue_message("second")
    dispatcher.enqueue_message("third")

    dispatcher.shutdown(timeout=1.0)

    assert len(send_times) == 3
    gaps = [b - a for a, b in zip(send_times, send_times[1:])]
    assert all(gap >= 0.018 for gap in gaps)


def test_dispatcher_drops_oldest_when_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    recorded_logs: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **payload: object) -> None:
        recorded_logs.append((event, payload))

    monkeypatch.setattr(telegram_notify, "log", fake_log)

    block_event = threading.Event()
    processing_event = threading.Event()
    sent: list[str] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        processing_event.set()
        if not block_event.is_set():
            block_event.wait(timeout=1.0)
        sent.append(str(json.get("text", "")))
        return _FakeResponse()

    dispatcher = telegram_notify.TelegramDispatcher(
        http_post=fake_post,
        rate_guard=lambda: None,
        queue_maxsize=3,
    )

    dispatcher.enqueue_message("msg-0")
    assert processing_event.wait(timeout=1.0)

    for idx in range(1, 6):
        dispatcher.enqueue_message(f"msg-{idx}")

    block_event.set()
    dispatcher.shutdown(timeout=2.0)

    overflow_events = [payload for event, payload in recorded_logs if event == "telegram.queue_overflow"]

    assert len(overflow_events) == 2
    assert [payload["dropped_message"] for payload in overflow_events] == ["msg-1", "msg-2"]
    assert [payload["dropped_total"] for payload in overflow_events] == [1, 2]
    assert all(payload["queue_maxsize"] == 3 for payload in overflow_events)

    # Messages newer than the dropped ones must still be delivered without leaks.
    assert set(sent) == {"msg-0", "msg-3", "msg-4", "msg-5"}
