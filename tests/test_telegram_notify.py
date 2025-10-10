from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from bybit_app.utils import telegram_notify


class _FakeResponse:
    def __init__(
        self,
        *,
        ok: bool = True,
        payload: dict[str, object] | None = None,
        status_code: int = 200,
        text: str = "",
        reason: str = "",
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": ok}
        self.text = text
        self.reason = reason

    def json(self) -> dict[str, object]:
        return self._payload


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        telegram_notify,
        "get_settings",
        lambda: SimpleNamespace(telegram_token="token", telegram_chat_id="chat"),
    )


def _make_dispatcher(**kwargs: object) -> telegram_notify.TelegramDispatcher:
    params = {
        "http_post": kwargs.get("http_post"),
        "rate_guard": kwargs.get("rate_guard", lambda: None),
        "max_attempts": kwargs.get("max_attempts", 5),
        "initial_backoff": kwargs.get("initial_backoff", 0.0),
        "max_backoff": kwargs.get("max_backoff", 0.0),
        "sleep": kwargs.get("sleep", lambda _delay: None),
    }
    if "queue_maxsize" in kwargs:
        params["queue_maxsize"] = kwargs["queue_maxsize"]
    return telegram_notify.TelegramDispatcher(**params)


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

    dispatcher = _make_dispatcher(http_post=fake_post, rate_guard=fake_rate_guard)

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

    dispatcher = _make_dispatcher(http_post=fake_post, rate_guard=fake_rate_guard)

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

    dispatcher = _make_dispatcher(http_post=fake_post, queue_maxsize=3)

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


def test_dispatcher_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    attempts: list[str] = []
    success_event = threading.Event()

    responses = [
        _FakeResponse(ok=False, payload={"ok": False, "description": "fail"}),
        _FakeResponse(ok=True, payload={"ok": True}),
    ]

    def fake_post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        attempts.append(json.get("text", ""))
        response = responses.pop(0)
        if response.ok:
            success_event.set()
        return response

    dispatcher = _make_dispatcher(http_post=fake_post)

    dispatcher.enqueue_message("retry-me")
    assert success_event.wait(timeout=1.0)

    dispatcher.shutdown(timeout=1.0)

    assert attempts == ["retry-me", "retry-me"]
    assert not responses


def test_dispatcher_marks_failed_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    recorded_logs: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **payload: object) -> None:
        recorded_logs.append((event, payload))

    monkeypatch.setattr(telegram_notify, "log", fake_log)

    def fake_post(url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        return _FakeResponse(ok=False, payload={"ok": False, "description": "boom"})

    dispatcher = _make_dispatcher(http_post=fake_post, max_attempts=3)

    dispatcher.enqueue_message("explode")
    dispatcher.shutdown(timeout=1.0)

    failure_logs = [payload for event, payload in recorded_logs if event == "telegram.delivery_failed"]
    assert len(failure_logs) == 1
    failure = failure_logs[0]
    assert failure["text"] == "explode"
    assert failure["attempts"] == 3
    assert failure["error"] == "boom"
