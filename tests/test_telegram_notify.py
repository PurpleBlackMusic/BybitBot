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
