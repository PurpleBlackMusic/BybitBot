import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from bybit_app.utils import error_handling


def _collectors(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[str, dict[str, object]]], list[str]]:
    recorded_logs: list[tuple[str, dict[str, object]]] = []
    telegram_messages: list[str] = []

    def fake_log(event: str, **payload: object) -> None:
        recorded_logs.append((event, payload))

    def fake_notify(text: str) -> None:
        telegram_messages.append(text)

    monkeypatch.setattr(error_handling, "log", fake_log)
    monkeypatch.setattr(error_handling, "enqueue_telegram_message", fake_notify)

    return recorded_logs, telegram_messages


def test_sys_excepthook_sends_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    logs, messages = _collectors(monkeypatch)
    monkeypatch.setattr(error_handling, "_previous_sys_hook", lambda *args, **kwargs: None)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_type, exc_value, tb = sys.exc_info()

    assert exc_type and exc_value

    error_handling._handle_sys_exception(exc_type, exc_value, tb)

    assert logs, "log entry missing for unhandled exception"
    event, payload = logs[0]
    assert event == "runtime.unhandled_exception"
    assert payload["origin"] == "sys"

    assert messages and "RuntimeError" in messages[0]


def test_threading_excepthook_sends_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    logs, messages = _collectors(monkeypatch)
    monkeypatch.setattr(error_handling, "_previous_thread_hook", lambda *_: None)

    try:
        raise ValueError("thread fail")
    except ValueError:
        exc_type, exc_value, tb = sys.exc_info()

    assert exc_type and exc_value

    args = SimpleNamespace(
        exc_type=exc_type,
        exc_value=exc_value,
        exc_traceback=tb,
        thread=threading.current_thread(),
    )

    error_handling._handle_thread_exception(args)

    assert logs and logs[0][0] == "runtime.unhandled_exception"
    assert logs[0][1]["origin"] == "thread"
    assert messages and "ValueError" in messages[0]


def test_asyncio_handler_handles_context(monkeypatch: pytest.MonkeyPatch) -> None:
    logs, messages = _collectors(monkeypatch)

    loop = asyncio.new_event_loop()
    try:
        handler = loop.get_exception_handler()
        assert handler is not None

        context = {"message": "timer task crashed", "future": object()}
        handler(loop, context)
    finally:
        loop.close()

    assert logs and logs[0][0] == "runtime.asyncio_error"
    assert logs[0][1]["origin"] == "asyncio"
    assert messages and "Asyncio error" in messages[0]
