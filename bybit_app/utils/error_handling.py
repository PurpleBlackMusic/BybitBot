from __future__ import annotations

import asyncio
import sys
import threading
import traceback
import weakref
import time
from types import TracebackType
from typing import Any, Callable, Dict, Mapping, MutableMapping

from .log import log
from .telegram_notify import enqueue_telegram_message
from .background import restart_automation, restart_guardian, restart_websockets

_installed = False
_previous_sys_hook: Callable[[type[BaseException], BaseException, TracebackType | None], None] | None = None
_previous_thread_hook: Callable[[threading.ExceptHookArgs], None] | None = None
_original_new_event_loop: Callable[[], asyncio.AbstractEventLoop] | None = None
_policy_patched = False
_asyncio_previous_handlers: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Callable[[asyncio.AbstractEventLoop, Dict[str, object]], None] | None]" = weakref.WeakKeyDictionary()

_TELEGRAM_LIMIT = 3600
_TRACEBACK_TAIL = 1600
_RECOVERY_THROTTLE = 30.0
_last_recovery_attempt = 0.0


def install_global_exception_handlers(*, force: bool = False) -> None:
    """Install fail-safe exception handlers emitting Telegram alerts."""

    global _installed

    if _installed and not force:
        return

    if force:
        _restore_previous_hooks()

    _install_sys_hook()
    _install_threading_hook()
    _install_asyncio_hooks()

    _installed = True


def _restore_previous_hooks() -> None:
    global _installed, _policy_patched, _original_new_event_loop

    if _previous_sys_hook is not None:
        sys.excepthook = _previous_sys_hook

    if _previous_thread_hook is not None and hasattr(threading, "excepthook"):
        threading.excepthook = _previous_thread_hook

    for loop, previous in list(_asyncio_previous_handlers.items()):
        try:
            loop.set_exception_handler(previous)
        except Exception:
            pass
    _asyncio_previous_handlers.clear()

    if _policy_patched and _original_new_event_loop is not None:
        try:
            policy = asyncio.get_event_loop_policy()
            policy.new_event_loop = _original_new_event_loop  # type: ignore[assignment]
        except Exception:
            pass

    _policy_patched = False
    _original_new_event_loop = None
    _installed = False


def _install_sys_hook() -> None:
    global _previous_sys_hook

    _previous_sys_hook = sys.excepthook
    sys.excepthook = _handle_sys_exception


def _install_threading_hook() -> None:
    global _previous_thread_hook

    if not hasattr(threading, "excepthook"):
        _previous_thread_hook = None
        return

    _previous_thread_hook = threading.excepthook
    threading.excepthook = _handle_thread_exception  # type: ignore[assignment]


def _install_asyncio_hooks() -> None:
    global _policy_patched, _original_new_event_loop

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        _attach_asyncio_handler(loop)

    try:
        policy = asyncio.get_event_loop_policy()
    except Exception:
        return

    if _policy_patched:
        return

    original_new_loop = getattr(policy, "new_event_loop", None)
    if not callable(original_new_loop):
        return

    _original_new_event_loop = original_new_loop  # type: ignore[assignment]

    def _wrapped_new_loop() -> asyncio.AbstractEventLoop:
        new_loop = original_new_loop()
        _attach_asyncio_handler(new_loop)
        return new_loop

    try:
        policy.new_event_loop = _wrapped_new_loop  # type: ignore[assignment]
    except Exception:
        return

    _policy_patched = True


def _attach_asyncio_handler(loop: asyncio.AbstractEventLoop) -> None:
    if loop in _asyncio_previous_handlers:
        return

    previous = loop.get_exception_handler()
    _asyncio_previous_handlers[loop] = previous

    def _handler(current_loop: asyncio.AbstractEventLoop, context: Dict[str, object]) -> None:
        try:
            _handle_asyncio_exception(current_loop, context)
        finally:
            if previous is not None:
                try:
                    previous(current_loop, context)
                except Exception:
                    pass

    loop.set_exception_handler(_handler)


def _handle_sys_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    try:
        _process_exception(
            exc_type,
            exc_value,
            exc_traceback,
            origin="sys",
            thread_name=threading.current_thread().name,
        )
    finally:
        if _previous_sys_hook is not None:
            try:
                _previous_sys_hook(exc_type, exc_value, exc_traceback)
            except Exception:
                pass


def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
    try:
        _process_exception(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            origin="thread",
            thread_name=getattr(args.thread, "name", None),
        )
    finally:
        if _previous_thread_hook is not None:
            try:
                _previous_thread_hook(args)
            except Exception:
                pass


def _handle_asyncio_exception(
    loop: asyncio.AbstractEventLoop,
    context: MutableMapping[str, object],
) -> None:
    exc = context.get("exception")
    message = _coerce_str(context.get("message"))
    normalised_context = _normalise_context(context)

    if isinstance(exc, BaseException):
        _process_exception(
            type(exc),
            exc,
            exc.__traceback__,
            origin="asyncio",
            thread_name=threading.current_thread().name,
            message=message,
            context=normalised_context,
        )
    else:
        _log_asyncio_message(message, normalised_context)


def _log_asyncio_message(message: str | None, context: Mapping[str, object] | None) -> None:
    payload: Dict[str, object] = {"origin": "asyncio"}
    if message:
        payload["message"] = message
    if context:
        payload["context"] = context

    try:
        log("runtime.asyncio_error", **payload)
    except Exception:
        pass

    body_lines = ["🔥 Asyncio error detected"]
    if message:
        body_lines.append(f"Message: {message}")
    if context:
        body_lines.append(_format_context_lines(context))

    text = "\n".join(body_lines)
    _safe_notify(text)


def _process_exception(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_traceback: TracebackType | None,
    *,
    origin: str,
    thread_name: str | None = None,
    message: str | None = None,
    context: Mapping[str, object] | None = None,
) -> None:
    if exc_type is None or exc_value is None:
        return

    if _is_ignorable(exc_type):
        return

    payload: Dict[str, object] = {"origin": origin}
    if thread_name:
        payload["thread"] = thread_name
    if message:
        payload["message"] = message
    if context:
        payload["context"] = context

    try:
        log("runtime.unhandled_exception", exc=(exc_type, exc_value, exc_traceback), **payload)
    except Exception:
        pass

    text = _build_exception_message(
        exc_type,
        exc_value,
        exc_traceback,
        origin=origin,
        thread_name=thread_name,
        message=message,
        context=context,
    )
    _safe_notify(text)
    try:
        _maybe_trigger_recovery(origin, exc_value)
    except Exception:
        pass


def _safe_notify(text: str | None) -> None:
    if not text:
        return
    try:
        enqueue_telegram_message(text)
    except Exception:
        pass


def _maybe_trigger_recovery(origin: str, exc: BaseException) -> None:
    global _last_recovery_attempt

    try:
        now = time.monotonic()
    except Exception:  # pragma: no cover - monotonic edge cases
        return

    if now - _last_recovery_attempt < _RECOVERY_THROTTLE:
        return

    _last_recovery_attempt = now

    try:
        ws_restarted = bool(restart_websockets())
    except Exception as ws_exc:
        log("runtime.recovery.ws.error", err=str(ws_exc), origin=origin)
        ws_restarted = False

    try:
        automation_restarted = bool(restart_automation())
    except Exception as auto_exc:
        log("runtime.recovery.automation.error", err=str(auto_exc), origin=origin)
        automation_restarted = False

    try:
        guardian_restarted = bool(restart_guardian())
    except Exception as guardian_exc:
        log("runtime.recovery.guardian.error", err=str(guardian_exc), origin=origin)
        guardian_restarted = False

    log(
        "runtime.recovery",
        origin=origin,
        ws=ws_restarted,
        automation=automation_restarted,
        guardian=guardian_restarted,
        exc_type=type(exc).__name__,
    )


def _build_exception_message(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
    *,
    origin: str,
    thread_name: str | None,
    message: str | None,
    context: Mapping[str, object] | None,
) -> str:
    header = ["🔥 Unhandled exception"]
    details: list[str] = []

    if origin:
        details.append(f"origin={origin}")
    if thread_name:
        details.append(f"thread={thread_name}")
    if message:
        details.append(f"message={message}")

    if details:
        header.append("(" + ", ".join(details) + ")")

    headline = " ".join(header)

    exception_line = f"{exc_type.__name__}: {_coerce_str(exc_value)}"

    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)).strip()
    tb_text = _trim_traceback(tb_text)

    parts = [headline, exception_line]
    if context:
        parts.append(_format_context_lines(context))
    if tb_text:
        parts.append(tb_text)

    message_text = "\n\n".join(part for part in parts if part)
    return _truncate(message_text, _TELEGRAM_LIMIT)


def _format_context_lines(context: Mapping[str, object]) -> str:
    if not context:
        return ""

    lines = ["Context:"]
    for key, value in context.items():
        lines.append(f"- {key}: {_coerce_str(value)}")
    return "\n".join(lines)


def _trim_traceback(text: str) -> str:
    if not text:
        return ""

    if len(text) <= _TRACEBACK_TAIL:
        return text

    head = text[: _TRACEBACK_TAIL // 2].rstrip()
    tail = text[-_TRACEBACK_TAIL // 2 :].lstrip()
    return f"{head}\n...\n{tail}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        result = str(value)
    except Exception:
        result = "<unrepresentable>"
    if len(result) > 500:
        result = result[:497] + "…"
    return result


def _normalise_context(context: Mapping[str, object]) -> Dict[str, object] | None:
    if not context:
        return None

    cleaned: Dict[str, object] = {}
    for key, value in context.items():
        if key == "exception":
            continue
        cleaned[str(key)] = _coerce_context_value(value)
    return cleaned or None


def _coerce_context_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _coerce_context_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_context_value(item) for item in list(value)[:5]]
    return _coerce_str(value)


def _is_ignorable(exc_type: type[BaseException]) -> bool:
    try:
        return issubclass(exc_type, (KeyboardInterrupt, SystemExit))
    except Exception:
        return False

