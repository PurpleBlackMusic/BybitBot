from __future__ import annotations

import atexit
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import requests

from .envs import get_settings
from .log import log


class TelegramDispatcher:
    """Asynchronous dispatcher for Telegram notifications."""

    DEFAULT_QUEUE_MAXSIZE = 200

    def __init__(
        self,
        http_post: Callable[..., object] | None = None,
        rate_guard: Callable[[], None] | None = None,
        queue_maxsize: int | None = None,
        *,
        max_attempts: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._http_post = http_post or requests.post
        self._rate_guard = rate_guard or _rate_guard
        maxsize = queue_maxsize or self.DEFAULT_QUEUE_MAXSIZE
        if maxsize <= 0:
            raise ValueError("queue_maxsize must be positive")
        self._queue_maxsize = maxsize
        self._queue: "queue.Queue[tuple[str, int] | None]" = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._overflow_lock = threading.Lock()
        self._dropped_messages = 0
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if initial_backoff < 0:
            raise ValueError("initial_backoff must be non-negative")
        if max_backoff < 0:
            raise ValueError("max_backoff must be non-negative")
        if initial_backoff > max_backoff and max_backoff != 0:
            raise ValueError("initial_backoff must be <= max_backoff or max_backoff must be zero")
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._sleep = sleep or time.sleep

    def enqueue_message(self, text: str) -> None:
        """Schedule message for asynchronous delivery without blocking."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self._ensure_thread()
        self._put_with_overflow_handling((text, 0))

    def shutdown(self, *, timeout: float | None = None) -> None:
        """Stop dispatcher thread waiting for outstanding work."""

        thread = self._thread
        if not thread:
            return

        self._stop_event.set()
        self._put_with_overflow_handling(None, record_overflow=False, allow_drop=False)
        thread.join(timeout=timeout)

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(target=self._run, name="telegram-dispatcher", daemon=True)
            self._thread = thread
            thread.start()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                if self._stop_event.is_set() and self._queue.empty():
                    break
                continue

            self._deliver(item)

        # Drain leftover sentinel items to keep queue consistent.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _deliver(self, item: tuple[str, int]) -> None:
        text, attempts = item
        try:
            self._rate_guard()
            result = _send_telegram_http(text, http_post=self._http_post)
        except Exception as exc:  # pragma: no cover - defensive catch for worker loop
            log("telegram.error", error=str(exc))
            self._handle_delivery_failure(text, attempts, error=str(exc))
            return

        ok = bool(getattr(result, "get", lambda *_: False)("ok")) if result else False
        if not ok:
            error: str | None = None
            if isinstance(result, dict):
                error_value = result.get("error") or result.get("description")
                if error_value:
                    error = str(error_value)
            if error is None:
                error = "telegram_delivery_failed"
            self._handle_delivery_failure(text, attempts, error=error)

    def _handle_delivery_failure(self, text: str, attempts: int, *, error: str) -> None:
        next_attempt = attempts + 1
        if next_attempt >= self._max_attempts:
            log(
                "telegram.delivery_failed",
                text=text,
                attempts=next_attempt,
                error=error,
            )
            return

        delay = 0.0
        if self._initial_backoff > 0:
            delay = self._initial_backoff * (2**attempts)
            if self._max_backoff > 0:
                delay = min(delay, self._max_backoff)
        if delay > 0:
            self._sleep(delay)

        self._put_with_overflow_handling((text, next_attempt))

    def _put_with_overflow_handling(
        self,
        item: tuple[str, int] | None,
        *,
        record_overflow: bool = True,
        allow_drop: bool = True,
    ) -> None:
        while True:
            try:
                if allow_drop:
                    self._queue.put_nowait(item)
                else:
                    self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if not allow_drop:
                    continue

                dropped: tuple[str, int] | None
                try:
                    dropped = self._queue.get_nowait()
                except queue.Empty:  # pragma: no cover - defensive guard
                    time.sleep(0)
                    continue

                if dropped is None:
                    # Sentinel should stay at the back of the queue. Skip logging.
                    continue

                if record_overflow:
                    self._record_overflow(dropped)

    def _record_overflow(self, dropped: tuple[str, int]) -> None:
        with self._overflow_lock:
            self._dropped_messages += 1
            total = self._dropped_messages

        log(
            "telegram.queue_overflow",
            dropped_message=dropped[0],
            dropped_total=total,
            queue_maxsize=self._queue_maxsize,
        )


def send_telegram(text: str):
    """Send Telegram message synchronously respecting rate limits."""

    _rate_guard()
    return _send_telegram_http(text)


# Soft rate limiting to respect Telegram Bot API guidance
# - ~1 msg/second per chat; ~20 msg/minute in groups; ~30 msg/sec global
# We implement per-chat guard (simple) to avoid spamming when loops misbehave.
RATE_STATE = {"last_ts": 0.0, "window": deque(maxlen=60)}  # last 60 timestamps
_RATE_LOCK = threading.Lock()


def _rate_guard() -> None:
    while True:
        with _RATE_LOCK:
            now = time.time()

            # purge old timestamps first
            window = RATE_STATE["window"]
            while window and now - window[0] > 60.0:
                window.popleft()

            wait_for = 0.0
            gap = now - RATE_STATE["last_ts"]
            if gap < 1.0:
                wait_for = max(wait_for, 1.0 - gap)

            if len(window) >= 20 and window:
                wait_for = max(wait_for, 60.0 - (now - window[0]))

            if wait_for <= 0.0:
                RATE_STATE["last_ts"] = now
                window.append(now)
                return

        time.sleep(wait_for)


def _send_telegram_http(text: str, http_post: Callable[..., object] | None = None):
    http = http_post or requests.post

    s = get_settings()
    if not s.telegram_token or not s.telegram_chat_id:
        return {"ok": False, "error": "telegram_not_configured"}

    url = f"https://api.telegram.org/bot{s.telegram_token}/sendMessage"
    payload = {"chat_id": s.telegram_chat_id, "text": text}

    try:
        response = http(url, json=payload, timeout=10)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        log("telegram.error", error=str(exc))
        return {"ok": False, "error": str(exc)}

    response_data: dict | None
    try:
        response_data = response.json()
    except ValueError:
        response_data = None

    success = getattr(response, "ok", False) and not (
        isinstance(response_data, dict) and response_data.get("ok") is False
    )

    if not success:
        description = ""
        if isinstance(response_data, dict):
            description = str(
                response_data.get("description")
                or response_data.get("error")
                or ""
            )
        if not description:
            description = getattr(response, "text", "") or getattr(response, "reason", "")
        log(
            "telegram.error",
            status=getattr(response, "status_code", None),
            error=description,
        )
        return {
            "ok": False,
            "status": getattr(response, "status_code", None),
            "error": description,
            "response": response_data,
        }

    log("telegram.send", status=getattr(response, "status_code", None))
    return {"ok": True, "status": getattr(response, "status_code", None), "response": response_data}


dispatcher = TelegramDispatcher()


def enqueue_telegram_message(text: str) -> None:
    dispatcher.enqueue_message(text)


def shutdown_telegram_dispatcher(*, timeout: float | None = None) -> None:
    dispatcher.shutdown(timeout=timeout)


atexit.register(dispatcher.shutdown)
