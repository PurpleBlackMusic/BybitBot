from __future__ import annotations

import atexit
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import requests

try:
    from tenacity import (
        RetryError,
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
        wait_none,
        wait_random,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback path
    import random

    class _FallbackAttempt:
        """Lightweight attempt state compatible with :class:`RetryError`."""

        def __init__(self, attempt_number: int, exc: Exception | None) -> None:
            self.attempt_number = attempt_number
            self._exception = exc

        def failed(self) -> bool:
            return self._exception is not None

        def exception(self) -> Exception | None:
            return self._exception

    class RetryError(RuntimeError):
        """Fallback ``RetryError`` mirroring the subset used by the app."""

        def __init__(self, last_attempt: _FallbackAttempt) -> None:
            self.last_attempt = last_attempt
            message = f"Retry failed after {last_attempt.attempt_number} attempts"
            if last_attempt.exception() is not None:
                message = f"{message}: {last_attempt.exception()}"
            super().__init__(message)

    class _WaitStrategy:
        def __call__(self, attempt: int) -> float:  # pragma: no cover - simple fallback
            return 0.0

        def __add__(self, other: "_WaitStrategy") -> "_WaitStrategy":
            return _CompositeWait((self, other))

    class _CompositeWait(_WaitStrategy):
        def __init__(self, strategies: tuple[_WaitStrategy, ...]) -> None:
            self._strategies = strategies

        def __call__(self, attempt: int) -> float:
            total = 0.0
            for strategy in self._strategies:
                try:
                    total += float(strategy(attempt))
                except Exception:
                    continue
            return max(0.0, total)

        def __add__(self, other: "_WaitStrategy") -> "_WaitStrategy":
            return _CompositeWait(self._strategies + (other,))

    class _WaitNone(_WaitStrategy):
        def __call__(self, attempt: int) -> float:
            return 0.0

    class _WaitRandom(_WaitStrategy):
        def __init__(self, lower: float, upper: float) -> None:
            self._lower = float(lower)
            self._upper = float(max(upper, lower))

        def __call__(self, attempt: int) -> float:
            return random.uniform(self._lower, self._upper)

    class _WaitExponential(_WaitStrategy):
        def __init__(self, multiplier: float, minimum: float, maximum: float | None) -> None:
            self._multiplier = float(multiplier)
            self._minimum = float(minimum)
            self._maximum = float(maximum) if maximum is not None else None

        def __call__(self, attempt: int) -> float:
            delay = self._multiplier * (2 ** max(attempt - 1, 0))
            delay = max(self._minimum, delay)
            if self._maximum is not None:
                delay = min(delay, self._maximum)
            return max(0.0, delay)

    class Retrying:
        """Minimal retry helper used when ``tenacity`` is unavailable."""

        def __init__(
            self,
            *,
            stop,
            wait,
            retry,
            sleep: Callable[[float], None] | None = None,
            reraise: bool = True,
            before_sleep=None,
        ) -> None:
            self._stop = stop
            self._wait = wait
            self._retry = retry
            self._sleep = sleep or time.sleep
            self._reraise = bool(reraise)
            self._before_sleep = before_sleep

        def __call__(self, fn: Callable[..., object], *args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - exercised indirectly
                    attempt += 1
                    should_retry = True
                    if self._retry is not None:
                        try:
                            should_retry = bool(self._retry(exc))
                        except Exception:
                            should_retry = False
                    if not should_retry:
                        if self._reraise:
                            raise
                        raise RetryError(_FallbackAttempt(attempt, exc)) from exc

                    stop_now = False
                    if self._stop is not None:
                        try:
                            stop_now = bool(self._stop(attempt))
                        except Exception:
                            stop_now = True
                    if stop_now:
                        if self._reraise:
                            raise
                        raise RetryError(_FallbackAttempt(attempt, exc)) from exc

                    delay = 0.0
                    if self._wait is not None:
                        try:
                            delay = float(self._wait(attempt))
                        except Exception:
                            delay = 0.0
                    if self._before_sleep is not None:
                        try:
                            self._before_sleep(_FallbackAttempt(attempt, exc))
                        except Exception:
                            pass
                    if delay > 0:
                        self._sleep(delay)

    def retry_if_exception_type(exc_type):
        if not isinstance(exc_type, tuple):
            exc_tuple = (exc_type,)
        else:
            exc_tuple = exc_type

        def _predicate(exc: Exception) -> bool:
            return isinstance(exc, exc_tuple)

        return _predicate

    def stop_after_attempt(limit: int):
        limit_value = max(int(limit), 1)

        def _stopper(attempt: int) -> bool:
            return attempt >= limit_value

        return _stopper

    def wait_none() -> _WaitStrategy:
        return _WaitNone()

    def wait_random(lower: float = 0.0, upper: float = 1.0) -> _WaitStrategy:
        return _WaitRandom(lower, upper)

    def wait_exponential(
        *,
        multiplier: float = 1.0,
        min: float = 0.0,
        max: float | None = None,
    ) -> _WaitStrategy:
        return _WaitExponential(multiplier, min, max)

from .envs import get_settings
from .log import log

_DEFAULT_SHUTDOWN_TIMEOUT = 1.0


class _RetryableTelegramError(RuntimeError):
    """Internal marker indicating a transient Telegram delivery failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


_REQUEST_SEMAPHORE = threading.BoundedSemaphore(value=1)


def _now() -> float:
    return time.time()


_LAST_ACTIVITY_LOCK = threading.Lock()
_LAST_ACTIVITY_TS = _now()


def _record_activity(ts: float | None = None) -> None:
    """Record the timestamp of the latest successful Telegram delivery."""

    with _LAST_ACTIVITY_LOCK:
        global _LAST_ACTIVITY_TS
        _LAST_ACTIVITY_TS = float(ts) if ts is not None else _now()


def _last_activity() -> float:
    with _LAST_ACTIVITY_LOCK:
        return _LAST_ACTIVITY_TS


def _acquire_request_slot() -> threading.Semaphore:
    _REQUEST_SEMAPHORE.acquire()
    return _REQUEST_SEMAPHORE


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
        self._wait_strategy = self._build_wait_strategy()

    def _build_wait_strategy(self):
        if self._initial_backoff <= 0:
            return wait_none()

        max_backoff: float | None = self._max_backoff if self._max_backoff > 0 else None
        base = wait_exponential(
            multiplier=self._initial_backoff,
            min=self._initial_backoff,
            max=max_backoff,
        )
        jitter_upper = min(0.25, self._initial_backoff)
        return base + wait_random(0, jitter_upper)

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

        join_timeout = timeout if timeout is not None else _DEFAULT_SHUTDOWN_TIMEOUT

        self._stop_event.set()
        self._put_with_overflow_handling(None, record_overflow=False, allow_drop=False)
        thread.join(timeout=join_timeout)

        if thread.is_alive():  # pragma: no cover - defensive
            log(
                "telegram.dispatcher.shutdown_timeout",
                timeout=join_timeout,
                pending=self._queue.qsize(),
            )
        else:
            self._thread = None

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
        text, _ = item

        def _attempt() -> dict[str, object]:
            semaphore = _acquire_request_slot()
            try:
                try:
                    self._rate_guard()
                    result = _send_telegram_http(text, http_post=self._http_post)
                except Exception as exc:  # pragma: no cover - defensive guard
                    log("telegram.error", error=str(exc))
                    raise _RetryableTelegramError(str(exc)) from exc
            finally:
                semaphore.release()

            ok = bool(getattr(result, "get", lambda *_: False)("ok")) if result else False
            if ok:
                _record_activity()
                return result  # type: ignore[return-value]

            error: str | None = None
            if isinstance(result, dict):
                error_value = result.get("error") or result.get("description")
                if error_value:
                    error = str(error_value)
            if error is None:
                error = "telegram_delivery_failed"
            raise _RetryableTelegramError(error)

        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait_strategy,
            retry=retry_if_exception_type(_RetryableTelegramError),
            sleep=self._sleep,
            reraise=False,
        )

        try:
            retrying(_attempt)
        except RetryError as exc:
            attempts = exc.last_attempt.attempt_number
            error_message = str(exc.last_attempt.exception()) or "telegram_delivery_failed"
            log(
                "telegram.delivery_failed",
                text=text,
                attempts=attempts,
                error=error_message,
            )

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


class TelegramHeartbeat:
    """Background monitor that emits periodic heartbeats and silence alerts."""

    def __init__(
        self,
        *,
        send: Callable[[str], None],
        get_settings_func: Callable[[], object] = get_settings,
        log_func: Callable[..., None] = log,
        time_func: Callable[[], float] = _now,
        silence_threshold: float = 60.0,
        min_poll_interval: float = 5.0,
    ) -> None:
        if silence_threshold <= 0:
            raise ValueError("silence_threshold must be positive")
        if min_poll_interval <= 0:
            raise ValueError("min_poll_interval must be positive")

        self._send = send
        self._get_settings = get_settings_func
        self._log = log_func
        self._time = time_func
        self._silence_threshold = float(silence_threshold)
        self._min_poll_interval = float(min_poll_interval)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_sent: Optional[float] = None
        self._alerted = False
        self._last_activity_seen: Optional[float] = None

    def ensure_started(self) -> None:
        with self._lock:
            thread = self._thread
            if thread and thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(target=self._run, name="telegram-heartbeat", daemon=True)
            self._thread = thread
            thread.start()

    def stop(self, *, timeout: float | None = None) -> None:
        thread = self._thread
        if not thread:
            return
        join_timeout = timeout if timeout is not None else _DEFAULT_SHUTDOWN_TIMEOUT
        self._stop_event.set()
        thread.join(timeout=join_timeout)

        if thread.is_alive():  # pragma: no cover - defensive
            self._log("telegram.heartbeat.shutdown_timeout", timeout=join_timeout)
        else:
            self._thread = None

    def run_cycle(self) -> float:
        settings = self._safe_settings()
        if not settings or not bool(getattr(settings, "heartbeat_enabled", False)):
            self._alerted = False
            self._last_sent = None
            self._last_activity_seen = None
            return self._min_poll_interval

        interval_seconds = self._resolve_interval_seconds(settings)
        threshold = self._effective_silence_threshold(interval_seconds)
        now = self._time()

        if self._should_emit_heartbeat(settings, now, interval_seconds):
            message = self._format_message(now)
            self._safe_send(message, interval_seconds, event="telegram.heartbeat.sent")

        self._check_silence(now, threshold)

        return self._next_sleep(interval_seconds, threshold)

    def _effective_silence_threshold(self, interval_seconds: float) -> float:
        return max(self._silence_threshold, interval_seconds * 1.5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sleep_for = self.run_cycle()
            if sleep_for <= 0:
                sleep_for = self._min_poll_interval
            self._stop_event.wait(timeout=sleep_for)

    def _safe_settings(self) -> object | None:
        try:
            return self._get_settings()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log("telegram.heartbeat.settings_error", err=str(exc))
            return None

    def _resolve_interval_seconds(self, settings: object) -> float:
        candidates = (
            ("seconds", getattr(settings, "heartbeat_seconds", None)),
            ("seconds", getattr(settings, "heartbeat_interval_seconds", None)),
            ("minutes", getattr(settings, "heartbeat_interval_min", None)),
            ("minutes", getattr(settings, "heartbeat_minutes", None)),
        )
        for kind, raw in candidates:
            try:
                if raw is None:
                    continue
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            seconds = value if kind == "seconds" else value * 60.0
            return max(seconds, self._min_poll_interval)

        return max(60.0, self._min_poll_interval)

    def _should_emit_heartbeat(
        self, settings: object, now: float, interval_seconds: float
    ) -> bool:
        token = getattr(settings, "telegram_token", "")
        chat_id = getattr(settings, "telegram_chat_id", "")
        if not token or not chat_id:
            return False
        if self._last_sent is None:
            return True
        return now - self._last_sent >= interval_seconds

    def _format_message(self, now: float) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
        return f"🤖 Heartbeat · alive @ {timestamp} UTC"

    def _safe_send(self, text: str, interval_seconds: float, *, event: str, mark_last: bool = True) -> None:
        try:
            self._send(text)
        except Exception as exc:  # pragma: no cover - network/runtime guard
            self._log("telegram.heartbeat.error", err=str(exc), text=text)
            return

        if mark_last:
            self._last_sent = self._time()

        self._log(event, interval=interval_seconds, text=text)

    def _check_silence(self, now: float, threshold: float) -> None:
        last = _last_activity()
        if last < 0:
            return
        previous_seen = self._last_activity_seen
        if previous_seen is None or last > previous_seen:
            self._alerted = False
        self._last_activity_seen = last

        silence = now - last

        if silence > threshold:
            if not self._alerted:
                self._alerted = True
                self._handle_silence(silence, threshold)
        else:
            if self._alerted:
                self._log(
                    "telegram.heartbeat.recovered",
                    silence=silence,
                    threshold=threshold,
                )
            self._alerted = False

    def _handle_silence(self, silence: float, threshold: float) -> None:
        payload = {"silence": round(silence, 3), "threshold": round(threshold, 3)}
        self._log("telegram.heartbeat.silence", **payload)
        alert_text = (
            "⚠️ Telegram heartbeat silence "
            f"> {threshold:.0f}s (actual {silence:.0f}s)."
        )
        self._safe_send(
            alert_text,
            threshold,
            event="telegram.heartbeat.alert",
            mark_last=False,
        )

    def _next_sleep(self, interval_seconds: float, threshold: float) -> float:
        candidate = interval_seconds / 4.0
        candidate = max(candidate, self._min_poll_interval)
        candidate = min(candidate, max(threshold / 2.0, self._min_poll_interval))
        return candidate

def send_telegram(text: str):
    """Send Telegram message synchronously respecting rate limits."""

    heartbeat.ensure_started()

    def _attempt() -> dict[str, object]:
        semaphore = _acquire_request_slot()
        try:
            try:
                _rate_guard()
                return _send_telegram_http(text)
            except Exception as exc:  # pragma: no cover - defensive guard
                log("telegram.error", error=str(exc))
                raise _RetryableTelegramError(str(exc)) from exc
        finally:
            semaphore.release()

    wait_strategy = wait_exponential(multiplier=1.0, min=1.0, max=30.0) + wait_random(0, 0.25)

    retrying = Retrying(
        stop=stop_after_attempt(5),
        wait=wait_strategy,
        retry=retry_if_exception_type(_RetryableTelegramError),
        sleep=time.sleep,
        reraise=False,
    )

    last_result: dict[str, object] | None = None

    def _wrapped_attempt() -> dict[str, object]:
        nonlocal last_result
        result = _attempt()
        last_result = result if isinstance(result, dict) else None
        ok = bool(getattr(result, "get", lambda *_: False)("ok")) if result else False
        if ok:
            _record_activity()
            return result

        error: str | None = None
        if isinstance(result, dict):
            error_value = result.get("error") or result.get("description")
            if error_value:
                error = str(error_value)
        if error is None:
            error = "telegram_delivery_failed"
        raise _RetryableTelegramError(error)

    try:
        return retrying(_wrapped_attempt)
    except RetryError as exc:
        attempts = exc.last_attempt.attempt_number
        error_message = str(exc.last_attempt.exception()) or "telegram_delivery_failed"
        log(
            "telegram.delivery_failed",
            text=text,
            attempts=attempts,
            error=error_message,
        )
        if last_result is not None:
            return last_result
        return {"ok": False, "error": error_message}


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
heartbeat = TelegramHeartbeat(send=dispatcher.enqueue_message)


def enqueue_telegram_message(text: str) -> None:
    heartbeat.ensure_started()
    dispatcher.enqueue_message(text)


def shutdown_telegram_dispatcher(*, timeout: float | None = None) -> None:
    heartbeat.stop(timeout=timeout)
    dispatcher.shutdown(timeout=timeout)


def _auto_start_heartbeat() -> None:
    try:
        settings = get_settings()
    except Exception:  # pragma: no cover - defensive guard
        return

    if getattr(settings, "heartbeat_enabled", False):
        heartbeat.ensure_started()


_auto_start_heartbeat()


atexit.register(shutdown_telegram_dispatcher)
