"""Light-weight automation loop for the signal executor."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

from .log import log

from . import signal_executor as signal_executor_module
from .signal_executor import ExecutionResult, SignalExecutor


class AutomationLoop:
    """Keep executing trading signals until stopped explicitly."""

    _SUCCESSFUL_STATUSES = {"filled", "dry_run"}

    def __init__(
        self,
        executor: SignalExecutor,
        *,
        poll_interval: float = 15.0,
        success_cooldown: float = 120.0,
        error_backoff: float = 5.0,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None]
        | None = None,
        sweeper: Callable[[], bool] | None = None,
    ) -> None:
        self.executor = executor
        self.poll_interval = max(float(poll_interval), 0.0)
        self.success_cooldown = max(float(success_cooldown), 0.0)
        self.error_backoff = max(float(error_backoff), 0.0)
        self._last_key: Optional[Tuple[Optional[str], Tuple[bool, bool, bool]]] = None
        self._last_status: Optional[str] = None
        self._last_result: Optional[ExecutionResult] = None
        self._on_cycle = on_cycle
        self._last_attempt_ts: Optional[float] = None
        self._next_retry_ts: Optional[float] = None
        self._sweeper = sweeper

    def _invoke_sweeper(self) -> None:
        if self._sweeper is None:
            return
        try:
            triggered = bool(self._sweeper())
        except Exception as exc:  # pragma: no cover - defensive callback guard
            log("guardian.auto.loop.sweeper.error", err=str(exc))
            return
        if triggered:
            self._last_key = None
            self._last_status = None
            self._last_result = None
            self._last_attempt_ts = None
            self._next_retry_ts = None

    def _should_execute(
        self, signature: Optional[str], settings_marker: Tuple[bool, bool, bool]
    ) -> bool:
        key = (signature, settings_marker)
        if self._last_key != key:
            self._next_retry_ts = None
            return True
        if self._last_status in self._SUCCESSFUL_STATUSES:
            return False

        if self.success_cooldown <= 0.0:
            return True

        if self._next_retry_ts is None:
            return True

        now = signal_executor_module.time.monotonic()
        return now >= self._next_retry_ts

    def _tick(self) -> float:
        self._invoke_sweeper()
        signature = self.executor.current_signature()
        settings_marker = self.executor.settings_marker()
        key = (signature, settings_marker)

        if self._should_execute(signature, settings_marker):
            attempt_started = signal_executor_module.time.monotonic()
            try:
                result = self.executor.execute_once()
            except Exception as exc:  # pragma: no cover - defensive
                log("guardian.auto.loop.error", err=str(exc))
                result = ExecutionResult(status="error", reason=str(exc))

            self._last_status = result.status
            self._last_key = key
            self._last_result = result
            self._last_attempt_ts = attempt_started

            if self._on_cycle is not None:
                try:
                    self._on_cycle(result, signature, settings_marker)
                except Exception:  # pragma: no cover - defensive callback guard
                    log("guardian.auto.loop.callback.error")

            if result.status in self._SUCCESSFUL_STATUSES:
                self._next_retry_ts = None
                return self.success_cooldown or self.poll_interval
            if result.status == "error":
                self._next_retry_ts = None
                return self.error_backoff or self.poll_interval or 1.0

            if self.success_cooldown > 0.0:
                self._next_retry_ts = attempt_started + self.success_cooldown
                return self.success_cooldown

            self._next_retry_ts = attempt_started
            return self.poll_interval
        elif (
            self._last_status not in self._SUCCESSFUL_STATUSES
            and self.success_cooldown > 0.0
            and self._next_retry_ts is not None
        ):
            remaining = self._next_retry_ts - signal_executor_module.time.monotonic()
            if remaining > 0.0:
                return remaining

        return self.poll_interval

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Process trading signals until ``stop_event`` is set."""

        event = stop_event or threading.Event()
        while not event.is_set():
            delay = self._tick()
            if delay <= 0:
                continue
            event.wait(delay)

