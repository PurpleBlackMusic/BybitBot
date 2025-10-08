from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .guardian_bot import GuardianBot
from .log import log
from .signal_executor import AutomationLoop, ExecutionResult, SignalExecutor
from .ws_manager import manager as ws_manager
from .realtime_cache import get_realtime_cache

BotFactory = Callable[[], GuardianBot]
ExecutorFactory = Callable[[GuardianBot], SignalExecutor]
LoopFactory = Callable[
    [SignalExecutor, Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None]],
    AutomationLoop,
]


class BackgroundServices:
    """Manage long-lived background services independent of Streamlit reruns."""

    def __init__(
        self,
        *,
        bot_factory: Optional[BotFactory] = None,
        executor_factory: Optional[ExecutorFactory] = None,
        loop_factory: Optional[LoopFactory] = None,
        public_stale_after: float = 60.0,
        private_stale_after: float = 90.0,
        automation_stale_after: float = 300.0,
    ) -> None:
        self._ws_lock = threading.Lock()
        self._ws_started = False
        self._ws_error: Optional[str] = None
        self._ws_last_started_at: float = 0.0
        self._ws_restart_count: int = 0
        self._ws_public_stale_after = max(float(public_stale_after), 0.0)
        self._ws_private_stale_after = max(float(private_stale_after), 0.0)

        self._automation_lock = threading.Lock()
        self._automation_thread: Optional[threading.Thread] = None
        self._automation_stop_event: Optional[threading.Event] = None
        self._automation_state: Dict[str, Any] = {}
        self._automation_error: Optional[str] = None
        self._automation_started_at: float = 0.0
        self._automation_last_cycle: float = 0.0
        self._automation_restart_count: int = 0
        self._automation_stale_after = max(float(automation_stale_after), 0.0)

        self._bot_factory: BotFactory = bot_factory or GuardianBot
        self._executor_factory: ExecutorFactory = executor_factory or (
            lambda bot: SignalExecutor(bot)
        )
        self._loop_factory = loop_factory
        self._automation_poll_interval = 15.0
        self._automation_success_cooldown = 120.0
        self._automation_error_backoff = 5.0

    # ------------------------------------------------------------------
    # Lifecycle
    def ensure_started(self) -> None:
        ws_ok = self.ensure_ws_started()
        if not ws_ok:
            return

        if not self._await_private_ready():
            return

        self.ensure_automation_loop()

    def _await_private_ready(self, timeout: float | None = None) -> bool:
        """Wait until the private websocket reports a fresh heartbeat."""

        if self._ws_private_stale_after <= 0:
            return True

        deadline = time.time() + (
            float(timeout)
            if isinstance(timeout, (int, float)) and float(timeout) > 0
            else min(self._ws_private_stale_after, 15.0)
        )

        def _coerce_float(value: object | None) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        while True:
            status = self._safe_ws_status()
            private = status.get("private") if isinstance(status, dict) else None
            if isinstance(private, dict):
                running = bool(private.get("running"))
                connected_field = private.get("connected")
                connected = True if connected_field is None else bool(connected_field)
                last_beat = _coerce_float(private.get("last_beat"))
                age = _coerce_float(private.get("age_seconds"))

                fresh_age = (
                    age is not None
                    and (
                        age <= self._ws_private_stale_after
                        or (self._ws_private_stale_after == 0 and age == 0)
                    )
                )

                if running and connected and (last_beat or fresh_age):
                    return True

            if time.time() >= deadline:
                return False

            time.sleep(0.2)

    def _safe_ws_status(self) -> Dict[str, Any]:
        try:
            return ws_manager.status()
        except Exception as exc:  # pragma: no cover - defensive guard
            log("background.ws.status.error", err=str(exc))
            return {}

    def _ws_needs_restart(self, status: Dict[str, Any]) -> bool:
        if not status:
            return True

        public = status.get("public") or {}
        private = status.get("private") or {}

        public_running = bool(public.get("running"))
        public_age = public.get("age_seconds")
        if not public_running:
            return True
        if (
            self._ws_public_stale_after
            and isinstance(public_age, (int, float))
            and public_age > self._ws_public_stale_after
        ):
            return True

        private_running = bool(private.get("running"))
        private_age = private.get("age_seconds")
        if private and not private_running:
            return True
        if (
            self._ws_private_stale_after
            and isinstance(private_age, (int, float))
            and private_age > self._ws_private_stale_after
        ):
            return True

        return False

    def ensure_ws_started(self, *, force: bool = False) -> bool:
        with self._ws_lock:
            status: Dict[str, Any] | None = None
            if not force and self._ws_started:
                status = self._safe_ws_status()
                if self._ws_needs_restart(status):
                    force = True

            if force:
                try:
                    ws_manager.stop_all()
                except Exception as exc:  # pragma: no cover - defensive guard
                    log("background.ws.stop.error", err=str(exc))
                self._ws_started = False

            previous_started = self._ws_started
            try:
                ok = ws_manager.start()
            except Exception as exc:  # pragma: no cover - defensive guard
                self._ws_error = str(exc)
                self._ws_started = False
                log("background.ws.start.error", err=str(exc))
                return False

            if ok:
                self._ws_started = True
                self._ws_error = None
                self._ws_last_started_at = time.time()
                if force or not previous_started:
                    self._ws_restart_count += 1
            else:
                self._ws_started = False
                self._ws_error = "manager.start returned False"
            return self._ws_started

    def restart_ws(self) -> bool:
        return self.ensure_ws_started(force=True)

    def ensure_automation_loop(self, *, force: bool = False) -> bool:
        join_thread: Optional[threading.Thread] = None
        with self._automation_lock:
            thread = self._automation_thread
            alive = bool(thread and thread.is_alive())

            stale = False
            if (
                alive
                and not force
                and self._automation_stale_after
                and self._automation_last_cycle
            ):
                age = time.time() - self._automation_last_cycle
                if age > self._automation_stale_after:
                    stale = True

            if not alive and self._automation_stale_after and self._automation_started_at:
                # if the loop exited silently, consider it stale so we restart immediately
                stale = True

            if force or not alive or stale:
                if thread and alive:
                    stop_event = self._automation_stop_event
                    if stop_event is not None:
                        stop_event.set()
                    if hasattr(thread, "join"):
                        join_thread = thread
                self._automation_thread = None
                self._automation_state = {}
                self._automation_error = None

                stop_event = threading.Event()
                self._automation_stop_event = stop_event
                self._automation_started_at = time.time()
                self._automation_last_cycle = self._automation_started_at

                thread = threading.Thread(
                    target=self._run_automation_loop,
                    args=(stop_event,),
                    daemon=True,
                )
                self._automation_thread = thread
                self._automation_restart_count += 1
                thread.start()

        if join_thread is not None:
            try:
                join_thread.join(timeout=5)
            except Exception:  # pragma: no cover - defensive guard
                pass
        return True

    def restart_automation_loop(self) -> bool:
        return self.ensure_automation_loop(force=True)

    # ------------------------------------------------------------------
    # Automation
    def _create_loop(
        self,
        executor: SignalExecutor,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None],
    ) -> AutomationLoop:
        if self._loop_factory is not None:
            return self._loop_factory(executor, on_cycle)
        return AutomationLoop(
            executor,
            poll_interval=self._automation_poll_interval,
            success_cooldown=self._automation_success_cooldown,
            error_backoff=self._automation_error_backoff,
            on_cycle=on_cycle,
        )

    def _run_automation_loop(self, stop_event: threading.Event) -> None:
        bot = self._bot_factory()
        executor = self._executor_factory(bot)

        def handle_cycle(
            result: ExecutionResult,
            signature: Optional[str],
            marker: Tuple[bool, bool, bool],
        ) -> None:
            payload: Dict[str, Any] = {"status": result.status}
            if result.reason is not None:
                payload["reason"] = result.reason
            if result.order is not None:
                payload["order"] = copy.deepcopy(result.order)
            if result.response is not None:
                payload["response"] = copy.deepcopy(result.response)
            if result.context is not None:
                payload["context"] = copy.deepcopy(result.context)

            ts = time.time()
            state = {
                "result": payload,
                "signature": signature,
                "settings_marker": marker,
                "ts": ts,
            }
            with self._automation_lock:
                self._automation_state = state
                self._automation_last_cycle = ts

        loop = self._create_loop(executor, handle_cycle)

        current_thread = threading.current_thread()
        try:
            loop.run(stop_event=stop_event)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("background.automation.error", err=str(exc))
            with self._automation_lock:
                self._automation_error = str(exc)
        finally:
            with self._automation_lock:
                if self._automation_thread is current_thread:
                    self._automation_thread = None
                    self._automation_stop_event = None

    def automation_snapshot(self) -> Dict[str, Any]:
        with self._automation_lock:
            thread = self._automation_thread
            state = copy.deepcopy(self._automation_state)
            error = self._automation_error
            started_at = self._automation_started_at
            last_cycle = self._automation_last_cycle
            restart_count = self._automation_restart_count
            stale_after = self._automation_stale_after

        snapshot: Dict[str, Any] = {
            "thread_alive": bool(thread and thread.is_alive()),
            "error": error,
            "started_at": started_at or None,
            "last_cycle_at": last_cycle or None,
            "restart_count": restart_count,
            "stale_after": stale_after,
        }
        if state:
            snapshot.update(
                {
                    "last_result": state.get("result"),
                    "signature": state.get("signature"),
                    "settings_marker": state.get("settings_marker"),
                    "last_run_at": state.get("ts"),
                }
            )
        last_ts = snapshot.get("last_run_at")
        if not last_ts and last_cycle:
            last_ts = last_cycle
        try:
            last_ts_float = float(last_ts) if last_ts is not None else None
        except (TypeError, ValueError):
            last_ts_float = None
        is_stale = False
        if (
            last_ts_float is not None
            and stale_after
            and stale_after > 0
        ):
            age = time.time() - last_ts_float
            is_stale = age > stale_after
        snapshot["stale"] = is_stale
        return snapshot

    # ------------------------------------------------------------------
    # WebSocket helpers
    def ws_snapshot(self) -> Dict[str, Any]:
        status = self._safe_ws_status()

        order_update = ws_manager.latest_order_update()
        execution = ws_manager.latest_execution()
        realtime = get_realtime_cache().snapshot(
            public_ttl=self._ws_public_stale_after or None,
            private_ttl=self._ws_private_stale_after or None,
        )

        public = status.get("public") or {}
        private = status.get("private") or {}
        public_age = public.get("age_seconds")
        private_age = private.get("age_seconds")
        public_stale = (
            isinstance(public_age, (int, float))
            and self._ws_public_stale_after
            and public_age > self._ws_public_stale_after
        )
        private_stale = (
            isinstance(private_age, (int, float))
            and self._ws_private_stale_after
            and private_age > self._ws_private_stale_after
        )

        return {
            "started": self._ws_started,
            "last_error": self._ws_error,
            "status": status,
            "last_order": order_update,
            "last_execution": execution,
            "realtime": realtime,
            "public_stale": bool(public_stale),
            "private_stale": bool(private_stale),
            "public_stale_after": self._ws_public_stale_after,
            "private_stale_after": self._ws_private_stale_after,
            "last_started_at": self._ws_last_started_at or None,
            "restart_count": self._ws_restart_count,
        }


_state = BackgroundServices()


def ensure_background_services() -> None:
    _state.ensure_started()


def get_automation_status() -> Dict[str, Any]:
    return _state.automation_snapshot()


def get_ws_snapshot() -> Dict[str, Any]:
    return _state.ws_snapshot()


def restart_automation() -> bool:
    return _state.restart_automation_loop()


def restart_websockets() -> bool:
    return _state.restart_ws()
