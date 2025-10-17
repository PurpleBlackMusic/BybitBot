from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .envs import get_api_client, get_settings, creds_ok
from .guardian_bot import GuardianBot
from .hygiene import cancel_stale_orders, cancel_twap_leftovers
from .log import log
from .signal_executor import AutomationLoop, ExecutionResult, SignalExecutor
from .ws_manager import manager as ws_manager
from .realtime_cache import get_realtime_cache
from .preflight import collect_preflight_snapshot

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
        self._automation_executor: Optional[SignalExecutor] = None
        self._automation_error: Optional[str] = None
        self._automation_started_at: float = 0.0
        self._automation_last_cycle: float = 0.0
        self._automation_restart_count: int = 0
        self._automation_stale_after = max(float(automation_stale_after), 0.0)
        self._executor_state: Dict[str, Any] = {}
        self._automation_sweeper: Dict[str, Any] = {}

        self._bot_factory: BotFactory = bot_factory or GuardianBot
        self._executor_factory: ExecutorFactory = executor_factory or (
            lambda bot: SignalExecutor(bot)
        )
        self._loop_factory = loop_factory
        self._automation_poll_interval = 15.0
        self._automation_success_cooldown = 120.0
        self._automation_error_backoff = 5.0

        self._hygiene_lock = threading.Lock()
        self._hygiene_performed = False

        self._order_sweep_interval = 20.0  # sweep idle orders roughly every 20s
        self._order_sweep_stale_after = 600.0  # cancel GTC orders older than 10 minutes
        self._order_sweep_batch_size = 20
        self._order_sweep_last = 0.0
        self._order_sweep_lock = threading.Lock()
        self._order_sweep_prefixes: tuple[str, ...] = ("AI-",)
        self._order_sweep_config_last = 0.0

        # guardian snapshot worker
        self._guardian_lock = threading.Lock()
        self._guardian_thread: Optional[threading.Thread] = None
        self._guardian_stop_event: Optional[threading.Event] = None
        self._guardian_refresh_event: Optional[threading.Event] = None
        self._guardian_state: Dict[str, Any] = {}
        self._guardian_error: Optional[str] = None
        self._guardian_started_at: float = 0.0
        self._guardian_last_refresh: float = 0.0
        self._guardian_restart_count: int = 0
        self._guardian_poll_interval = 12.0
        self._guardian_error_backoff = 5.0

        self._preflight_lock = threading.Lock()
        self._preflight_state: Dict[str, Any] = {}
        self._preflight_last_run: float = 0.0
        self._preflight_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    def ensure_started(self) -> None:
        ws_ok = self.ensure_ws_started()
        if not ws_ok:
            return

        self._run_startup_hygiene()

        self._run_preflight_healthcheck()

        if not self._await_private_ready():
            return

        self.ensure_automation_loop()
        self.ensure_guardian_worker()

    def _await_private_ready(self, timeout: float | None = None) -> bool:
        """Wait until the private websocket reports a fresh heartbeat."""

        if self._ws_private_stale_after <= 0:
            return True

        start_wait = time.time()
        deadline = start_wait + (
            float(timeout)
            if isinstance(timeout, (int, float)) and float(timeout) > 0
            else min(self._ws_private_stale_after, 15.0)
        )

        def _coerce_float(value: object | None) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        logged_wait = False
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
                    if logged_wait:
                        log(
                            "background.ws.private.wait.ready",
                            waited=round(time.time() - start_wait, 3),
                        )
                    return True

            if not logged_wait:
                log("background.ws.private.waiting")
                logged_wait = True

            if time.time() >= deadline:
                log(
                    "background.ws.private.wait.timeout",
                    waited=round(time.time() - start_wait, 3),
                    status=status,
                )
                return False

            time.sleep(0.2)

    def _safe_ws_status(self) -> Dict[str, Any]:
        try:
            return ws_manager.status()
        except Exception as exc:  # pragma: no cover - defensive guard
            log("background.ws.status.error", err=str(exc))
            return {}

    def _analyze_ws_status(
        self, status: Dict[str, Any]
    ) -> tuple[bool, bool, Optional[Dict[str, object]]]:
        if not status:
            return True, True, None

        public = status.get("public") or {}
        private = status.get("private") or {}

        def _coerce_age(value: object) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        restart_public = False
        restart_private = False
        fallback_payload: Optional[Dict[str, object]] = None

        public_running = bool(public.get("running"))
        public_age = _coerce_age(public.get("age_seconds"))
        if not public_running:
            restart_public = True
        elif (
            self._ws_public_stale_after
            and public_age is not None
            and public_age > self._ws_public_stale_after
        ):
            restart_public = True
            fallback_payload = {
                "age": round(public_age, 3),
                "threshold": self._ws_public_stale_after,
            }

        private_running = bool(private.get("running"))
        private_age = _coerce_age(private.get("age_seconds"))
        if private and not private_running:
            restart_private = True
        if (
            self._ws_private_stale_after
            and private_age is not None
            and private_age > self._ws_private_stale_after
        ):
            restart_private = True

        return restart_public, restart_private, fallback_payload

    def ensure_ws_started(self, *, force: bool = False) -> bool:
        with self._ws_lock:
            status: Dict[str, Any] | None = None
            restart_public = False
            restart_private = False
            fallback_payload: Optional[Dict[str, object]] = None

            if not force and self._ws_started:
                status = self._safe_ws_status()
                (
                    restart_public,
                    restart_private,
                    fallback_payload,
                ) = self._analyze_ws_status(status)

            if force:
                restart_public = True
                restart_private = True

            if restart_public:
                try:
                    ws_manager.stop_all()
                except Exception as exc:  # pragma: no cover - defensive guard
                    log("background.ws.stop.error", err=str(exc))
                self._ws_started = False
                self._hygiene_performed = False
            elif restart_private:
                stop_private = getattr(ws_manager, "stop_private", None)
                if callable(stop_private):
                    try:
                        stop_private()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        log("background.ws.private.stop.error", err=str(exc))

            if fallback_payload and hasattr(ws_manager, "force_public_fallback"):
                try:
                    ws_manager.force_public_fallback(
                        "public_channel_stale",
                        **fallback_payload,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    log("background.ws.fallback.error", err=str(exc))
                else:
                    log("background.ws.fallback.engaged", **fallback_payload)

            previous_started = self._ws_started
            start_mode = None
            if not self._ws_started or restart_public:
                start_mode = "full"
            elif restart_private:
                start_mode = "private"

            if start_mode is None:
                return self._ws_started

            try:
                if start_mode == "full":
                    ok = ws_manager.start()
                else:
                    start_private = getattr(ws_manager, "start_private", None)
                    if not callable(start_private):
                        return self._ws_started
                    ok = start_private()
            except Exception as exc:  # pragma: no cover - defensive guard
                self._ws_error = str(exc)
                if start_mode == "full":
                    self._ws_started = False
                log("background.ws.start.error", err=str(exc))
                return False

            if start_mode == "full":
                if ok:
                    self._ws_started = True
                    self._ws_error = None
                    self._ws_last_started_at = time.time()
                    if restart_public or force or not previous_started:
                        self._ws_restart_count += 1
                    log(
                        "background.ws.start.success",
                        mode=start_mode,
                        restart_count=self._ws_restart_count,
                    )
                else:
                    self._ws_started = False
                    self._ws_error = "manager.start returned False"
                    log("background.ws.start.failed", mode=start_mode)
                return self._ws_started

            # private-only restart path
            if ok:
                self._ws_error = None
                log("background.ws.start.success", mode=start_mode)
                return True

            self._ws_error = "manager.start_private returned False"
            log("background.ws.private.restart.failed")
            return False

    def _run_startup_hygiene(self) -> None:
        with self._hygiene_lock:
            if self._hygiene_performed:
                return

            try:
                settings = get_settings()
            except Exception as exc:  # pragma: no cover - defensive guard
                log("background.hygiene.settings.error", err=str(exc))
                return

            if not creds_ok(settings):
                self._hygiene_performed = True
                return

            try:
                api = get_api_client()
            except Exception as exc:  # pragma: no cover - defensive guard
                log("background.hygiene.api.error", err=str(exc))
                return

            try:
                result = cancel_twap_leftovers(api, category="spot")
            except Exception as exc:  # pragma: no cover - defensive guard
                log("background.hygiene.twap.error", err=str(exc))
                self._hygiene_performed = True
                return

            total = 0
            try:
                total = int(result.get("total", 0))
            except Exception:
                total = 0

            if total > 0:
                log("background.hygiene.twap.cleaned", total=total)
            else:
                log("background.hygiene.twap.cleaned", total=0)

            self._hygiene_performed = True

    def _run_preflight_healthcheck(self) -> None:
        now = time.time()
        with self._preflight_lock:
            if self._preflight_state and now - self._preflight_last_run < 15.0:
                return
            self._preflight_last_run = now

        try:
            settings = get_settings()
        except Exception as exc:  # pragma: no cover - defensive guard
            with self._preflight_lock:
                self._preflight_error = str(exc)
            log("background.preflight.settings_error", err=str(exc))
            return

        try:
            api = get_api_client()
        except Exception as exc:  # pragma: no cover - defensive guard
            with self._preflight_lock:
                self._preflight_error = str(exc)
            log("background.preflight.api_unavailable", err=str(exc))
            return

        ws_status = self._safe_ws_status()

        try:
            snapshot = collect_preflight_snapshot(
                settings,
                api=api,
                ws_status=ws_status,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            with self._preflight_lock:
                self._preflight_error = str(exc)
            log("background.preflight.error", err=str(exc))
            return

        with self._preflight_lock:
            self._preflight_state = snapshot
            self._preflight_error = None
            self._preflight_last_run = time.time()

    def restart_ws(self) -> bool:
        return self.ensure_ws_started(force=True)

    def ensure_automation_loop(self, *, force: bool = False) -> bool:
        join_thread: Optional[threading.Thread] = None
        should_start = False
        executor: Optional[SignalExecutor] = None

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
                executor = self._automation_executor
                should_start = True
                if thread and alive:
                    stop_event = self._automation_stop_event
                    if stop_event is not None:
                        stop_event.set()
                    if hasattr(thread, "join"):
                        join_thread = thread

        if join_thread is not None:
            try:
                join_thread.join(timeout=5)
            except Exception:  # pragma: no cover - defensive guard
                pass
            if join_thread.is_alive():
                log(
                    "background.automation.restart_blocked",
                    reason="join_timeout",
                )
                should_start = False

        if not should_start:
            return True

        with self._automation_lock:
            thread = self._automation_thread
            if thread and thread.is_alive():
                log(
                    "background.automation.restart_blocked",
                    reason="thread_still_alive",
                )
                return True

            stop_event = threading.Event()
            self._automation_stop_event = stop_event
            self._automation_started_at = time.time()
            self._automation_last_cycle = self._automation_started_at
            self._automation_error = None

            state: Dict[str, Any] = {}
            state["executor"] = executor
            self._automation_state = state
            self._automation_executor = executor
            self._automation_sweeper = {}
            self._order_sweep_last = 0.0

            thread = threading.Thread(
                target=self._run_automation_loop,
                args=(stop_event, executor),
                daemon=True,
            )
            self._automation_thread = thread
            self._automation_restart_count += 1

        thread.start()
        return True

    def restart_automation_loop(self) -> bool:
        return self.ensure_automation_loop(force=True)

    # ------------------------------------------------------------------
    # Guardian snapshot worker
    def ensure_guardian_worker(self, *, force: bool = False) -> bool:
        with self._guardian_lock:
            thread = self._guardian_thread
            if thread and thread.is_alive() and not force:
                return True

            if thread and thread.is_alive():
                stop_event = self._guardian_stop_event
                if stop_event:
                    stop_event.set()
                thread.join(timeout=2.0)

            stop_event = threading.Event()
            refresh_event = threading.Event()
            self._guardian_stop_event = stop_event
            self._guardian_refresh_event = refresh_event
            self._guardian_thread = threading.Thread(
                target=self._run_guardian_worker,
                args=(stop_event, refresh_event),
                daemon=True,
            )
            self._guardian_restart_count += 1
            self._guardian_started_at = time.time()
            self._guardian_error = None

            thread = self._guardian_thread

        thread.start()
        return True

    def restart_guardian_worker(self) -> bool:
        return self.ensure_guardian_worker(force=True)

    # ------------------------------------------------------------------
    # Automation
    def _create_loop(
        self,
        executor: SignalExecutor,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None],
        *,
        sweeper: Callable[[], bool] | None = None,
    ) -> AutomationLoop:
        if self._loop_factory is not None:
            return self._loop_factory(executor, on_cycle)
        return AutomationLoop(
            executor,
            poll_interval=self._automation_poll_interval,
            success_cooldown=self._automation_success_cooldown,
            error_backoff=self._automation_error_backoff,
            on_cycle=on_cycle,
            sweeper=sweeper,
        )

    def _run_automation_loop(
        self, stop_event: threading.Event, executor: Optional[SignalExecutor] = None
    ) -> None:
        if executor is None:
            bot = self._bot_factory()
            executor = self._executor_factory(bot)

        self._automation_executor = executor

        with self._automation_lock:
            state_snapshot = copy.deepcopy(self._executor_state)
        if state_snapshot:
            try:
                executor.restore_state(state_snapshot)
            except Exception:  # pragma: no cover - defensive guard
                pass

        cycle_counter = 0

        def handle_cycle(
            result: ExecutionResult,
            signature: Optional[str],
            marker: Tuple[bool, bool, bool],
        ) -> None:
            nonlocal cycle_counter
            cycle_counter += 1
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
                state["executor"] = self._automation_executor
                self._automation_state = state
                self._automation_last_cycle = ts

            log(
                "background.automation.loop.cycle",
                cycle=cycle_counter,
                status=result.status,
                reason=result.reason,
                signature=signature,
            )

        def sweep_orders() -> bool:
            return self._maybe_sweep_orders()

        loop = self._create_loop(executor, handle_cycle, sweeper=sweep_orders)

        current_thread = threading.current_thread()
        started_at = time.time()
        log(
            "background.automation.loop.start",
            thread=current_thread.name,
            has_executor=bool(self._automation_executor),
        )
        try:
            loop.run(stop_event=stop_event)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("background.automation.error", err=str(exc))
            with self._automation_lock:
                self._automation_error = str(exc)
        else:
            with self._automation_lock:
                self._automation_error = None
        finally:
            log(
                "background.automation.loop.stop",
                thread=current_thread.name,
                duration=round(time.time() - started_at, 3),
                cycles=cycle_counter,
            )
            with self._automation_lock:
                try:
                    self._executor_state = executor.export_state()
                except Exception:  # pragma: no cover - defensive guard
                    self._executor_state = {}
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
            sweeper_snapshot = copy.deepcopy(self._automation_sweeper)

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
        if sweeper_snapshot:
            snapshot["sweeper"] = sweeper_snapshot
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

    def _run_guardian_worker(
        self,
        stop_event: threading.Event,
        refresh_event: threading.Event,
    ) -> None:
        bot: GuardianBot | None = None
        backoff_until = 0.0

        while not stop_event.is_set():
            now = time.time()
            if now < backoff_until:
                sleep_for = max(0.1, backoff_until - now)
                if stop_event.wait(timeout=sleep_for):
                    break
                continue

            if bot is None:
                try:
                    bot = self._bot_factory()
                except Exception as exc:  # pragma: no cover - defensive guard
                    backoff_until = time.time() + self._guardian_error_backoff
                    with self._guardian_lock:
                        self._guardian_error = str(exc)
                    continue

            cycle_started = time.time()
            state_payload: Dict[str, Any] = {}
            error: Optional[str] = None

            try:
                bot.refresh()
                report = bot.unified_report()
                brief = bot.generate_brief()
                scorecard = bot.signal_scorecard(brief)
                plan_steps = bot.plan_steps(brief)
                safety_notes = bot.safety_notes()
                risk_summary = bot.risk_summary()
            except Exception as exc:  # pragma: no cover - defensive guard
                error = str(exc)
                backoff_until = time.time() + self._guardian_error_backoff
            else:
                state_payload = {
                    "report": copy.deepcopy(report),
                    "brief": brief.to_dict(),
                    "scorecard": copy.deepcopy(scorecard),
                    "plan_steps": list(plan_steps),
                    "safety_notes": list(safety_notes),
                    "risk_summary": str(risk_summary),
                }
                backoff_until = 0.0

            with self._guardian_lock:
                if state_payload:
                    self._guardian_state = state_payload
                    self._guardian_last_refresh = time.time()
                if error:
                    self._guardian_error = error
                elif state_payload:
                    self._guardian_error = None

            elapsed = time.time() - cycle_started
            wait_for = max(0.0, self._guardian_poll_interval - elapsed)

            if refresh_event.is_set():
                refresh_event.clear()
                wait_for = 0.0

            if wait_for <= 0:
                continue

            if stop_event.wait(timeout=wait_for):
                break

    def guardian_snapshot(self) -> Dict[str, Any]:
        with self._guardian_lock:
            state = copy.deepcopy(self._guardian_state)
            error = self._guardian_error
            started_at = self._guardian_started_at
            last_refresh = self._guardian_last_refresh
            restart_count = self._guardian_restart_count
            thread_alive = self._guardian_thread and self._guardian_thread.is_alive()

        snapshot: Dict[str, Any] = {
            "thread_alive": bool(thread_alive),
            "started_at": started_at or None,
            "last_refresh_at": last_refresh or None,
            "restart_count": restart_count,
            "error": error,
            "state": state,
        }
        if last_refresh:
            snapshot["age_seconds"] = max(0.0, time.time() - last_refresh)
        return snapshot

    def request_guardian_refresh(self) -> None:
        with self._guardian_lock:
            event = self._guardian_refresh_event
            if event is not None:
                event.set()

    def preflight_snapshot(self) -> Dict[str, Any]:
        with self._preflight_lock:
            state = copy.deepcopy(self._preflight_state)
            error = self._preflight_error
            last_run = self._preflight_last_run

        snapshot: Dict[str, Any] = {
            "state": state,
            "last_run_at": last_run or None,
            "error": error,
        }
        if last_run:
            snapshot["age_seconds"] = max(0.0, time.time() - last_run)
        if isinstance(state, Mapping):
            snapshot.update(state)
        return snapshot

    def _should_sweep_orders(self) -> bool:
        self._refresh_order_sweeper_settings()
        if self._order_sweep_interval <= 0 or self._order_sweep_stale_after <= 0:
            return False
        now = time.time()
        with self._order_sweep_lock:
            if now - self._order_sweep_last < self._order_sweep_interval:
                return False
            self._order_sweep_last = now
        return True

    def _refresh_order_sweeper_settings(self) -> None:
        if self._order_sweep_interval <= 0:
            return

        now = time.time()
        refresh_after = max(self._order_sweep_interval, 30.0)
        if now - self._order_sweep_config_last < refresh_after:
            return

        self._order_sweep_config_last = now

        try:
            settings = get_settings()
        except Exception as exc:
            log("background.automation.sweeper.settings_error", err=str(exc))
            return

        candidates: list[float] = []
        for attr in ("reprice_unfilled_after_sec", "execution_watchdog_max_age_sec"):
            value = getattr(settings, attr, None)
            if isinstance(value, (int, float)):
                candidates.append(float(value))

        if not candidates:
            return

        positive = [value for value in candidates if value > 0]
        if positive:
            new_value = max(max(positive), 60.0)
        else:
            new_value = 0.0

        if abs(new_value - self._order_sweep_stale_after) < 1e-6:
            return

        self._order_sweep_stale_after = new_value

    def _maybe_sweep_orders(self) -> bool:
        if not self._should_sweep_orders():
            return False

        try:
            api = get_api_client()
        except Exception as exc:
            log("background.automation.sweeper.api_unavailable", err=str(exc))
            with self._automation_lock:
                self._automation_sweeper = {
                    "last_error": str(exc),
                    "last_run_at": time.time(),
                    "interval": self._order_sweep_interval,
                    "stale_after": self._order_sweep_stale_after,
                }
            return False

        try:
            result = cancel_stale_orders(
                api,
                category="spot",
                older_than_sec=int(self._order_sweep_stale_after),
                batch_size=self._order_sweep_batch_size,
                time_in_force="GTC",
                order_types="limit",
                link_prefixes=self._order_sweep_prefixes,
            )
        except Exception as exc:
            log("background.automation.sweeper.error", err=str(exc))
            with self._automation_lock:
                self._automation_sweeper = {
                    "last_error": str(exc),
                    "last_run_at": time.time(),
                    "interval": self._order_sweep_interval,
                    "stale_after": self._order_sweep_stale_after,
                }
            return False

        total = 0
        batches = 0
        stale_total = 0
        stale_symbols: tuple[str, ...] = ()
        if isinstance(result, Mapping):
            try:
                total = int(result.get("total", 0))
            except Exception:
                total = 0
            batches_payload = result.get("batches")
            if isinstance(batches_payload, Sequence):
                batches = len(batches_payload)
            try:
                stale_total = int(result.get("stale_total", 0))
            except Exception:
                stale_total = 0
            symbols_payload = result.get("stale_symbols")
            if isinstance(symbols_payload, Sequence) and not isinstance(symbols_payload, (str, bytes)):
                stale_symbols = tuple(str(sym) for sym in symbols_payload if sym)

        now = time.time()
        sweeper_snapshot = {
            "last_run_at": now,
            "interval": self._order_sweep_interval,
            "stale_after": self._order_sweep_stale_after,
            "cancelled": total,
            "batches": batches,
            "stale_total": stale_total,
            "triggered_retry": total > 0,
        }
        if stale_symbols:
            sweeper_snapshot["stale_symbols"] = stale_symbols

        with self._automation_lock:
            self._automation_sweeper = sweeper_snapshot

        if total > 0:
            log(
                "background.automation.sweeper.cancelled",
                total=total,
                batches=batches,
            )
            return True

        return False

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
        event_stats = ws_manager.private_event_stats()

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
            "private_event_cursor": event_stats.get("latest_id"),
            "private_event_backlog": event_stats.get("size"),
            "private_event_dropped": event_stats.get("dropped"),
        }

    def ws_events(
        self,
        *,
        since: int | None = None,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        events = ws_manager.private_events(since=since, limit=limit)
        stats = ws_manager.private_event_stats()
        return {"events": events, "stats": stats}


_state = BackgroundServices()


def ensure_background_services() -> None:
    _state.ensure_started()


def get_automation_status() -> Dict[str, Any]:
    return _state.automation_snapshot()


def get_ws_snapshot() -> Dict[str, Any]:
    return _state.ws_snapshot()


def get_guardian_state() -> Dict[str, Any]:
    return _state.guardian_snapshot()


def get_preflight_snapshot() -> Dict[str, Any]:
    return _state.preflight_snapshot()


def restart_automation() -> bool:
    return _state.restart_automation_loop()


def restart_websockets() -> bool:
    return _state.restart_ws()


def restart_guardian() -> bool:
    return _state.restart_guardian_worker()


def request_guardian_refresh() -> None:
    _state.request_guardian_refresh()


def get_ws_events(*, since: int | None = None, limit: int | None = 100) -> Dict[str, Any]:
    return _state.ws_events(since=since, limit=limit)
