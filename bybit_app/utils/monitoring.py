"""Runtime monitoring helpers for Bybit Spot Guardian."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - gracefully degrade when psutil missing
    psutil = None

if psutil is not None:  # pragma: no cover - defensive initialisation
    try:
        _PROCESS = psutil.Process()  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - psutil defensive
        _PROCESS = None
else:  # pragma: no cover - fallback when psutil missing
    _PROCESS = None


@dataclass
class MetricsSnapshot:
    """Lightweight container for the runtime metrics we emit."""

    cpu_percent: float
    memory_mb: float
    threads: int
    trades_count: int
    automation_lag_seconds: float | None
    stale: bool
    error: Optional[str]
    timestamp: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "threads": self.threads,
            "trades_count": self.trades_count,
            "automation_lag_seconds": self.automation_lag_seconds,
            "stale": self.stale,
            "error": self.error,
            "timestamp": int(self.timestamp),
        }


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _collect_process_metrics() -> tuple[float, float]:
    process = _PROCESS

    if process is None:  # pragma: no cover - fallback path for minimal envs
        try:
            import resource  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - ultimate fallback
            return 0.0, 0.0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = float(usage.ru_maxrss) / 1024.0
        return 0.0, memory_mb

    try:
        cpu = float(process.cpu_percent(interval=None))
    except Exception:  # pragma: no cover - psutil defensive
        cpu = 0.0
    try:
        memory = float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover - psutil defensive
        memory = 0.0
    return cpu, memory


def collect_system_metrics(snapshot: Mapping[str, Any] | None = None) -> MetricsSnapshot:
    """Aggregate system and automation metrics from the latest snapshot."""

    cpu_percent, memory_mb = _collect_process_metrics()
    threads = threading.active_count()

    trades_count = 0
    error: str | None = None
    lag_seconds: float | None = None
    stale = False

    if snapshot:
        result = snapshot.get("last_result")
        if isinstance(result, Mapping):
            executions = result.get("executions")
            if isinstance(executions, (list, tuple)):
                trades_count = len(executions)
            elif result.get("order"):
                trades_count = 1
            error_value = result.get("reason")
            if error_value:
                error = str(error_value)
        else:
            order = getattr(result, "order", None)
            if order:
                trades_count = 1
            reason = getattr(result, "reason", None)
            if reason:
                error = str(reason)

        stale = bool(snapshot.get("stale"))
        explicit_error = snapshot.get("error")
        if explicit_error:
            error = str(explicit_error)

        last_cycle = snapshot.get("last_cycle_at") or snapshot.get("last_run_at")
        last_cycle_value = _safe_float(last_cycle)
        if last_cycle_value is not None:
            lag = max(time.time() - last_cycle_value, 0.0)
            lag_seconds = round(lag, 3)

    return MetricsSnapshot(
        cpu_percent=cpu_percent,
        memory_mb=memory_mb,
        threads=threads,
        trades_count=trades_count,
        automation_lag_seconds=lag_seconds,
        stale=stale,
        error=error,
        timestamp=time.time(),
    )


class MonitoringReporter:
    """Emit metrics periodically and raise alerts on critical anomalies."""

    def __init__(
        self,
        logger: Callable[..., None],
        *,
        alert_callback: Callable[[str], None] | None = None,
        metrics_interval: float = 60.0,
        alert_cooldown: float = 300.0,
        cpu_threshold: float = 85.0,
        memory_threshold_mb: float = 1024.0,
    ) -> None:
        self._logger = logger
        self._alert_callback = alert_callback
        self._metrics_interval = max(float(metrics_interval), 5.0)
        self._alert_cooldown = max(float(alert_cooldown), 30.0)
        self._cpu_threshold = float(cpu_threshold)
        self._memory_threshold = float(memory_threshold_mb)
        self._last_metrics_ts = 0.0
        self._alert_history: MutableMapping[str, float] = {}

    # ------------------------------------------------------------------
    def _should_emit_metrics(self, now: float) -> bool:
        return now - self._last_metrics_ts >= self._metrics_interval

    def _can_alert(self, code: str, now: float) -> bool:
        last = self._alert_history.get(code, 0.0)
        if now - last < self._alert_cooldown:
            return False
        self._alert_history[code] = now
        return True

    def _dispatch_alert(self, message: str) -> None:
        if not self._alert_callback:
            return
        try:
            self._alert_callback(message)
        except Exception:
            # Alert channels are best-effort; we do not want to raise.
            self._logger("monitoring.alert.dispatch_error", severity="warning", message=message)

    def _build_alert(self, metrics: MetricsSnapshot) -> tuple[str, str] | None:
        if metrics.error:
            return "automation_error", f"⚠️ Automation reported error: {metrics.error}"
        if metrics.stale:
            return "automation_stale", "⚠️ Automation loop appears stale"
        if metrics.automation_lag_seconds and metrics.automation_lag_seconds > self._metrics_interval * 3:
            return (
                "automation_lag",
                f"⚠️ Automation lag {metrics.automation_lag_seconds:.0f}s exceeds expected interval",
            )
        if metrics.cpu_percent > self._cpu_threshold:
            return (
                "high_cpu",
                f"⚠️ CPU usage {metrics.cpu_percent:.1f}% exceeds {self._cpu_threshold}%",
            )
        if metrics.memory_mb > self._memory_threshold:
            return (
                "high_memory",
                f"⚠️ Memory usage {metrics.memory_mb:.0f}MB exceeds {self._memory_threshold:.0f}MB",
            )
        return None

    # ------------------------------------------------------------------
    def process(self, snapshot: Mapping[str, Any] | None) -> MetricsSnapshot:
        metrics = collect_system_metrics(snapshot)
        now = time.time()

        if self._should_emit_metrics(now):
            self._logger("monitoring.metrics", severity="info", **metrics.as_dict())
            self._last_metrics_ts = now

        alert = self._build_alert(metrics)
        if alert and self._can_alert(alert[0], now):
            code, message = alert
            self._logger("monitoring.alert", severity="warning", code=code, message=message)
            self._dispatch_alert(message)

        return metrics


__all__ = ["MonitoringReporter", "MetricsSnapshot", "collect_system_metrics"]
