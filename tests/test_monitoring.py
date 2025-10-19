from __future__ import annotations

import time

from bybit_app.utils import monitoring


def test_collect_system_metrics_parses_snapshot(monkeypatch):
    monkeypatch.setattr(monitoring, "_collect_process_metrics", lambda: (12.5, 256.0))
    snapshot = {
        "last_result": {"executions": [1, 2], "reason": ""},
        "stale": True,
        "last_cycle_at": time.time() - 5,
    }
    metrics = monitoring.collect_system_metrics(snapshot)
    assert metrics.trades_count == 2
    assert metrics.stale is True
    assert metrics.automation_lag_seconds is not None
    assert metrics.cpu_percent == 12.5
    assert metrics.memory_mb == 256.0


def test_monitoring_reporter_emits_alert(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    alerts: list[str] = []

    reporter = monitoring.MonitoringReporter(
        lambda event, **payload: events.append((event, payload)),
        alert_callback=alerts.append,
        metrics_interval=0.0,
        alert_cooldown=0.0,
        cpu_threshold=0.0,
        memory_threshold_mb=0.0,
    )

    monkeypatch.setattr(monitoring, "_collect_process_metrics", lambda: (5.0, 5.0))
    snapshot = {"error": "boom"}
    reporter.process(snapshot)

    assert alerts and "boom" in alerts[0]
    emitted = {event for event, _ in events}
    assert "monitoring.metrics" in emitted
    assert "monitoring.alert" in emitted
