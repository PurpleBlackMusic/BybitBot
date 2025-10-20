import json
import time
from pathlib import Path

import pytest

from bybit_app.utils.ai.deepseek_ops import DeepSeekRuntimeSupervisor
from bybit_app.utils.signal_executor_models import ExecutionResult


class FakeAdapter:
    def __init__(self, *, api_key: str | None = "test") -> None:
        self.api_key = api_key
        self.calls: list[str] = []

    def get_signal(self, symbol: str) -> dict[str, object]:
        self.calls.append(symbol)
        return {"symbol": symbol, "deepseek_confidence": 0.5}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def test_runtime_supervisor_records_and_refreshes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trades_path = tmp_path / "trades.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    state_path = tmp_path / "state.json"
    status_path = tmp_path / "status.json"
    model_path = tmp_path / "model.joblib"

    status_payload = {
        "watchlist": ["BTCUSDT", "ETHUSDT"],
        "risk": {"max_drawdown_alert_pct": 7.5},
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    adapter = FakeAdapter()

    supervisor = DeepSeekRuntimeSupervisor(
        adapter_factory=lambda: adapter,
        refresh_interval=0.01,
        trades_path=trades_path,
        metrics_path=metrics_path,
        state_path=state_path,
        status_path=status_path,
        model_path=model_path,
        model_stale_after=0.1,
        alert_cooldown=0.0,
    )

    result = ExecutionResult(
        status="filled",
        reason=None,
        order={"symbol": "BTCUSDT", "qty": 1.0},
        response=None,
        context={"deepseek_signal": {"score": 0.82}},
    )

    supervisor.process_cycle(result, "sig-1", (True, False, True))

    trades = _read_jsonl(trades_path)
    metrics = _read_jsonl(metrics_path)
    assert len(trades) == 1
    assert len(metrics) == 1
    assert trades[0]["status"] == "filled"
    assert "deepseek" in trades[0]
    assert metrics[0]["drawdown_limit_pct"] == pytest.approx(7.5)
    assert adapter.calls == ["BTCUSDT", "ETHUSDT"]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("watchlist") == ["BTCUSDT", "ETHUSDT"]
    assert state.get("last_refresh_ts")
    assert state.get("model_alert_ts")


def test_runtime_supervisor_skips_refresh_without_key(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    state_path = tmp_path / "state.json"
    adapter = FakeAdapter(api_key=None)
    supervisor = DeepSeekRuntimeSupervisor(
        adapter_factory=lambda: adapter,
        refresh_interval=0.0,
        trades_path=trades_path,
        metrics_path=metrics_path,
        state_path=state_path,
        status_path=tmp_path / "status.json",
        model_path=tmp_path / "model.joblib",
        model_stale_after=0.0,
    )

    result = ExecutionResult(status="dry_run", reason=None, order=None, response=None, context=None)
    supervisor.process_cycle(result, None, (False, False, False))

    trades = _read_jsonl(trades_path)
    metrics = _read_jsonl(metrics_path)
    assert len(trades) == 1
    assert trades[0]["status"] == "dry_run"
    assert len(metrics) == 1
    assert not adapter.calls
    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    assert not state_data.get("refreshed_symbols")
