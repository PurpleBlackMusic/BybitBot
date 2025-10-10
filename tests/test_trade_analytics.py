from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bybit_app.utils.trade_analytics import aggregate_execution_metrics, load_executions


def _write_ledger(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "pnl" / "executions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return path


def test_load_executions_normalises_json(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    events = [
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "execPrice": "27000",
            "execQty": "0.01",
            "execTime": now.timestamp(),
            "execFee": "0.00001",
            "feeCurrency": "BTC",
            "isMaker": True,
        },
        {
            "symbol": "ETHUSDT",
            "side": "sell",
            "execPrice": "1800",
            "execQty": "0.5",
            "execTimeNs": int(now.timestamp() * 1e9),
        },
    ]
    path = _write_ledger(tmp_path, events)

    records = load_executions(path)
    assert len(records) == 2
    assert records[0].symbol == "BTCUSDT"
    assert records[0].side == "buy"
    assert records[0].notional == 270.0
    assert records[0].raw_fee == pytest.approx(0.00001)
    assert records[0].fee == pytest.approx(0.27)
    assert records[0].is_maker is True
    assert records[0].timestamp is not None


def test_aggregate_execution_metrics(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    events = [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": 27000,
            "execQty": 0.01,
            "execTime": (now - timedelta(minutes=5)).timestamp(),
            "execFee": 0.12,
            "isMaker": True,
        },
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execPrice": 27400,
            "execQty": 0.005,
            "execTime": (now - timedelta(minutes=2)).timestamp(),
            "execFee": 0.06,
            "isMaker": False,
        },
        {
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": 1850,
            "execQty": 0.4,
            "execTime": (now - timedelta(hours=2)).timestamp(),
        },
    ]
    path = _write_ledger(tmp_path, events)

    records = load_executions(path)
    metrics = aggregate_execution_metrics(records)

    assert metrics["trades"] == 3
    assert "BTCUSDT" in metrics["symbols"]
    assert metrics["gross_volume"] > 0
    assert metrics["last_trade_ts"] is not None
    assert metrics["activity"]["15m"] >= 2
    assert metrics["activity"]["1h"] >= 2
    assert metrics["per_symbol"]
    btc_row = next(row for row in metrics["per_symbol"] if row["symbol"] == "BTCUSDT")
    assert btc_row["trades"] == 2
    assert btc_row["volume"] > 0
