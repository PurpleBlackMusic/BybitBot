from __future__ import annotations

import json
from pathlib import Path

from bybit_app.utils.spot_pnl import spot_inventory_and_pnl


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def test_spot_pnl_skips_invalid_lines(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("{\n", encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "10000",
            "execQty": "0.1",
            "execFee": "0",
        }) + "\n")

    inv = spot_inventory_and_pnl(ledger_path=ledger_path)
    assert "BTCUSDT" in inv
    assert inv["BTCUSDT"]["position_qty"] == 0.1


def test_spot_pnl_caps_sell_volume(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    events = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execPrice": "1000",
            "execQty": "1",
            "execFee": "0",
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "execPrice": "1200",
            "execQty": "2",
            "execFee": "0",
        },
    ]
    _write_events(ledger_path, events)

    inv = spot_inventory_and_pnl(ledger_path=ledger_path)
    eth = inv["ETHUSDT"]
    assert eth["position_qty"] == 0.0
    assert eth["avg_cost"] == 0.0
    assert abs(eth["realized_pnl"] - 200.0) < 1e-9
