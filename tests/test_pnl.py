from __future__ import annotations

import json
from pathlib import Path

from bybit_app.utils import pnl as pnl_module
from bybit_app.utils.envs import Settings


def test_add_execution_seeds_legacy_ledger(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "pnl" / "executions.jsonl"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    legacy_event = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderId": "legacy-1",
        "execId": "fill-1",
        "execPrice": "10000",
        "execQty": "0.1",
        "execFee": "0",
        "execTime": 111,
        "category": "spot",
    }
    exec_key = pnl_module._execution_key(legacy_event)
    base_path.write_text(
        json.dumps({**legacy_event, "execKey": exec_key}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(pnl_module, "LEDGER", base_path)
    monkeypatch.setattr(pnl_module, "_RECENT_KEYS", {})
    monkeypatch.setattr(pnl_module, "_RECENT_KEY_SET", {})
    monkeypatch.setattr(pnl_module, "_RECENT_WARMED", set())

    settings = Settings(testnet=True)

    pnl_module.add_execution(legacy_event, settings=settings)

    network_path = pnl_module.ledger_path(settings)
    assert network_path != base_path
    assert not network_path.exists()

    assert base_path.read_text(encoding="utf-8").count("\n") == 1
