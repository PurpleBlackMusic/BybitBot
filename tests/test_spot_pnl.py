from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bybit_app.utils import pnl as pnl_module
from bybit_app.utils.pnl import add_execution, read_ledger
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


def test_spot_pnl_network_isolation(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "data" / "pnl"
    monkeypatch.setattr(pnl_module, "_LEDGER_DIR", base_dir)
    pnl_module._RECENT_KEYS.clear()
    pnl_module._RECENT_KEY_SET.clear()
    pnl_module._RECENT_WARMED.clear()

    testnet_settings = SimpleNamespace(testnet=True)
    mainnet_settings = SimpleNamespace(testnet=False)

    add_execution(
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderLinkId": "tn-1",
            "execPrice": "10000",
            "execQty": "0.5",
            "execFee": "0.1",
            "execTime": 1,
            "category": "spot",
        },
        settings=testnet_settings,
    )

    add_execution(
        {
            "symbol": "ETHUSDT",
            "side": "Buy",
            "orderLinkId": "mn-1",
            "execPrice": "2000",
            "execQty": "1",
            "execFee": "0.2",
            "execTime": 2,
            "category": "spot",
        },
        settings=mainnet_settings,
    )

    testnet_inventory = spot_inventory_and_pnl(settings=testnet_settings)
    mainnet_inventory = spot_inventory_and_pnl(settings=mainnet_settings)

    assert set(testnet_inventory) == {"BTCUSDT"}
    assert testnet_inventory["BTCUSDT"]["position_qty"] == 0.5
    assert set(mainnet_inventory) == {"ETHUSDT"}
    assert mainnet_inventory["ETHUSDT"]["position_qty"] == 1.0


def test_read_ledger_supports_last_exec_id(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "data" / "pnl"
    monkeypatch.setattr(pnl_module, "_LEDGER_DIR", base_dir)

    ledger_path = base_dir / "executions.testnet.jsonl"
    events = [
        {"execId": "1", "symbol": "BTCUSDT"},
        {"execId": "2", "symbol": "ETHUSDT"},
        {"execId": "3", "symbol": "SOLUSDT"},
    ]
    _write_events(ledger_path, events)

    rows, last_id, marker_found = read_ledger(
        None,
        return_meta=True,
        last_exec_id="2",
        settings=SimpleNamespace(testnet=True),
    )

    assert marker_found is True
    assert last_id == "3"
    assert rows == [events[-1]]

    rows, last_id, marker_found = read_ledger(
        None,
        return_meta=True,
        last_exec_id="missing",
        settings=SimpleNamespace(testnet=True),
    )
    assert marker_found is False
    assert last_id == "3"
    assert rows == events


def test_spot_inventory_handles_old_positions(tmp_path: Path) -> None:
    ledger_path = tmp_path / "pnl" / "executions.jsonl"
    base_event = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "execPrice": "10000",
        "execQty": "1",
        "execFee": "0",
        "execId": "1",
    }
    filler = {
        "category": "spot",
        "symbol": "ETHUSDT",
        "side": "Buy",
        "execPrice": "2000",
        "execQty": "1",
        "execFee": "0",
    }

    events = [base_event]
    for idx in range(5000):
        filler_event = dict(filler)
        filler_event["execId"] = f"f{idx}"
        events.append(filler_event)

    _write_events(ledger_path, events)

    inventory = spot_inventory_and_pnl(ledger_path=ledger_path)
    assert inventory["BTCUSDT"]["position_qty"] == 1.0

    cache_path = ledger_path.with_name(ledger_path.name + ".spot_cache.json")
    assert cache_path.exists()

    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "category": "spot",
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "execPrice": "11000",
                    "execQty": "0.4",
                    "execFee": "0",
                    "execId": "latest",
                }
            )
            + "\n"
        )

    updated = spot_inventory_and_pnl(ledger_path=ledger_path)
    btc = updated["BTCUSDT"]
    assert abs(btc["position_qty"] - 0.6) < 1e-12
    assert abs(btc["realized_pnl"] - 400.0) < 1e-9
