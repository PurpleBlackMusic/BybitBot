import json
from pathlib import Path

import pytest

from bybit_app.utils.spot_fifo import spot_fifo_pnl


def test_spot_fifo_pnl_fifo_accounting(tmp_path: Path):
    ledger = tmp_path / "executions.jsonl"
    events = [
        {"category": "spot", "symbol": "BTCUSDT", "side": "Buy", "execPrice": "100", "execQty": "1", "execFee": "0.1"},
        {"category": "spot", "symbol": "BTCUSDT", "side": "Buy", "execPrice": "105", "execQty": "0.5", "execFee": "0.05"},
        {"category": "spot", "symbol": "BTCUSDT", "side": "Sell", "execPrice": "110", "execQty": "1.2", "execFee": "0.12"},
        {"category": "linear", "symbol": "BTCUSDT", "side": "Buy", "execPrice": "1", "execQty": "1", "execFee": "0"},
    ]
    ledger.write_text("\n".join(json.dumps(ev) for ev in events), encoding="utf-8")

    res = spot_fifo_pnl(ledger)
    btc = res["BTCUSDT"]

    assert btc["realized_pnl"] == pytest.approx(10.76)
    assert btc["position_qty"] == pytest.approx(0.3)
    assert btc["layers"] == [[pytest.approx(0.3), pytest.approx(105.1)]]


def test_spot_fifo_pnl_handles_missing_file(tmp_path: Path):
    ledger = tmp_path / "executions.jsonl"
    assert spot_fifo_pnl(ledger) == {}
