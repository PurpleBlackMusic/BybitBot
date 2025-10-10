from __future__ import annotations

import json
from pathlib import Path

import pytest

from bybit_app.utils import pnl as pnl_module
from bybit_app.utils.pnl import read_ledger


def test_read_ledger_streams_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    total = 500
    with ledger_path.open("w", encoding="utf-8") as handle:
        for idx in range(total):
            handle.write(json.dumps({"execId": str(idx), "value": idx}) + "\n")

    call_count = 0
    original_loads = pnl_module.json.loads

    def counting_loads(payload: str, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_loads(payload, *args, **kwargs)

    monkeypatch.setattr(pnl_module.json, "loads", counting_loads)

    n = 25
    rows, last_seen, marker_found = read_ledger(
        n,
        ledger_path=ledger_path,
        return_meta=True,
    )

    assert marker_found is True
    assert last_seen == str(total - 1)
    assert len(rows) == n
    expected_ids = [str(total - n + i) for i in range(n)]
    assert [row["execId"] for row in rows] == expected_ids
    assert call_count == n


def test_execution_fee_in_quote_infers_base_for_low_price_fill() -> None:
    execution = {
        "execFee": "0.1",
        "execQty": "100",
        "execPrice": "0.02",
        "symbol": "ABCUSDT",
    }

    fee_in_quote = pnl_module.execution_fee_in_quote(execution)

    assert fee_in_quote == pytest.approx(0.1 * 0.02)
    assert execution.get("feeCurrency") == "ABC"
