import json
from pathlib import Path

import pytest

from bybit_app.utils import trade_pairs


@pytest.fixture
def pnl_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "_data"
    pnl = base / "pnl"
    pnl.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(trade_pairs, "DATA_DIR", base)
    trade_pairs.DEC = pnl / "decisions.jsonl"
    trade_pairs.LED = pnl / "executions.testnet.jsonl"
    trade_pairs.TRD = pnl / "trades.jsonl"
    trade_pairs._PAIR_CACHE.clear()
    return pnl


def test_pair_trades_allocates_buy_fees_proportionally(pnl_dir: Path):
    executions = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execTime": 1000,
            "execPrice": 100,
            "execQty": 10,
            "execFee": 5,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "execTime": 2000,
            "execPrice": 110,
            "execQty": 4,
            "execFee": 0.2,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "execTime": 3000,
            "execPrice": 120,
            "execQty": 3,
            "execFee": 0.3,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    first, second = trades

    assert first["qty"] == pytest.approx(4)
    assert first["fees"] == pytest.approx(2.2)
    assert first["entry_value"] == pytest.approx(400)
    assert first["exit_value"] == pytest.approx(440)
    assert first["gross_pnl"] == pytest.approx(40)
    assert first["net_pnl"] == pytest.approx(37.8)
    assert first["slippage_bps"] is None

    assert second["qty"] == pytest.approx(3)
    assert second["fees"] == pytest.approx(1.8)
    assert second["entry_value"] == pytest.approx(300)
    assert second["exit_value"] == pytest.approx(360)
    assert second["gross_pnl"] == pytest.approx(60)
    assert second["net_pnl"] == pytest.approx(58.2)

    remaining_fee = trade_pairs._read_jsonl(trade_pairs.TRD)
    assert remaining_fee == trades


def test_pair_trades_preserves_fee_signs(pnl_dir: Path) -> None:
    executions = [
        {
            "category": "spot",
            "symbol": "ARBUSDT",
            "side": "Buy",
            "orderLinkId": "ARB-1",
            "execTime": 100,
            "execPrice": 1.0,
            "execQty": 10,
            "execFee": -0.05,
        },
        {
            "category": "spot",
            "symbol": "ARBUSDT",
            "side": "Sell",
            "orderLinkId": "ARB-1",
            "execTime": 200,
            "execPrice": 1.1,
            "execQty": 10,
            "execFee": -0.02,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["fees"] == pytest.approx(-0.07)
    assert trade["entry_value"] == pytest.approx(10)
    assert trade["exit_value"] == pytest.approx(11)
    assert trade["gross_pnl"] == pytest.approx(1)
    assert trade["net_pnl"] == pytest.approx(1.07)


def test_pair_trades_appends_completed_trades(pnl_dir: Path) -> None:
    decisions_path = pnl_dir / "decisions.jsonl"
    ledger_path = pnl_dir / "executions.testnet.jsonl"
    trades_path = pnl_dir / "trades.jsonl"

    decisions_path.write_text("", encoding="utf-8")

    first_wave = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "SOL-1",
            "execTime": 1000,
            "execPrice": 20,
            "execQty": 1,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "SOL-1",
            "execTime": 2000,
            "execPrice": 25,
            "execQty": 1,
            "execFee": 0.03,
        },
    ]

    ledger_path.write_text(
        "\n".join(json.dumps(entry) for entry in first_wave),
        encoding="utf-8",
    )

    trades_a = trade_pairs.pair_trades()
    assert len(trades_a) == 1
    assert trades_a[0]["gross_pnl"] == pytest.approx(5)
    assert trades_a[0]["net_pnl"] == pytest.approx(4.95)

    stored_after_first = [json.loads(line) for line in trades_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(stored_after_first) == 1
    assert stored_after_first[0]["net_pnl"] == pytest.approx(4.95)

    second_wave = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "SOL-2",
            "execTime": 3000,
            "execPrice": 18,
            "execQty": 2,
            "execFee": 0.04,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "SOL-2",
            "execTime": 4000,
            "execPrice": 22,
            "execQty": 2,
            "execFee": 0.05,
        },
    ]

    with ledger_path.open("a", encoding="utf-8") as handle:
        for entry in second_wave:
            handle.write("\n" + json.dumps(entry))

    trades_b = trade_pairs.pair_trades()
    assert len(trades_b) == 2
    assert trades_b[0]["net_pnl"] == pytest.approx(4.95)
    assert trades_b[1]["net_pnl"] == pytest.approx(7.91)

    stored_after_second = [
        json.loads(line)
        for line in trades_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_after_second) == 2
    assert {row["orderLinkId"] for row in stored_after_second} == {"SOL-1", "SOL-2"}
    net_values = {row["orderLinkId"]: row["net_pnl"] for row in stored_after_second}
    assert net_values["SOL-1"] == pytest.approx(4.95)
    assert net_values["SOL-2"] == pytest.approx(7.91)

    trade_pairs.pair_trades()
    stored_after_third = [
        json.loads(line)
        for line in trades_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_after_third) == 2


def test_pair_trades_skips_placeholder_links_on_mainnet(
    pnl_dir: Path,
) -> None:
    decisions_path = pnl_dir / "decisions.jsonl"
    ledger_path = pnl_dir / "executions.mainnet.jsonl"
    trades_path = pnl_dir / "trades.jsonl"

    decisions_path.write_text("", encoding="utf-8")

    events = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderLinkId": "abc",
            "execTime": 1000,
            "execPrice": 10,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderLinkId": "abc",
            "execTime": 2000,
            "execPrice": 11,
            "execQty": 1,
            "execFee": 0.01,
        },
    ]

    ledger_path.write_text(
        "\n".join(json.dumps(entry) for entry in events),
        encoding="utf-8",
    )

    trade_pairs.LED = ledger_path
    trade_pairs.pair_trades(network="mainnet")

    assert (
        not trades_path.exists()
        or trades_path.read_text(encoding="utf-8").strip() == ""
    )


def test_pair_trades_prefers_matching_order_link_id(pnl_dir: Path):
    decisions = [
        {"symbol": "ETHUSDT", "ts": 90, "decision_mid": "mid-a", "sl": 90, "rr": 2},
        {"symbol": "ETHUSDT", "ts": 190, "decision_mid": "mid-b", "sl": 95, "rr": 3},
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-A",
            "execTime": 100,
            "execPrice": 100,
            "execQty": 1,
            "execFee": 0.1,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-B",
            "execTime": 200,
            "execPrice": 200,
            "execQty": 1,
            "execFee": 0.2,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-B",
            "execTime": 300,
            "execPrice": 300,
            "execQty": 1,
            "execFee": 0.3,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-A",
            "execTime": 400,
            "execPrice": 150,
            "execQty": 1,
            "execFee": 0.15,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    first, second = trades

    assert first["entry_vwap"] == pytest.approx(200)
    assert first["exit_vwap"] == pytest.approx(300)
    assert first["fees"] == pytest.approx(0.2 + 0.3)
    assert first["decision_mid"] == "mid-b"
    assert first["gross_pnl"] == pytest.approx(100)
    assert first["net_pnl"] == pytest.approx(99.5)

    assert second["entry_vwap"] == pytest.approx(100)
    assert second["exit_vwap"] == pytest.approx(150)
    assert second["fees"] == pytest.approx(0.1 + 0.15)
    assert second["decision_mid"] == "mid-a"
    assert second["gross_pnl"] == pytest.approx(50)
    assert second["net_pnl"] == pytest.approx(49.75)


def test_pair_trades_calculates_slippage(pnl_dir: Path) -> None:
    decisions = [
        {"symbol": "BTCUSDT", "ts": 50, "orderLinkId": "BTC-1", "decision_mid": 100.0},
        {"symbol": "BTCUSDT", "ts": 150, "orderLinkId": "BTC-1", "decision_mid": 105.0},
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderLinkId": "BTC-1",
            "execTime": 100,
            "execPrice": 101.0,
            "execQty": 1,
            "execFee": 0.05,
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderLinkId": "BTC-1",
            "execTime": 200,
            "execPrice": 104.0,
            "execQty": 1,
            "execFee": 0.04,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["entry_mid"] == pytest.approx(100.0)
    assert trade["exit_mid"] == pytest.approx(105.0)
    assert trade["entry_slippage_bps"] == pytest.approx(100.0)
    assert trade["exit_slippage_bps"] == pytest.approx((104.0 / 105.0 - 1.0) * 10000.0)
    assert trade["slippage_bps"] == trade["exit_slippage_bps"]
    assert trade["gross_pnl"] == pytest.approx(3.0)
    assert trade["net_pnl"] == pytest.approx(2.91)


def test_pair_trades_discards_buys_outside_window(pnl_dir: Path):
    executions = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execTime": 0,
            "execPrice": 100,
            "execQty": 1,
            "execFee": 0.1,
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execTime": 8 * 24 * 3600 * 1000,
            "execPrice": 110,
            "execQty": 1,
            "execFee": 0.11,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades(window_ms=7 * 24 * 3600 * 1000)

    assert trades == []


def test_pair_trades_retains_recent_fill_when_pruning_stale_link(pnl_dir: Path):
    day = 24 * 3600 * 1000
    executions = [
        {
            "category": "spot",
            "symbol": "ADAUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-ADA",
            "execTime": 0,
            "execPrice": 0.3,
            "execQty": 1,
            "execFee": 0.001,
        },
        {
            "category": "spot",
            "symbol": "ADAUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-ADA",
            "execTime": 9 * day,
            "execPrice": 0.33,
            "execQty": 1,
            "execFee": 0.0011,
        },
        {
            "category": "spot",
            "symbol": "ADAUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-ADA",
            "execTime": 12 * day,
            "execPrice": 0.4,
            "execQty": 1,
            "execFee": 0.0012,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades(window_ms=7 * day)

    assert len(trades) == 1
    trade = trades[0]

    assert trade["qty"] == pytest.approx(1)
    assert trade["entry_vwap"] == pytest.approx(0.33)
    assert trade["fees"] == pytest.approx(0.000363 + 0.00048)
    assert trade["gross_pnl"] == pytest.approx(0.4 - 0.33)
    assert trade["net_pnl"] == pytest.approx((0.4 - 0.33) - (0.000363 + 0.00048))

    expected_hold = int(max(0, (12 * day - 9 * day) / 1000))
    assert trade["hold_sec"] == expected_hold


def test_linked_sell_preserves_lot_outside_window(pnl_dir: Path) -> None:
    day = 24 * 3600 * 1000
    executions = [
        {
            "category": "spot",
            "symbol": "XLMUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-XLM",
            "execTime": 0,
            "execPrice": 0.1,
            "execQty": 2,
            "execFee": 0.0002,
        },
        {
            "category": "spot",
            "symbol": "XLMUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-XLM",
            "execTime": 10 * day,
            "execPrice": 0.12,
            "execQty": 2,
            "execFee": 0.00024,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades(window_ms=7 * day)

    assert len(trades) == 1
    trade = trades[0]

    assert trade["qty"] == pytest.approx(2)
    assert trade["entry_vwap"] == pytest.approx(0.1)


def test_pair_trades_merges_context_when_extending_link_lot(pnl_dir: Path):
    decisions = [
        {"symbol": "SOLUSDT", "ts": 50, "decision_mid": "mid-initial"},
        {"symbol": "SOLUSDT", "ts": 150, "decision_mid": "mid-final", "sl": 80, "rr": 4},
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 60,
            "execPrice": 20,
            "execQty": 2,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 160,
            "execPrice": 22,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 200,
            "execPrice": 30,
            "execQty": 3,
            "execFee": 0.03,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]
    assert trade["decision_mid"] == "mid-final"
    assert trade["qty"] == pytest.approx(3)
    assert trade["fees"] == pytest.approx(0.02 + 0.01 + 0.03)
    assert trade["entry_vwap"] == pytest.approx((2 * 20 + 1 * 22) / 3)


def test_pair_trades_computes_r_mult_when_sl_or_rr_is_zero(pnl_dir: Path):
    decisions = [
        {"symbol": "XRPUSDT", "ts": 10, "sl": 0.0, "rr": 0},
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "XRPUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-XRP",
            "execTime": 20,
            "execPrice": 100,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "XRPUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-XRP",
            "execTime": 40,
            "execPrice": 110,
            "execQty": 1,
            "execFee": 0.02,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["r_mult"] == pytest.approx((110 - 100) / (100 - 0))


def test_targeted_context_can_clear_sl_and_rr(pnl_dir: Path) -> None:
    decisions = [
        {"symbol": "BTCUSDT", "ts": 5, "sl": 95.0, "rr": 3.0},
        {"symbol": "BTCUSDT", "ts": 15, "orderLinkId": "BTC-LINK", "sl": 90.0, "rr": 2.0},
        {
            "symbol": "BTCUSDT",
            "ts": 25,
            "orderLinkId": "BTC-LINK",
            "sl": None,
            "rr": None,
        },
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderLinkId": "BTC-LINK",
            "execTime": 10,
            "execPrice": 100.0,
            "execQty": 1.0,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "orderLinkId": "BTC-LINK",
            "execTime": 40,
            "execPrice": 110.0,
            "execQty": 1.0,
            "execFee": 0.02,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["sl"] is None
    assert trade["rr"] is None


def test_symbol_context_persists_after_targeted_decision(pnl_dir: Path):
    decisions = [
        {"symbol": "ETHUSDT", "ts": 10, "decision_mid": "broad"},
        {
            "symbol": "ETHUSDT",
            "ts": 18,
            "decision_mid": "targeted",
            "orderLinkId": "LINK-ETH",
        },
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "orderLinkId": "LINK-ETH",
            "execTime": 20,
            "execPrice": 100,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Buy",
            "execTime": 25,
            "execPrice": 90,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "orderLinkId": "LINK-ETH",
            "execTime": 30,
            "execPrice": 110,
            "execQty": 1,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "ETHUSDT",
            "side": "Sell",
            "execTime": 35,
            "execPrice": 95,
            "execQty": 1,
            "execFee": 0.02,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    linked_trade = next(t for t in trades if t["orderLinkId"] == "LINK-ETH")
    unlinked_trade = next(t for t in trades if t["orderLinkId"] is None)

    assert linked_trade["decision_mid"] == "targeted"
    assert unlinked_trade["decision_mid"] == "broad"


def test_pair_trades_refreshes_context_before_sell(pnl_dir: Path):
    decisions = [
        {"symbol": "BNBUSDT", "ts": 10, "decision_mid": "initial", "sl": 150, "rr": 1, "orderLinkId": "ENTRY-BNB"},
        {
            "symbol": "BNBUSDT",
            "ts": 80,
            "decision_mid": "updated",
            "sl": 190,
            "rr": 2,
            "orderLinkId": "ENTRY-BNB",
        },
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "BNBUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-BNB",
            "execTime": 20,
            "execPrice": 200,
            "execQty": 1,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "BNBUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-BNB",
            "execTime": 120,
            "execPrice": 220,
            "execQty": 1,
            "execFee": 0.03,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["decision_mid"] == "updated"
    assert trade["r_mult"] == pytest.approx((220 - 200) / (200 - 190))


def test_pair_trades_skips_targeted_context_for_unlinked_entry(pnl_dir: Path):
    decisions = [
        {
            "symbol": "DOGEUSDT",
            "ts": 10,
            "decision_mid": "targeted",
            "sl": 0.05,
            "rr": 3,
            "orderLinkId": "ENTRY-DOGE",
        }
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "DOGEUSDT",
            "side": "Buy",
            "execTime": 20,
            "execPrice": 0.06,
            "execQty": 100,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "DOGEUSDT",
            "side": "Sell",
            "execTime": 40,
            "execPrice": 0.065,
            "execQty": 100,
            "execFee": 0.01,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["decision_mid"] is None
    assert trade["sl"] is None
    assert trade["rr"] is None


def test_pair_trades_applies_symbol_context_to_linked_exit(pnl_dir: Path):
    decisions = [
        {"symbol": "NEARUSDT", "ts": 10, "decision_mid": "initial", "sl": 6.5, "rr": 1},
        {"symbol": "NEARUSDT", "ts": 80, "decision_mid": "global-update", "sl": 7.5, "rr": 2},
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "NEARUSDT",
            "side": "Buy",
            "execTime": 20,
            "execPrice": 10,
            "execQty": 2,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "NEARUSDT",
            "side": "Sell",
            "execTime": 120,
            "execPrice": 11,
            "execQty": 2,
            "execFee": 0.022,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["decision_mid"] == "global-update"
    assert trade["sl"] == pytest.approx(7.5)
    assert trade["rr"] == pytest.approx(2)


def test_pair_trades_does_not_apply_targeted_context_to_unlinked_lot(pnl_dir: Path):
    decisions = [
        {"symbol": "LTCUSDT", "ts": 90, "decision_mid": "global"},
        {
            "symbol": "LTCUSDT",
            "ts": 120,
            "decision_mid": "targeted",
            "orderLinkId": "ENTRY-LTC",
            "sl": 40,
            "rr": 3,
        },
    ]

    executions = [
        {
            "category": "spot",
            "symbol": "LTCUSDT",
            "side": "Buy",
            "execTime": 100,
            "execPrice": 30,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "LTCUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-LTC",
            "execTime": 130,
            "execPrice": 35,
            "execQty": 1,
            "execFee": 0.015,
        },
        {
            "category": "spot",
            "symbol": "LTCUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-LTC",
            "execTime": 140,
            "execPrice": 40,
            "execQty": 1,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "LTCUSDT",
            "side": "Sell",
            "execTime": 150,
            "execPrice": 32,
            "execQty": 1,
            "execFee": 0.012,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions), encoding="utf-8"
    )
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    targeted_trade, unlinked_trade = trades

    assert targeted_trade["decision_mid"] == "targeted"
    assert targeted_trade["sl"] == pytest.approx(40)
    assert targeted_trade["rr"] == pytest.approx(3)

    assert unlinked_trade["decision_mid"] == "global"
    assert unlinked_trade["sl"] is None
    assert unlinked_trade["rr"] is None


def test_pair_trades_records_order_link_id_on_trades(pnl_dir: Path):
    executions = [
        {
            "category": "spot",
            "symbol": "LINKUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-LINK",
            "execTime": 10,
            "execPrice": 7.5,
            "execQty": 2,
            "execFee": 0.02,
        },
        {
            "category": "spot",
            "symbol": "LINKUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-LINK",
            "execTime": 20,
            "execPrice": 8.0,
            "execQty": 2,
            "execFee": 0.021,
        },
        {
            "category": "spot",
            "symbol": "LINKUSDT",
            "side": "Buy",
            "execTime": 30,
            "execPrice": 6.5,
            "execQty": 1,
            "execFee": 0.01,
        },
        {
            "category": "spot",
            "symbol": "LINKUSDT",
            "side": "Sell",
            "execTime": 40,
            "execPrice": 6.75,
            "execQty": 1,
            "execFee": 0.01,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    linked_trade, unlinked_trade = trades

    assert linked_trade["orderLinkId"] == "ENTRY-LINK"
    assert unlinked_trade["orderLinkId"] is None


def test_pair_trades_skips_mismatched_link_sell_for_unlinked_lot(pnl_dir: Path):
    executions = [
        {
            "category": "spot",
            "symbol": "ATOMUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-ATOM",
            "execTime": 10,
            "execPrice": 12,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "ATOMUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-MISMATCH",
            "execTime": 20,
            "execPrice": 14,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "ATOMUSDT",
            "side": "Sell",
            "execTime": 30,
            "execPrice": 13,
            "execQty": 1,
            "execFee": 0.0,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["entry_ts"] == 10
    assert trade["exit_ts"] == 30
    assert trade["orderLinkId"] == "ENTRY-ATOM"
    assert trade["exit_vwap"] == pytest.approx(13)


def test_pair_trades_skips_targeted_sell_when_link_qty_is_insufficient(
    pnl_dir: Path,
) -> None:
    executions = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 10,
            "execPrice": 20,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 20,
            "execPrice": 22,
            "execQty": 2,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "execTime": 30,
            "execPrice": 21,
            "execQty": 1,
            "execFee": 0.0,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 1
    trade = trades[0]

    assert trade["entry_ts"] == 10
    assert trade["exit_ts"] == 30
    assert trade["orderLinkId"] == "ENTRY-SOL"
    assert trade["exit_vwap"] == pytest.approx(21)


def test_unlinked_sell_prefers_unlinked_lot_when_available(pnl_dir: Path) -> None:
    executions = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 10,
            "execPrice": 20,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "execTime": 20,
            "execPrice": 21,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "execTime": 30,
            "execPrice": 22,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 40,
            "execPrice": 23,
            "execQty": 1,
            "execFee": 0.0,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2

    unlinked_trade = next(t for t in trades if t["orderLinkId"] is None)
    linked_trade = next(t for t in trades if t["orderLinkId"] == "ENTRY-SOL")

    assert unlinked_trade["entry_ts"] == 20
    assert unlinked_trade["exit_ts"] == 30
    assert unlinked_trade["exit_vwap"] == pytest.approx(22)

    assert linked_trade["entry_ts"] == 10
    assert linked_trade["exit_ts"] == 40
    assert linked_trade["exit_vwap"] == pytest.approx(23)


def test_unlinked_sell_skips_when_unlinked_qty_is_insufficient(pnl_dir: Path) -> None:
    executions = [
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "execTime": 10,
            "execPrice": 20,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Buy",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 20,
            "execPrice": 21,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "execTime": 30,
            "execPrice": 22,
            "execQty": 2,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "execTime": 40,
            "execPrice": 23,
            "execQty": 1,
            "execFee": 0.0,
        },
        {
            "category": "spot",
            "symbol": "SOLUSDT",
            "side": "Sell",
            "orderLinkId": "ENTRY-SOL",
            "execTime": 50,
            "execPrice": 24,
            "execQty": 1,
            "execFee": 0.0,
        },
    ]

    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    (pnl_dir / "executions.testnet.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in executions), encoding="utf-8"
    )

    trades = trade_pairs.pair_trades()

    assert len(trades) == 2
    unlinked_trade = next(t for t in trades if t["orderLinkId"] is None)
    linked_trade = next(t for t in trades if t["orderLinkId"] == "ENTRY-SOL")

    assert unlinked_trade["entry_ts"] == 10
    assert unlinked_trade["exit_ts"] == 40
    assert linked_trade["entry_ts"] == 20
    assert linked_trade["exit_ts"] == 50


def test_pair_trades_uses_cache_for_repeated_calls(
    pnl_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = [{"symbol": "BTCUSDT", "ts": 1}]
    large_ledger: list[dict[str, object]] = []
    for idx in range(100):
        ts_base = idx * 1000
        large_ledger.append(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execTime": ts_base + 10,
                "execPrice": 100 + idx,
                "execQty": 1.0,
                "execFee": 0.1,
            }
        )
        large_ledger.append(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execTime": ts_base + 20,
                "execPrice": 105 + idx,
                "execQty": 1.0,
                "execFee": 0.05,
            }
        )

    (pnl_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in decisions), encoding="utf-8"
    )
    ledger_path = pnl_dir / "executions.testnet.jsonl"
    ledger_path.write_text(
        "\n".join(json.dumps(row) for row in large_ledger), encoding="utf-8"
    )

    original_read = trade_pairs._read_jsonl
    read_calls: list[Path] = []

    def counting_read(path: Path, *args, **kwargs) -> list[dict]:
        read_calls.append(Path(path))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(trade_pairs, "_read_jsonl", counting_read)

    first = trade_pairs.pair_trades()
    second = trade_pairs.pair_trades()

    assert first is not second
    assert first == second
    assert len(read_calls) == 3
    assert read_calls.count(ledger_path) == 1
    assert read_calls.count(pnl_dir / "decisions.jsonl") == 1
    assert read_calls.count(trade_pairs.TRD) == 1

    # Modify ledger to invalidate cache and force another read
    extra_buy = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "execTime": 200000,
        "execPrice": 110,
        "execQty": 1.0,
        "execFee": 0.02,
    }
    extra_sell = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Sell",
        "execTime": 200010,
        "execPrice": 125,
        "execQty": 1.0,
        "execFee": 0.02,
    }
    large_ledger.extend([extra_buy, extra_sell])
    ledger_path.write_text(
        "\n".join(json.dumps(row) for row in large_ledger), encoding="utf-8"
    )

    third = trade_pairs.pair_trades()
    assert len(read_calls) == 6
    assert len(third) == len(first) + 1


def test_pair_trades_reads_recent_slice_for_large_ledger(
    pnl_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (pnl_dir / "decisions.jsonl").write_text("", encoding="utf-8")

    gap = 60_000
    total_trades = 600
    recent_count = 5
    ledger_rows: list[dict[str, object]] = []

    for idx in range(total_trades):
        base_ts = idx * gap
        ledger_rows.append(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "execTime": base_ts,
                "execPrice": 100 + idx,
                "execQty": 1.0,
                "execFee": 0.01,
            }
        )
        ledger_rows.append(
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execTime": base_ts + gap // 2,
                "execPrice": 101 + idx,
                "execQty": 1.0,
                "execFee": 0.01,
            }
        )

    ledger_path = pnl_dir / "executions.testnet.jsonl"
    ledger_path.write_text(
        "\n".join(json.dumps(row) for row in ledger_rows), encoding="utf-8"
    )

    original_read = trade_pairs._read_jsonl
    ledger_rows_read: list[int] = []

    def recording_read(path: Path, *args, **kwargs):
        rows = original_read(path, *args, **kwargs)
        if Path(path) == trade_pairs.LED:
            ledger_rows_read.append(len(rows))
        return rows

    monkeypatch.setattr(trade_pairs, "_read_jsonl", recording_read)

    window_ms = 5 * gap - 1_000
    trades = trade_pairs.pair_trades(window_ms=window_ms)

    assert len(trades) == recent_count

    start_recent_base = (total_trades - recent_count) * gap
    assert min(trade["entry_ts"] for trade in trades) >= start_recent_base

    latest_exec_ts = (total_trades - 1) * gap + gap // 2
    cutoff = latest_exec_ts - window_ms
    assert min(trade["exit_ts"] for trade in trades) >= cutoff

    assert ledger_rows_read
    assert ledger_rows_read[0] < len(ledger_rows)
