import json
import time
from pathlib import Path

import pytest

from bybit_app.utils.market_scanner import MarketScannerError, scan_market_opportunities


def _write_snapshot(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    snapshot = {"ts": time.time(), "rows": rows}
    path = tmp_path / "ai" / "market_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")


def test_market_scanner_ranks_opportunities(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "BTCUSDT",
            "turnover24h": "8000000",
            "price24hPcnt": "2.5",
            "bestBidPrice": "27000",
            "bestAskPrice": "27001",
            "volume24h": "3500",
        },
        {
            "symbol": "ETHUSDT",
            "turnover24h": "5000000",
            "price24hPcnt": "1.4",
            "bestBidPrice": "1800",
            "bestAskPrice": "1800.5",
            "volume24h": "2800",
        },
        {
            "symbol": "ADAUSDT",
            "turnover24h": "2200000",
            "price24hPcnt": "-3.2",
            "bestBidPrice": "0.5",
            "bestAskPrice": "0.501",
            "volume24h": "5000000",
        },
        {
            "symbol": "LOWUSDT",
            "turnover24h": "10000",
            "price24hPcnt": "6.0",
            "bestBidPrice": "0.1",
            "bestAskPrice": "0.11",
            "volume24h": "200000",
        },
    ]

    _write_snapshot(tmp_path, rows)

    opportunities = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        limit=3,
        min_turnover=1_000_000.0,
        min_change_pct=1.0,
        max_spread_bps=50.0,
    )

    symbols = [entry["symbol"] for entry in opportunities]
    assert symbols == ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    assert opportunities[0]["source"] == "market_scanner"
    assert opportunities[2]["trend"] == "sell"
    assert opportunities[0]["note"]


def test_market_scanner_respects_whitelist(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "LOWUSDT",
            "turnover24h": "20000",
            "price24hPcnt": "0.8",
            "bestBidPrice": "0.1",
            "bestAskPrice": "0.1005",
            "volume24h": "120000",
        }
    ]

    _write_snapshot(tmp_path, rows)

    opportunities = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        whitelist=["LOWUSDT"],
        min_turnover=1_000_000.0,
        min_change_pct=2.0,
        max_spread_bps=40.0,
        limit=5,
    )

    assert len(opportunities) == 1
    entry = opportunities[0]
    assert entry["symbol"] == "LOWUSDT"
    assert entry["actionable"] is False
    assert entry["probability"] is None or entry["probability"] >= 0.5


def test_market_scanner_honors_blacklist(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "SOLUSDT",
            "turnover24h": "5500000",
            "price24hPcnt": "1.2",
            "bestBidPrice": "20.1",
            "bestAskPrice": "20.13",
            "volume24h": "840000",
        },
        {
            "symbol": "XRPUSDT",
            "turnover24h": "4200000",
            "price24hPcnt": "2.4",
            "bestBidPrice": "0.5",
            "bestAskPrice": "0.5006",
            "volume24h": "12000000",
        },
    ]

    _write_snapshot(tmp_path, rows)

    opportunities = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        blacklist=["XRPUSDT"],
        min_turnover=1_000_000.0,
        min_change_pct=0.5,
        max_spread_bps=40.0,
    )

    assert [entry["symbol"] for entry in opportunities] == ["SOLUSDT"]


def test_market_scanner_raises_when_snapshot_missing(tmp_path: Path) -> None:
    with pytest.raises(MarketScannerError):
        scan_market_opportunities(api=None, data_dir=tmp_path)


def test_market_scanner_converts_usdc_symbols(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "SOLUSDC",
            "turnover24h": "2000000",
            "price24hPcnt": "1.2",
            "bestBidPrice": "19.8",
            "bestAskPrice": "19.85",
            "volume24h": "3200000",
        },
        {
            "symbol": "BTCEUR",
            "turnover24h": "5000000",
            "price24hPcnt": "0.4",
            "bestBidPrice": "27000",
            "bestAskPrice": "27010",
            "volume24h": "1000",
        },
    ]

    _write_snapshot(tmp_path, rows)

    opportunities = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        min_turnover=1000.0,
        min_change_pct=0.5,
        max_spread_bps=100.0,
        limit=5,
    )

    assert [entry["symbol"] for entry in opportunities] == ["SOLUSDT"]
    conversion = opportunities[0].get("quote_conversion")
    assert conversion == {"from": "USDC", "to": "USDT", "original": "SOLUSDC"}
