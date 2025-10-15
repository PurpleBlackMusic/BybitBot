import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import numpy as np

from bybit_app.utils.ai import models as ai_models
from bybit_app.utils.market_scanner import (
    MarketScanner,
    MarketScannerError,
    scan_market_opportunities,
    _CandleCache,
)
from bybit_app.utils.portfolio_manager import PortfolioManager
from bybit_app.utils.symbol_resolver import SymbolResolver
from bybit_app.utils.trade_analytics import ExecutionRecord


def _write_snapshot(
    tmp_path: Path, rows: list[dict[str, object]], *, testnet: bool = False
) -> None:
    snapshot = {"ts": time.time(), "rows": rows}
    filename = "market_snapshot_testnet.json" if testnet else "market_snapshot.json"
    path = tmp_path / "ai" / filename
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
            "price1hPcnt": "1.8",
            "price4hPcnt": "2.1",
            "price7dPcnt": "6.2",
            "highPrice24h": "28200",
            "lowPrice24h": "26000",
            "lastPrice": "27100",
            "highPrice1h": "27500",
            "lowPrice1h": "26800",
            "closePrice1h": "27100",
            "highPrice4h": "28400",
            "lowPrice4h": "26300",
            "closePrice4h": "27120",
            "highPrice7d": "30000",
            "lowPrice7d": "24800",
            "closePrice7d": "27200",
            "volume1h": "400",
            "prevVolume1h": "150",
            "volume4h": "1200",
            "prevVolume4h": "500",
            "prevVolume24h": "300",
            "bid1Size": "150",
            "ask1Size": "120",
            "corr_btc": "0.45",
            "corr_market": "0.35",
        },
        {
            "symbol": "ETHUSDT",
            "turnover24h": "5000000",
            "price24hPcnt": "1.4",
            "bestBidPrice": "1800",
            "bestAskPrice": "1800.5",
            "volume24h": "2800",
            "price1hPcnt": "1.3",
            "price4hPcnt": "1.6",
            "price7dPcnt": "5.1",
            "highPrice24h": "1880",
            "lowPrice24h": "1720",
            "lastPrice": "1805",
            "highPrice1h": "1825",
            "lowPrice1h": "1780",
            "closePrice1h": "1805",
            "highPrice4h": "1875",
            "lowPrice4h": "1710",
            "closePrice4h": "1800",
            "volume1h": "350",
            "prevVolume1h": "420",
            "volume4h": "1400",
            "prevVolume4h": "1550",
            "prevVolume24h": "900",
            "bid1Size": "90",
            "ask1Size": "95",
            "corr_btc": "0.92",
            "corr_market": "0.88",
        },
        {
            "symbol": "ADAUSDT",
            "turnover24h": "2200000",
            "price24hPcnt": "-3.2",
            "bestBidPrice": "0.5",
            "bestAskPrice": "0.501",
            "volume24h": "5000000",
            "price1hPcnt": "-1.5",
            "price4hPcnt": "-2.5",
            "price7dPcnt": "-4.2",
            "highPrice24h": "0.56",
            "lowPrice24h": "0.45",
            "lastPrice": "0.50",
            "highPrice1h": "0.53",
            "lowPrice1h": "0.48",
            "closePrice1h": "0.50",
            "highPrice4h": "0.57",
            "lowPrice4h": "0.44",
            "closePrice4h": "0.50",
            "volume1h": "800000",
            "prevVolume1h": "920000",
            "volume4h": "2500000",
            "prevVolume4h": "2750000",
            "prevVolume24h": "650000",
            "bid1Size": "4000000",
            "ask1Size": "5000000",
            "corr_btc": "0.28",
            "corr_market": "0.31",
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
    assert next(entry for entry in opportunities if entry["symbol"] == "ADAUSDT")[
        "trend"
    ] == "sell"
    assert opportunities[0]["note"]

    top = opportunities[0]
    assert top["volatility_pct"] and top["volatility_pct"] > 5
    assert top["volatility_windows"]["1h"] is not None
    assert top["volume_spike_score"] and top["volume_spike_score"] > 0
    assert top["volume_impulse"]["1h"] and top["volume_impulse"]["1h"] > 0
    assert top["correlations"] and "btc" in top["correlations"]
    assert top["model_metrics"]["bias"] < 0
    assert 0 <= top["probability"] <= 1
    assert top["score"] > 0
    assert "волатильность" in top["note"]
    assert "импульс объёма" in top["note"]

    ada = next(entry for entry in opportunities if entry["symbol"] == "ADAUSDT")
    assert ada["probability"] is not None
    assert ada["probability"] < opportunities[0]["probability"]
    assert ada["probability"] < opportunities[1]["probability"]
    assert ada["depth_imbalance"] is not None and ada["depth_imbalance"] < 0
    assert ada["model_metrics"]["correlation"] >= -0.5


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_training_dataset_prorates_fee_for_oversized_sell(tmp_path: Path) -> None:
    state = ai_models._SymbolState()
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    buy = ExecutionRecord(
        symbol="BTCUSDT",
        side="buy",
        qty=1.5,
        price=100.0,
        fee=0.0,
        timestamp=timestamp,
    )
    state.register_buy(buy)

    sell = ExecutionRecord(
        symbol="BTCUSDT",
        side="sell",
        qty=2.0,
        price=110.0,
        fee=2.0,
        timestamp=timestamp,
    )

    realised = state.realise_sell(sell)
    assert realised is not None
    _, pnl, _ = realised

    qty_to_close = min(abs(sell.qty), abs(buy.qty))
    expected_fee = sell.fee * (qty_to_close / abs(sell.qty))
    expected_pnl = sell.price * qty_to_close - expected_fee - buy.price * qty_to_close

    assert pnl == pytest.approx(expected_pnl)

    assert state.position_qty == 0.0


def test_default_ledger_path_prefers_most_recent(tmp_path: Path) -> None:
    data_dir = tmp_path / "ai"
    pnl_dir = data_dir / "pnl"
    testnet_path = pnl_dir / "executions.testnet.jsonl"
    mainnet_path = pnl_dir / "executions.mainnet.jsonl"

    for path in (testnet_path, mainnet_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(testnet_path, (now - 60, now - 60))
    os.utime(mainnet_path, (now - 120, now - 120))

    assert ai_models._default_ledger_path(data_dir) == testnet_path

    os.utime(mainnet_path, (now + 60, now + 60))

    assert ai_models._default_ledger_path(data_dir) == mainnet_path

    os.utime(testnet_path, (now + 120, now + 120))
    os.utime(mainnet_path, (now + 120, now + 120))

    assert ai_models._default_ledger_path(data_dir) == mainnet_path


def test_build_training_dataset_emits_recency_weights(tmp_path: Path) -> None:
    now = time.time()
    ledger_path = tmp_path / "executions.jsonl"
    records = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "execQty": "0.1",
            "execPrice": "10000",
            "execFee": "-0.1",
            "isMaker": True,
            "execTime": now - 5 * 3600,
        },
        {
            "symbol": "BTCUSDT",
            "side": "sell",
            "execQty": "0.1",
            "execPrice": "10300",
            "execFee": "0.1",
            "isMaker": False,
            "execTime": now - 5 * 3600 + 120,
        },
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "execQty": "0.2",
            "execPrice": "11000",
            "execFee": "-0.2",
            "isMaker": True,
            "execTime": now - 600,
        },
        {
            "symbol": "BTCUSDT",
            "side": "sell",
            "execQty": "0.2",
            "execPrice": "10800",
            "execFee": "0.2",
            "isMaker": False,
            "execTime": now - 480,
        },
    ]
    _write_ledger(ledger_path, records)

    matrix, labels, recency = ai_models.build_training_dataset(ledger_path=ledger_path)

    assert matrix.shape[0] == 2
    assert labels.tolist() == [1, 0]
    assert np.all(recency > 0)
    assert recency[0] < recency[1] <= 1.0


def test_build_training_dataset_handles_negative_fees(tmp_path: Path) -> None:
    now = time.time()
    ledger_path = tmp_path / "executions.jsonl"
    records = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "execQty": "1",
            "execPrice": "100",
            "execFee": "-0.1",
            "feeCurrency": "BTC",
            "isMaker": True,
            "execTime": now - 120,
        },
        {
            "symbol": "BTCUSDT",
            "side": "sell",
            "execQty": "1",
            "execPrice": "100",
            "execFee": "-0.1",
            "feeCurrency": "BTC",
            "isMaker": True,
            "execTime": now - 60,
        },
    ]
    _write_ledger(ledger_path, records)

    executions = ai_models.load_executions(ledger_path)
    assert executions[0].raw_fee == pytest.approx(-0.1)
    assert executions[0].fee == pytest.approx(-10.0)
    assert executions[1].raw_fee == pytest.approx(-0.1)
    assert executions[1].fee == pytest.approx(-10.0)

    matrix, labels, recency = ai_models.build_training_dataset(ledger_path=ledger_path)

    assert matrix.shape == (1, len(ai_models.MODEL_FEATURES))
    assert labels.tolist() == [1]
    assert recency.shape == (1,)
    assert recency[0] > 0


def test_train_market_model_logs_metrics_and_uses_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    ledger_path = tmp_path / "executions.jsonl"
    records = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "execQty": "0.1",
            "execPrice": "10000",
            "execFee": "-0.1",
            "isMaker": True,
            "execTime": now - 6 * 3600,
        },
        {
            "symbol": "BTCUSDT",
            "side": "sell",
            "execQty": "0.1",
            "execPrice": "10500",
            "execFee": "0.1",
            "isMaker": False,
            "execTime": now - 6 * 3600 + 60,
        },
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "execQty": "0.2",
            "execPrice": "11000",
            "execFee": "-0.2",
            "isMaker": True,
            "execTime": now - 900,
        },
        {
            "symbol": "BTCUSDT",
            "side": "sell",
            "execQty": "0.2",
            "execPrice": "10700",
            "execFee": "0.2",
            "isMaker": False,
            "execTime": now - 840,
        },
    ]
    _write_ledger(ledger_path, records)

    matrix, labels, recency = ai_models.build_training_dataset(ledger_path=ledger_path)
    class_weights = ai_models._balanced_sample_weights(labels)
    expected = class_weights * recency

    captured_weights: dict[str, np.ndarray] = {}
    original_fit = ai_models._WeightedLogisticRegression.fit

    def capture_fit(self, X, y, sample_weight=None):  # type: ignore[override]
        if sample_weight is not None:
            captured_weights["passed"] = np.array(sample_weight, copy=True)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(ai_models._WeightedLogisticRegression, "fit", capture_fit)

    loss_weights: list[np.ndarray] = []
    original_loss = ai_models._weighted_log_loss

    def capture_loss(
        captured_labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
    ) -> float:
        loss_weights.append(np.array(weights, copy=True))
        return original_loss(captured_labels, probabilities, weights)

    monkeypatch.setattr(ai_models, "_weighted_log_loss", capture_loss)

    logged: list[tuple[str, dict[str, object]]] = []

    def fake_log(
        event: str,
        *,
        severity: str | None = None,
        exc: BaseException | None = None,
        **payload: object,
    ) -> None:
        logged.append((event, payload))

    monkeypatch.setattr(ai_models, "log", fake_log)

    model_path = tmp_path / "ai" / "model.joblib"
    model = ai_models.train_market_model(
        data_dir=tmp_path,
        ledger_path=ledger_path,
        model_path=model_path,
        min_samples=1,
    )

    assert model is not None
    assert "passed" in captured_weights
    np.testing.assert_allclose(captured_weights["passed"], expected)
    assert loss_weights
    for weights in loss_weights:
        assert np.isclose(weights.mean(), 1.0)

    assert logged
    event, payload = logged[-1]
    assert event == "market_model.training_metrics"
    assert payload["samples"] == 2
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["log_loss"] >= 0.0
    assert payload["positive_rate"] == pytest.approx(float(labels.mean()))


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


def test_weighted_model_sorts_by_risk_features(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "ALPHAUSDT",
            "turnover24h": "3200000",
            "price24hPcnt": "3.2",
            "bestBidPrice": "14.8",
            "bestAskPrice": "14.82",
            "volume24h": "980000",
            "price1hPcnt": "1.1",
            "price4hPcnt": "2.4",
            "highPrice24h": "15.4",
            "lowPrice24h": "13.2",
            "lastPrice": "14.9",
            "highPrice1h": "15.0",
            "lowPrice1h": "14.2",
            "closePrice1h": "14.8",
            "volume1h": "52000",
            "prevVolume1h": "18000",
            "prevVolume24h": "35000",
            "bid1Size": "12000",
            "ask1Size": "8000",
            "corr_btc": "0.28",
            "corr_market": "0.32",
        },
        {
            "symbol": "BETAUSDT",
            "turnover24h": "4500000",
            "price24hPcnt": "3.5",
            "bestBidPrice": "8.1",
            "bestAskPrice": "8.12",
            "volume24h": "1200000",
            "price1hPcnt": "1.4",
            "price4hPcnt": "3.1",
            "highPrice24h": "8.9",
            "lowPrice24h": "7.0",
            "lastPrice": "8.05",
            "highPrice1h": "8.4",
            "lowPrice1h": "7.7",
            "closePrice1h": "8.0",
            "volume1h": "62000",
            "prevVolume1h": "91000",
            "prevVolume24h": "180000",
            "bid1Size": "5000",
            "ask1Size": "7500",
            "corr_btc": "0.95",
            "corr_market": "0.91",
            "highPrice4h": "9.1",
            "lowPrice4h": "6.8",
            "closePrice4h": "8.0",
        },
        {
            "symbol": "GAMMAUSDT",
            "turnover24h": "9000000",
            "price24hPcnt": "0.9",
            "bestBidPrice": "2.02",
            "bestAskPrice": "2.025",
            "volume24h": "4500000",
            "price1hPcnt": "0.4",
            "price4hPcnt": "0.8",
            "highPrice24h": "2.2",
            "lowPrice24h": "1.8",
            "lastPrice": "2.03",
            "highPrice1h": "2.05",
            "lowPrice1h": "1.95",
            "closePrice1h": "2.02",
            "volume1h": "210000",
            "prevVolume1h": "200000",
            "prevVolume24h": "430000",
            "bid1Size": "180000",
            "ask1Size": "160000",
            "corr_btc": "0.55",
            "corr_market": "0.52",
        },
    ]

    _write_snapshot(tmp_path, rows)

    opportunities = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        min_turnover=2_000_000.0,
        min_change_pct=0.5,
        max_spread_bps=60.0,
    )

    ordered = [entry["symbol"] for entry in opportunities]
    assert ordered[:3] == ["ALPHAUSDT", "GAMMAUSDT", "BETAUSDT"]

    alpha = opportunities[0]
    beta = next(entry for entry in opportunities if entry["symbol"] == "BETAUSDT")

    assert alpha["model_metrics"]["correlation"] > beta["model_metrics"]["correlation"]
    assert beta["model_metrics"]["correlation"] < -0.4
    assert alpha["volume_impulse"]["1h"] > 0
    assert beta["volume_impulse"]["1h"] is not None and beta["volume_impulse"]["1h"] < 0
    assert opportunities[0]["probability"] > opportunities[-1]["probability"]


def test_market_scanner_uses_network_specific_snapshot(tmp_path: Path) -> None:
    mainnet_rows = [
        {
            "symbol": "MAINUSDT",
            "turnover24h": "3000000",
            "price24hPcnt": "1.5",
            "bestBidPrice": "1.0",
            "bestAskPrice": "1.01",
            "volume24h": "1200000",
        }
    ]
    testnet_rows = [
        {
            "symbol": "TESTUSDT",
            "turnover24h": "4000000",
            "price24hPcnt": "2.5",
            "bestBidPrice": "2.0",
            "bestAskPrice": "2.01",
            "volume24h": "2200000",
        }
    ]

    _write_snapshot(tmp_path, mainnet_rows)
    _write_snapshot(tmp_path, testnet_rows, testnet=True)

    mainnet_opps = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        min_turnover=1_000_000.0,
        min_change_pct=0.5,
        max_spread_bps=100.0,
        limit=5,
        testnet=False,
    )

    testnet_opps = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        min_turnover=1_000_000.0,
        min_change_pct=0.5,
        max_spread_bps=100.0,
        limit=5,
        testnet=True,
    )

    assert [entry["symbol"] for entry in mainnet_opps] == ["MAINUSDT"]
    assert [entry["symbol"] for entry in testnet_opps] == ["TESTUSDT"]


class _DummyAPI:
    def __init__(self) -> None:
        self.kline_calls: list[tuple[str, int]] = []

    def kline(self, category: str, symbol: str, interval: int, limit: int):  # noqa: D401 - test stub
        self.kline_calls.append((symbol, interval))
        return {"result": {"list": [[1, "1", "1", "1", "1", "100", "1000"]]}}


def _resolver_rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BBSOLUSDT",
            "baseCoin": "BBSOL",
            "quoteCoin": "USDT",
            "status": "Trading",
            "alias": "SOL",
            "lotSizeFilter": {"minOrderQty": "0.1", "basePrecision": "0.1", "minOrderAmt": "3"},
            "priceFilter": {"tickSize": "0.01"},
        },
        {
            "symbol": "WBTCUSDT",
            "baseCoin": "WBTC",
            "quoteCoin": "USDT",
            "status": "Trading",
            "alias": "BTC",
            "lotSizeFilter": {"minOrderQty": "0.0001", "basePrecision": "0.0001", "minOrderAmt": "5"},
            "priceFilter": {"tickSize": "0.5"},
        },
        {
            "symbol": "ETHUSDC",
            "baseCoin": "ETH",
            "quoteCoin": "USDC",
            "status": "Trading",
            "lotSizeFilter": {"minOrderQty": "0.01", "basePrecision": "0.01", "minOrderAmt": "10"},
            "priceFilter": {"tickSize": "0.1"},
        },
        {
            "symbol": "APTUSDT",
            "baseCoin": "APT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {"minOrderQty": "1", "basePrecision": "1", "minOrderAmt": "1"},
            "priceFilter": {"tickSize": "0.01"},
        },
    ]


def test_stateful_market_scanner_enriches_candidates(tmp_path: Path) -> None:
    resolver = SymbolResolver(api=None, refresh=False, bootstrap_rows=_resolver_rows())
    api = _DummyAPI()
    messages: list[str] = []
    scan_calls: list[dict[str, object]] = []

    def fake_scanner(api_client, **kwargs):
        scan_calls.append(kwargs)
        return [
            {"symbol": "SOLUSDT", "score": 9.0, "trend": "buy"},
            {"symbol": "WBTCUSDT", "score": 8.0},
            {"symbol": "ETHUSDC", "score": 7.0},
            {"symbol": "APTUSDT", "score": 6.0},
        ]

    manager = PortfolioManager(total_capital=1000, max_positions=5, risk_per_trade=0.1, min_allocation=10.0)

    scanner = MarketScanner(
        api=api,
        symbol_resolver=resolver,
        scanner=fake_scanner,
        scanner_kwargs={"data_dir": tmp_path},
        portfolio_manager=manager,
        telegram_sender=messages.append,
        refresh_interval=(5.0, 5.0),
    )

    results = scanner.refresh(force=True, now=1000.0)
    assert len(results) == 4
    first = results[0]
    assert first["instrument"]["symbol"] == "BBSOLUSDT"
    assert "1m" in first["candles"] and "5m" in first["candles"]

    assert len(api.kline_calls) == 8  # four symbols, two intervals each
    assert scan_calls  # scanner executed at least once
    assert messages
    assert messages[0].startswith("🏁 Scanner: TOP5 → BBSOL, WBTC, ETH")
    assert "активных: 0/5" in messages[0]

    # Within the refresh window the cached payload should be reused
    cached = scanner.refresh(force=False, now=1002.0)
    assert cached == results
    assert len(scan_calls) == 1
    assert len(api.kline_calls) == 8
def test_candle_cache_filters_unclosed_payload() -> None:
    now = 10_000.0
    interval = 1
    interval_ms = interval * 60_000
    base_ms = int(now * 1000.0)

    closed_start = base_ms - 2 * interval_ms
    almost_closed_start = base_ms - interval_ms
    open_with_flag = base_ms - interval_ms // 2
    open_by_time = base_ms - interval_ms + 10_000

    class DummyAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def kline(self, category: str, symbol: str, interval: int, limit: int):
            self.calls.append((symbol, interval))
            return {
                "result": {
                    "list": [
                        [closed_start, "1", "1", "1", "1", "10", "100", 1],
                        [almost_closed_start, "1", "1", "1", "1", "11", "110", 1],
                        [open_with_flag, "1", "1", "1", "1", "12", "120", 0],
                        [open_by_time, "1", "1", "1", "1", "13", "130"],
                    ]
                }
            }

    api = DummyAPI()
    cache = _CandleCache(api, intervals=(interval,), ttl=0.0, limit=10)

    bundle = cache.fetch("BTCUSDT", now)

    assert "1m" in bundle
    candles = bundle["1m"]
    starts = [candle["start"] for candle in candles]

    assert starts == [closed_start, almost_closed_start]
    assert all(start + interval_ms <= base_ms for start in starts)
    assert api.calls == [("BTCUSDT", interval)]

