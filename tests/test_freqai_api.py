from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bybit_app.utils.envs import Settings
from bybit_app.utils.freqai.api import create_app
from bybit_app.utils.freqai.store import FreqAIPredictionStore


def _write_snapshot(
    base: Path,
    rows: list[dict[str, object]],
    *,
    testnet: bool | None = None,
) -> None:
    payload = {"ts": time.time(), "category": "spot", "rows": rows}
    filename = "market_snapshot.json"
    if testnet:
        if "." in filename:
            stem, ext = filename.rsplit(".", 1)
            filename = f"{stem}_testnet.{ext}"
        else:
            filename = f"{filename}_testnet"
    path = base / "ai" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sample_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "turnover24h": "3200000",
        "price24hPcnt": "2.5",
        "price1hPcnt": "0.6",
        "price4hPcnt": "1.2",
        "bestBidPrice": "10.0",
        "bestAskPrice": "10.05",
        "volume24h": "1500000",
        "volume1h": "120000",
        "prevVolume1h": "60000",
        "highPrice24h": "10.8",
        "lowPrice24h": "8.9",
        "lastPrice": "10.02",
        "highPrice1h": "10.2",
        "lowPrice1h": "9.8",
        "closePrice1h": "10.0",
        "bid1Size": "5000",
        "ask1Size": "4800",
        "corr_btc": "0.45",
        "corr_market": "0.52",
    }


@pytest.fixture()
def freqai_settings() -> Settings:
    settings = Settings(
        ai_min_turnover_usd=0.0,
        ai_min_change_volatility_ratio=0.0,
        ai_min_turnover_ratio=0.0,
        ai_min_top_quote_usd=0.0,
        ai_min_ev_bps=0.0,
        ai_max_spread_bps=200.0,
        ai_min_top_quote_ratio=0.0,
        ai_market_scan_enabled=True,
        ai_enabled=True,
    )
    settings.testnet = False
    settings.freqai_api_token = "test-token"
    return settings


def test_freqai_api_features_and_predictions(tmp_path: Path, freqai_settings: Settings) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    _write_snapshot(tmp_path, [_sample_row("SOLUSDT")])

    prediction_payload = {
        "generated_at": 1700000100.0,
        "source": "freqtrade",
        "predictions": {"SOLUSDT": {"probability": 0.82, "ev_bps": 160.0}},
    }
    store.update(prediction_payload)

    app = create_app(store=store, settings_provider=lambda: freqai_settings)
    client = TestClient(app)
    client.headers.update({"X-API-Token": freqai_settings.freqai_api_token})

    health = client.get("/health")
    assert health.status_code == 200
    prediction_summary = health.json()["predictions"]
    assert prediction_summary["total_pairs"] == 1
    assert prediction_summary["stale"] is False
    assert prediction_summary["recency_seconds"] == pytest.approx(3600.0)

    response = client.get("/features", params={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert payload["executions_path"].endswith("executions.mainnet.jsonl")

    pair = payload["pairs"][0]
    assert pair["symbol"] == "SOLUSDT"
    assert pair["features"]
    freqai_meta = pair.get("freqai")
    assert freqai_meta is not None
    assert pytest.approx(freqai_meta["probability"], rel=1e-6) == 0.82
    assert pytest.approx(freqai_meta["ev_bps"], rel=1e-6) == 160.0

    predictions = client.get("/predictions")
    assert predictions.status_code == 200
    assert predictions.json()["pairs"]["SOLUSDT"]["probability"] == pytest.approx(0.82)


def test_freqai_features_reports_network_ledger(tmp_path: Path, freqai_settings: Settings) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    _write_snapshot(tmp_path, [_sample_row("ADAUSDT")], testnet=True)

    store.update(
        {
            "generated_at": 1700000300.0,
            "predictions": {"ADAUSDT": {"probability": 0.51, "ev_bps": 120.0}},
        }
    )

    freqai_settings.testnet = True

    pnl_dir = tmp_path / "pnl"
    pnl_dir.mkdir(parents=True, exist_ok=True)
    testnet_path = pnl_dir / "executions.testnet.jsonl"
    testnet_path.write_text("{}", encoding="utf-8")

    app = create_app(store=store, settings_provider=lambda: freqai_settings)
    client = TestClient(app)
    client.headers.update({"X-API-Token": freqai_settings.freqai_api_token})

    response = client.get("/features", params={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["executions_path"].endswith("executions.testnet.jsonl")


def test_freqai_api_requires_token(tmp_path: Path, freqai_settings: Settings) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    app = create_app(store=store, settings_provider=lambda: freqai_settings)
    client = TestClient(app)

    unauthorised = client.get("/health")
    assert unauthorised.status_code == 401

    authorised = client.get(
        "/health",
        headers={"X-API-Token": freqai_settings.freqai_api_token},
    )
    assert authorised.status_code == 200
