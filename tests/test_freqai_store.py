from __future__ import annotations

from pathlib import Path

import pytest

from bybit_app.utils.freqai import store as freqai_store
from bybit_app.utils.freqai.store import FreqAIPredictionStore


def _sample_payload() -> dict[str, object]:
    return {
        "generated_at": 1700000000.0,
        "source": "freqtrade",
        "horizon_minutes": 30,
        "predictions": {
            "BTC/USDT": {"probability": 0.66, "ev_bps": 120.0, "confidence": 0.85},
            "ETHUSDT": {"probability_pct": 72.0, "expected_value_bps": 180.0},
        },
    }


def test_freqai_store_roundtrip(tmp_path: Path) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    snapshot = store.update(_sample_payload())

    assert store.path.exists()
    assert snapshot["source"] == "freqtrade"
    pairs = snapshot["pairs"]
    assert isinstance(pairs, dict)
    btc = pairs["BTCUSDT"]
    assert pytest.approx(btc["probability"], rel=1e-6) == 0.66
    assert pytest.approx(btc["ev_bps"], rel=1e-6) == 120.0
    eth = pairs["ETHUSDT"]
    assert pytest.approx(eth["probability"], rel=1e-6) == 0.72
    assert pytest.approx(eth["ev_bps"], rel=1e-6) == 180.0

    snapshot_reload = store.snapshot()
    assert snapshot_reload["pairs"]["BTCUSDT"]["probability"] == pytest.approx(0.66)


def test_freqai_store_custom_path(tmp_path: Path) -> None:
    custom_path = tmp_path / "alt" / "predictions.json"
    store = FreqAIPredictionStore(data_dir=tmp_path, predictions_path=custom_path)
    store.update(_sample_payload())

    assert custom_path.exists()
    assert store.path == custom_path.resolve()


def test_get_prediction_store_uses_override(tmp_path: Path) -> None:
    override = tmp_path / "shared" / "freqai.json"
    store = freqai_store.get_prediction_store(tmp_path, prediction_path=override)
    again = freqai_store.get_prediction_store(tmp_path, prediction_path=override)

    assert store is again
    assert store.path == override.resolve()

    default_store = freqai_store.get_prediction_store(tmp_path)
    assert default_store.path != store.path


def test_freqai_store_top_pairs(tmp_path: Path) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    payload = _sample_payload()
    payload["predictions"]["XRPUSDT"] = {"probability": 0.9, "ev_bps": 90.0}
    store.update(payload)
    top = store.top_pairs(limit=2)
    assert len(top) == 2
    assert top[0]["symbol"] in {"ETHUSDT", "XRPUSDT"}
    assert top[0]["ev_bps"] >= top[1]["ev_bps"]


def test_freqai_store_accepts_logit(tmp_path: Path) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    payload = {
        "generated_at": 1700000001.0,
        "predictions": {
            "SOLUSDT": {"logit": 0.7, "expected_value_pct": 1.4},
        },
    }
    snapshot = store.update(payload)
    sol = snapshot["pairs"]["SOLUSDT"]
    assert sol["probability"] > 0.0
    assert pytest.approx(sol["ev_bps"], rel=1e-6) == 140.0


def test_freqai_store_accepts_sequence_payload(tmp_path: Path) -> None:
    store = FreqAIPredictionStore(data_dir=tmp_path)
    payload = {
        "predictions": [
            {"symbol": "ADA/USDT", "probability": "0.55", "ev_bps": "90"},
            {
                "pair": "XLMUSDT",
                "prediction": {"probability_pct": 68.0, "expected_value": 150.0},
                "fold": "v1",
            },
            {"pair": None, "probability": 0.1},
        ]
    }

    snapshot = store.update(payload)

    pairs = snapshot["pairs"]
    assert "ADAUSDT" in pairs
    ada = pairs["ADAUSDT"]
    assert pytest.approx(ada["probability"], rel=1e-6) == 0.55
    assert pytest.approx(ada["ev_bps"], rel=1e-6) == 90.0

    assert "XLMUSDT" in pairs
    xlm = pairs["XLMUSDT"]
    assert pytest.approx(xlm["probability"], rel=1e-6) == 0.68
    assert pytest.approx(xlm["ev_bps"], rel=1e-6) == 150.0
    assert xlm.get("meta", {}).get("fold") == "v1"


def test_freqai_store_filters_and_cache_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_time = 1_700_000_000.0
    monkeypatch.setattr(freqai_store, "_now", lambda: base_time)

    store = FreqAIPredictionStore(data_dir=tmp_path)
    payload = _sample_payload()
    payload["predictions"]["BTC/USDT"]["generated_at"] = base_time - 90.0
    payload["predictions"]["ETHUSDT"]["generated_at"] = base_time - 10.0
    store.update(payload)

    file_mtime = store.path.stat().st_mtime
    assert store._cache_mtime == pytest.approx(file_mtime, rel=1e-9)

    monkeypatch.setattr(freqai_store, "_now", lambda: base_time + 10.0)

    top = store.top_pairs(limit=5, min_probability=0.7, min_ev_bps=150.0, max_age=60.0)
    assert len(top) == 1
    assert top[0]["symbol"] == "ETHUSDT"

    monkeypatch.setattr(freqai_store, "_now", lambda: base_time + 120.0)
    assert store.is_stale(max_age=60.0) is True


def test_freqai_store_reuses_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_time = 1_700_000_500.0
    monkeypatch.setattr(freqai_store, "_now", lambda: base_time)

    store = FreqAIPredictionStore(data_dir=tmp_path)
    snapshot = store.update(_sample_payload())

    def _explode() -> None:  # pragma: no cover - sanity guard
        raise AssertionError("snapshot should not be invoked when provided")

    monkeypatch.setattr(store, "snapshot", _explode)

    pairs = store.top_pairs(limit=1, snapshot=snapshot, now=base_time + 1.0)
    assert len(pairs) == 1


def test_freqai_store_reuses_snapshot_for_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_time = 1_700_001_000.0
    monkeypatch.setattr(freqai_store, "_now", lambda: base_time)

    store = FreqAIPredictionStore(data_dir=tmp_path)
    snapshot = store.update(_sample_payload())

    def _explode() -> None:  # pragma: no cover - sanity guard
        raise AssertionError("snapshot should not be invoked when provided")

    monkeypatch.setattr(store, "snapshot", _explode)

    assert store.is_stale(max_age=60.0, snapshot=snapshot, now=base_time + 30.0) is False
    assert store.is_stale(max_age=60.0, snapshot=snapshot, now=base_time + 120.0) is True
