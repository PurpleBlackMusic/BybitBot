from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bybit_app.utils.ai.models import MODEL_FEATURES, liquidity_feature
from bybit_app.utils.market_scanner import (
    TURNOVER_AVG_TRADES_PER_DAY,
    scan_market_opportunities,
)


FEATURE_NAMES = list(MODEL_FEATURES)


def _build_pipeline(
    *,
    weight: float,
    intercept: float,
    means: Optional[Sequence[float]],
    stds: Optional[Sequence[float]],
    weights: Optional[Sequence[float]],
) -> Pipeline:
    feature_count = len(FEATURE_NAMES)
    mean_vector = np.array(means or ([0.0] * feature_count), dtype=float)
    scale_vector = np.array(stds or ([1.0] * feature_count), dtype=float)
    scale_vector = np.where(np.abs(scale_vector) < 1e-6, 1.0, scale_vector)

    scaler = StandardScaler()
    scaler.mean_ = mean_vector
    scaler.scale_ = scale_vector
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = feature_count
    scaler.n_samples_seen_ = 1
    # Align with production where numpy arrays without named columns are used.

    classifier = LogisticRegression()
    classifier.classes_ = np.array([0, 1], dtype=int)
    if weights is not None:
        coefficients = np.array(list(weights), dtype=float)
    else:
        coefficients = np.array([weight] + [0.0] * (feature_count - 1), dtype=float)
    classifier.coef_ = coefficients.reshape(1, -1)
    classifier.intercept_ = np.array([intercept], dtype=float)
    classifier.n_iter_ = np.array([1], dtype=int)

    return Pipeline([
        ("scaler", scaler),
        ("classifier", classifier),
    ])


def _write_model(
    path: Path,
    *,
    weight: float,
    intercept: float = 0.0,
    means: Optional[Sequence[float]] = None,
    stds: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
) -> None:
    pipeline = _build_pipeline(
        weight=weight,
        intercept=intercept,
        means=means,
        stds=stds,
        weights=weights,
    )

    payload = {
        "feature_names": FEATURE_NAMES,
        "trained_at": time.time(),
        "samples": 100,
        "pipeline": pipeline,
    }
    joblib.dump(payload, path)


def _write_snapshot(path: Path) -> None:
    snapshot = {
        "ts": time.time(),
        "rows": [
            {
                "symbol": "AAAUSDT",
                "price24hPcnt": 5.0,
                "turnover24h": 2_000_000.0,
                "bestBidPrice": 1.0,
                "bestAskPrice": 1.01,
            },
            {
                "symbol": "BBBUSDT",
                "price24hPcnt": -4.0,
                "turnover24h": 2_000_000.0,
                "bestBidPrice": 2.0,
                "bestAskPrice": 2.02,
            },
        ],
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def test_model_weights_affect_ranking(tmp_path: Path) -> None:
    ai_dir = tmp_path / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = ai_dir / "market_snapshot.json"
    _write_snapshot(snapshot_path)

    model_path = ai_dir / "model.joblib"

    _write_model(model_path, weight=2.0)
    first = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        cache_ttl=9999.0,
        min_turnover=0.0,
        max_spread_bps=120.0,
    )
    assert first, "scanner should return opportunities"
    first_symbols = [entry["symbol"] for entry in first]
    assert first_symbols[0] == "AAAUSDT"

    _write_model(model_path, weight=-3.0)
    second = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        cache_ttl=9999.0,
        min_turnover=0.0,
        max_spread_bps=120.0,
    )
    assert second, "scanner should return opportunities with updated weights"
    second_symbols = [entry["symbol"] for entry in second]
    assert second_symbols[0] == "BBBUSDT"


def test_turnover_feature_consistency(tmp_path: Path) -> None:
    ai_dir = tmp_path / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = ai_dir / "market_snapshot.json"

    notional = 750.0
    turnover = notional * TURNOVER_AVG_TRADES_PER_DAY
    snapshot = {
        "ts": time.time(),
        "rows": [
            {
                "symbol": "AAAUSDT",
                "price24hPcnt": 1.5,
                "turnover24h": turnover,
                "bestBidPrice": 1.0,
                "bestAskPrice": 1.01,
            }
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    turnover_feature = liquidity_feature(notional)
    means = [0.0] * len(FEATURE_NAMES)
    means[2] = turnover_feature
    stds = [1.0] * len(FEATURE_NAMES)
    stds[2] = 0.5
    weights = [0.0] * len(FEATURE_NAMES)
    weights[2] = 2.0
    _write_model(
        ai_dir / "model.joblib",
        weight=0.0,
        intercept=-1.0,
        means=means,
        stds=stds,
        weights=weights,
    )

    entries = scan_market_opportunities(
        api=None,
        data_dir=tmp_path,
        cache_ttl=9999.0,
        min_turnover=0.0,
        max_spread_bps=120.0,
    )

    assert entries, "scanner should produce entries"
    entry = entries[0]
    features = entry["model_metrics"]["features"]

    assert features["turnover_log"] == pytest.approx(turnover_feature, rel=1e-6)
    assert 0.05 < entry["probability"] < 0.95
