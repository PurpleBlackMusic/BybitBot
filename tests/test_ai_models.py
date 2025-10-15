"""Tests for feature engineering in ``bybit_app.utils.ai.models``."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bybit_app.utils.ai.models import (
    MODEL_FEATURES,
    MarketModel,
    _SymbolState,
    _WeightedLogisticRegression,
    _cross_sectional_weights,
    _rolling_out_of_sample_metrics,
    liquidity_feature,
)
from bybit_app.utils.trade_analytics import ExecutionRecord


def _record(
    *,
    side: str,
    qty: float,
    price: float,
    timestamp: datetime,
) -> ExecutionRecord:
    return ExecutionRecord(
        symbol="BTCUSDT",
        side=side,
        qty=qty,
        price=price,
        fee=0.0,
        is_maker=True,
        timestamp=timestamp,
    )


def test_model_features_layout() -> None:
    assert MODEL_FEATURES == (
        "directional_change_pct",
        "multiframe_change_pct",
        "turnover_log",
        "volatility_pct",
        "volume_impulse",
        "depth_imbalance",
        "spread_bps",
        "correlation_strength",
    )


def test_sell_vector_contains_expected_metrics() -> None:
    state = _SymbolState()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    state.register_buy(_record(side="buy", qty=2.0, price=100.0, timestamp=start))
    state.register_buy(
        _record(side="buy", qty=1.0, price=102.0, timestamp=start + timedelta(minutes=10))
    )

    sell_record = _record(
        side="sell",
        qty=2.5,
        price=110.0,
        timestamp=start + timedelta(minutes=16),
    )

    realised = state.realise_sell(sell_record)
    assert realised is not None
    vector, *_ = realised
    assert len(vector) == len(MODEL_FEATURES)
    features = dict(zip(MODEL_FEATURES, vector))

    avg_cost = (2.0 * 100.0 + 1.0 * 102.0) / 3.0
    last_seen_price = 102.0  # последняя цена до продажи
    history_avg = (100.0 + 102.0) / 2.0
    expected_change = (last_seen_price - avg_cost) / avg_cost * 100.0
    assert features["directional_change_pct"] == pytest.approx(expected_change)
    assert features["multiframe_change_pct"] == pytest.approx(
        (last_seen_price - history_avg) / history_avg * 100.0
    )
    assert features["volume_impulse"] > 0.0
    assert features["turnover_log"] == pytest.approx(
        liquidity_feature(last_seen_price * 2.5)
    )

    # Ensure remaining buys are preserved for the open portion of the position.
    assert state.position_qty == pytest.approx(0.5)


def test_multiframe_change_differs_from_directional() -> None:
    state = _SymbolState()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    state.register_buy(_record(side="buy", qty=1.0, price=50.0, timestamp=start))
    state.register_buy(
        _record(side="buy", qty=9.0, price=110.0, timestamp=start + timedelta(minutes=1))
    )

    sell_record = _record(
        side="sell",
        qty=10.0,
        price=120.0,
        timestamp=start + timedelta(minutes=2),
    )

    realised = state.realise_sell(sell_record)
    assert realised is not None
    vector, *_ = realised
    features = dict(zip(MODEL_FEATURES, vector))

    directional = features["directional_change_pct"]
    multiframe = features["multiframe_change_pct"]

    last_seen_price = 110.0
    assert directional == pytest.approx((last_seen_price - 104.0) / 104.0 * 100.0, rel=1e-4)
    assert multiframe == pytest.approx((last_seen_price - 80.0) / 80.0 * 100.0, rel=1e-4)
    assert abs(multiframe - directional) > 1.0


def test_weighted_logistic_handles_single_class() -> None:
    classifier = _WeightedLogisticRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    matrix = np.ones((5, len(MODEL_FEATURES)), dtype=float)
    labels = np.zeros(5, dtype=int)
    pipeline.fit(matrix, labels)

    model = MarketModel(
        feature_names=MODEL_FEATURES,
        pipeline=pipeline,
        trained_at=0.0,
        samples=len(labels),
    )

    probability = model.predict_proba({name: 0.0 for name in MODEL_FEATURES})

    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_liquidity_feature_matches_logarithm() -> None:
    assert liquidity_feature(0.0) == 0.0
    assert liquidity_feature(-10.0) == 0.0

    value = 512.5
    expected = math.log10(value + 1.0)
    assert liquidity_feature(value) == pytest.approx(expected)


def test_market_model_predict_proba_handles_zero_std() -> None:
    classifier = _WeightedLogisticRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

    features = np.zeros((2, 1), dtype=float)
    labels = np.array([0, 1], dtype=int)
    pipeline.fit(features, labels)

    model = MarketModel(
        feature_names=("directional_change_pct",),
        pipeline=pipeline,
        trained_at=0.0,
        samples=len(labels),
    )

    probability = model.predict_proba({"directional_change_pct": 1.0})

    assert math.isfinite(probability)
    assert 0.0 < probability <= 1.0


def test_cross_sectional_weights_equalise_symbols() -> None:
    symbols = ["BTCUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
    weights = _cross_sectional_weights(symbols)

    assert weights.shape == (4,)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] != pytest.approx(weights[0])
    assert weights.mean() == pytest.approx(1.0)
    assert weights[2] > weights[0]


def test_rolling_out_of_sample_metrics_generates_windows() -> None:
    samples = 12
    feature_template = np.linspace(-1.0, 1.0, samples)
    matrix = np.tile(feature_template.reshape(-1, 1), (1, len(MODEL_FEATURES)))
    matrix = np.array(matrix, dtype=float)
    labels = np.array([0, 1] * (samples // 2), dtype=int)
    base_weights = np.ones(samples, dtype=float)
    timestamps = np.linspace(1_700_000_000, 1_700_000_000 + samples, samples)
    symbols = ["BTCUSDT" if index % 2 == 0 else "ETHUSDT" for index in range(samples)]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", _WeightedLogisticRegression()),
    ])

    metrics = _rolling_out_of_sample_metrics(
        pipeline,
        matrix,
        labels,
        base_weights,
        timestamps,
        symbols,
        min_train=4,
        max_splits=3,
    )

    assert metrics is not None
    assert metrics["splits"] >= 1
    assert metrics["avg_accuracy"] >= 0.0
    assert metrics["avg_accuracy"] <= 1.0
    assert metrics["avg_log_loss"] >= 0.0
    assert metrics["windows"]
    for window in metrics["windows"]:
        assert window["samples"] >= 1
        assert 0.0 <= window["accuracy"] <= 1.0
        assert window["log_loss"] >= 0.0
