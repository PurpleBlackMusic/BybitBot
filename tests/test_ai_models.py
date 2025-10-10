"""Tests for feature engineering in ``bybit_app.utils.ai.models``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bybit_app.utils.ai.models import MODEL_FEATURES, _SymbolState, _ensure_scaling
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
        "maker_flag",
        "hold_minutes",
        "position_closed_fraction",
    )


def test_hold_duration_and_fraction_features() -> None:
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

    assert features["hold_minutes"] == pytest.approx(14.0, abs=1e-6)
    assert features["position_closed_fraction"] == pytest.approx(2.5 / 3.0, abs=1e-6)

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

    assert directional == pytest.approx((120.0 - 104.0) / 104.0 * 100.0, rel=1e-4)
    assert multiframe == pytest.approx((120.0 - 80.0) / 80.0 * 100.0, rel=1e-4)
    assert abs(multiframe - directional) > 1.0


def test_scaling_output_matches_feature_count() -> None:
    matrix = np.ones((4, len(MODEL_FEATURES)), dtype=float)
    normalized, means, stds = _ensure_scaling(matrix)

    assert normalized.shape[1] == len(MODEL_FEATURES)
    assert means.shape[0] == len(MODEL_FEATURES)
    assert stds.shape[0] == len(MODEL_FEATURES)
