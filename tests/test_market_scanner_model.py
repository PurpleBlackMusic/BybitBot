from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import pytest

from bybit_app.utils.ai.models import MODEL_FEATURES, liquidity_feature
from bybit_app.utils.market_scanner import (
    TURNOVER_AVG_TRADES_PER_DAY,
    scan_market_opportunities,
)


FEATURE_NAMES = list(MODEL_FEATURES)


def _write_model(
    path: Path,
    *,
    weight: float,
    intercept: float = 0.0,
    means: Optional[Sequence[float]] = None,
    stds: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
) -> None:
    if weights is not None:
        coefficients = list(weights)
    else:
        coefficients = [weight] + [0.0] * (len(FEATURE_NAMES) - 1)

    payload = {
        "feature_names": FEATURE_NAMES,
        "coefficients": coefficients,
        "intercept": intercept,
        "feature_means": list(means or ([0.0] * len(FEATURE_NAMES))),
        "feature_stds": list(stds or ([1.0] * len(FEATURE_NAMES))),
        "trained_at": time.time(),
        "samples": 100,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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

    model_path = ai_dir / "model.json"

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
        ai_dir / "model.json",
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
