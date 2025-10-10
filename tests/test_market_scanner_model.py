from __future__ import annotations

import json
import time
from pathlib import Path

from bybit_app.utils.ai.models import MODEL_FEATURES
from bybit_app.utils.market_scanner import scan_market_opportunities


FEATURE_NAMES = list(MODEL_FEATURES)


def _write_model(path: Path, *, weight: float) -> None:
    payload = {
        "feature_names": FEATURE_NAMES,
        "coefficients": [weight] + [0.0] * (len(FEATURE_NAMES) - 1),
        "intercept": 0.0,
        "feature_means": [0.0] * len(FEATURE_NAMES),
        "feature_stds": [1.0] * len(FEATURE_NAMES),
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
        cache_ttl=None,
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
        cache_ttl=None,
        min_turnover=0.0,
        max_spread_bps=120.0,
    )
    assert second, "scanner should return opportunities with updated weights"
    second_symbols = [entry["symbol"] for entry in second]
    assert second_symbols[0] == "BBBUSDT"
