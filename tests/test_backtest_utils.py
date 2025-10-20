import numpy as np
import pandas as pd

from bybit_app.utils.ai import models as ai_models
from bybit_app.utils.backtest import (
    optimise_indicator_grid,
    simulate_probability_strategy,
)


def _build_mock_model() -> ai_models.MarketModel:
    feature_names = ai_models.MODEL_FEATURES
    sample_count = len(feature_names)
    X = np.zeros((sample_count, len(feature_names)), dtype=float)
    for idx in range(sample_count):
        X[idx, idx] = 1.0
    y = np.array([1 if idx % 2 == 0 else 0 for idx in range(sample_count)], dtype=int)
    pipeline = ai_models.Pipeline(
        [
            ("scaler", ai_models.StandardScaler()),
            ("classifier", ai_models._WeightedLogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    return ai_models.MarketModel(
        feature_names=feature_names,
        pipeline=pipeline,
        trained_at=0.0,
        samples=sample_count,
    )


def test_simulate_probability_strategy_returns_metrics() -> None:
    model = _build_mock_model()
    feature_columns = list(model.feature_names)
    frame = pd.DataFrame(0.0, index=range(10), columns=feature_columns)
    frame.loc[:, feature_columns[0]] = np.linspace(0.1, 1.0, 10)
    frame.loc[:, feature_columns[1]] = np.linspace(-0.5, 0.5, 10)
    frame["return_pct"] = np.linspace(-0.5, 1.5, 10)
    frame["timestamp"] = np.arange(10)

    baseline = simulate_probability_strategy(
        model,
        frame,
        threshold=0.4,
        deepseek_enabled=False,
    )
    enhanced = simulate_probability_strategy(
        model,
        frame,
        threshold=0.4,
        deepseek_enabled=True,
    )

    assert baseline.metrics.trades >= 0
    assert enhanced.metrics.trades >= 0
    assert baseline.name == "baseline"
    assert enhanced.name == "deepseek"


def test_optimise_indicator_grid_filters_with_deepseek_threshold() -> None:
    periods = 80
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="h")
    close = pd.Series(np.linspace(100, 140, periods))
    frame = pd.DataFrame(
        {
            "close": close.values,
            "return_pct": np.linspace(0.2, 1.2, periods),
            "timestamp": timestamps.astype("int64") // 1_000_000_000,
            "deepseek_score": np.where(np.arange(periods) < 40, 0.8, 0.4),
        }
    )

    results = optimise_indicator_grid(
        frame,
        sma_fast=[5],
        sma_slow=[15],
        ema_fast=[5],
        ema_slow=[15],
        rsi_period=[14],
        rsi_buy=[60.0],
        macd_fast=[12],
        macd_slow=[26],
        macd_signal=[9],
        deepseek_thresholds=[0.7],
    )

    assert results
    entry = results[0]
    assert "baseline" in entry and "deepseek" in entry
    baseline_trades = entry["baseline"]["trades"]
    deepseek_trades = entry["deepseek"]["trades"]
    assert baseline_trades >= deepseek_trades
