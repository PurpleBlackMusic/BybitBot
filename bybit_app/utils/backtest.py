"""Offline backtesting helpers for DeepSeek-enhanced strategies."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .ai import models as ai_models

_DEEPSEEK_COLUMNS: Tuple[str, ...] = (
    "deepseek_score",
    "deepseek_stop_loss",
    "deepseek_take_profit",
)


@dataclass(slots=True)
class BacktestMetrics:
    trades: int
    avg_return_pct: float
    win_rate: float
    max_drawdown_pct: float
    stability: float
    cumulative_return_pct: float


@dataclass(slots=True)
class ScenarioResult:
    name: str
    threshold: float
    metrics: BacktestMetrics


@dataclass(slots=True)
class IndicatorParameters:
    sma_fast: int
    sma_slow: int
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_buy: float
    macd_fast: int
    macd_slow: int
    macd_signal: int


def _ensure_numpy(series: Sequence[float] | np.ndarray) -> np.ndarray:
    data = np.asarray(series, dtype=float)
    if data.ndim == 0:
        return data.reshape(1)
    return data


def _equity_curve_metrics(returns_pct: Sequence[float]) -> BacktestMetrics:
    trades = len(returns_pct)
    if trades == 0:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = _ensure_numpy(returns_pct)
    avg_return = float(np.mean(returns)) if trades else 0.0
    wins = float(np.sum(returns > 0.0))
    win_rate = wins / trades if trades else 0.0

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + (value / 100.0)
        if equity <= 0.0:
            equity = 1e-9
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    std = float(np.std(returns)) if trades > 1 else 0.0
    stability = avg_return / std if std > 1e-9 else (avg_return if avg_return > 0 else 0.0)
    cumulative_return = (equity - 1.0) * 100.0

    return BacktestMetrics(
        trades=trades,
        avg_return_pct=avg_return,
        win_rate=win_rate,
        max_drawdown_pct=max_drawdown * 100.0,
        stability=stability,
        cumulative_return_pct=cumulative_return,
    )


def simulate_probability_strategy(
    model: ai_models.MarketModel,
    frame: pd.DataFrame,
    *,
    threshold: float,
    deepseek_enabled: bool,
    deepseek_columns: Sequence[str] = _DEEPSEEK_COLUMNS,
) -> ScenarioResult:
    """Evaluate a probability-driven strategy using *model* over *frame*."""

    if frame.empty:
        metrics = BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return ScenarioResult(
            name="deepseek" if deepseek_enabled else "baseline",
            threshold=float(threshold),
            metrics=metrics,
        )

    feature_columns = list(model.feature_names)
    if deepseek_enabled:
        features = frame[feature_columns].to_numpy(dtype=float)
    else:
        features = frame[feature_columns].copy().to_numpy(dtype=float)
        for column in deepseek_columns:
            if column in feature_columns:
                idx = feature_columns.index(column)
                features[:, idx] = 0.0

    probabilities = model.pipeline.predict_proba(features)
    positive_index = ai_models._positive_index(model.pipeline)
    positive = probabilities[:, positive_index]
    returns = frame["return_pct"].to_numpy(dtype=float)
    executed = returns[positive >= float(threshold)]
    metrics = _equity_curve_metrics(executed)
    scenario_name = "deepseek" if deepseek_enabled else "baseline"
    return ScenarioResult(name=scenario_name, threshold=float(threshold), metrics=metrics)


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def evaluate_indicator_parameters(
    frame: pd.DataFrame,
    params: IndicatorParameters,
    *,
    deepseek_threshold: float = 0.0,
    deepseek_column: str = "deepseek_score",
) -> Dict[str, BacktestMetrics]:
    if frame.empty:
        empty = BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return {"baseline": empty, "deepseek": empty}

    working = frame.sort_values("timestamp").reset_index(drop=True)
    close = working["close"].astype(float)

    sma_fast = close.rolling(window=params.sma_fast, min_periods=params.sma_fast).mean()
    sma_slow = close.rolling(window=params.sma_slow, min_periods=params.sma_slow).mean()
    ema_fast = close.ewm(span=params.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.ema_slow, adjust=False).mean()
    macd, macd_signal = _compute_macd(close, params.macd_fast, params.macd_slow, params.macd_signal)
    rsi = _compute_rsi(close, max(params.rsi_period, 2))

    base_signal = (
        (ema_fast > ema_slow)
        & (sma_fast > sma_slow)
        & (macd > macd_signal)
        & (rsi <= params.rsi_buy)
    )

    returns = working.loc[base_signal.fillna(False), "return_pct"].to_numpy(dtype=float)
    baseline_metrics = _equity_curve_metrics(returns)

    if deepseek_column in working.columns:
        deepseek_scores = working.loc[base_signal.fillna(False), deepseek_column].astype(float)
        filtered_returns = returns[deepseek_scores >= float(deepseek_threshold)]
    else:
        filtered_returns = np.array([], dtype=float)

    deepseek_metrics = _equity_curve_metrics(filtered_returns)
    return {"baseline": baseline_metrics, "deepseek": deepseek_metrics}


def optimise_indicator_grid(
    frame: pd.DataFrame,
    *,
    sma_fast: Sequence[int],
    sma_slow: Sequence[int],
    ema_fast: Sequence[int],
    ema_slow: Sequence[int],
    rsi_period: Sequence[int],
    rsi_buy: Sequence[float],
    macd_fast: Sequence[int],
    macd_slow: Sequence[int],
    macd_signal: Sequence[int],
    deepseek_thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for combo in product(
        sma_fast,
        sma_slow,
        ema_fast,
        ema_slow,
        rsi_period,
        rsi_buy,
        macd_fast,
        macd_slow,
        macd_signal,
        deepseek_thresholds,
    ):
        params = IndicatorParameters(
            sma_fast=combo[0],
            sma_slow=combo[1],
            ema_fast=combo[2],
            ema_slow=combo[3],
            rsi_period=combo[4],
            rsi_buy=combo[5],
            macd_fast=combo[6],
            macd_slow=combo[7],
            macd_signal=combo[8],
        )
        threshold = combo[9]
        metrics = evaluate_indicator_parameters(
            frame,
            params,
            deepseek_threshold=threshold,
        )
        results.append(
            {
                "parameters": dataclasses.asdict(params),
                "deepseek_threshold": float(threshold),
                "baseline": dataclasses.asdict(metrics["baseline"]),
                "deepseek": dataclasses.asdict(metrics["deepseek"]),
            }
        )

    results.sort(
        key=lambda entry: entry["deepseek"]["avg_return_pct"],
        reverse=True,
    )
    return results


__all__ = [
    "BacktestMetrics",
    "ScenarioResult",
    "IndicatorParameters",
    "simulate_probability_strategy",
    "evaluate_indicator_parameters",
    "optimise_indicator_grid",
]
