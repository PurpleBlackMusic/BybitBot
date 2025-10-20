#!/usr/bin/env python3
"""Train the market-scanner model using DeepSeek-derived features."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from bybit_app.utils.ai import models as ai_models
from bybit_app.utils.paths import DATA_DIR


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _load_signal_records(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, Mapping):
        candidates = [payload]
    else:
        candidates = []
        for line in text.splitlines():
            snippet = line.strip()
            if not snippet:
                continue
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                candidates.append(parsed)

    return [item for item in candidates if isinstance(item, Mapping)]


def load_deepseek_signals(signals_dir: Optional[Path]) -> Dict[str, pd.DataFrame]:
    """Load DeepSeek signal history grouped by symbol."""

    store: Dict[str, List[Dict[str, Any]]] = {}
    if signals_dir is None:
        return {}
    for path in sorted(signals_dir.glob("*.json*")):
        for record in _load_signal_records(path):
            symbol = str(record.get("symbol") or "").upper()
            timestamp = _safe_float(record.get("timestamp"))
            if not symbol or timestamp is None:
                continue
            payload = {
                "timestamp": timestamp,
                "deepseek_score": _safe_float(record.get("deepseek_score"), 0.0),
                "deepseek_stop_loss": _safe_float(record.get("deepseek_stop_loss"), 0.0),
                "deepseek_take_profit": _safe_float(record.get("deepseek_take_profit"), 0.0),
            }
            store.setdefault(symbol, []).append(payload)
    frames: Dict[str, pd.DataFrame] = {}
    for symbol, rows in store.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        frames[symbol] = df.reset_index(drop=True)
    return frames


def load_ohlcv_frame(base_dir: Path, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    ohlcv_path = base_dir / symbol / f"{symbol}_{interval}.csv"
    if not ohlcv_path.exists():
        return None
    try:
        frame = pd.read_csv(ohlcv_path)
    except Exception:
        return None
    if frame.empty or "timestamp" not in frame.columns:
        return None
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", errors="coerce")


def build_feature_frame(
    symbol: str,
    frame: pd.DataFrame,
    signals: Optional[pd.DataFrame],
    lookahead: int,
) -> pd.DataFrame:
    if lookahead <= 0:
        raise ValueError("lookahead must be a positive integer")

    work = frame.copy()
    work["symbol"] = symbol
    work["timestamp"] = _ensure_datetime(work["timestamp"])
    work = work.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    if work.empty:
        return np.empty((0, len(ai_models.MODEL_FEATURES))), np.empty((0,), dtype=int)

    if signals is not None and not signals.empty:
        merged = signals.copy()
        merged["timestamp"] = _ensure_datetime(merged["timestamp"])
        work = work.merge(merged, on="timestamp", how="left")
    else:
        work["deepseek_score"] = 0.0
        work["deepseek_stop_loss"] = 0.0
        work["deepseek_take_profit"] = 0.0

    work["directional_change_pct"] = work["close"].pct_change().fillna(0.0) * 100.0
    work["multiframe_change_pct"] = (
        work["close"].pct_change(lookahead * 4).fillna(work["directional_change_pct"]) * 100.0
    )
    work["turnover_log"] = np.log10(np.clip(work.get("volume", 0.0).fillna(0.0), 0, None) + 1.0)
    price_range = work.get("high", 0.0) - work.get("low", 0.0)
    work["volatility_pct"] = (
        (price_range / work["close"].replace(0.0, np.nan)).fillna(0.0).clip(lower=-1000, upper=1000) * 100.0
    )
    vol_mean = work.get("volume", 0.0).rolling(max(lookahead, 2)).mean().replace(0.0, np.nan)
    work["volume_impulse"] = np.log1p((work.get("volume", 0.0) / vol_mean) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    )
    work["volatility_ratio"] = (
        work["volatility_pct"].rolling(max(lookahead, 2)).mean().pct_change().fillna(0.0)
    )
    work["volume_trend"] = work.get("volume", 0.0).rolling(max(lookahead, 2)).mean().pct_change()
    work["volume_trend"] = work["volume_trend"].fillna(0.0) * 100.0

    seconds = (
        work["timestamp"].dt.hour * 3600
        + work["timestamp"].dt.minute * 60
        + work["timestamp"].dt.second
    )
    angle = seconds / 86_400.0 * 2.0 * math.pi
    work["session_hour_sin"] = np.sin(angle)
    work["session_hour_cos"] = np.cos(angle)
    work["macro_regime_score"] = np.tanh(work["multiframe_change_pct"].fillna(0.0) / 50.0)
    work["holding_period_minutes"] = float(lookahead * 60)

    feature_frame = pd.DataFrame(0.0, index=work.index, columns=ai_models.MODEL_FEATURES)
    for column in feature_frame.columns:
        if column in work.columns:
            feature_frame[column] = work[column].fillna(0.0).astype(float)

    future_price = work["close"].shift(-lookahead)
    targets = (future_price > work["close"]).astype(int)
    valid = targets.notna()
    if not valid.any():
        return pd.DataFrame(columns=(*ai_models.MODEL_FEATURES, "label", "return_pct", "timestamp", "close", "future_close"))

    subset = feature_frame.loc[valid].replace([np.inf, -np.inf], 0.0).fillna(0.0).copy()
    subset["label"] = targets.loc[valid].to_numpy(dtype=int)
    current_close = work.loc[valid, "close"].astype(float)
    next_close = future_price.loc[valid].astype(float)
    subset["close"] = current_close.to_numpy()
    subset["future_close"] = next_close.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = ((next_close - current_close) / current_close).replace([np.inf, -np.inf], np.nan) * 100.0
    subset["return_pct"] = returns.fillna(0.0).to_numpy()
    timestamps = work.loc[valid, "timestamp"]
    subset["timestamp"] = pd.to_datetime(timestamps).astype("int64") / 1_000_000_000
    subset["symbol"] = symbol
    return subset.reset_index(drop=True)


def build_feature_matrix(
    symbol: str,
    frame: pd.DataFrame,
    signals: Optional[pd.DataFrame],
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_frame = build_feature_frame(symbol, frame, signals, lookahead)
    if feature_frame.empty:
        return np.empty((0, len(ai_models.MODEL_FEATURES))), np.empty((0,), dtype=int)

    matrix = feature_frame[list(ai_models.MODEL_FEATURES)].to_numpy(dtype=float)
    labels = feature_frame["label"].to_numpy(dtype=int)
    return matrix, labels


def discover_symbols(ohlcv_root: Path) -> List[str]:
    if not ohlcv_root.exists():
        return []
    symbols = [entry.name for entry in ohlcv_root.iterdir() if entry.is_dir()]
    return sorted({symbol.upper() for symbol in symbols})


def build_dataset(
    ohlcv_root: Path,
    symbols: Sequence[str],
    signals: Dict[str, pd.DataFrame],
    interval: str,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrices: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for symbol in symbols:
        frame = load_ohlcv_frame(ohlcv_root, symbol, interval)
        if frame is None:
            continue
        signal_frame = signals.get(symbol.upper())
        matrix, labels = build_feature_matrix(symbol, frame, signal_frame, lookahead)
        if matrix.size == 0 or labels.size == 0:
            continue
        matrices.append(matrix)
        labels_list.append(labels)
    if not matrices:
        return np.empty((0, len(ai_models.MODEL_FEATURES))), np.empty((0,), dtype=int)
    combined_matrix = np.vstack(matrices)
    combined_labels = np.concatenate(labels_list)
    return combined_matrix, combined_labels


def train_pipeline(matrix: np.ndarray, labels: np.ndarray) -> ai_models.Pipeline:
    classifier = ai_models._WeightedLogisticRegression()
    pipeline = ai_models.Pipeline([("scaler", ai_models.StandardScaler()), ("classifier", classifier)])
    pipeline.fit(matrix, labels)
    return pipeline


def compute_training_metrics(pipeline: ai_models.Pipeline, matrix: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    probabilities = pipeline.predict_proba(matrix)[:, ai_models._positive_index(pipeline)]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float((predictions == labels).mean()) if labels.size else 0.0
    avg_probability = float(probabilities.mean()) if probabilities.size else 0.0
    return {
        "samples": int(labels.size),
        "accuracy": accuracy,
        "avg_probability": avg_probability,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Base data directory")
    parser.add_argument("--signals", type=Path, default=None, help="Directory containing DeepSeek signals")
    parser.add_argument("--interval", default="1h", help="OHLCV interval filename suffix (e.g. 1h)")
    parser.add_argument("--symbols", nargs="*", help="Optional list of symbols to train on")
    parser.add_argument("--lookahead", type=int, default=1, help="Prediction horizon in bars")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum samples required to train")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override path for the trained model (defaults to data_dir/ai/model.joblib)",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser()
    ohlcv_root = data_dir / "ohlcv" / "spot"
    if args.symbols:
        symbols = [str(symbol).upper() for symbol in args.symbols]
    else:
        symbols = discover_symbols(ohlcv_root)
    if not symbols:
        print("No symbols available for training", file=sys.stderr)
        return 1

    signals_dir = args.signals.expanduser() if isinstance(args.signals, Path) else None
    signals = load_deepseek_signals(signals_dir)

    matrix, labels = build_dataset(ohlcv_root, symbols, signals, args.interval, args.lookahead)
    if labels.size < max(int(args.min_rows), 1):
        print(
            f"Insufficient samples ({labels.size}) for training. Collect more data or lower --min-rows.",
            file=sys.stderr,
        )
        return 1

    pipeline = train_pipeline(matrix, labels)
    metrics = compute_training_metrics(pipeline, matrix, labels)
    feature_stats = ai_models._summarise_feature_distribution(matrix)

    model = ai_models.MarketModel(
        feature_names=tuple(ai_models.MODEL_FEATURES),
        pipeline=pipeline,
        trained_at=float(time.time()),
        samples=int(labels.size),
        training_metrics=metrics,
        feature_stats=feature_stats,
    )

    output_path = args.output if args.output is not None else data_dir / "ai" / ai_models.MODEL_FILENAME
    resolved = ai_models.save_model(model, output_path, data_dir=data_dir)
    print(f"Model trained with {labels.size} samples and stored at {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
