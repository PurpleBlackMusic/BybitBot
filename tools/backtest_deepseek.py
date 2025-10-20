#!/usr/bin/env python3
"""Run offline backtests comparing DeepSeek-assisted strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from bybit_app.utils.ai import models as ai_models
from bybit_app.utils.backtest import (
    optimise_indicator_grid,
    simulate_probability_strategy,
)
from bybit_app.utils.paths import DATA_DIR

from train_deepseek_model import (
    build_feature_frame,
    discover_symbols,
    load_deepseek_signals,
    load_ohlcv_frame,
)
def _assemble_feature_frame(
    *,
    data_dir: Path,
    signals: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    interval: str,
    lookahead: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        source = load_ohlcv_frame(data_dir, symbol, interval)
        if source is None or source.empty:
            continue
        signal_frame = signals.get(symbol.upper())
        feature_frame = build_feature_frame(symbol, source, signal_frame, lookahead)
        if feature_frame.empty:
            continue
        frames.append(feature_frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def _default_grid_arguments(parser: argparse.ArgumentParser) -> None:
    grid = parser.add_argument_group("Grid search")
    grid.add_argument(
        "--grid-sma-fast",
        nargs="+",
        type=int,
        default=[5, 8, 13],
        help="Candidate windows for the fast SMA.",
    )
    grid.add_argument(
        "--grid-sma-slow",
        nargs="+",
        type=int,
        default=[21, 34, 55],
        help="Candidate windows for the slow SMA.",
    )
    grid.add_argument(
        "--grid-ema-fast",
        nargs="+",
        type=int,
        default=[8, 12, 21],
        help="Candidate spans for the fast EMA.",
    )
    grid.add_argument(
        "--grid-ema-slow",
        nargs="+",
        type=int,
        default=[34, 55, 89],
        help="Candidate spans for the slow EMA.",
    )
    grid.add_argument(
        "--grid-rsi-period",
        nargs="+",
        type=int,
        default=[14, 21],
        help="RSI lookback periods to evaluate.",
    )
    grid.add_argument(
        "--grid-rsi-buy",
        nargs="+",
        type=float,
        default=[25.0, 30.0, 35.0],
        help="RSI over-sold thresholds for entry.",
    )
    grid.add_argument(
        "--grid-macd-fast",
        nargs="+",
        type=int,
        default=[12, 15],
        help="MACD fast EMA spans.",
    )
    grid.add_argument(
        "--grid-macd-slow",
        nargs="+",
        type=int,
        default=[26, 30],
        help="MACD slow EMA spans.",
    )
    grid.add_argument(
        "--grid-macd-signal",
        nargs="+",
        type=int,
        default=[9],
        help="MACD signal EMA spans.",
    )
    grid.add_argument(
        "--grid-deepseek-threshold",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7],
        help="DeepSeek score thresholds to filter trades during grid search.",
    )
    grid.add_argument(
        "--grid-top",
        type=int,
        default=10,
        help="Number of top parameter combinations to display.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Base data directory")
    parser.add_argument("--signals", type=Path, default=None, help="Directory with DeepSeek signals")
    parser.add_argument("--model", type=Path, default=None, help="Override model path")
    parser.add_argument("--interval", default="1h", help="OHLCV interval suffix (default: 1h)")
    parser.add_argument("--lookahead", type=int, default=1, help="Prediction horizon in bars")
    parser.add_argument("--symbols", nargs="*", help="Optional list of symbols to evaluate")
    parser.add_argument("--threshold", type=float, default=0.55, help="Probability threshold for trades")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run indicator grid search using DeepSeek guidance.",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=None,
        help="Path to write full grid results as JSON",
    )
    _default_grid_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser()
    ohlcv_root = data_dir / "ohlcv" / "spot"
    if args.symbols:
        symbols = [str(symbol).upper() for symbol in args.symbols]
    else:
        symbols = discover_symbols(ohlcv_root)
    if not symbols:
        print("No symbols available for backtesting", flush=True)
        return 1

    signals_dir = args.signals.expanduser() if isinstance(args.signals, Path) else None
    signals = load_deepseek_signals(signals_dir)
    feature_frame = _assemble_feature_frame(
        data_dir=ohlcv_root,
        signals=signals,
        symbols=symbols,
        interval=args.interval,
        lookahead=args.lookahead,
    )
    if feature_frame.empty:
        print("No feature rows constructed for the requested configuration", flush=True)
        return 1

    model_path = args.model if args.model is not None else data_dir / "ai" / ai_models.MODEL_FILENAME
    model = ai_models.load_model(model_path, data_dir=data_dir)
    if model is None:
        print(f"Model not found at {model_path}", flush=True)
        return 1

    baseline = simulate_probability_strategy(
        model,
        feature_frame,
        threshold=args.threshold,
        deepseek_enabled=False,
    )
    enhanced = simulate_probability_strategy(
        model,
        feature_frame,
        threshold=args.threshold,
        deepseek_enabled=True,
    )

    summary = {
        "baseline": baseline.metrics.__dict__,
        "deepseek": enhanced.metrics.__dict__,
        "threshold": args.threshold,
        "rows": int(len(feature_frame)),
        "symbols": symbols,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.optimize:
        results = optimise_indicator_grid(
            feature_frame,
            sma_fast=args.grid_sma_fast,
            sma_slow=args.grid_sma_slow,
            ema_fast=args.grid_ema_fast,
            ema_slow=args.grid_ema_slow,
            rsi_period=args.grid_rsi_period,
            rsi_buy=args.grid_rsi_buy,
            macd_fast=args.grid_macd_fast,
            macd_slow=args.grid_macd_slow,
            macd_signal=args.grid_macd_signal,
            deepseek_thresholds=args.grid_deepseek_threshold,
        )
        top = args.grid_top if args.grid_top > 0 else 10
        top_results = results[:top]
        output = {
            "top": top_results,
            "total_combinations": len(results),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        if args.grid_output is not None:
            args.grid_output.parent.mkdir(parents=True, exist_ok=True)
            args.grid_output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
