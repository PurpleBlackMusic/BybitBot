"""Utilities for adaptive self-learning and performance tracking."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .ai.models import MODEL_FILENAME, MarketModel, train_market_model, load_model
from .envs import Settings
from .log import log
from .paths import DATA_DIR
from .trade_pairs import pair_trades, pair_trades_cache_signature


@dataclass(frozen=True)
class TradePerformanceSnapshot:
    """Summary statistics of recent completed trades."""

    results: Tuple[int, ...]
    win_streak: int
    loss_streak: int
    realised_sum: float
    average_pnl: float
    sample_count: int
    last_exit_ts: Optional[float]
    window_ms: int

    def recent_results(self, limit: int = 10) -> Tuple[int, ...]:
        if limit <= 0:
            return ()
        return self.results[-limit:]


_PerformanceCacheKey = Tuple[Path, Tuple[int, int, int], int]
_PERFORMANCE_CACHE: Dict[_PerformanceCacheKey, TradePerformanceSnapshot] = {}
_TRAINING_STATE_CACHE: Dict[Path, Dict[str, object]] = {}


def _coerce_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _trade_realised_pnl(trade: Mapping[str, object]) -> Optional[float]:
    qty = _coerce_float(trade.get("qty"))
    entry = _coerce_float(trade.get("entry_vwap"))
    exit_price = _coerce_float(trade.get("exit_vwap"))
    fees = _coerce_float(trade.get("fees")) or 0.0
    if qty is None or entry is None or exit_price is None:
        return None
    if qty <= 0 or entry <= 0 or exit_price <= 0:
        return None
    return (exit_price - entry) * qty - fees


def _trade_timestamp(trade: Mapping[str, object]) -> Optional[float]:
    for key in ("exit_ts", "entry_ts"):
        ts_value = trade.get(key)
        ts = _coerce_float(ts_value)
        if ts is None:
            continue
        if ts > 1e12:
            ts /= 1000.0
        return ts
    return None


def load_trade_performance(
    *,
    data_dir: Path | str = DATA_DIR,
    settings: Optional[Settings] = None,
    limit: int = 200,
    window_days: int = 30,
) -> Optional[TradePerformanceSnapshot]:
    """Return cached statistics for recent trades."""

    base_dir = Path(data_dir)
    window_ms = max(int(window_days * 24 * 3600 * 1000), 0)
    signature = pair_trades_cache_signature(window_ms=window_ms, settings=settings)
    cache_key: _PerformanceCacheKey = (base_dir.resolve(), signature, max(limit, 0))
    cached = _PERFORMANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    trades = pair_trades(window_ms=window_ms, settings=settings)
    if not trades:
        return None

    trades.sort(key=lambda row: _coerce_float(row.get("exit_ts")) or _coerce_float(row.get("entry_ts")) or 0.0)
    selected: Sequence[Mapping[str, object]]
    if limit > 0:
        selected = trades[-limit:]
    else:
        selected = trades

    results: list[int] = []
    realised_values: list[float] = []
    last_exit_ts: Optional[float] = None
    for trade in selected:
        pnl = _trade_realised_pnl(trade)
        if pnl is None:
            continue
        realised_values.append(pnl)
        if pnl > 0:
            results.append(1)
        elif pnl < 0:
            results.append(-1)
        else:
            results.append(0)
        ts = _trade_timestamp(trade)
        if ts is not None:
            last_exit_ts = ts

    if not results:
        return None

    def _streak(target: int) -> int:
        streak = 0
        for outcome in reversed(results):
            if outcome == target:
                streak += 1
            elif outcome == 0:
                continue
            else:
                break
        return streak

    win_streak = _streak(1)
    loss_streak = _streak(-1)
    sample_count = len(results)
    realised_sum = sum(realised_values)
    average = realised_sum / sample_count if sample_count else 0.0

    snapshot = TradePerformanceSnapshot(
        results=tuple(results),
        win_streak=win_streak,
        loss_streak=loss_streak,
        realised_sum=realised_sum,
        average_pnl=average,
        sample_count=sample_count,
        last_exit_ts=last_exit_ts,
        window_ms=window_ms,
    )
    _PERFORMANCE_CACHE[cache_key] = snapshot
    return snapshot


def _training_state_path(data_dir: Path) -> Path:
    return data_dir / "ai" / "self_learning.json"


def _load_training_state(path: Path) -> Dict[str, object]:
    cached = _TRAINING_STATE_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.exists():
        state: Dict[str, object] = {}
    else:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    _TRAINING_STATE_CACHE[path] = state
    return state


def _store_training_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _TRAINING_STATE_CACHE[path] = state


def maybe_retrain_market_model(
    *,
    data_dir: Path | str = DATA_DIR,
    settings: Optional[Settings] = None,
    now: Optional[float] = None,
    force: bool = False,
    min_samples: int = 25,
) -> Optional[Dict[str, object]]:
    """Retrain the market model on a rolling window of executions when stale."""

    base_dir = Path(data_dir)
    now_ts = float(now if now is not None else time.time())

    interval_minutes = _coerce_float(getattr(settings, "ai_retrain_minutes", None))
    if interval_minutes is None or interval_minutes <= 0:
        retrain_interval = 7 * 24 * 3600.0
    else:
        retrain_interval = max(interval_minutes * 60.0, 3600.0)

    trade_limit = getattr(settings, "ai_training_trade_limit", 0) if settings else 0
    if not isinstance(trade_limit, (int, float)) or trade_limit <= 0:
        trade_limit = 400
    trade_limit = int(trade_limit)

    state_path = _training_state_path(base_dir)
    state = _load_training_state(state_path)
    last_retrain = _coerce_float(state.get("last_retrain_ts")) or 0.0

    model_path = base_dir / "ai" / MODEL_FILENAME
    model = load_model(model_path, data_dir=base_dir)
    if model is None:
        force = True

    if not force and last_retrain > 0.0 and (now_ts - last_retrain) < retrain_interval:
        return state or None

    trained: Optional[MarketModel]
    try:
        trained = train_market_model(
            data_dir=base_dir,
            model_path=model_path,
            min_samples=min_samples,
            limit=trade_limit,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log("market_model.retrain.error", err=str(exc))
        log(
            "market_model.retrain.fallback",
            severity="warning",
            reason="training_failed",
            using_cached_model=model is not None,
            previous_trained_at=getattr(model, "trained_at", None),
            previous_samples=getattr(model, "samples", None),
        )
        return state or None

    if trained is None:
        log(
            "market_model.retrain.skipped",
            reason="insufficient_samples",
            samples=0,
            trade_limit=trade_limit,
        )
        return state or None

    metrics = dict(getattr(trained, "training_metrics", {}) or {})
    new_state = {
        "last_retrain_ts": now_ts,
        "samples": trained.samples,
        "trade_limit": trade_limit,
        "metrics": metrics,
    }
    _store_training_state(state_path, new_state)
    log(
        "market_model.retrain.complete",
        samples=trained.samples,
        trade_limit=trade_limit,
        metrics=metrics,
    )
    return new_state
