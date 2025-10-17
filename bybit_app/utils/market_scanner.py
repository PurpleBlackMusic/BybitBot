from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from statistics import fmean, pstdev
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd

from .bybit_api import BybitAPI
from .ai.external import ExternalFeatureProvider
from .ai.models import (
    MarketModel,
    ensure_market_model,
    initialise_feature_map,
    liquidity_feature,
)
from .fees import fee_rate_for_symbol
from .freqai import get_prediction_store
from .log import log
from .paths import DATA_DIR
from .market_features import build_feature_bundle
from .ohlcv import normalise_ohlcv_frame
from .symbols import ensure_usdt_symbol
from .telegram_notify import enqueue_telegram_message
from .trade_analytics import load_executions

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .portfolio_manager import PortfolioManager
    from .symbol_resolver import InstrumentMetadata, SymbolResolver
    from .envs import Settings
    from .trade_analytics import ExecutionRecord

SNAPSHOT_FILENAME = "market_snapshot.json"
DEFAULT_CACHE_TTL = 300.0
_TRADE_COST_SAMPLE = 600
_DEFAULT_MAKER_RATIO = 0.25
_DEFAULT_TAKER_FEE_BPS = 5.0

_OUTLIER_PERCENT_FIELDS: Tuple[str, ...] = (
    "price5mPcnt",
    "price15mPcnt",
    "price1hPcnt",
    "price4hPcnt",
    "price24hPcnt",
    "price7dPcnt",
)
_OUTLIER_Z_THRESHOLD = 4.5
_OUTLIER_IQR_MULTIPLIER = 3.5
_OUTLIER_MIN_MAGNITUDE = 80.0
_MAINTENANCE_KEYS: Tuple[str, ...] = (
    "maintenance",
    "maintenanceSymbols",
    "maintenance_symbols",
    "maintenance_list",
)
_DELIST_KEYS: Tuple[str, ...] = (
    "delist",
    "delists",
    "delisting",
    "delistingSymbols",
    "delisting_symbols",
    "delist_symbols",
    "delistList",
)

_DEFAULT_DAILY_SURGE_LIMIT = 12.0
_DEFAULT_RSI_OVERBOUGHT = 72.0
_DEFAULT_STOCH_OVERBOUGHT = 85.0

_HOURLY_SIGNAL_CACHE_TTL = 300.0
_VOLATILITY_CACHE_TTL = 900.0
_HOURLY_LOOKBACK_HOURS = 180
_MAX_CROSS_LOOKBACK = 6
_MAX_VOL_SAMPLE = 20
_VOLATILITY_LOOKBACK_DAYS = 30
_RSI_CONFIRMATION_BUY = 55.0
_RSI_CONFIRMATION_SELL = 45.0
_MIN_HOURLY_MOMENTUM = 0.1
_CROSS_CONFIRMATION_WINDOW = 3

_HOURLY_SIGNAL_CACHE: Dict[Tuple[str, str], Tuple[float, Optional[Dict[str, object]]]] = {}
_VOLATILITY_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}
_IMPULSE_SIGNAL_THRESHOLD = math.log(1.8)


def _ledger_path_for_costs(data_dir: Path, testnet: Optional[bool]) -> Path:
    data_dir = Path(data_dir)
    pnl_dir = data_dir / "pnl"
    legacy = pnl_dir / "executions.jsonl"
    if testnet is None:
        # fall back to whichever ledger exists, preferring the freshest network specific file
        testnet_path = pnl_dir / "executions.testnet.jsonl"
        mainnet_path = pnl_dir / "executions.mainnet.jsonl"
        candidates = [path for path in (testnet_path, mainnet_path) if path.exists()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 2:
            try:
                testnet_mtime = testnet_path.stat().st_mtime
                mainnet_mtime = mainnet_path.stat().st_mtime
            except OSError:
                testnet_mtime = mainnet_mtime = 0.0
            return testnet_path if testnet_mtime >= mainnet_mtime else mainnet_path
        return legacy

    marker = "testnet" if bool(testnet) else "mainnet"
    candidate = pnl_dir / f"executions.{marker}.jsonl"
    if candidate.exists():
        return candidate
    if legacy.exists():
        return legacy
    return candidate


def _ledger_signature(path: Path) -> Tuple[str, int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return str(path), 0, 0

    mtime_ns = getattr(stat, "st_mtime_ns", None)
    if mtime_ns is None:
        mtime_ns = int(stat.st_mtime * 1_000_000_000)
    size = int(getattr(stat, "st_size", 0))
    return str(path), int(mtime_ns), size


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _extract_training_metric(
    model: Optional[MarketModel], name: str
) -> Optional[float]:
    if model is None:
        return None
    metrics = getattr(model, "training_metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    return _safe_float(metrics.get(name))


def _calibrate_probability(
    probability: Optional[float], model: Optional[MarketModel]
) -> Optional[float]:
    if probability is None:
        return None
    bias = _extract_training_metric(model, "calibration_bias")
    if bias is None:
        return probability
    adjusted = float(probability) + float(bias) * 0.5
    return max(min(adjusted, 0.995), 0.005)


def _decision_threshold(model: Optional[MarketModel], default: float = 0.5) -> float:
    positive_rate = _extract_training_metric(model, "positive_rate")
    if positive_rate is None:
        return default
    candidate = float(positive_rate) + 0.08
    return max(0.4, min(0.6, candidate))


def _safe_setting_float(settings: Optional["Settings"], name: str, default: float) -> float:
    if settings is None:
        return default
    try:
        value = getattr(settings, name)
    except AttributeError:
        return default
    numeric = _safe_float(value)
    if numeric is None or not math.isfinite(numeric):
        return default
    return float(numeric)


def _maker_fee_hint(settings: Optional["Settings"], taker_hint: float) -> float:
    if settings is None:
        return taker_hint * 0.5
    raw = _safe_float(getattr(settings, "ai_maker_fee_bps", None))
    if raw is None or not math.isfinite(raw):
        return taker_hint * 0.5
    return float(raw)


def _extract_fee_components(records: Sequence["ExecutionRecord"]) -> Tuple[Optional[float], Optional[float], Optional[float], int, int]:
    maker_notional = 0.0
    maker_fees = 0.0
    taker_notional = 0.0
    taker_fees = 0.0
    maker_samples = 0
    taker_samples = 0

    for record in records:
        notional = float(getattr(record, "notional", 0.0) or 0.0)
        if notional <= 0:
            continue
        fee = float(getattr(record, "fee", 0.0) or 0.0)
        if fee < 0:
            fee = 0.0
        is_maker = getattr(record, "is_maker", None)
        if is_maker is True:
            maker_notional += notional
            maker_fees += fee
            maker_samples += 1
        else:
            taker_notional += notional
            taker_fees += fee
            taker_samples += 1

    maker_fee_bps: Optional[float]
    if maker_notional > 0 and maker_fees > 0:
        maker_fee_bps = (maker_fees / maker_notional) * 10_000.0
    else:
        maker_fee_bps = None

    taker_fee_bps: Optional[float]
    if taker_notional > 0 and taker_fees > 0:
        taker_fee_bps = (taker_fees / taker_notional) * 10_000.0
    else:
        taker_fee_bps = None

    total_notional = maker_notional + taker_notional
    maker_ratio = maker_notional / total_notional if total_notional > 0 else None

    return maker_fee_bps, taker_fee_bps, maker_ratio, maker_samples, taker_samples


@lru_cache(maxsize=16)
def _cached_cost_profile(
    signature: Tuple[str, int, int],
    taker_hint: float,
    maker_hint: float,
    slippage_hint: float,
) -> Dict[str, float]:
    path_str, _, _ = signature
    path = Path(path_str)
    records = load_executions(path, limit=_TRADE_COST_SAMPLE) if path_str else []
    maker_fee_raw, taker_fee_raw, maker_ratio_raw, maker_samples, taker_samples = _extract_fee_components(records)

    taker_fee = taker_fee_raw if taker_fee_raw is not None else taker_hint
    if taker_fee <= 0:
        taker_fee = taker_hint if taker_hint > 0 else _DEFAULT_TAKER_FEE_BPS
    taker_fee = _clamp(float(taker_fee), 0.0, 50.0)

    maker_fee = maker_fee_raw if maker_fee_raw is not None else maker_hint
    if maker_fee <= 0:
        maker_fee = maker_hint
    if maker_fee <= 0:
        maker_fee = taker_fee * 0.5
    maker_fee = _clamp(float(maker_fee), 0.0, taker_fee)

    maker_ratio = maker_ratio_raw if maker_ratio_raw is not None else _DEFAULT_MAKER_RATIO
    maker_ratio = _clamp(float(maker_ratio), 0.0, 1.0)

    effective_fee = maker_fee * maker_ratio + taker_fee * (1.0 - maker_ratio)
    round_trip_fee = effective_fee * 2.0
    slippage = _clamp(float(slippage_hint), 0.0, 500.0)

    return {
        "maker_fee_bps": maker_fee,
        "taker_fee_bps": taker_fee,
        "maker_ratio": maker_ratio,
        "effective_fee_bps": effective_fee,
        "round_trip_fee_bps": round_trip_fee,
        "slippage_bps": slippage,
        "sample_size": maker_samples + taker_samples,
        "fee_source": "history",
    }


def _resolve_trade_cost_profile(
    data_dir: Path,
    settings: Optional["Settings"],
    testnet: Optional[bool],
) -> Dict[str, float]:
    taker_hint = _safe_setting_float(settings, "ai_fee_bps", _DEFAULT_TAKER_FEE_BPS)
    maker_hint = _maker_fee_hint(settings, taker_hint)
    slippage_hint = _safe_setting_float(settings, "ai_slippage_bps", 0.0)

    ledger_path = _ledger_path_for_costs(Path(data_dir), testnet)
    signature = _ledger_signature(ledger_path)
    return _cached_cost_profile(signature, taker_hint, maker_hint, slippage_hint)


def _default_fee_guard_bps(settings: Optional["Settings"]) -> float:
    """Return the configured fallback fee guard expressed in basis points."""

    return max(_safe_setting_float(settings, "spot_tp_fee_guard_bps", 20.0), 0.0)


def _resolve_symbol_fee_guard_bps(
    symbol: str,
    *,
    settings: Optional["Settings"],
    api: Optional[BybitAPI],
    cache: Dict[str, float],
) -> float:
    """Resolve the dynamic fee guard for a symbol in basis points."""

    if not symbol:
        return 0.0

    cached = cache.get(symbol)
    if cached is not None:
        return cached

    sentinel = _default_fee_guard_bps(settings)
    override_value: Optional[float] = None
    if settings is not None:
        candidate = _safe_float(getattr(settings, "spot_tp_fee_guard_bps", None))
        if candidate is not None and candidate > 0:
            override_value = float(candidate)

    if override_value is not None and override_value > 0 and override_value != sentinel:
        guard_bps = override_value
    else:
        guard_bps = sentinel
        snapshot = fee_rate_for_symbol(category="spot", symbol=symbol)
        combined: Optional[float] = None
        if snapshot is not None:
            taker_bps = snapshot.taker_fee_bps
            maker_bps = snapshot.maker_fee_bps
            if taker_bps is not None and taker_bps != 0:
                combined = abs(float(taker_bps)) * 2.0
            elif maker_bps is not None and maker_bps != 0:
                combined = abs(float(maker_bps)) * 2.0
        if combined is not None and combined > 0:
            guard_bps = combined
        elif override_value is not None and override_value > 0:
            guard_bps = override_value

    guard_bps = max(float(guard_bps), 0.0)
    cache[symbol] = guard_bps
    return guard_bps


class MarketScannerError(RuntimeError):
    """Raised when the market snapshot cannot be loaded."""


def _network_snapshot_filename(
    *, testnet: Optional[bool] = None, settings: Optional["Settings"] = None
) -> str:
    """Return the snapshot filename matching the active network."""

    resolved = testnet
    if resolved is None and settings is not None:
        try:
            resolved = bool(getattr(settings, "testnet"))
        except Exception:
            resolved = None

    if resolved:
        if "." in SNAPSHOT_FILENAME:
            stem, ext = SNAPSHOT_FILENAME.rsplit(".", 1)
            return f"{stem}_testnet.{ext}"
        return f"{SNAPSHOT_FILENAME}_testnet"
    return SNAPSHOT_FILENAME


def _snapshot_path(
    data_dir: Path, *, testnet: Optional[bool] = None, settings: Optional["Settings"] = None
) -> Path:
    filename = _network_snapshot_filename(testnet=testnet, settings=settings)
    return Path(data_dir) / "ai" / filename


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _normalise_percent(value: object) -> Optional[float]:
    pct = _safe_float(value)
    if pct is None:
        return None
    if abs(pct) <= 1.0:
        pct *= 100.0
    return pct


def _ohlcv_hourly_path(data_dir: Path, symbol: str) -> Path:
    base = Path(data_dir) / "ohlcv" / "spot" / symbol.upper()
    return base / f"{symbol.upper()}_1h.csv"


def _load_hourly_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    working = frame.copy()
    if "start" not in working.columns:
        if "timestamp" in working.columns:
            working = working.rename(columns={"timestamp": "start"})
        elif "time" in working.columns:
            working = working.rename(columns={"time": "start"})
    try:
        normalised = normalise_ohlcv_frame(working, timestamp_col="start")
    except Exception:
        return None
    if "close" not in normalised.columns:
        return None
    return normalised


def _compute_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    if closes.empty:
        return pd.Series(dtype="float64")
    delta = closes.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def _detect_recent_cross(diff: pd.Series) -> Tuple[Optional[str], Optional[int]]:
    if diff.empty:
        return None, None
    max_offset = min(len(diff) - 1, _MAX_CROSS_LOOKBACK)
    if max_offset <= 0:
        return None, None
    for offset in range(1, max_offset + 1):
        current = diff.iloc[-offset]
        previous = diff.iloc[-(offset + 1)]
        if current > 0 and previous <= 0:
            return "bullish", offset - 1
        if current < 0 and previous >= 0:
            return "bearish", offset - 1
    return None, None


def _load_hourly_indicator_bundle(
    symbol: str,
    *,
    data_dir: Path = DATA_DIR,
    now: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    key = (str(Path(data_dir).resolve()), symbol.upper())
    now_ts = now if isinstance(now, (int, float)) else time.time()
    cached = _HOURLY_SIGNAL_CACHE.get(key)
    if cached and now_ts - cached[0] <= _HOURLY_SIGNAL_CACHE_TTL:
        payload = cached[1]
        return copy.deepcopy(payload) if payload is not None else None

    path = _ohlcv_hourly_path(data_dir, symbol)
    frame = _load_hourly_frame(path)
    if frame is None or frame.empty:
        _HOURLY_SIGNAL_CACHE[key] = (now_ts, None)
        return None

    frame = frame.tail(max(_HOURLY_LOOKBACK_HOURS + 10, 60))
    if frame.empty or "close" not in frame.columns:
        _HOURLY_SIGNAL_CACHE[key] = (now_ts, None)
        return None

    working = frame.set_index("start")
    assert isinstance(working.index, pd.DatetimeIndex)

    closes = working["close"].astype(float)
    if closes.size < 55:
        _HOURLY_SIGNAL_CACHE[key] = (now_ts, None)
        return None

    ema_fast = closes.ewm(span=21, adjust=False).mean()
    ema_slow = closes.ewm(span=50, adjust=False).mean()
    diff = ema_fast - ema_slow
    cross_label, cross_age = _detect_recent_cross(diff)

    rsi_series = _compute_rsi_series(closes)
    rsi_value = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

    momentum_pct: Optional[float] = None
    if closes.size >= 2 and closes.iloc[-2] != 0:
        momentum_pct = (closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0

    high_series = working.get("high")
    low_series = working.get("low")
    lookback_hours = max(_HOURLY_LOOKBACK_HOURS, 48)
    end_ts = working.index[-1]
    start_ts = end_ts - pd.Timedelta(hours=lookback_hours)
    recent_window = working.loc[working.index >= start_ts]
    last_close = float(closes.iloc[-1])
    breakout = False
    breakdown = False
    recent_high = None
    recent_low = None
    if isinstance(high_series, pd.Series) and not high_series.empty:
        recent_high = float(recent_window["high"].max()) if not recent_window.empty else None
        if recent_high and recent_high > 0:
            breakout = last_close >= recent_high * 0.999
    if isinstance(low_series, pd.Series) and not low_series.empty:
        recent_low = float(recent_window["low"].min()) if not recent_window.empty else None
        if recent_low and recent_low > 0:
            breakdown = last_close <= recent_low * 1.001

    payload: Dict[str, object] = {
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "ema_cross": cross_label,
        "ema_cross_bars_ago": cross_age,
        "ema_fast_above": bool(diff.iloc[-1] > 0),
        "rsi": rsi_value,
        "momentum_pct": float(momentum_pct) if momentum_pct is not None else None,
        "breakout": breakout,
        "breakdown": breakdown,
        "last_close": last_close,
    }
    if recent_high is not None:
        payload["recent_high"] = recent_high
    if recent_low is not None:
        payload["recent_low"] = recent_low

    _HOURLY_SIGNAL_CACHE[key] = (now_ts, payload)
    return copy.deepcopy(payload)


def _sanitise_hourly_signal(bundle: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not bundle:
        return None
    allowed = {
        "ema_fast",
        "ema_slow",
        "ema_cross",
        "ema_cross_bars_ago",
        "ema_fast_above",
        "rsi",
        "momentum_pct",
        "breakout",
        "breakdown",
        "last_close",
        "recent_high",
        "recent_low",
    }
    result: Dict[str, object] = {}
    for key, value in bundle.items():
        if key not in allowed:
            continue
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result if result else None


def _compute_dynamic_min_change(
    data_dir: Path,
    ratio: float,
    fallback: float,
    *,
    now: Optional[float] = None,
) -> float:
    if ratio is None or ratio <= 0:
        return max(fallback, 0.05)

    resolved_dir = str(Path(data_dir).resolve())
    now_ts = now if isinstance(now, (int, float)) else time.time()
    cached = _VOLATILITY_CACHE.get(resolved_dir)
    if cached and now_ts - cached[0] <= _VOLATILITY_CACHE_TTL:
        cached_value = cached[1]
        if cached_value is None:
            return max(fallback, 0.05)
        return max(cached_value, 0.05)

    base_dir = Path(data_dir) / "ohlcv" / "spot"
    if not base_dir.exists():
        _VOLATILITY_CACHE[resolved_dir] = (now_ts, None)
        return max(fallback, 0.05)

    changes: List[float] = []
    for symbol_dir in sorted(base_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        path = symbol_dir / f"{symbol_dir.name}_1h.csv"
        frame = _load_hourly_frame(path)
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        working = frame.set_index("start")
        assert isinstance(working.index, pd.DatetimeIndex)
        if working.empty:
            continue
        window_start = working.index.max() - pd.Timedelta(days=_VOLATILITY_LOOKBACK_DAYS)
        subset = working.loc[working.index >= window_start]
        closes = subset["close"].astype(float)
        if closes.empty:
            continue
        daily = closes.resample("1D").last().dropna()
        if daily.size <= 1:
            continue
        pct = daily.pct_change().dropna()
        if pct.empty:
            continue
        changes.append(float(pct.abs().mean() * 100.0))
        if len(changes) >= _MAX_VOL_SAMPLE:
            break

    if not changes:
        _VOLATILITY_CACHE[resolved_dir] = (now_ts, None)
        return max(fallback, 0.05)

    avg_change = fmean(changes)
    dynamic = max(avg_change * ratio, 0.0)
    if dynamic <= 0:
        _VOLATILITY_CACHE[resolved_dir] = (now_ts, None)
        return max(fallback, 0.05)

    final_value = max(dynamic, 0.05)
    _VOLATILITY_CACHE[resolved_dir] = (now_ts, final_value)
    return final_value


def _resolve_min_turnover_threshold(
    base_min_turnover: float,
    rows: Sequence[Mapping[str, object]],
    ratio: float,
) -> float:
    base_value = max(float(base_min_turnover or 0.0), 0.0)
    if ratio is None or ratio <= 0:
        return base_value
    turnovers: List[float] = []
    for row in rows:
        value = _safe_float(row.get("turnover24h"))
        if value is None or value <= 0:
            continue
        turnovers.append(float(value))
    if not turnovers:
        return base_value
    turnovers.sort(reverse=True)
    sample = turnovers[: _MAX_VOL_SAMPLE]
    average = fmean(sample)
    dynamic = average * ratio
    if base_value > 0:
        dynamic = max(dynamic, base_value * 0.5)
    return max(dynamic, 0.0)


def _estimate_top_depth_threshold(
    rows: Sequence[Mapping[str, object]],
    ratio: float,
    fallback: Optional[float],
) -> float:
    base_value = max(float(fallback or 0.0), 0.0)
    if ratio is None or ratio <= 0:
        return base_value
    totals: List[float] = []
    for row in rows:
        bid_price = _safe_float(row.get("bestBidPrice") or row.get("bid1Price"))
        ask_price = _safe_float(row.get("bestAskPrice") or row.get("ask1Price"))
        bid_size = _safe_float(row.get("bid1Size") or row.get("bidSize"))
        ask_size = _safe_float(row.get("ask1Size") or row.get("askSize"))
        bid_quote = (
            bid_price * bid_size
            if bid_price and bid_size and bid_price > 0 and bid_size > 0
            else None
        )
        ask_quote = (
            ask_price * ask_size
            if ask_price and ask_size and ask_price > 0 and ask_size > 0
            else None
        )
        total_quote: Optional[float] = None
        if bid_quote is not None or ask_quote is not None:
            total_quote = (bid_quote or 0.0) + (ask_quote or 0.0)
        if total_quote is not None and total_quote > 0:
            totals.append(float(total_quote))
    if not totals:
        return base_value
    totals.sort(reverse=True)
    sample = totals[: _MAX_VOL_SAMPLE]
    average = fmean(sample)
    dynamic = average * ratio
    if base_value > 0:
        dynamic = max(dynamic, base_value * 0.5)
    return max(dynamic, 0.0)


def _resolve_trend_with_confirmation(
    change_pct: Optional[float],
    effective_change: float,
    indicator: Optional[Mapping[str, object]],
    *,
    force_include: bool = False,
) -> str:
    if change_pct is None:
        return "wait"
    if change_pct >= effective_change:
        base_trend = "buy"
    elif change_pct <= -effective_change:
        base_trend = "sell"
    else:
        return "wait"

    if force_include or not indicator:
        return base_trend

    rsi_value = indicator.get("rsi") if isinstance(indicator, Mapping) else None
    momentum = indicator.get("momentum_pct") if isinstance(indicator, Mapping) else None
    cross_label = indicator.get("ema_cross") if isinstance(indicator, Mapping) else None
    cross_age = indicator.get("ema_cross_bars_ago") if isinstance(indicator, Mapping) else None
    breakout = bool(indicator.get("breakout")) if isinstance(indicator, Mapping) else False
    breakdown = bool(indicator.get("breakdown")) if isinstance(indicator, Mapping) else False

    if isinstance(cross_age, (int, float)) and cross_age >= 0:
        cross_recent = cross_age <= _CROSS_CONFIRMATION_WINDOW
    else:
        cross_recent = True if cross_label in {"bullish", "bearish"} else False

    momentum_ok_buy = (
        momentum is None
        or (isinstance(momentum, (int, float)) and float(momentum) >= _MIN_HOURLY_MOMENTUM)
    )
    momentum_ok_sell = (
        momentum is None
        or (isinstance(momentum, (int, float)) and float(momentum) <= -_MIN_HOURLY_MOMENTUM)
    )

    if base_trend == "buy":
        rsi_ok = isinstance(rsi_value, (int, float)) and float(rsi_value) >= _RSI_CONFIRMATION_BUY
        if cross_label == "bullish" and cross_recent and rsi_ok and momentum_ok_buy:
            return "buy"
        if breakout and rsi_ok and momentum_ok_buy:
            return "buy"
        return "wait"

    rsi_ok = isinstance(rsi_value, (int, float)) and float(rsi_value) <= _RSI_CONFIRMATION_SELL
    if cross_label == "bearish" and cross_recent and rsi_ok and momentum_ok_sell:
        return "sell"
    if breakdown and rsi_ok and momentum_ok_sell:
        return "sell"
    return "wait"

def _spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or ask <= 0:
        return None
    spread = (ask - bid) / ask * 10000.0
    if spread < 0:
        spread = 0.0
    return spread


def _strength_from_change(change_pct: Optional[float]) -> float:
    if change_pct is None:
        return 0.0
    # smooth growth curve that approaches 1 for very strong moves
    return math.tanh(abs(change_pct) / 5.0)


TURNOVER_AVG_TRADES_PER_DAY = 4_000.0
"""Approximate number of trades represented in a 24h turnover snapshot."""


def _normalised_turnover(value: Optional[float]) -> float:
    """Convert a 24h turnover metric to a single-trade estimate."""

    if value is None or value <= 0:
        return 0.0
    return float(value) / TURNOVER_AVG_TRADES_PER_DAY


def _score_turnover(turnover: Optional[float]) -> float:
    normalised = _normalised_turnover(turnover)
    return liquidity_feature(normalised)


def _quantile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    fraction = position - lower_index
    return lower + (upper - lower) * fraction


def _normalise_epoch_seconds(value: object) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    absolute = abs(numeric)
    if absolute >= 1e18:
        seconds = numeric / 1_000_000_000.0
    elif absolute >= 1e15:
        seconds = numeric / 1_000_000.0
    elif absolute >= 1e12:
        seconds = numeric / 1000.0
    elif absolute >= 1e9:
        seconds = float(numeric)
    elif absolute >= 1e6:
        seconds = float(numeric)
    else:
        seconds = float(numeric)
    return seconds


def _collect_symbol_candidates(value: object) -> Set[str]:
    symbols: Set[str] = set()
    if value is None:
        return symbols
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return symbols
        parts = re.split(r"[\s,;]+", text)
        for part in parts:
            candidate, _ = ensure_usdt_symbol(part)
            if candidate:
                symbols.add(candidate)
        return symbols
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("symbol", "pair", "instrument", "asset", "name")):
                symbols.update(_collect_symbol_candidates(nested))
            elif isinstance(nested, (Mapping, list, tuple, set, frozenset)):
                symbols.update(_collect_symbol_candidates(nested))
        return symbols
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            symbols.update(_collect_symbol_candidates(item))
        return symbols
    candidate, _ = ensure_usdt_symbol(value)
    if candidate:
        symbols.add(candidate)
    return symbols


def _extract_snapshot_stoplist(snapshot: Mapping[str, object], keys: Sequence[str]) -> Set[str]:
    stoplist: Set[str] = set()
    for key in keys:
        value = snapshot.get(key)
        if value is None:
            continue
        stoplist.update(_collect_symbol_candidates(value))
    return stoplist


def _detect_outlier_symbols(rows: Sequence[Mapping[str, object]]) -> Set[str]:
    field_samples: Dict[str, List[Tuple[str, float]]] = {field: [] for field in _OUTLIER_PERCENT_FIELDS}
    for raw in rows:
        symbol, _ = ensure_usdt_symbol(raw.get("symbol"))
        if not symbol:
            continue
        for field in _OUTLIER_PERCENT_FIELDS:
            value = _normalise_percent(raw.get(field))
            if value is None or not math.isfinite(value):
                continue
            field_samples[field].append((symbol, float(value)))

    outliers: Set[str] = set()
    for field, samples in field_samples.items():
        if len(samples) < 5:
            continue
        values = [value for _, value in samples]
        sorted_values = sorted(values)
        lower_quartile = _quantile(sorted_values, 0.25)
        upper_quartile = _quantile(sorted_values, 0.75)
        if lower_quartile is None or upper_quartile is None:
            continue
        iqr = upper_quartile - lower_quartile
        lower_bound = lower_quartile - _OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = upper_quartile + _OUTLIER_IQR_MULTIPLIER * iqr
        mean_value = fmean(values)
        std_dev = pstdev(values) if len(values) > 1 else 0.0

        for symbol, value in samples:
            magnitude = abs(value)
            if magnitude < _OUTLIER_MIN_MAGNITUDE:
                continue
            flagged = False
            if std_dev > 1e-9:
                z_score = abs((value - mean_value) / std_dev)
                if z_score > _OUTLIER_Z_THRESHOLD:
                    flagged = True
            if not flagged and iqr > 0:
                if value < lower_bound or value > upper_bound:
                    flagged = True
            if flagged:
                outliers.add(symbol)
    return outliers


def _row_in_maintenance(row: Mapping[str, object]) -> bool:
    status_keys = ("status", "symbolStatus", "tradingStatus", "state")
    for key in status_keys:
        raw = row.get(key)
        if not isinstance(raw, str):
            continue
        text = raw.strip().lower()
        if not text:
            continue
        if any(word in text for word in ("maint", "suspend", "halt", "pause", "down")):
            return True
    flag_keys = ("maintenance", "underMaintenance", "isSuspended", "suspended")
    for key in flag_keys:
        raw = row.get(key)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
        elif raw:
            return True
    return False


def _row_delisted(row: Mapping[str, object], *, now: Optional[float] = None) -> bool:
    status_keys = ("status", "symbolStatus", "state", "tradingStatus")
    for key in status_keys:
        raw = row.get(key)
        if isinstance(raw, str):
            text = raw.strip().lower()
            if "delist" in text or "terminate" in text:
                return True

    timestamp_keys = (
        "delistingTime",
        "delistTime",
        "deliveryTime",
        "expiryTime",
        "expireTime",
    )
    reference = now if now is not None else time.time()
    for key in timestamp_keys:
        ts = _normalise_epoch_seconds(row.get(key))
        if ts is None:
            continue
        if ts <= reference + 24 * 3600:
            return True
    return False


def _apply_stoplists_from_settings(
    settings: Optional["Settings"],
    *,
    maintenance: Set[str],
    delisted: Set[str],
) -> None:
    if settings is None:
        return
    manual_maintenance = getattr(settings, "ai_maintenance_stoplist", None)
    if manual_maintenance is not None:
        maintenance.update(_collect_symbol_candidates(manual_maintenance))
    manual_delist = getattr(settings, "ai_delist_stoplist", None)
    if manual_delist is not None:
        delisted.update(_collect_symbol_candidates(manual_delist))


def _logit(probability: float) -> float:
    probability = max(min(probability, 1.0 - 1e-6), 1e-6)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _best_volume_impulse(impulses: Mapping[str, Optional[float]] | None) -> float:
    if not impulses:
        return 0.0
    best = 0.0
    for value in impulses.values():
        if value is None:
            continue
        if abs(value) > abs(best):
            best = float(value)
    return best


def _weighted_opportunity_model(
    *,
    change_pct: Optional[float],
    trend: str,
    turnover: Optional[float],
    spread_bps: Optional[float],
    features: Mapping[str, object],
    force_include: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    base_probability = 0.5
    if change_pct is not None:
        strength = _strength_from_change(change_pct)
        if trend == "buy":
            base_probability = 0.5 + strength / 2.0
        elif trend == "sell":
            base_probability = 0.5 - strength / 2.0
    base_logit = _logit(base_probability)

    direction = 0
    if trend == "buy":
        direction = 1
    elif trend == "sell":
        direction = -1

    components: Dict[str, float] = {"bias": -0.2}

    blended = features.get("blended_change_pct")
    if blended is not None and direction != 0:
        components["multi_tf"] = direction * math.tanh(float(blended) / 8.0) * 0.8
    else:
        components["multi_tf"] = 0.0

    strength_component = 0.0
    if change_pct is not None and direction != 0:
        strength_component = direction * _strength_from_change(change_pct)
    components["momentum"] = strength_component

    impulses = features.get("volume_impulse")
    best_impulse = _best_volume_impulse(impulses if isinstance(impulses, Mapping) else None)
    components["volume"] = max(min(best_impulse, 1.5), -1.5) * 0.7

    volatility_pct = features.get("volatility_pct")
    if isinstance(volatility_pct, (int, float)):
        components["volatility"] = -math.tanh(float(volatility_pct) / 70.0) * 0.9
    else:
        components["volatility"] = 0.0

    depth_imbalance = features.get("depth_imbalance")
    if isinstance(depth_imbalance, (int, float)) and direction != 0:
        components["orderbook"] = float(depth_imbalance) * direction * 1.2
    else:
        components["orderbook"] = 0.0

    order_flow_ratio = features.get("order_flow_ratio")
    if isinstance(order_flow_ratio, (int, float)) and direction != 0:
        flow_component = float(order_flow_ratio) * direction * 1.1
        components["order_flow"] = max(min(flow_component, 1.8), -1.8)
    else:
        components["order_flow"] = 0.0

    cvd_score = features.get("cvd_score")
    if isinstance(cvd_score, (int, float)) and direction != 0:
        cvd_component = float(cvd_score) * direction * 1.0
        components["cvd"] = max(min(cvd_component, 1.5), -1.5)
    else:
        components["cvd"] = 0.0

    top_depth_imbalance = features.get("top_depth_imbalance")
    if isinstance(top_depth_imbalance, (int, float)) and direction != 0:
        components["top_depth"] = max(min(float(top_depth_imbalance) * direction * 0.9, 1.2), -1.2)
    else:
        components["top_depth"] = 0.0

    correlation_strength = features.get("correlation_strength")
    if isinstance(correlation_strength, (int, float)):
        components["correlation"] = -float(correlation_strength) * 0.6
    else:
        components["correlation"] = 0.0

    liquidity_raw = _score_turnover(turnover)
    components["liquidity"] = min(liquidity_raw / 6.0, 1.2)

    if spread_bps is not None:
        components["spread"] = -min(max(spread_bps, 0.0) / 120.0, 1.0)
    else:
        components["spread"] = 0.0

    if force_include:
        components["bias"] += 0.4

    logit = base_logit + sum(components.values())
    probability = _sigmoid(logit)

    score = probability * 100.0
    score += max(0.0, liquidity_raw) * 3.0
    score += max(0.0, best_impulse) * 15.0
    if isinstance(volatility_pct, (int, float)):
        score -= max(0.0, float(volatility_pct)) * 0.4
    if isinstance(depth_imbalance, (int, float)) and direction != 0:
        score += max(-10.0, min(10.0, float(depth_imbalance) * direction * 50.0))

    if score < 0:
        score = 0.0

    metrics = dict(components)
    metrics.update(
        {
            "base_probability": base_probability,
            "base_logit": base_logit,
            "final_logit": logit,
            "liquidity_raw": liquidity_raw,
            "best_volume_impulse": best_impulse,
        }
    )

    return probability, score, metrics


def _resolve_correlation_value(
    correlations: Mapping[str, object] | None, aliases: Sequence[str]
) -> Optional[float]:
    if not isinstance(correlations, Mapping):
        return None
    lowered_aliases = [alias.strip().lower() for alias in aliases if alias]
    for key, value in correlations.items():
        if not isinstance(key, str):
            continue
        cleaned = key.replace("-", "_").replace("/", "_").lower()
        for alias in lowered_aliases:
            if cleaned == alias:
                return _safe_float(value)
            if cleaned.replace("usdt", "") == alias:
                return _safe_float(value)
            if cleaned.endswith(alias):
                return _safe_float(value)
    return None


def _model_feature_vector(
    *,
    trend: str,
    change_pct: Optional[float],
    blended_change: Optional[float],
    turnover: Optional[float],
    volatility_pct: Optional[float],
    volume_impulse: Mapping[str, Optional[float]] | None,
    depth_imbalance: Optional[float],
    spread_bps: Optional[float],
    correlation_strength: Optional[float],
    order_flow_ratio: Optional[float],
    top_depth_imbalance: Optional[float],
    volatility_ratio: Optional[float],
    volume_trend: Optional[float],
    sentiment_score: Optional[float],
    news_heat: Optional[float],
    macro_regime_score: Optional[float],
    event_timestamp: Optional[float],
) -> Dict[str, float]:
    direction = 0
    if trend == "buy":
        direction = 1
    elif trend == "sell":
        direction = -1

    signed_change = 0.0
    if change_pct is not None:
        signed_change = float(change_pct)
        if direction < 0:
            signed_change = -signed_change

    multi_tf = blended_change
    if multi_tf is None:
        multi_tf = change_pct
    if multi_tf is None:
        multi_tf_value = signed_change
    else:
        multi_tf_value = float(multi_tf)
        if direction < 0:
            multi_tf_value = -multi_tf_value

    turnover_log = liquidity_feature(_normalised_turnover(turnover))

    vol_value = 0.0
    if isinstance(volatility_pct, (int, float)):
        vol_value = float(volatility_pct)

    best_impulse = _best_volume_impulse(volume_impulse if isinstance(volume_impulse, Mapping) else None)

    depth_value = 0.0
    if isinstance(depth_imbalance, (int, float)):
        depth_value = float(depth_imbalance)
        if direction != 0:
            depth_value *= direction

    spread_value = 0.0
    if isinstance(spread_bps, (int, float)):
        spread_value = float(spread_bps)

    correlation_value = 0.0
    if isinstance(correlation_strength, (int, float)):
        correlation_value = float(correlation_strength)

    session_hour_sin = 0.0
    session_hour_cos = 1.0
    if isinstance(event_timestamp, (int, float)) and math.isfinite(event_timestamp):
        seconds = float(event_timestamp) % 86_400.0
        angle = (seconds / 86_400.0) * 2.0 * math.pi
        session_hour_sin = math.sin(angle)
        session_hour_cos = math.cos(angle)

    order_flow_value = 0.0
    if isinstance(order_flow_ratio, (int, float)):
        order_flow_value = float(order_flow_ratio)
        if direction != 0:
            order_flow_value *= direction

    top_depth_value = 0.0
    if isinstance(top_depth_imbalance, (int, float)):
        top_depth_value = float(top_depth_imbalance)
        if direction != 0:
            top_depth_value *= direction

    volatility_ratio_value = 0.0
    if isinstance(volatility_ratio, (int, float)):
        volatility_ratio_value = float(volatility_ratio)

    volume_trend_value = 0.0
    if isinstance(volume_trend, (int, float)):
        volume_trend_value = float(volume_trend)

    sentiment_value = float(sentiment_score) if isinstance(sentiment_score, (int, float)) else 0.0
    news_value = float(news_heat) if isinstance(news_heat, (int, float)) else 0.0
    macro_value = float(macro_regime_score) if isinstance(macro_regime_score, (int, float)) else 0.0

    features = initialise_feature_map()
    features["directional_change_pct"] = signed_change
    features["multiframe_change_pct"] = multi_tf_value
    features["turnover_log"] = turnover_log
    features["volatility_pct"] = vol_value
    features["volume_impulse"] = best_impulse
    features["depth_imbalance"] = depth_value
    features["spread_bps"] = spread_value
    features["correlation_strength"] = correlation_value
    features["session_hour_sin"] = session_hour_sin
    features["session_hour_cos"] = session_hour_cos
    features["volatility_ratio"] = volatility_ratio_value
    features["volume_trend"] = volume_trend_value
    features["order_flow_ratio"] = order_flow_value
    features["top_depth_imbalance"] = top_depth_value
    features["sentiment_score"] = sentiment_value
    features["news_heat"] = news_value
    features["macro_regime_score"] = macro_value
    return features


def load_market_snapshot(
    data_dir: Path = DATA_DIR,
    *,
    testnet: Optional[bool] = None,
    settings: Optional["Settings"] = None,
) -> Optional[Dict[str, object]]:
    path = _snapshot_path(data_dir, testnet=testnet, settings=settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_market_snapshot(
    snapshot: Dict[str, object],
    data_dir: Path = DATA_DIR,
    *,
    testnet: Optional[bool] = None,
    settings: Optional["Settings"] = None,
) -> None:
    path = _snapshot_path(data_dir, testnet=testnet, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_market_snapshot(api: BybitAPI, category: str = "spot") -> Dict[str, object]:
    response = api.tickers(category=category)
    rows: List[Dict[str, object]] = []
    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, dict):
            rows = result.get("list") or []  # type: ignore[assignment]
        elif isinstance(response.get("list"), list):
            rows = response.get("list")  # type: ignore[assignment]
    snapshot = {
        "ts": time.time(),
        "category": category,
        "rows": rows,
    }
    return snapshot


def scan_market_opportunities(
    api: Optional[BybitAPI],
    *,
    data_dir: Path = DATA_DIR,
    limit: int = 25,
    min_turnover: float = 1_000_000.0,
    min_change_pct: float = 0.5,
    max_spread_bps: float = 60.0,
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
    cache_ttl: float = DEFAULT_CACHE_TTL,
    settings: Optional["Settings"] = None,
    testnet: Optional[bool] = None,
    min_top_quote: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Rank spot symbols by liquidity and momentum to surface opportunities.

    Parameters
    ----------
    cache_ttl:
        Lifetime in seconds for a cached market snapshot. A value of ``0`` forces
        a refresh on every call.
    """

    if testnet is None and settings is not None:
        try:
            testnet = bool(getattr(settings, "testnet"))
        except Exception:
            testnet = None

    testnet_active = bool(testnet)

    limit_value = int(limit) if isinstance(limit, (int, float)) else 0
    if limit_value <= 0:
        limit_value = 25
    if testnet_active:
        limit_value = max(limit_value, 50)
    limit = limit_value

    min_turnover = max(0.0, float(min_turnover))
    if testnet_active and min_turnover > 50_000.0:
        min_turnover = 50_000.0

    min_ev_bps_threshold = max(
        _safe_setting_float(settings, "ai_min_ev_bps", 80.0),
        0.0,
    )

    max_daily_surge_pct = max(
        _safe_setting_float(settings, "ai_max_daily_surge_pct", _DEFAULT_DAILY_SURGE_LIMIT),
        0.0,
    )
    rsi_overbought_threshold = max(
        _safe_setting_float(
            settings,
            "ai_overbought_rsi_threshold",
            _DEFAULT_RSI_OVERBOUGHT,
        ),
        0.0,
    )
    stochastic_overbought_threshold = max(
        _safe_setting_float(
            settings,
            "ai_overbought_stochastic_threshold",
            _DEFAULT_STOCH_OVERBOUGHT,
        ),
        0.0,
    )

    effective_change = float(min_change_pct) if min_change_pct is not None else 0.5
    if effective_change < 0.05:
        effective_change = 0.05
    max_spread_bps = float(max_spread_bps)
    if testnet_active:
        if max_spread_bps <= 0:
            max_spread_bps = 120.0
        else:
            max_spread_bps = max(max_spread_bps, 120.0)

    if min_top_quote is None and settings is not None:
        try:
            min_top_quote = float(getattr(settings, "ai_min_top_quote_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_top_quote = 0.0
    elif min_top_quote is None:
        min_top_quote = 0.0
    else:
        try:
            min_top_quote = float(min_top_quote)
        except (TypeError, ValueError):
            min_top_quote = 0.0
    min_top_quote = max(min_top_quote or 0.0, 0.0)

    if cache_ttl is None:
        ttl_value = DEFAULT_CACHE_TTL
    else:
        try:
            ttl_value = float(cache_ttl)
        except (TypeError, ValueError):
            ttl_value = DEFAULT_CACHE_TTL
    cache_ttl = max(ttl_value, 0.0)

    snapshot = load_market_snapshot(data_dir, settings=settings, testnet=testnet)
    now = time.time()
    if snapshot is not None and cache_ttl >= 0:
        ts = _safe_float(snapshot.get("ts"))
        if ts is not None and now - ts > cache_ttl:
            snapshot = None

    if snapshot is None and api is not None:
        try:
            snapshot = fetch_market_snapshot(api)
        except Exception:
            snapshot = None
        else:
            save_market_snapshot(
                snapshot,
                data_dir=data_dir,
                settings=settings,
                testnet=testnet,
            )

    if snapshot is None:
        raise MarketScannerError(
            "Рыночный снапшот недоступен — проверьте подключение к API."
        )

    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        result = snapshot.get("result")
        if isinstance(result, dict):
            rows = result.get("list")  # type: ignore[assignment]
        else:
            rows = []
    if not isinstance(rows, list):
        rows = []

    turnover_ratio = _safe_setting_float(settings, "ai_min_turnover_ratio", 0.3)
    min_turnover = _resolve_min_turnover_threshold(min_turnover, rows, turnover_ratio)

    top_quote_ratio = _safe_setting_float(settings, "ai_min_top_quote_ratio", 0.2)
    min_top_quote = _estimate_top_depth_threshold(rows, top_quote_ratio, min_top_quote)

    volatility_ratio = _safe_setting_float(
        settings, "ai_min_change_volatility_ratio", 0.6
    )
    effective_change = _compute_dynamic_min_change(
        data_dir,
        volatility_ratio,
        effective_change,
    )

    total_turnover_usd = 0.0
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            sym, _ = ensure_usdt_symbol(item.get("symbol"))
            if not sym:
                continue
            turnover_value = _safe_float(item.get("turnover24h"))
            if turnover_value is not None and turnover_value > 0:
                total_turnover_usd += turnover_value

    def _normalise_symbol_set(symbols: Iterable[object]) -> Set[str]:
        normalised: Set[str] = set()
        for item in symbols:
            candidate, _ = ensure_usdt_symbol(item)
            if candidate:
                normalised.add(candidate)
        return normalised

    entries: List[Dict[str, object]] = []
    fee_guard_cache: Dict[str, float] = {}
    wset = _normalise_symbol_set(whitelist or ())
    bset = _normalise_symbol_set(blacklist or ())

    if settings is not None:
        forced_candidates = _collect_symbol_candidates(
            getattr(settings, "ai_force_include", None)
        )
        if forced_candidates:
            wset.update(forced_candidates)

    maintenance_symbols: Set[str] = set()
    delist_symbols: Set[str] = set()
    if isinstance(snapshot, Mapping):
        maintenance_symbols = _extract_snapshot_stoplist(snapshot, _MAINTENANCE_KEYS)
        delist_symbols = _extract_snapshot_stoplist(snapshot, _DELIST_KEYS)
    _apply_stoplists_from_settings(settings, maintenance=maintenance_symbols, delisted=delist_symbols)

    outlier_symbols = _detect_outlier_symbols(rows if isinstance(rows, list) else [])

    data_warning_cache: Set[Tuple[str, str]] = set()

    def _log_data_warning(symbol: str, issue: str, **details: object) -> None:
        key = (symbol, issue)
        if not symbol or key in data_warning_cache:
            return
        data_warning_cache.add(key)
        log(
            "market_scanner.data_warning",
            symbol=symbol,
            issue=issue,
            severity="warning",
            **details,
        )

    retrain_minutes = _safe_setting_float(settings, "ai_retrain_minutes", 0.0)
    if retrain_minutes is None or retrain_minutes <= 0:
        retrain_interval = 7 * 24 * 3600.0
    else:
        retrain_interval = max(float(retrain_minutes) * 60.0, 3600.0)

    training_limit_setting = _safe_setting_float(
        settings, "ai_training_trade_limit", 0.0
    )
    if training_limit_setting is None or training_limit_setting <= 0:
        training_limit = 400
    else:
        training_limit = max(int(training_limit_setting), 50)

    try:
        model: Optional[MarketModel] = ensure_market_model(
            data_dir=data_dir,
            max_age=retrain_interval,
            limit=training_limit,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log("market_scanner.model.error", err=str(exc))
        model = None

    external_provider = ExternalFeatureProvider(data_dir=data_dir)
    try:  # pragma: no cover - best effort logging
        external_provider.log_health()
    except Exception:
        pass
    macro_regime_hint = external_provider.macro_regime_score()

    prediction_path_override: Optional[str] = None
    if settings is not None:
        raw_override = getattr(settings, "freqai_prediction_path", None)
        if isinstance(raw_override, str):
            cleaned = raw_override.strip()
            if cleaned:
                prediction_path_override = cleaned
        elif raw_override:
            prediction_path_override = str(raw_override)

    freqai_store = get_prediction_store(data_dir, prediction_path=prediction_path_override)
    freqai_snapshot = freqai_store.snapshot()
    raw_freqai_pairs = freqai_snapshot.get("pairs")
    if isinstance(raw_freqai_pairs, Mapping):
        freqai_pairs: Mapping[str, Mapping[str, object]] = raw_freqai_pairs
    else:
        freqai_pairs = {}
    freqai_source = str(freqai_snapshot.get("source") or "freqai")
    freqai_generated_at = _safe_float(freqai_snapshot.get("generated_at"))
    freqai_updated_at = _safe_float(freqai_snapshot.get("updated_at"))

    def _freqai_lookup(symbol: str, *, raw_symbol: object | None = None) -> Optional[Mapping[str, object]]:
        canonical = freqai_store.canonical_symbol(symbol)
        prediction = freqai_pairs.get(canonical)
        if prediction is None and raw_symbol:
            prediction = freqai_pairs.get(freqai_store.canonical_symbol(str(raw_symbol)))
        return prediction

    snapshot_ts = None
    if isinstance(snapshot, Mapping):
        snapshot_ts = _safe_float(snapshot.get("ts"))

    cost_profile = _resolve_trade_cost_profile(data_dir, settings, testnet)
    base_maker_fee_bps = float(cost_profile.get("maker_fee_bps", 0.0) or 0.0)
    base_taker_fee_bps = float(
        cost_profile.get("taker_fee_bps", _DEFAULT_TAKER_FEE_BPS) or _DEFAULT_TAKER_FEE_BPS
    )
    maker_ratio_hint = float(cost_profile.get("maker_ratio", _DEFAULT_MAKER_RATIO) or _DEFAULT_MAKER_RATIO)
    slippage_cost_bps = float(cost_profile.get("slippage_bps", 0.0) or 0.0)
    historical_fee_source = str(cost_profile.get("fee_source") or "history")

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        raw_symbol = raw.get("symbol")
        symbol, quote_source = ensure_usdt_symbol(raw_symbol)
        if not symbol:
            continue
        upper_source = str(raw_symbol).strip().upper() if isinstance(raw_symbol, str) else None
        if bset and symbol in bset:
            continue

        force_include = symbol in wset

        if not force_include and symbol in maintenance_symbols:
            log("market_scanner.filter.maintenance", symbol=symbol, source="snapshot")
            continue
        if not force_include and _row_in_maintenance(raw):
            log("market_scanner.filter.maintenance", symbol=symbol, source="row")
            continue
        if not force_include and symbol in delist_symbols:
            log("market_scanner.filter.delist", symbol=symbol, source="snapshot")
            continue
        if not force_include and _row_delisted(raw, now=now):
            log("market_scanner.filter.delist", symbol=symbol, source="row")
            continue
        if not force_include and symbol in outlier_symbols:
            log("market_scanner.filter.outlier", symbol=symbol)
            continue

        turnover = _safe_float(raw.get("turnover24h"))
        change_pct = _normalise_percent(raw.get("price24hPcnt"))
        volume = _safe_float(raw.get("volume24h"))
        bid = _safe_float(raw.get("bestBidPrice"))
        ask = _safe_float(raw.get("bestAskPrice"))
        spread_bps = _spread_bps(bid, ask)
        if turnover is None or turnover <= 0:
            _log_data_warning(
                symbol,
                "missing_turnover",
                raw_turnover=raw.get("turnover24h"),
            )
        if volume is None or volume <= 0:
            _log_data_warning(
                symbol,
                "missing_volume",
                raw_volume=raw.get("volume24h"),
            )
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            _log_data_warning(symbol, "missing_orderbook", bid=bid, ask=ask)

        if total_turnover_usd > 0 and turnover is not None and turnover > 0:
            market_dominance_pct: Optional[float] = (turnover / total_turnover_usd) * 100.0
        else:
            market_dominance_pct = None

        feature_bundle = build_feature_bundle(raw)
        blended_change = feature_bundle.get("blended_change_pct")
        volatility_pct = feature_bundle.get("volatility_pct")
        volatility_windows = feature_bundle.get("volatility_windows")
        volume_spike_score = feature_bundle.get("volume_spike_score")
        volume_impulse = feature_bundle.get("volume_impulse")
        depth_imbalance = feature_bundle.get("depth_imbalance")
        order_flow_ratio = feature_bundle.get("order_flow_ratio")
        cvd_score = feature_bundle.get("cvd_score")
        cvd_windows = feature_bundle.get("cvd_windows")
        top_depth_quote = feature_bundle.get("top_depth_quote")
        top_depth_imbalance = feature_bundle.get("top_depth_imbalance")
        correlations = feature_bundle.get("correlations")
        correlation_strength = feature_bundle.get("correlation_strength")
        social_trend_score = feature_bundle.get("social_trend_score")
        overbought_indicators = feature_bundle.get("overbought_indicators")
        volatility_ratio_value = feature_bundle.get("volatility_ratio")
        volume_trend_value = feature_bundle.get("volume_trend")

        rsi_value: Optional[float] = None
        stochastic_value: Optional[float] = None
        distance_from_high: Optional[float] = None
        if isinstance(overbought_indicators, Mapping):
            rsi_value = _safe_float(overbought_indicators.get("rsi"))
            stochastic_value = _safe_float(
                overbought_indicators.get("stochastic_pct")
            )
            distance_from_high = _safe_float(
                overbought_indicators.get("distance_from_high_pct")
            )

        if not force_include and turnover is not None and turnover < min_turnover:
            turnover_ok = False
        else:
            turnover_ok = True

        top_bid_quote = None
        top_ask_quote = None
        top_total_quote = None
        if isinstance(top_depth_quote, Mapping):
            top_bid_quote = _safe_float(top_depth_quote.get("bid"))
            top_ask_quote = _safe_float(top_depth_quote.get("ask"))
            total_hint = _safe_float(top_depth_quote.get("total"))
            if total_hint is None and (
                top_bid_quote is not None or top_ask_quote is not None
            ):
                total_hint = (top_bid_quote or 0.0) + (top_ask_quote or 0.0)
            top_total_quote = total_hint

        liquidity_ok = True
        if not force_include and min_top_quote > 0:
            available_quote = top_total_quote if top_total_quote is not None else 0.0
            if available_quote < min_top_quote:
                liquidity_ok = False

        indicator_bundle: Optional[Dict[str, object]] = None
        if change_pct is None:
            trend = "wait"
        else:
            base_trend = "wait"
            if change_pct >= effective_change:
                base_trend = "buy"
            elif change_pct <= -effective_change:
                base_trend = "sell"
            if base_trend in {"buy", "sell"}:
                indicator_bundle = _load_hourly_indicator_bundle(
                    symbol,
                    data_dir=data_dir,
                )
                confirmed_trend = _resolve_trend_with_confirmation(
                    change_pct,
                    effective_change,
                    indicator_bundle,
                    force_include=force_include,
                )
                if confirmed_trend == "wait":
                    if indicator_bundle is None or force_include:
                        trend = base_trend
                    else:
                        trend = "wait"
                else:
                    trend = confirmed_trend
            else:
                trend = base_trend

        raw_change_value = _safe_float(raw.get("price24hPcnt"))
        if raw_change_value is not None and abs(raw_change_value) <= 1.0:
            daily_change_for_guard = abs(raw_change_value)
        else:
            daily_change_for_guard = change_pct

        if (
            not force_include
            and trend == "buy"
            and max_daily_surge_pct > 0
            and daily_change_for_guard is not None
            and daily_change_for_guard >= max_daily_surge_pct
        ):
            log(
                "market_scanner.filter.daily_surge",
                symbol=symbol,
                change_pct=daily_change_for_guard,
                limit=max_daily_surge_pct,
            )
            continue

        actionable = False
        spread_ok = True
        if spread_bps is not None and max_spread_bps > 0:
            spread_ok = spread_bps <= max_spread_bps

        if trend in {"buy", "sell"}:
            change_ok = change_pct is not None and abs(change_pct) >= effective_change
            actionable = turnover_ok and spread_ok and change_ok and liquidity_ok

        if not actionable and not force_include:
            strength = _strength_from_change(change_pct)
            if strength < 0.1:
                continue

        event_ts = _safe_float(
            raw.get("updateTime")
            or raw.get("updatedTime")
            or raw.get("timestamp")
            or raw.get("time")
        )
        if event_ts is None:
            event_ts = snapshot_ts

        sentiment_value = external_provider.sentiment_for(symbol)
        social_value = external_provider.social_score_for(symbol)
        if math.isfinite(social_value):
            sentiment_value = sentiment_value * 0.7 + social_value * 0.3
        news_heat_value = external_provider.news_heat_for(symbol)

        feature_vector = _model_feature_vector(
            trend=trend,
            change_pct=change_pct,
            blended_change=blended_change,
            turnover=turnover,
            volatility_pct=volatility_pct,
            volume_impulse=volume_impulse if isinstance(volume_impulse, Mapping) else None,
            depth_imbalance=depth_imbalance,
            spread_bps=spread_bps,
            correlation_strength=correlation_strength,
            order_flow_ratio=order_flow_ratio if isinstance(order_flow_ratio, (int, float)) else None,
            top_depth_imbalance=top_depth_imbalance,
            volatility_ratio=volatility_ratio_value if isinstance(volatility_ratio_value, (int, float)) else None,
            volume_trend=volume_trend_value,
            sentiment_score=sentiment_value,
            news_heat=news_heat_value,
            macro_regime_score=macro_regime_hint,
            event_timestamp=event_ts,
        )

        overbought_flags: List[str] = []
        overbought_details: Dict[str, float] = {}
        if (
            rsi_overbought_threshold > 0
            and rsi_value is not None
            and rsi_value >= rsi_overbought_threshold
        ):
            overbought_flags.append("rsi")
            overbought_details["rsi_excess"] = float(
                rsi_value - rsi_overbought_threshold
            )
        if (
            stochastic_overbought_threshold > 0
            and stochastic_value is not None
            and stochastic_value >= stochastic_overbought_threshold
        ):
            overbought_flags.append("stochastic")
            overbought_details["stochastic_excess"] = float(
                stochastic_value - stochastic_overbought_threshold
            )

        decision_threshold = _decision_threshold(model, default=0.5)
        raw_logistic_probability: Optional[float] = None

        if model is not None:
            probability = model.predict_proba(feature_vector)
            raw_logistic_probability = float(probability)
            score = probability * 100.0
            prob_clamped = min(max(probability, 1e-6), 1.0 - 1e-6)
            logit = math.log(prob_clamped / (1.0 - prob_clamped))
            model_metrics: Dict[str, object] = {
                "model": "logistic_regression",
                "features": feature_vector,
                "logit": logit,
                "trained_at": model.trained_at,
                "samples": model.samples,
            }
            model_metrics["raw_probability"] = raw_logistic_probability
        else:
            probability, score, fallback_metrics = _weighted_opportunity_model(
                change_pct=change_pct,
                trend=trend,
                turnover=turnover,
                spread_bps=spread_bps,
                features=feature_bundle,
                force_include=force_include,
            )
            fallback_details = dict(fallback_metrics)
            model_metrics = dict(fallback_details)
            model_metrics.update(
                {
                    "model": "logistic_regression",
                    "features": feature_vector,
                    "warning": "model_unavailable",
                    "fallback": fallback_details,
                }
            )

        logistic_probability: Optional[float]
        logistic_score: Optional[float]
        if model is not None:
            calibrated = _calibrate_probability(raw_logistic_probability, model)
            if calibrated is not None:
                probability = calibrated
                score = calibrated * 100.0
            logistic_probability = probability if probability is not None else calibrated
            logistic_score = score
            if calibrated is not None:
                model_metrics["calibrated_probability"] = calibrated
            model_metrics["decision_threshold"] = decision_threshold
        else:
            logistic_probability = probability
            logistic_score = score

        freqai_prediction = _freqai_lookup(symbol, raw_symbol=raw_symbol)
        freqai_payload: Optional[Dict[str, object]] = None

        direction = 0
        if trend == "buy":
            direction = 1
        elif trend == "sell":
            direction = -1

        gross_bps: Optional[float] = None
        spread_component = max(float(spread_bps or 0.0), 0.0)

        maker_fee_bps = base_maker_fee_bps
        taker_fee_bps = base_taker_fee_bps
        maker_ratio = maker_ratio_hint
        fee_source = historical_fee_source

        snapshot = fee_rate_for_symbol(category="spot", symbol=symbol)
        if snapshot is not None:
            snapshot_taker = snapshot.taker_fee_bps
            snapshot_maker = snapshot.maker_fee_bps
            if snapshot_taker is not None:
                taker_fee_bps = _clamp(float(snapshot_taker), -100.0, 100.0)
                fee_source = "api"
            if snapshot_maker is not None:
                maker_fee_bps = _clamp(float(snapshot_maker), -100.0, 100.0)
                fee_source = "api"

        maker_fee_bps = _clamp(float(maker_fee_bps), -100.0, 100.0)
        taker_fee_bps = _clamp(float(taker_fee_bps), -100.0, 100.0)
        maker_ratio = _clamp(float(maker_ratio), 0.0, 1.0)

        effective_fee_bps = maker_fee_bps * maker_ratio + taker_fee_bps * (1.0 - maker_ratio)
        fee_round_trip_bps = effective_fee_bps * 2.0

        base_cost_bps = fee_round_trip_bps + slippage_cost_bps + spread_component
        guard_key = symbol if symbol else ""
        fee_guard_bps_value = (
            _resolve_symbol_fee_guard_bps(
                guard_key,
                settings=settings,
                api=api,
                cache=fee_guard_cache,
            )
            if guard_key
            else 0.0
        )
        fee_guard_bps_value = max(float(fee_guard_bps_value), 0.0)
        total_cost_bps = max(base_cost_bps + fee_guard_bps_value, 0.0)

        if change_pct is not None:
            directional_change = float(change_pct)
            if direction != 0:
                directional_change *= direction
            ev_pct_raw = directional_change
            gross_bps = ev_pct_raw * 100.0
            total_cost_pct = total_cost_bps / 100.0
            ev_pct = ev_pct_raw - total_cost_pct
            ev_bps = ev_pct * 100.0
        else:
            ev_bps = None
            total_cost_bps = total_cost_bps if total_cost_bps > 0 else 0.0

        logistic_ev_bps = ev_bps
        if isinstance(freqai_prediction, Mapping):
            freqai_payload = {
                "symbol": freqai_prediction.get("symbol") or symbol,
                "canonical": freqai_prediction.get("canonical")
                or freqai_store.canonical_symbol(symbol),
                "source": str(freqai_prediction.get("source") or freqai_source),
                "generated_at": freqai_prediction.get("generated_at") or freqai_generated_at,
                "updated_at": freqai_updated_at,
            }

            override_prob = _safe_float(freqai_prediction.get("probability"))
            if override_prob is not None:
                probability = max(0.0, min(override_prob, 1.0))
                score_candidate = _safe_float(freqai_prediction.get("score"))
                if score_candidate is not None:
                    score = float(score_candidate)
                else:
                    score = probability * 100.0
                freqai_payload["probability"] = probability
                freqai_payload["probability_pct"] = probability * 100.0
            elif logistic_probability is not None:
                freqai_payload["probability"] = float(logistic_probability)
                freqai_payload["probability_pct"] = float(logistic_probability) * 100.0

            override_ev = _safe_float(freqai_prediction.get("ev_bps"))
            if override_ev is not None:
                ev_bps = float(override_ev)
            if ev_bps is not None:
                freqai_payload["ev_bps"] = float(ev_bps)
                freqai_payload["ev_pct"] = float(ev_bps) / 100.0
            elif logistic_ev_bps is not None:
                freqai_payload["ev_bps"] = float(logistic_ev_bps)
                freqai_payload["ev_pct"] = float(logistic_ev_bps) / 100.0

            confidence = _safe_float(freqai_prediction.get("confidence"))
            if confidence is not None:
                freqai_payload["confidence"] = confidence
                freqai_payload["confidence_pct"] = confidence * 100.0

            for key in ("horizon_minutes", "window_minutes"):
                value = _safe_float(freqai_prediction.get(key))
                if value is not None:
                    freqai_payload[key] = value

            meta = freqai_prediction.get("meta")
            if isinstance(meta, Mapping):
                freqai_payload["meta"] = dict(meta)

            freqai_payload["logistic_probability"] = (
                float(logistic_probability) if logistic_probability is not None else None
            )
            freqai_payload["logistic_score"] = float(logistic_score) if logistic_score is not None else None
            freqai_payload["logistic_ev_bps"] = (
                float(logistic_ev_bps) if logistic_ev_bps is not None else None
            )

        probability_context: Optional[float] = None
        if freqai_payload and "probability" in freqai_payload:
            probability_context = _safe_float(freqai_payload.get("probability"))
        if probability_context is None and logistic_probability is not None:
            probability_context = float(logistic_probability)

        overbought_blocked = (
            len(overbought_flags) >= 2 and not force_include and trend == "buy"
        )
        if overbought_blocked:
            rsi_margin = float(overbought_details.get("rsi_excess", 0.0))
            stochastic_margin = float(
                overbought_details.get("stochastic_excess", 0.0)
            )
            if max(rsi_margin, stochastic_margin) < 3.0:
                overbought_blocked = False
            elif (
                probability_context is not None
                and (
                    model is not None
                    or (freqai_payload is not None and "probability" in freqai_payload)
                )
                and probability_context >= decision_threshold
            ):
                overbought_blocked = False
        if overbought_blocked:
            log(
                "market_scanner.filter.overbought",
                symbol=symbol,
                rsi=rsi_value,
                stochastic=stochastic_value,
                reasons=overbought_flags,
                probability=probability_context,
                decision_threshold=decision_threshold,
            )
            continue

        if logistic_probability is not None:
            model_metrics["logistic_probability"] = float(logistic_probability)
        if logistic_score is not None:
            model_metrics["logistic_score"] = float(logistic_score)
        if logistic_ev_bps is not None:
            model_metrics["logistic_ev_bps"] = float(logistic_ev_bps)
        if freqai_payload:
            model_metrics["freqai_override"] = dict(freqai_payload)
            model_metrics["probability_source"] = freqai_payload.get("source")
            model_metrics["ev_source"] = freqai_payload.get("source")

        if not force_include and min_ev_bps_threshold > 0.0:
            if ev_bps is None:
                log(
                    "market_scanner.filter.ev_threshold",
                    symbol=symbol,
                    reason="missing_ev",
                    min_ev_bps=min_ev_bps_threshold,
                )
                continue
            if ev_bps < min_ev_bps_threshold:
                log(
                    "market_scanner.filter.ev_threshold",
                    symbol=symbol,
                    ev_bps=ev_bps,
                    min_ev_bps=min_ev_bps_threshold,
                )
                continue

        note_parts: List[str] = []
        if change_pct is not None:
            note_parts.append(f"24ч {change_pct:+.2f}%")
        if turnover is not None and turnover > 0:
            note_parts.append(f"оборот ${turnover / 1_000_000:.2f}M")
        if spread_bps is not None:
            note_parts.append(f"спред {spread_bps:.1f} б.п.")
        if total_cost_bps is not None:
            note_parts.append(f"издержки ≈ {total_cost_bps:.1f} б.п.")
        if quote_source == "USDC":
            note_parts.append("конвертировано из USDC")
        if volatility_pct is not None:
            note_parts.append(f"волатильность {volatility_pct:.1f}%")
        if rsi_value is not None:
            note_parts.append(f"RSI≈{rsi_value:.1f}")
        if stochastic_value is not None:
            note_parts.append(f"Стох≈{stochastic_value:.0f}%")
        if (
            distance_from_high is not None
            and distance_from_high > 0
            and distance_from_high < 20.0
        ):
            note_parts.append(f"от хая {distance_from_high:.1f}%")
        if volume_spike_score is not None and volume_spike_score > 0:
            note_parts.append(f"всплеск объёма ×{math.exp(volume_spike_score):.2f}")
        if depth_imbalance is not None and direction != 0:
            imbalance_pct = depth_imbalance * 100.0
            side = "покупателей" if imbalance_pct > 0 else "продавцов"
            note_parts.append(f"преимущество {side} {abs(imbalance_pct):.1f}%")
        if isinstance(order_flow_ratio, (int, float)) and direction != 0:
            flow_pct = float(order_flow_ratio) * 100.0 * direction
            if abs(flow_pct) >= 5.0:
                side = "покупателей" if flow_pct > 0 else "продавцов"
                note_parts.append(f"поток ордеров за {side} {abs(flow_pct):.1f}%")
        if isinstance(cvd_score, (int, float)):
            cvd_pct = float(cvd_score) * 100.0
            if abs(cvd_pct) >= 5.0:
                side = "покупателей" if cvd_pct > 0 else "продавцов"
                note_parts.append(f"CVD на стороне {side} {abs(cvd_pct):.1f}%")
        if correlation_strength is not None and correlation_strength > 0:
            note_parts.append(f"корреляция {correlation_strength * 100:.0f}%")
        if indicator_bundle:
            cross_label = indicator_bundle.get("ema_cross")
            if cross_label == "bullish":
                note_parts.append("EMA21↑EMA50")
            elif cross_label == "bearish":
                note_parts.append("EMA21↓EMA50")
            rsi_hint = indicator_bundle.get("rsi")
            if isinstance(rsi_hint, (int, float)):
                note_parts.append(f"RSI₁ₕ≈{float(rsi_hint):.1f}")
            momentum_hint = indicator_bundle.get("momentum_pct")
            if isinstance(momentum_hint, (int, float)):
                note_parts.append(f"Δ1ч≈{float(momentum_hint):+.2f}%")
        best_impulse_window = None
        best_impulse_value: Optional[float] = None
        if isinstance(volume_impulse, Mapping):
            for window, value in volume_impulse.items():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                if best_impulse_value is None or abs(numeric) > abs(best_impulse_value):
                    best_impulse_window = window
                    best_impulse_value = float(numeric)
        impulse_signal = bool(
            best_impulse_value is not None
            and best_impulse_value >= _IMPULSE_SIGNAL_THRESHOLD
        )
        if (
            best_impulse_window is not None
            and best_impulse_value is not None
            and best_impulse_value > 0
        ):
            note_parts.append(
                f"импульс объёма {best_impulse_window} ×{math.exp(best_impulse_value):.2f}"
            )
        if not liquidity_ok and min_top_quote > 0:
            available_quote = top_total_quote if top_total_quote is not None else 0.0
            note_parts.append(
                "тонкий стакан {available:.1f} USDT (< {required:.1f})".format(
                    available=available_quote,
                    required=min_top_quote,
                )
            )
        if overbought_flags:
            note_parts.append("⚠️ перекупленность")

        entry = {
            "symbol": symbol,
            "trend": trend,
            "probability": probability,
            "ev_bps": ev_bps,
            "gross_ev_bps": gross_bps,
            "costs_bps": {
                "total": total_cost_bps,
                "fees": fee_round_trip_bps,
                "slippage": slippage_cost_bps,
                "spread": spread_component,
                "fee_guard_bps": fee_guard_bps_value,
                "maker_fee_bps": maker_fee_bps,
                "taker_fee_bps": taker_fee_bps,
                "maker_ratio": maker_ratio,
                "fee_source": fee_source,
            }
            if total_cost_bps is not None
            else None,
            "score": score,
            "note": ", ".join(note_parts) or None,
            "turnover_usd": turnover,
            "change_pct": change_pct,
            "spread_bps": spread_bps,
            "volume": volume,
            "volatility_pct": volatility_pct,
            "volatility_windows": volatility_windows,
            "volume_spike_score": volume_spike_score,
            "volume_impulse": volume_impulse,
            "depth_imbalance": depth_imbalance,
            "order_flow_ratio": order_flow_ratio,
            "cvd_score": cvd_score,
            "cvd_windows": cvd_windows,
            "top_depth_quote": {
                "bid": top_bid_quote,
                "ask": top_ask_quote,
                "total": top_total_quote,
            },
            "top_depth_imbalance": top_depth_imbalance,
            "blended_change_pct": blended_change,
            "correlations": correlations,
            "correlation_strength": correlation_strength,
            "overbought_indicators": overbought_indicators
            if isinstance(overbought_indicators, Mapping)
            else None,
            "overbought_flags": tuple(overbought_flags) if overbought_flags else None,
            "distance_from_high_pct": distance_from_high,
            "impulse_signal": impulse_signal,
            "impulse_strength": best_impulse_value,
            "model_metrics": model_metrics,
            "source": "market_scanner",
            "actionable": actionable,
            "liquidity_ok": liquidity_ok,
            "min_top_quote_usd": min_top_quote if min_top_quote > 0 else None,
        }
        hourly_signal = _sanitise_hourly_signal(indicator_bundle)
        if hourly_signal:
            entry["hourly_signal"] = hourly_signal
        if quote_source == "USDC" and upper_source:
            entry["quote_conversion"] = {"from": "USDC", "to": "USDT", "original": upper_source}

        if freqai_payload:
            entry["freqai"] = freqai_payload
            entry["probability_source"] = freqai_payload.get("source") or "freqai"
            entry["ev_source"] = freqai_payload.get("source") or "freqai"
        elif logistic_probability is not None:
            entry["probability_source"] = "logistic_regression"
            entry["ev_source"] = "market_scanner"

        entries.append(entry)

    entries.sort(
        key=lambda item: (
            0 if item.get("actionable") else 1,
            -(item.get("score") or 0.0),
            -abs(item.get("change_pct") or 0.0),
            item.get("symbol", ""),
        )
    )

    if limit and limit > 0:
        entries = entries[: int(limit)]

    return entries


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_kline_rows(payload: object) -> Sequence[object]:
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, Mapping):
            rows = result.get("list")
            if isinstance(rows, Sequence):
                return rows  # type: ignore[return-value]
        rows = payload.get("list")
        if isinstance(rows, Sequence):
            return rows  # type: ignore[return-value]
    elif isinstance(payload, Sequence):
        return payload
    return []


def _interpret_confirm_flag(value: object) -> Optional[bool]:
    """Best effort conversion of Bybit candle ``confirm`` fields to ``bool``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        return bool(numeric)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return None
        if text in {"1", "true", "closed", "confirm", "yes"}:
            return True
        if text in {"0", "false", "open", "no"}:
            return False
    return None


def _normalise_candles(payload: object) -> List[Dict[str, object]]:
    candles: List[Dict[str, object]] = []
    for entry in _extract_kline_rows(payload):
        start: Optional[int]
        open_: Optional[float]
        high: Optional[float]
        low: Optional[float]
        close: Optional[float]
        volume: Optional[float]
        turnover: Optional[float] = None
        confirm: Optional[bool] = None

        if isinstance(entry, Mapping):
            start = _safe_int(entry.get("start") or entry.get("openTime") or entry.get("timestamp"))
            open_ = _safe_float(entry.get("open"))
            high = _safe_float(entry.get("high"))
            low = _safe_float(entry.get("low"))
            close = _safe_float(entry.get("close"))
            volume = _safe_float(entry.get("volume"))
            turnover = _safe_float(entry.get("turnover"))
            confirm = _interpret_confirm_flag(
                entry.get("confirm")
                or entry.get("isClosed")
                or entry.get("is_close")
                or entry.get("closed")
            )
        elif isinstance(entry, Sequence):
            sequence = list(entry)
            if not sequence:
                continue
            start = _safe_int(sequence[0])
            open_ = _safe_float(sequence[1]) if len(sequence) > 1 else None
            high = _safe_float(sequence[2]) if len(sequence) > 2 else None
            low = _safe_float(sequence[3]) if len(sequence) > 3 else None
            close = _safe_float(sequence[4]) if len(sequence) > 4 else None
            volume = _safe_float(sequence[5]) if len(sequence) > 5 else None
            turnover = _safe_float(sequence[6]) if len(sequence) > 6 else None
            confirm = _interpret_confirm_flag(sequence[7]) if len(sequence) > 7 else None
        else:
            continue

        if start is None:
            continue

        if confirm is False:
            # Explicitly flagged by the API as "not yet closed" – skip.
            continue

        record: Dict[str, object] = {"start": start}
        if open_ is not None:
            record["open"] = open_
        if high is not None:
            record["high"] = high
        if low is not None:
            record["low"] = low
        if close is not None:
            record["close"] = close
        if volume is not None:
            record["volume"] = volume
        if turnover is not None:
            record["turnover"] = turnover
        candles.append(record)

    candles.sort(key=lambda row: row.get("start") or 0)
    return candles


def _select_closed_candles(
    candles: Sequence[Mapping[str, object]], *, interval_minutes: int, now: float
) -> List[Dict[str, object]]:
    """Filter out candles that are still forming at *now*."""

    if not candles:
        return []

    try:
        interval_ms = int(max(interval_minutes, 0)) * 60_000
    except (TypeError, ValueError):  # pragma: no cover - defensive
        interval_ms = 0

    now_ms = int(float(now) * 1000.0)
    closed: List[Dict[str, object]] = []

    for candle in candles:
        start_raw = candle.get("start") if isinstance(candle, Mapping) else None
        start = _safe_int(start_raw) if start_raw is not None else None
        if start is None:
            continue
        if interval_ms > 0 and start + interval_ms > now_ms:
            continue
        closed.append(dict(candle))

    return closed


class _CandleCache:
    """Small helper that caches recent kline snapshots per symbol."""

    def __init__(
        self,
        api: Optional[BybitAPI],
        *,
        category: str = "spot",
        intervals: Sequence[int] = (1, 5),
        ttl: float = 45.0,
        limit: int = 100,
    ) -> None:
        self.api = api
        self.category = category
        self.intervals = tuple(int(interval) for interval in intervals)
        self.ttl = max(float(ttl), 1.0)
        self.limit = max(int(limit), 1)
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, object]]]] = {}

    def fetch(self, symbol: str, now: float) -> Dict[str, List[Dict[str, object]]]:
        bundle: Dict[str, List[Dict[str, object]]] = {}
        if self.api is None:
            return bundle

        for interval in self.intervals:
            key = (symbol, interval)
            cached = self._cache.get(key)
            if cached is not None:
                ts, candles = cached
                if now - ts <= self.ttl and candles:
                    bundle[f"{interval}m"] = candles
                    continue

            try:
                payload = self.api.kline(
                    category=self.category,
                    symbol=symbol,
                    interval=interval,
                    limit=self.limit,
                )
            except Exception as exc:  # pragma: no cover - network/runtime guard
                log("market_scanner.candles.error", symbol=symbol, interval=interval, err=str(exc))
                continue

            candles = _normalise_candles(payload)
            closed = _select_closed_candles(candles, interval_minutes=interval, now=now)
            self._cache[key] = (now, closed)
            if closed:
                bundle[f"{interval}m"] = closed

        return bundle


class MarketScanner:
    """Stateful helper that keeps the multi-asset opportunity universe fresh."""

    def __init__(
        self,
        api: Optional[BybitAPI],
        symbol_resolver: Optional["SymbolResolver"] = None,
        *,
        data_dir: Path = DATA_DIR,
        scanner: Callable[..., List[Dict[str, object]]] = scan_market_opportunities,
        scanner_kwargs: Optional[Mapping[str, object]] = None,
        refresh_interval: Tuple[float, float] = (60.0, 120.0),
        candle_ttl: float = 45.0,
        candle_limit: int = 120,
        mode: str = "breakout",
        portfolio_manager: Optional["PortfolioManager"] = None,
        telegram_sender: Callable[[str], object] = enqueue_telegram_message,
        category: str = "spot",
    ) -> None:
        self.api = api
        self.symbol_resolver = symbol_resolver
        self.data_dir = Path(data_dir)
        self._scanner = scanner
        self._scanner_kwargs = dict(scanner_kwargs or {})
        self._refresh_interval = (
            max(float(refresh_interval[0]), 5.0),
            max(float(refresh_interval[1]), float(refresh_interval[0])),
        )
        self._next_refresh: float = 0.0
        self._last_update: float = 0.0
        self._top_candidates: List[Dict[str, object]] = []
        self._last_leader_digest: Optional[Tuple[str, ...]] = None
        self.mode = mode
        self.portfolio_manager = portfolio_manager
        self._telegram_sender = telegram_sender
        self._category = category
        self._candle_cache = _CandleCache(
            api,
            category=category,
            intervals=(1, 5),
            ttl=candle_ttl,
            limit=candle_limit,
        )

        # Always ensure the scanner uses the configured data directory
        if "data_dir" not in self._scanner_kwargs:
            self._scanner_kwargs["data_dir"] = self.data_dir

    # ------------------------------------------------------------------
    # public API
    def refresh(self, *, force: bool = False, now: Optional[float] = None) -> List[Dict[str, object]]:
        """Refresh the ranking when the refresh window elapsed."""

        current_ts = now if now is not None else time.time()
        if not force and self._next_refresh and current_ts < self._next_refresh and self._top_candidates:
            return copy.deepcopy(self._top_candidates)

        scan_kwargs = dict(self._scanner_kwargs)
        scan_kwargs.setdefault("data_dir", self.data_dir)
        opportunities = self._scanner(self.api, **scan_kwargs)

        enriched: List[Dict[str, object]] = []
        for entry in opportunities:
            enriched.append(self._enrich_entry(entry, current_ts))

        self._top_candidates = enriched
        self._last_update = current_ts
        self._schedule_next_refresh(current_ts)
        self._notify_leader_change()
        return copy.deepcopy(self._top_candidates)

    def top_candidates(self, limit: int = 5) -> List[Dict[str, object]]:
        entries = self._top_candidates[: max(int(limit), 0)]
        return copy.deepcopy(entries)

    @property
    def last_update(self) -> float:
        return self._last_update

    # ------------------------------------------------------------------
    # internal helpers
    def _schedule_next_refresh(self, now: float) -> None:
        low, high = self._refresh_interval
        if high <= low:
            delay = low
        else:
            delay = random.uniform(low, high)
        self._next_refresh = now + delay

    def _resolve_metadata(self, symbol: str) -> Optional["InstrumentMetadata"]:
        if not self.symbol_resolver:
            return None
        metadata = self.symbol_resolver.metadata(symbol)
        if metadata is None:
            metadata = self.symbol_resolver.resolve_symbol(symbol)
        return metadata

    def _enrich_entry(self, entry: Mapping[str, object], now: float) -> Dict[str, object]:
        enriched = copy.deepcopy(entry)
        symbol = str(entry.get("symbol") or "").strip().upper()
        metadata = self._resolve_metadata(symbol)
        if metadata is not None:
            enriched["instrument"] = metadata.as_dict()
            canonical = metadata.symbol
        else:
            enriched["instrument"] = None
            canonical = symbol

        if canonical:
            enriched["candles"] = self._candle_cache.fetch(canonical, now)
        else:
            enriched["candles"] = {}
        enriched.setdefault("source", "market_scanner")
        return enriched

    def _notify_leader_change(self) -> None:
        if not self._top_candidates:
            return

        symbols = []
        for entry in self._top_candidates[:5]:
            instrument = entry.get("instrument")
            if isinstance(instrument, Mapping):
                base = instrument.get("base")
                symbol = str(base or instrument.get("symbol") or instrument.get("base"))
            else:
                symbol = str(entry.get("symbol") or "")
            if symbol:
                symbols.append(symbol.upper())

        digest = tuple(symbols)
        if not symbols or digest == self._last_leader_digest:
            return

        self._last_leader_digest = digest
        active = 0
        capacity = len(symbols)
        if self.portfolio_manager is not None:
            active = self.portfolio_manager.active_positions
            capacity = self.portfolio_manager.max_positions

        top_line = _format_top_line(symbols)
        message = f"🏁 Scanner: TOP5 → {top_line} | режим {self.mode} | активных: {active}/{capacity}."
        log("market_scanner.leaderboard", top=symbols, mode=self.mode, active=active, capacity=capacity)
        try:
            self._telegram_sender(message)
        except Exception as exc:  # pragma: no cover - safeguard around external IO
            log("market_scanner.telegram.error", err=str(exc))


def _format_top_line(symbols: Sequence[str]) -> str:
    cleaned = [symbol for symbol in symbols if symbol]
    if not cleaned:
        return "—"
    preview = cleaned[:3]
    body = ", ".join(preview)
    if len(cleaned) > len(preview):
        body = f"{body}, …"
    return body
