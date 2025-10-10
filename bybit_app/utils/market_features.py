"""Feature engineering helpers for market scanner."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional


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


def _avg(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def compute_multi_timeframe_momentum(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    """Estimate momentum using multiple percentage change windows."""

    timeframe_fields = {
        "5m": "price5mPcnt",
        "15m": "price15mPcnt",
        "1h": "price1hPcnt",
        "4h": "price4hPcnt",
        "24h": "price24hPcnt",
        "7d": "price7dPcnt",
    }
    weighted_sum = 0.0
    weight_total = 0.0
    contributions: Dict[str, Optional[float]] = {}

    for weight_index, (name, field) in enumerate(timeframe_fields.items(), start=1):
        change_pct = _normalise_percent(row.get(field))
        contributions[name] = change_pct
        if change_pct is None:
            continue
        weight = 1.0 + weight_index * 0.3
        weighted_sum += change_pct * weight
        weight_total += weight

    blended = weighted_sum / weight_total if weight_total else None
    dominant = _avg([value for value in contributions.values() if value is not None])

    return {
        "blended_change_pct": blended,
        "dominant_change_pct": dominant,
        "timeframe_contributions": contributions,
    }


def compute_intraday_volatility(row: Mapping[str, object]) -> Optional[float]:
    """Compute an intraday volatility proxy based on price extremes."""

    high = _safe_float(
        row.get("highPrice24h")
        or row.get("high24h")
        or row.get("high")
        or row.get("highPrice")
    )
    low = _safe_float(
        row.get("lowPrice24h")
        or row.get("low24h")
        or row.get("low")
        or row.get("lowPrice")
    )
    close = _safe_float(row.get("lastPrice") or row.get("close") or row.get("closePrice"))
    if high is None or low is None or close is None or close <= 0:
        return None
    range_pct = (high - low) / close * 100.0
    if range_pct < 0:
        range_pct = 0.0
    return range_pct


def compute_volume_spike(row: Mapping[str, object]) -> Optional[float]:
    """Measure volume expansion versus previous periods."""

    vol_24h = _safe_float(row.get("volume24h") or row.get("turnover24h"))
    vol_1h = _safe_float(row.get("volume1h") or row.get("turnover1h"))
    prev_vol_24h = _safe_float(row.get("prevVolume24h") or row.get("volume24hPrev"))

    candidates = [value for value in [vol_1h, prev_vol_24h] if value is not None and value > 0]
    baseline = _avg(candidates)
    if vol_24h is None or vol_24h <= 0 or baseline is None or baseline <= 0:
        return None
    spike_ratio = vol_24h / baseline
    if spike_ratio <= 0:
        return None
    return math.log(spike_ratio)


def compute_orderbook_depth_signal(row: Mapping[str, object]) -> Optional[float]:
    """Evaluate orderbook imbalance using best bid/ask size when available."""

    bid_size = _safe_float(row.get("bid1Size") or row.get("bidSize"))
    ask_size = _safe_float(row.get("ask1Size") or row.get("askSize"))
    if bid_size is None or ask_size is None or bid_size <= 0 or ask_size <= 0:
        return None
    imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    return float(imbalance)


def build_feature_bundle(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    momentum = compute_multi_timeframe_momentum(row)
    volatility = compute_intraday_volatility(row)
    volume_spike = compute_volume_spike(row)
    depth_signal = compute_orderbook_depth_signal(row)

    bundle: Dict[str, Optional[float]] = {
        "blended_change_pct": momentum["blended_change_pct"],
        "dominant_change_pct": momentum["dominant_change_pct"],
        "volatility_pct": volatility,
        "volume_spike_score": volume_spike,
        "depth_imbalance": depth_signal,
    }
    if bundle["blended_change_pct"] is None:
        bundle["blended_change_pct"] = momentum["dominant_change_pct"]
    return bundle
