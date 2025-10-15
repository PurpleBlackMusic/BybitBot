"""Feature engineering helpers for market scanner."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence


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


def _range_volatility(high: Optional[float], low: Optional[float], ref: Optional[float]) -> Optional[float]:
    if high is None or low is None or ref is None or ref <= 0:
        return None
    value = (high - low) / ref * 100.0
    if value < 0:
        value = 0.0
    return value


def compute_multi_timeframe_momentum(row: Mapping[str, object]) -> Dict[str, object]:
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


def compute_volatility_windows(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    """Return volatility estimations for several rolling windows."""

    window_specs = {
        "1h": ("highPrice1h", "lowPrice1h", "closePrice1h"),
        "4h": ("highPrice4h", "lowPrice4h", "closePrice4h"),
        "24h": ("highPrice24h", "lowPrice24h", "lastPrice"),
        "7d": ("highPrice7d", "lowPrice7d", "closePrice7d"),
    }
    fallback_percent_fields = {
        "1h": "price1hPcnt",
        "4h": "price4hPcnt",
        "24h": "price24hPcnt",
        "7d": "price7dPcnt",
    }

    results: Dict[str, Optional[float]] = {}
    last_price = _safe_float(row.get("lastPrice") or row.get("close") or row.get("closePrice"))

    for window, (high_key, low_key, ref_key) in window_specs.items():
        high = _safe_float(row.get(high_key))
        low = _safe_float(row.get(low_key))
        reference = _safe_float(row.get(ref_key)) or last_price
        volatility = _range_volatility(high, low, reference)
        if volatility is None:
            pct_field = fallback_percent_fields.get(window)
            if pct_field:
                change_pct = _normalise_percent(row.get(pct_field))
                volatility = abs(change_pct) if change_pct is not None else None
        results[window] = volatility

    overall = results.get("24h")
    if overall is None:
        values = [value for value in results.values() if value is not None]
        overall = _avg(values)

    return {
        "windows": results,
        "overall": overall,
    }


def compute_volume_impulse(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    """Measure volume expansion versus previous periods and trend impulses."""

    window_pairs = {
        "1h": ("volume1h", "prevVolume1h"),
        "4h": ("volume4h", "prevVolume4h"),
        "24h": ("volume24h", "prevVolume24h"),
    }
    impulses: Dict[str, Optional[float]] = {}

    for window, (current_key, prev_key) in window_pairs.items():
        current = _safe_float(row.get(current_key) or row.get(current_key.replace("volume", "turnover")))
        previous = _safe_float(row.get(prev_key) or row.get(prev_key.replace("volume", "turnover")))
        if current is None or current <= 0:
            impulses[window] = None
            continue
        baseline_candidates = [value for value in [previous] if value is not None and value > 0]
        if window != "24h":
            # incorporate slower lookback where available
            longer_key = "prevVolume24h" if window == "1h" else "prevVolume7d"
            longer = _safe_float(row.get(longer_key))
            if longer is not None and longer > 0:
                baseline_candidates.append(longer)
        baseline = _avg(baseline_candidates)
        if baseline is None or baseline <= 0:
            impulses[window] = None
            continue
        ratio = current / baseline
        impulses[window] = math.log(ratio) if ratio > 0 else None

    # compatibility helper with older callers expecting "volume_spike_score"
    spike_candidates = [value for value in impulses.values() if value is not None]
    spike = max(spike_candidates) if spike_candidates else None

    return {
        "impulses": impulses,
        "spike_score": spike,
    }


def _first_float(row: Mapping[str, object], keys: Sequence[str]) -> Optional[float]:
    """Return the first valid float from ``row`` among ``keys``."""

    for key in keys:
        value = row.get(key)
        numeric = _safe_float(value)
        if numeric is not None:
            return numeric
    return None


def compute_orderbook_depth_signal(row: Mapping[str, object]) -> Optional[float]:
    """Evaluate orderbook imbalance using best bid/ask size when available."""

    bid_size = _safe_float(row.get("bid1Size") or row.get("bidSize"))
    ask_size = _safe_float(row.get("ask1Size") or row.get("askSize"))
    if bid_size is None or ask_size is None or bid_size <= 0 or ask_size <= 0:
        return None
    imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    return float(imbalance)


def compute_order_flow_metrics(row: Mapping[str, object]) -> Dict[str, object]:
    """Extract order-flow ratios and top-of-book liquidity metrics."""

    buy_turnover = _first_float(
        row,
        (
            "buyTurnover24h",
            "takerBuyQuoteVolume",
            "buyQuoteVolume24h",
            "buyVolumeQuote24h",
            "buyNotional24h",
        ),
    )
    sell_turnover = _first_float(
        row,
        (
            "sellTurnover24h",
            "takerSellQuoteVolume",
            "sellQuoteVolume24h",
            "sellVolumeQuote24h",
            "sellNotional24h",
        ),
    )

    order_flow_ratio: Optional[float] = None
    if (
        buy_turnover is not None
        and sell_turnover is not None
        and buy_turnover >= 0
        and sell_turnover >= 0
        and buy_turnover + sell_turnover > 0
    ):
        order_flow_ratio = (buy_turnover - sell_turnover) / (buy_turnover + sell_turnover)

    window_templates = {
        "1h": ("buyVolume1h", "sellVolume1h"),
        "4h": ("buyVolume4h", "sellVolume4h"),
        "24h": ("buyVolume24h", "sellVolume24h"),
    }

    cvd_windows: Dict[str, Optional[float]] = {}
    cvd_score: Optional[float] = None

    for window, (buy_key, sell_key) in window_templates.items():
        buy_volume = _first_float(
            row,
            (buy_key, buy_key.replace("Volume", "BaseVolume"), buy_key.replace("Volume", "Qty")),
        )
        sell_volume = _first_float(
            row,
            (
                sell_key,
                sell_key.replace("Volume", "BaseVolume"),
                sell_key.replace("Volume", "Qty"),
            ),
        )

        if buy_volume is None and sell_volume is None:
            cvd_windows[window] = None
            continue

        buy_volume = max(buy_volume or 0.0, 0.0)
        sell_volume = max(sell_volume or 0.0, 0.0)
        total = buy_volume + sell_volume
        if total <= 0:
            cvd_value = None
        else:
            cvd_value = (buy_volume - sell_volume) / total
            if cvd_score is None or abs(cvd_value) > abs(cvd_score):
                cvd_score = cvd_value
        cvd_windows[window] = cvd_value

    bid_price = _safe_float(row.get("bestBidPrice") or row.get("bid1Price"))
    ask_price = _safe_float(row.get("bestAskPrice") or row.get("ask1Price"))
    bid_size = _safe_float(row.get("bid1Size") or row.get("bidSize"))
    ask_size = _safe_float(row.get("ask1Size") or row.get("askSize"))

    bid_quote = bid_price * bid_size if bid_price and bid_size and bid_price > 0 and bid_size > 0 else None
    ask_quote = ask_price * ask_size if ask_price and ask_size and ask_price > 0 and ask_size > 0 else None

    top_total_quote: Optional[float] = None
    if bid_quote is not None or ask_quote is not None:
        top_total_quote = (bid_quote or 0.0) + (ask_quote or 0.0)

    top_imbalance: Optional[float] = None
    if (
        bid_quote is not None
        and ask_quote is not None
        and bid_quote + ask_quote > 0
    ):
        top_imbalance = (bid_quote - ask_quote) / (bid_quote + ask_quote)

    return {
        "order_flow_ratio": order_flow_ratio,
        "cvd_score": cvd_score,
        "cvd_windows": cvd_windows,
        "top_depth_quote": {
            "bid": bid_quote,
            "ask": ask_quote,
            "total": top_total_quote,
        },
        "top_depth_imbalance": top_imbalance,
    }


def compute_cross_market_correlation(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    """Extract correlation metrics against reference benchmarks when provided."""

    correlations: Dict[str, Optional[float]] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if "corr" not in lowered and not lowered.startswith("beta_"):
            continue
        cleaned = lowered
        if lowered.startswith("corr_"):
            cleaned = lowered.split("corr_", 1)[-1]
        elif lowered.endswith("corr"):
            cleaned = lowered.rsplit("corr", 1)[0]
        elif lowered.startswith("beta_"):
            cleaned = lowered.split("beta_", 1)[-1]
        cleaned = cleaned.strip("_") or lowered
        correlations[cleaned] = _safe_float(value)

    if not correlations:
        return {"correlations": {}, "avg_abs": None}

    magnitudes = [abs(val) for val in correlations.values() if val is not None]
    avg_abs = _avg(magnitudes) if magnitudes else None
    return {"correlations": correlations, "avg_abs": avg_abs}


def build_feature_bundle(row: Mapping[str, object]) -> Dict[str, object]:
    momentum = compute_multi_timeframe_momentum(row)
    volatility = compute_volatility_windows(row)
    volume = compute_volume_impulse(row)
    depth_signal = compute_orderbook_depth_signal(row)
    order_flow = compute_order_flow_metrics(row)
    correlation = compute_cross_market_correlation(row)

    blended_change = momentum["blended_change_pct"]
    if blended_change is None:
        blended_change = momentum["dominant_change_pct"]

    return {
        "blended_change_pct": blended_change,
        "dominant_change_pct": momentum["dominant_change_pct"],
        "timeframe_contributions": momentum["timeframe_contributions"],
        "volatility_pct": volatility["overall"],
        "volatility_windows": volatility["windows"],
        "volume_spike_score": volume["spike_score"],
        "volume_impulse": volume["impulses"],
        "depth_imbalance": depth_signal,
        "order_flow_ratio": order_flow["order_flow_ratio"],
        "cvd_score": order_flow["cvd_score"],
        "cvd_windows": order_flow["cvd_windows"],
        "top_depth_quote": order_flow["top_depth_quote"],
        "top_depth_imbalance": order_flow["top_depth_imbalance"],
        "correlations": correlation["correlations"],
        "correlation_strength": correlation["avg_abs"],
    }
