from __future__ import annotations

import copy
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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

from .bybit_api import BybitAPI
from .ai.models import (
    MarketModel,
    ensure_market_model,
    initialise_feature_map,
    liquidity_feature,
)
from .log import log
from .paths import DATA_DIR
from .market_features import build_feature_bundle
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
_MAINTENANCE_GRACE_SECONDS = 180.0
_MIN_CANDLE_SAMPLES = 12
_PRICE_Z_THRESHOLD = 3.5
_PRICE_IQR_MULTIPLIER = 3.0
_VOLUME_Z_THRESHOLD = 4.0
_VOLUME_IQR_MULTIPLIER = 3.5

_MAINTENANCE_KEYWORDS: Tuple[str, ...] = (
    "maint",
    "suspend",
    "halt",
    "upgrade",
    "pause",
    "offline",
)

_DELIST_KEYWORDS: Tuple[str, ...] = (
    "delist",
    "terminated",
    "expired",
    "cease",
    "settled",
    "closed",
)


@dataclass(frozen=True)
class _SymbolHealth:
    maintenance: bool
    delisted: bool
    reason: Optional[str] = None


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

    features = initialise_feature_map()
    features["directional_change_pct"] = signed_change
    features["multiframe_change_pct"] = multi_tf_value
    features["turnover_log"] = turnover_log
    features["volatility_pct"] = vol_value
    features["volume_impulse"] = best_impulse
    features["depth_imbalance"] = depth_value
    features["spread_bps"] = spread_value
    features["correlation_strength"] = correlation_value
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

    effective_change = float(min_change_pct) if min_change_pct is not None else 0.5
    if effective_change < 0.05:
        effective_change = 0.05
    max_spread_bps = float(max_spread_bps)
    if testnet_active:
        if max_spread_bps <= 0:
            max_spread_bps = 120.0
        else:
            max_spread_bps = max(max_spread_bps, 120.0)

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

    snapshot_ts = _safe_float(snapshot.get("ts"))
    if snapshot_ts is None:
        snapshot_ts = now

    def _normalise_symbol_set(symbols: Iterable[object]) -> Set[str]:
        normalised: Set[str] = set()
        for item in symbols:
            candidate, _ = ensure_usdt_symbol(item)
            if candidate:
                normalised.add(candidate)
        return normalised

    entries: List[Dict[str, object]] = []
    wset = _normalise_symbol_set(whitelist or ())
    bset = _normalise_symbol_set(blacklist or ())

    try:
        model: Optional[MarketModel] = ensure_market_model(data_dir=data_dir)
    except Exception as exc:  # pragma: no cover - defensive logging
        log("market_scanner.model.error", err=str(exc))
        model = None

    cost_profile = _resolve_trade_cost_profile(data_dir, settings, testnet)
    fee_round_trip_bps = float(cost_profile.get("round_trip_fee_bps", 0.0) or 0.0)
    slippage_cost_bps = float(cost_profile.get("slippage_bps", 0.0) or 0.0)

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

        health = _assess_symbol_health(raw, now=snapshot_ts)
        if health.delisted or health.maintenance:
            continue

        turnover = _safe_float(raw.get("turnover24h"))
        change_pct = _normalise_percent(raw.get("price24hPcnt"))
        volume = _safe_float(raw.get("volume24h"))
        bid = _safe_float(raw.get("bestBidPrice"))
        ask = _safe_float(raw.get("bestAskPrice"))
        spread_bps = _spread_bps(bid, ask)
        feature_bundle = build_feature_bundle(raw)
        blended_change = feature_bundle.get("blended_change_pct")
        volatility_pct = feature_bundle.get("volatility_pct")
        volatility_windows = feature_bundle.get("volatility_windows")
        volume_spike_score = feature_bundle.get("volume_spike_score")
        volume_impulse = feature_bundle.get("volume_impulse")
        depth_imbalance = feature_bundle.get("depth_imbalance")
        correlations = feature_bundle.get("correlations")
        correlation_strength = feature_bundle.get("correlation_strength")

        force_include = symbol in wset
        if not force_include and turnover is not None and turnover < min_turnover:
            turnover_ok = False
        else:
            turnover_ok = True

        if change_pct is None:
            trend = "wait"
        elif change_pct >= effective_change:
            trend = "buy"
        elif change_pct <= -effective_change:
            trend = "sell"
        else:
            trend = "wait"

        actionable = False
        spread_ok = True
        if spread_bps is not None and max_spread_bps > 0:
            spread_ok = spread_bps <= max_spread_bps

        if trend in {"buy", "sell"}:
            change_ok = change_pct is not None and abs(change_pct) >= effective_change
            actionable = turnover_ok and spread_ok and change_ok

        if not actionable and not force_include:
            strength = _strength_from_change(change_pct)
            if strength < 0.1:
                continue

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
        )

        if model is not None:
            probability = model.predict_proba(feature_vector)
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

        direction = 0
        if trend == "buy":
            direction = 1
        elif trend == "sell":
            direction = -1

        gross_bps: Optional[float] = None
        total_cost_bps: Optional[float] = None
        spread_component = max(float(spread_bps or 0.0), 0.0)

        if change_pct is not None:
            directional_change = float(change_pct)
            if direction != 0:
                directional_change *= direction
            gross_bps = directional_change * 100.0
            total_cost_bps = fee_round_trip_bps + slippage_cost_bps + spread_component
            ev_bps = gross_bps - total_cost_bps
        else:
            ev_bps = None

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
        if volume_spike_score is not None and volume_spike_score > 0:
            note_parts.append(f"всплеск объёма ×{math.exp(volume_spike_score):.2f}")
        if depth_imbalance is not None and direction != 0:
            imbalance_pct = depth_imbalance * 100.0
            side = "покупателей" if imbalance_pct > 0 else "продавцов"
            note_parts.append(f"преимущество {side} {abs(imbalance_pct):.1f}%")
        if correlation_strength is not None and correlation_strength > 0:
            note_parts.append(f"корреляция {correlation_strength * 100:.0f}%")
        if isinstance(volume_impulse, Mapping):
            best_window = None
            best_value = 0.0
            for window, value in volume_impulse.items():
                if value is None:
                    continue
                if abs(value) > abs(best_value):
                    best_window = window
                    best_value = float(value)
            if best_window is not None and best_value > 0:
                note_parts.append(f"импульс объёма {best_window} ×{math.exp(best_value):.2f}")

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
            "blended_change_pct": blended_change,
            "correlations": correlations,
            "correlation_strength": correlation_strength,
            "model_metrics": model_metrics,
            "source": "market_scanner",
            "actionable": actionable,
        }
        if quote_source == "USDC" and upper_source:
            entry["quote_conversion"] = {"from": "USDC", "to": "USDT", "original": upper_source}
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


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return None
        if lowered in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", "disabled"}:
            return False
    return None


def _parse_timestamp(value: object) -> Optional[float]:
    if value is None:
        return None
    candidate: Optional[float]
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            candidate = float(text)
        except ValueError:
            return None
    else:
        return None

    abs_value = abs(candidate)
    if abs_value >= 1e18:
        candidate /= 1_000_000_000.0
    elif abs_value >= 1e15:
        candidate /= 1_000_000.0
    elif abs_value >= 1e12:
        candidate /= 1_000.0
    return candidate


def _quartiles(sorted_values: Sequence[float]) -> Tuple[float, float]:
    count = len(sorted_values)
    if count == 0:
        return 0.0, 0.0
    midpoint = count // 2
    if count % 2 == 0:
        lower_half = sorted_values[:midpoint]
        upper_half = sorted_values[midpoint:]
    else:
        lower_half = sorted_values[:midpoint]
        upper_half = sorted_values[midpoint + 1 :]
    if not lower_half:
        lower_half = sorted_values[:1]
    if not upper_half:
        upper_half = sorted_values[-1:]
    q1 = statistics.median(lower_half)
    q3 = statistics.median(upper_half)
    return float(q1), float(q3)


def _detect_outlier_indices(
    values: Sequence[float],
    *,
    z_threshold: float,
    iqr_multiplier: float,
) -> Set[int]:
    size = len(values)
    if size < 4:
        return set()

    flagged: Set[int] = set()

    mean_value = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev > 1e-9:
        for idx, value in enumerate(values):
            z_score = abs((value - mean_value) / stdev)
            if z_score >= z_threshold:
                flagged.add(idx)

    sorted_values = sorted(values)
    q1, q3 = _quartiles(sorted_values)
    iqr = q3 - q1
    if iqr > 1e-9:
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        for idx, value in enumerate(values):
            if value < lower or value > upper:
                flagged.add(idx)

    if flagged:
        return flagged

    median_value = statistics.median(values)
    deviations = [abs(value - median_value) for value in values]
    mad = statistics.median(deviations)
    if mad <= 1e-9:
        return set()
    for idx, deviation in enumerate(deviations):
        modified_z = 0.6745 * deviation / mad
        if modified_z >= z_threshold:
            flagged.add(idx)
    return flagged


def _coalesce_price(entry: Mapping[str, object]) -> Optional[float]:
    for key in ("close", "open", "high", "low"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _filter_outlier_candles(candles: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    total = len(candles)
    if total < _MIN_CANDLE_SAMPLES:
        return list(candles)

    price_samples: List[Tuple[int, float]] = []
    volume_samples: List[Tuple[int, float]] = []
    for idx, entry in enumerate(candles):
        price = _coalesce_price(entry)
        if price is not None and math.isfinite(price):
            price_samples.append((idx, float(price)))
        volume = entry.get("volume")
        if isinstance(volume, (int, float)) and math.isfinite(volume):
            volume_samples.append((idx, float(volume)))

    flagged_indices: Set[int] = set()

    if len(price_samples) >= _MIN_CANDLE_SAMPLES:
        price_values = [value for _, value in price_samples]
        price_outliers = _detect_outlier_indices(
            price_values,
            z_threshold=_PRICE_Z_THRESHOLD,
            iqr_multiplier=_PRICE_IQR_MULTIPLIER,
        )
        if price_outliers:
            flagged_indices.update(price_samples[idx][0] for idx in price_outliers)

    if len(volume_samples) >= _MIN_CANDLE_SAMPLES:
        volume_values = [value for _, value in volume_samples]
        volume_outliers = _detect_outlier_indices(
            volume_values,
            z_threshold=_VOLUME_Z_THRESHOLD,
            iqr_multiplier=_VOLUME_IQR_MULTIPLIER,
        )
        if volume_outliers:
            flagged_indices.update(volume_samples[idx][0] for idx in volume_outliers)

    if not flagged_indices:
        return list(candles)

    remaining = total - len(flagged_indices)
    if remaining < max(_MIN_CANDLE_SAMPLES // 2, 4):
        return list(candles)

    cleaned = [entry for idx, entry in enumerate(candles) if idx not in flagged_indices]
    return cleaned


def _assess_symbol_health(
    row: Mapping[str, object],
    *,
    now: Optional[float] = None,
) -> _SymbolHealth:
    text_fragments: List[str] = []
    for key in (
        "status",
        "symbolStatus",
        "state",
        "listStatus",
        "suspendDesc",
        "note",
        "message",
    ):
        value = row.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                text_fragments.append(cleaned)

    text_blob = " ".join(text_fragments).lower()
    maintenance = False
    delisted = False
    reason: Optional[str] = None

    for keyword in _DELIST_KEYWORDS:
        if keyword in text_blob:
            delisted = True
            reason = f"status:{keyword}"
            break

    if not delisted:
        for keyword in _MAINTENANCE_KEYWORDS:
            if keyword in text_blob:
                maintenance = True
                reason = f"status:{keyword}"
                break

    if not delisted:
        for key in ("isTrading", "trading", "tradeAvailable", "tradeSwitch", "is_trading"):
            value = row.get(key)
            if value is None:
                continue
            flag = _coerce_bool(value)
            if flag is None:
                continue
            if not flag:
                maintenance = True
                if reason is None:
                    reason = f"{key}:off"
                break

    reference_now = float(now) if now is not None else time.time()

    start = None
    end = None
    for key in (
        "maintenanceStartTime",
        "maintenanceStart",
        "maintainStartTime",
        "maintainStartTs",
        "maintenance_window_start",
    ):
        start = _parse_timestamp(row.get(key))
        if start is not None:
            break
    for key in (
        "maintenanceEndTime",
        "maintenanceEnd",
        "maintainEndTime",
        "maintainEndTs",
        "maintenance_window_end",
    ):
        end = _parse_timestamp(row.get(key))
        if end is not None:
            break
    if start is not None and end is not None:
        if start > end:
            start, end = end, start
        if reference_now >= start - _MAINTENANCE_GRACE_SECONDS and reference_now <= end + _MAINTENANCE_GRACE_SECONDS:
            maintenance = True
            if reason is None:
                reason = "maintenance_window"

    if not delisted:
        for key in ("delistingTime", "delistTime", "expiryTime", "settleTime", "closeTime"):
            marker = _parse_timestamp(row.get(key))
            if marker is None:
                continue
            if marker <= reference_now:
                delisted = True
                reason = key
                break

    return _SymbolHealth(maintenance=maintenance, delisted=delisted, reason=reason)


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

        if isinstance(entry, Mapping):
            start = _safe_int(entry.get("start") or entry.get("openTime") or entry.get("timestamp"))
            open_ = _safe_float(entry.get("open"))
            high = _safe_float(entry.get("high"))
            low = _safe_float(entry.get("low"))
            close = _safe_float(entry.get("close"))
            volume = _safe_float(entry.get("volume"))
            turnover = _safe_float(entry.get("turnover"))
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
        else:
            continue

        if start is None:
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
    cleaned = _filter_outlier_candles(candles)
    return cleaned


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
            self._cache[key] = (now, candles)
            bundle[f"{interval}m"] = candles

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
