from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from fastapi import FastAPI, HTTPException

from ..envs import Settings, get_api_client, get_settings
from ..market_scanner import _ledger_path_for_costs, scan_market_opportunities
from ..paths import DATA_DIR
from ..ws_manager import manager as ws_manager
from .store import FreqAIPredictionStore, get_prediction_store


_PREDICTION_RECENCY_SECONDS = 3600.0


SettingsProvider = Callable[[], Settings]


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _serialise_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    serialised: Dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialised[str(key)] = value
        elif isinstance(value, Mapping):
            serialised[str(key)] = _serialise_mapping(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            serialised[str(key)] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return serialised


def _prediction_path_from_settings(settings: Optional[Settings]) -> Optional[str]:
    if settings is None:
        return None
    raw = getattr(settings, "freqai_prediction_path", None)
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or None
    if raw:
        return str(raw)
    return None


def _prediction_recency_seconds(settings: Optional[Settings]) -> Optional[float]:
    if settings is None:
        return None
    minutes = getattr(settings, "ai_retrain_minutes", None)
    numeric = _safe_float(minutes)
    if numeric is None:
        return None
    seconds = numeric * 60.0
    if seconds <= 0:
        return 0.0
    if seconds < _PREDICTION_RECENCY_SECONDS:
        return seconds
    return _PREDICTION_RECENCY_SECONDS


def _prediction_top_limit(settings: Optional[Settings]) -> Optional[int]:
    if settings is None:
        return None
    raw = getattr(settings, "freqai_top_pairs", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value


def _features_from_entry(entry: Mapping[str, object]) -> Dict[str, object]:
    features: Dict[str, object] = {}

    def _assign(key: str, source: object) -> None:
        numeric = _safe_float(source)
        if numeric is not None:
            features[key] = numeric

    for field in (
        "turnover_usd",
        "volume",
        "volatility_pct",
        "spread_bps",
        "change_pct",
        "score",
        "ev_bps",
        "gross_ev_bps",
    ):
        _assign(field, entry.get(field))

    model_metrics = entry.get("model_metrics")
    if isinstance(model_metrics, Mapping):
        model_features = model_metrics.get("features")
        if isinstance(model_features, Mapping):
            for key, value in model_features.items():
                _assign(f"model.{key}", value)
        fallback = model_metrics.get("fallback")
        if isinstance(fallback, Mapping):
            for key, value in fallback.items():
                _assign(f"fallback.{key}", value)

    volume_impulse = entry.get("volume_impulse")
    if isinstance(volume_impulse, Mapping):
        features["volume_impulse"] = {
            str(window): _safe_float(value)
            for window, value in volume_impulse.items()
            if _safe_float(value) is not None
        }

    volatility_windows = entry.get("volatility_windows")
    if isinstance(volatility_windows, Mapping):
        features["volatility_windows"] = {
            str(window): _safe_float(value)
            for window, value in volatility_windows.items()
            if _safe_float(value) is not None
        }

    order_flow_ratio = entry.get("order_flow_ratio")
    if order_flow_ratio is not None:
        _assign("order_flow_ratio", order_flow_ratio)

    depth_imbalance = entry.get("depth_imbalance")
    if depth_imbalance is not None:
        _assign("depth_imbalance", depth_imbalance)

    top_depth = entry.get("top_depth_quote")
    if isinstance(top_depth, Mapping):
        for side in ("bid", "ask", "total"):
            _assign(f"top_depth_quote.{side}", top_depth.get(side))

    hourly = entry.get("hourly_signal") or entry.get("hourly")
    if isinstance(hourly, Mapping):
        hourly_map = _serialise_mapping(hourly)
        if hourly_map:
            features["hourly_signal"] = hourly_map
        _assign("rsi_hourly", hourly.get("rsi"))
        _assign("ema_fast", hourly.get("ema_fast"))
        _assign("ema_slow", hourly.get("ema_slow"))
        _assign("momentum_pct", hourly.get("momentum_pct"))

    indicator = entry.get("overbought_indicators")
    if isinstance(indicator, Mapping):
        features["overbought_indicators"] = _serialise_mapping(indicator)

    correlations = entry.get("correlations")
    if isinstance(correlations, Mapping):
        features["correlations"] = _serialise_mapping(correlations)

    return features


def _predictions_summary(
    store: FreqAIPredictionStore,
    *,
    limit: Optional[int] = None,
    recency_seconds: Optional[float] = None,
    snapshot: Optional[Mapping[str, object]] = None,
    now: Optional[float] = None,
) -> Dict[str, object]:
    snapshot = snapshot if snapshot is not None else store.snapshot()
    current_time = now if now is not None else time.time()

    if recency_seconds is None:
        recency = _PREDICTION_RECENCY_SECONDS
    else:
        try:
            recency = float(recency_seconds)
        except (TypeError, ValueError):
            recency = _PREDICTION_RECENCY_SECONDS
    if recency < 0:
        recency = 0.0

    if limit is None:
        resolved_limit = 5
    else:
        try:
            resolved_limit = int(limit)
        except (TypeError, ValueError):
            resolved_limit = 0
    if resolved_limit < 0:
        resolved_limit = 0

    if resolved_limit == 0:
        top: Tuple[Dict[str, object], ...] = tuple()
    else:
        top = store.top_pairs(
            resolved_limit,
            max_age=recency,
            snapshot=snapshot,
            now=current_time,
        )

    return {
        "generated_at": snapshot.get("generated_at"),
        "source": snapshot.get("source"),
        "total_pairs": len(snapshot.get("pairs", {})),
        "stale": store.is_stale(
            max_age=recency,
            snapshot=snapshot,
            now=current_time,
        ),
        "recency_seconds": recency,
        "top": list(top),
    }


def create_app(
    *,
    store: Optional[FreqAIPredictionStore] = None,
    settings_provider: SettingsProvider | None = None,
) -> FastAPI:
    store_override = store
    provider = settings_provider or get_settings

    def _resolve_store(settings: Optional[Settings]) -> FreqAIPredictionStore:
        if store_override is not None:
            return store_override
        return get_prediction_store(
            DATA_DIR,
            prediction_path=_prediction_path_from_settings(settings),
        )

    app = FastAPI(title="BybitBot FreqAI Bridge", version="1.0.0")

    @app.get("/health")
    def health() -> Dict[str, object]:
        try:
            settings = provider()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=f"settings_error: {exc}") from exc

        prediction_store = _resolve_store(settings)
        snapshot = prediction_store.snapshot()
        recency = _prediction_recency_seconds(settings)
        top_limit = _prediction_top_limit(settings)
        current_time = time.time()

        ws_status: Optional[Mapping[str, object]]
        try:  # pragma: no cover - optional runtime state
            ws_status = ws_manager.status()
        except Exception:
            ws_status = None

        return {
            "status": "ok",
            "generated_at": snapshot.get("generated_at"),
            "updated_at": snapshot.get("updated_at"),
            "source": snapshot.get("source"),
            "symbols": len(snapshot.get("pairs", {})),
            "settings": {
                "testnet": getattr(settings, "testnet", True),
                "ai_enabled": getattr(settings, "ai_enabled", False),
                "ai_buy_threshold": getattr(settings, "ai_buy_threshold", None),
                "ai_sell_threshold": getattr(settings, "ai_sell_threshold", None),
                "freqai_enabled": getattr(settings, "freqai_enabled", False),
            },
            "predictions": _predictions_summary(
                prediction_store,
                limit=top_limit,
                recency_seconds=recency,
                snapshot=snapshot,
                now=current_time,
            ),
            "ws": ws_status,
        }

    @app.get("/predictions")
    def get_predictions() -> Dict[str, object]:
        settings: Optional[Settings]
        if store_override is not None:
            settings = None
        else:
            try:  # pragma: no cover - settings optional for custom paths
                settings = provider()
            except Exception:
                settings = None
        prediction_store = _resolve_store(settings)
        return prediction_store.snapshot()

    @app.post("/predictions")
    def post_predictions(payload: Mapping[str, object]) -> Dict[str, object]:
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="payload must be an object")
        settings: Optional[Settings]
        if store_override is not None:
            settings = None
        else:
            try:  # pragma: no cover - settings optional for custom paths
                settings = provider()
            except Exception:
                settings = None
        prediction_store = _resolve_store(settings)
        snapshot = prediction_store.update(payload)
        return {"status": "ok", "symbols": len(snapshot.get("pairs", {}))}

    @app.get("/features")
    def features(limit: int = 25, whitelist: Optional[str] = None) -> Dict[str, object]:
        try:
            settings = provider()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=f"settings_error: {exc}") from exc

        prediction_store = _resolve_store(settings)

        try:
            api = get_api_client()
        except Exception:  # pragma: no cover - optional runtime state
            api = None

        whitelist_symbols = []
        if whitelist:
            whitelist_symbols = [symbol.strip().upper() for symbol in whitelist.split(",") if symbol.strip()]

        feature_limit_setting = getattr(settings, "freqai_feature_limit", None)
        try:
            feature_limit = int(feature_limit_setting) if feature_limit_setting is not None else None
        except (TypeError, ValueError):
            feature_limit = None
        if feature_limit is not None and feature_limit <= 0:
            feature_limit = None

        resolved_limit = int(limit)
        if resolved_limit <= 0:
            resolved_limit = feature_limit or 25
        elif feature_limit is not None:
            resolved_limit = min(resolved_limit, feature_limit)

        opportunities = scan_market_opportunities(
            api,
            data_dir=prediction_store.data_dir,
            limit=max(int(resolved_limit), 1),
            min_turnover=1_000_000.0,
            min_change_pct=0.5,
            settings=settings,
            whitelist=whitelist_symbols or None,
            min_top_quote=getattr(settings, "ai_min_top_quote_usd", None),
        )

        pairs = []
        for entry in opportunities:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            features_payload = _features_from_entry(entry)
            freqai = entry.get("freqai")
            if isinstance(freqai, Mapping):
                freqai_payload = _serialise_mapping(freqai)
            else:
                freqai_payload = None

            pairs.append(
                {
                    "symbol": symbol,
                    "trend": entry.get("trend"),
                    "probability": entry.get("probability"),
                    "ev_bps": entry.get("ev_bps"),
                    "score": entry.get("score"),
                    "features": features_payload,
                    "freqai": freqai_payload,
                    "note": entry.get("note"),
                    "actionable": entry.get("actionable"),
                }
            )

        ledger_path = _ledger_path_for_costs(
            prediction_store.data_dir,
            getattr(settings, "testnet", None),
        )
        freqai_snapshot = prediction_store.snapshot()
        recency = _prediction_recency_seconds(settings)
        top_limit = _prediction_top_limit(settings)
        current_time = time.time()

        response = {
            "generated_at": time.time(),
            "count": len(pairs),
            "pairs": pairs,
            "executions_path": str(ledger_path),
            "predictions": _predictions_summary(
                prediction_store,
                limit=top_limit if top_limit is not None else 10,
                recency_seconds=recency,
                snapshot=freqai_snapshot,
                now=current_time,
            ),
        }
        return response

    return app


app = create_app()
