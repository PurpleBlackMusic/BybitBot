"""Helpers for loading external AI/ML feature sources."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from ..log import log
from ..paths import DATA_DIR


@dataclass
class ExternalFeatureSnapshot:
    """Container for sentiment/news metrics loaded from disk."""

    sentiment: Mapping[str, float]
    news_heat: Mapping[str, float]
    social_score: Mapping[str, float]
    macro_regime: float
    updated_at: Optional[float]


class ExternalFeatureProvider:
    """Load optional external data sources (sentiment, macro regime, etc.)."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._cache: Optional[ExternalFeatureSnapshot] = None
        self._cache_path: Optional[Path] = None

    def _snapshot_path(self) -> Path:
        return self.data_dir / "external" / "sentiment.json"

    def _load_snapshot(self) -> ExternalFeatureSnapshot:
        path = self._snapshot_path()
        if self._cache is not None and path == self._cache_path:
            return self._cache
        sentiment: Dict[str, float] = {}
        news_heat: Dict[str, float] = {}
        social_score: Dict[str, float] = {}
        macro_regime = 0.0
        updated_at: Optional[float] = None
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                sentiment = {
                    str(symbol).upper(): float(value)
                    for symbol, value in (payload.get("sentiment") or {}).items()
                    if _is_finite(value)
                }
                news_heat = {
                    str(symbol).upper(): float(value)
                    for symbol, value in (payload.get("news_heat") or {}).items()
                    if _is_finite(value)
                }
                social_score = {
                    str(symbol).upper(): float(value)
                    for symbol, value in (payload.get("social") or {}).items()
                    if _is_finite(value)
                }
                macro = payload.get("macro_regime")
                if _is_finite(macro):
                    macro_regime = float(macro)
                ts_value = payload.get("updated_at")
                if _is_finite(ts_value):
                    updated_at = float(ts_value)
        snapshot = ExternalFeatureSnapshot(
            sentiment=sentiment,
            news_heat=news_heat,
            social_score=social_score,
            macro_regime=macro_regime,
            updated_at=updated_at,
        )
        self._cache = snapshot
        self._cache_path = path
        return snapshot

    def sentiment_for(self, symbol: str) -> float:
        snapshot = self._load_snapshot()
        return float(snapshot.sentiment.get(symbol.upper(), 0.0))

    def news_heat_for(self, symbol: str) -> float:
        snapshot = self._load_snapshot()
        return float(snapshot.news_heat.get(symbol.upper(), 0.0))

    def social_score_for(self, symbol: str) -> float:
        snapshot = self._load_snapshot()
        return float(snapshot.social_score.get(symbol.upper(), 0.0))

    def macro_regime_score(self) -> float:
        snapshot = self._load_snapshot()
        return float(snapshot.macro_regime)

    def batch_features(self, symbols: Mapping[str, str] | Mapping[str, object]) -> Dict[str, Dict[str, float]]:
        snapshot = self._load_snapshot()
        features: Dict[str, Dict[str, float]] = {}
        for symbol in symbols:
            key = str(symbol).upper()
            features[key] = {
                "sentiment_score": float(snapshot.sentiment.get(key, 0.0)),
                "news_heat": float(snapshot.news_heat.get(key, 0.0)),
                "social_score": float(snapshot.social_score.get(key, 0.0)),
            }
        return features

    def log_health(self) -> None:
        snapshot = self._load_snapshot()
        log(
            "external_features.snapshot",
            symbols=len(snapshot.sentiment),
            macro_regime=float(snapshot.macro_regime),
            updated_at=snapshot.updated_at,
        )


def _is_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)
