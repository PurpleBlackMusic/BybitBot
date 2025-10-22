"""Helpers for loading external AI/ML feature sources."""

from __future__ import annotations

import asyncio
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..log import log
from ..paths import DATA_DIR
from .deepseek_adapter import DeepSeekAdapter


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

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        *,
        deepseek_adapter: DeepSeekAdapter | None = None,
        enable_deepseek: bool = True,
        deepseek_cache_ttl: float = 30.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self._cache: Optional[ExternalFeatureSnapshot] = None
        self._cache_path: Optional[Path] = None
        self._cache_mtime: Optional[float] = None
        self._enable_deepseek = bool(enable_deepseek)
        self._deepseek_batch_cache: Dict[str, Mapping[str, Any]] | None = None
        self._deepseek_batch_symbols: set[str] = set()
        self._deepseek_batch_timestamp: float = 0.0
        self._deepseek_cache_ttl = max(float(deepseek_cache_ttl), 0.0)
        self._executor: ThreadPoolExecutor | None = None
        if deepseek_adapter is not None:
            self._deepseek: DeepSeekAdapter | None = deepseek_adapter
        elif self._enable_deepseek:
            self._deepseek = DeepSeekAdapter()
        else:
            self._deepseek = None

    def _snapshot_path(self) -> Path:
        return self.data_dir / "external" / "sentiment.json"

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None or self._executor._shutdown:  # type: ignore[attr-defined]
            # allow up to 8 workers but avoid spinning too many threads for
            # smaller batches.
            self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="deepseek")
        return self._executor

    def _deepseek_cache_valid(self, symbols: Iterable[str]) -> bool:
        if self._deepseek_batch_cache is None:
            return False
        if self._deepseek_cache_ttl <= 0.0:
            return False
        now = time.time()
        if now - self._deepseek_batch_timestamp > self._deepseek_cache_ttl:
            return False
        requested = {str(symbol).upper() for symbol in symbols}
        return requested.issubset(self._deepseek_batch_symbols)

    async def _gather_deepseek_batch(self, symbols: tuple[str, ...]) -> list[Mapping[str, Any]]:
        loop = asyncio.get_running_loop()
        executor = self._ensure_executor()
        tasks = [
            loop.run_in_executor(executor, self._deepseek.get_signal, symbol)  # type: ignore[arg-type]
            for symbol in symbols
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _fetch_deepseek_batch(self, symbols: Iterable[str]) -> Dict[str, Mapping[str, Any]]:
        ordered = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))
        if not ordered or not self._enable_deepseek or self._deepseek is None:
            return {}

        if self._deepseek_cache_valid(ordered):
            assert self._deepseek_batch_cache is not None
            return {symbol: self._deepseek_batch_cache.get(symbol, {}) for symbol in ordered}

        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self._gather_deepseek_batch(ordered))
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except Exception as exc:  # pragma: no cover - defensive fallback
            log("external_features.deepseek.batch_error", error=str(exc))
            results = []

        payload: Dict[str, Mapping[str, Any]] = {}
        for symbol, result in zip(ordered, results):
            if isinstance(result, Exception):  # pragma: no cover - network failure
                log("external_features.deepseek.symbol_error", symbol=symbol, error=str(result))
                continue
            if isinstance(result, Mapping):
                payload[symbol] = result
            else:
                payload[symbol] = {}

        self._deepseek_batch_cache = payload
        self._deepseek_batch_symbols = set(ordered)
        self._deepseek_batch_timestamp = time.time()
        return {symbol: payload.get(symbol, {}) for symbol in ordered}

    def _load_snapshot(self) -> ExternalFeatureSnapshot:
        path = self._snapshot_path()
        mtime: Optional[float] = None
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            stat_result = None
        except OSError:
            stat_result = None
        else:
            mtime = float(getattr(stat_result, "st_mtime", None) or 0.0)

        if (
            self._cache is not None
            and path == self._cache_path
            and (
                (mtime is None and self._cache_mtime is None)
                or (mtime is not None and self._cache_mtime == mtime)
            )
        ):
            return self._cache
        sentiment: Dict[str, float] = {}
        news_heat: Dict[str, float] = {}
        social_score: Dict[str, float] = {}
        macro_regime = 0.0
        updated_at: Optional[float] = None
        if stat_result is not None:
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
        self._cache_mtime = mtime
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

    def batch_features(
        self,
        symbols: Mapping[str, str] | Mapping[str, object],
    ) -> Dict[str, Dict[str, Any]]:
        snapshot = self._load_snapshot()
        features: Dict[str, Dict[str, Any]] = {}
        symbol_list = [str(symbol).upper() for symbol in symbols]
        deepseek_batch: Dict[str, Mapping[str, Any]] = {}
        if self._enable_deepseek and self._deepseek is not None:
            deepseek_batch = self._fetch_deepseek_batch(symbol_list)
        for symbol in symbol_list:
            key = symbol
            features[key] = {
                "sentiment_score": float(snapshot.sentiment.get(key, 0.0)),
                "news_heat": float(snapshot.news_heat.get(key, 0.0)),
                "social_score": float(snapshot.social_score.get(key, 0.0)),
                "deepseek_score": 0.0,
            }
            deepseek_payload: Mapping[str, Any] = deepseek_batch.get(key, {})
            score_value = deepseek_payload.get("deepseek_confidence")
            try:
                score_numeric = float(score_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                score_numeric = 0.0
            features[key]["deepseek_score"] = score_numeric
            for extra_key, extra_value in deepseek_payload.items():
                if extra_key == "deepseek_confidence":
                    continue
                features[key][extra_key] = extra_value
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
