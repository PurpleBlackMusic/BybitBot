from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..log import log
from ..paths import DATA_DIR


def _now() -> float:
    return time.time()


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _coerce_probability(value: object) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric > 1.0:
        numeric /= 100.0
    if numeric < 0.0:
        numeric = 0.0
    if numeric > 1.0:
        numeric = 1.0
    return numeric


def _serialise_meta(mapping: Mapping[str, object]) -> Dict[str, object]:
    serialisable: Dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialisable[key] = value
        elif isinstance(value, Mapping):
            serialisable[key] = _serialise_meta(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            serialisable[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return serialisable


def _normalise_symbol(symbol: object) -> Tuple[str, str]:
    text = str(symbol or "").strip().upper()
    canonical = text.replace("/", "").replace("-", "")
    return canonical or text, text or canonical


class FreqAIPredictionStore:
    """Persist and expose predictions coming from a FreqAI sidecar."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        *,
        predictions_path: Optional[Path | str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if predictions_path:
            candidate = Path(predictions_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.data_dir / candidate
            self._path = candidate.resolve(strict=False)
        else:
            self._path = (self.data_dir / "ai" / "freqai_predictions.json").resolve(strict=False)
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_mtime: Optional[float] = None

    @property
    def path(self) -> Path:
        return self._path

    def canonical_symbol(self, symbol: str) -> str:
        canonical, _ = _normalise_symbol(symbol)
        return canonical

    def snapshot(self) -> Dict[str, Any]:
        path = self._path
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            stat_result = None
        except OSError:
            stat_result = None
        mtime = float(getattr(stat_result, "st_mtime", 0.0)) if stat_result else None

        with self._lock:
            if (
                self._cache is not None
                and self._cache_mtime is not None
                and mtime is not None
                and abs(self._cache_mtime - mtime) < 1e-9
            ):
                return copy.deepcopy(self._cache)

            if stat_result is None:
                self._cache = {
                    "generated_at": None,
                    "updated_at": None,
                    "source": None,
                    "pairs": {},
                }
                self._cache_mtime = None
                return copy.deepcopy(self._cache)

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

            snapshot = self._normalise_payload(payload)
            snapshot["updated_at"] = float(payload.get("updated_at") or payload.get("generated_at") or _now())
            self._cache = snapshot
            self._cache_mtime = mtime
            return copy.deepcopy(snapshot)

    def update(self, payload: Mapping[str, object]) -> Dict[str, Any]:
        snapshot = self._normalise_payload(payload)
        snapshot["updated_at"] = _now()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._cache = copy.deepcopy(snapshot)
            try:
                payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
                self._path.write_text(payload, encoding="utf-8")
            except Exception:
                # ensure cache is still available even if write fails
                self._cache_mtime = None
                raise
            else:
                try:
                    stat_result = self._path.stat()
                except OSError:
                    stat_result = None
                self._cache_mtime = (
                    float(getattr(stat_result, "st_mtime", 0.0)) if stat_result else snapshot["updated_at"]
                )
                if stat_result is None:
                    # fall back to the logical timestamp so snapshot consumers retain recency
                    self._cache["updated_at"] = snapshot["updated_at"]
        log(
            "freqai.predictions.updated",
            symbols=len(snapshot.get("pairs", {})),
            generated_at=snapshot.get("generated_at"),
            source=snapshot.get("source"),
        )
        return copy.deepcopy(snapshot)

    def top_pairs(
        self,
        limit: int = 5,
        *,
        min_probability: Optional[float] = None,
        min_ev_bps: Optional[float] = None,
        max_age: Optional[float] = None,
        snapshot: Optional[Mapping[str, object]] = None,
        now: Optional[float] = None,
    ) -> Tuple[Dict[str, object], ...]:
        snapshot = snapshot if snapshot is not None else self.snapshot()
        pairs = snapshot.get("pairs")
        if not isinstance(pairs, Mapping):
            return tuple()

        if now is None:
            now = _now()
        items = []
        for entry in pairs.values():
            if not isinstance(entry, Mapping):
                continue
            probability = _safe_float(entry.get("probability"))
            if min_probability is not None and (probability is None or probability < min_probability):
                continue
            ev_bps = _safe_float(entry.get("ev_bps"))
            if min_ev_bps is not None and (ev_bps is None or ev_bps < min_ev_bps):
                continue
            if max_age is not None and max_age > 0:
                generated_at = _safe_float(entry.get("generated_at"))
                if generated_at is None:
                    generated_at = _safe_float(snapshot.get("updated_at"))
                if generated_at is None or now - generated_at > max_age:
                    continue
            items.append(copy.deepcopy(entry))

        if not items:
            return tuple()

        def _sort_key(entry: Mapping[str, object]) -> Tuple[float, float]:
            ev = _safe_float(entry.get("ev_bps")) or 0.0
            prob = _safe_float(entry.get("probability")) or 0.0
            return (ev, prob)

        items.sort(key=_sort_key, reverse=True)
        if limit and limit > 0:
            items = items[: int(limit)]
        return tuple(items)

    def is_stale(
        self,
        *,
        max_age: float,
        snapshot: Optional[Mapping[str, object]] = None,
        now: Optional[float] = None,
    ) -> bool:
        if max_age <= 0:
            return False
        snapshot = snapshot if snapshot is not None else self.snapshot()
        updated_at = _safe_float(snapshot.get("updated_at"))
        if updated_at is None:
            return True
        if now is None:
            now = _now()
        return (now - updated_at) > max_age

    def _normalise_payload(self, payload: Mapping[str, object]) -> Dict[str, Any]:
        generated_at = _safe_float(payload.get("generated_at"))
        if generated_at is None:
            generated_at = _safe_float(payload.get("timestamp"))
        if generated_at is None:
            generated_at = _now()

        source = str(payload.get("source") or payload.get("provider") or "freqai").strip() or "freqai"
        horizon = _safe_float(payload.get("horizon_minutes"))
        if horizon is None:
            horizon = _safe_float(payload.get("horizon"))
        window = _safe_float(payload.get("window_minutes"))
        predictions_payload = (
            payload.get("predictions")
            or payload.get("pairs")
            or payload.get("symbols")
            or payload.get("data")
            or {}
        )

        def _prediction_items() -> Iterable[Tuple[object, Mapping[str, object]]]:
            if isinstance(predictions_payload, Mapping):
                for symbol, raw_prediction in predictions_payload.items():
                    if isinstance(raw_prediction, Mapping):
                        yield symbol, raw_prediction
                return

            if isinstance(predictions_payload, Iterable) and not isinstance(
                predictions_payload, (str, bytes, bytearray)
            ):
                for raw_prediction in predictions_payload:
                    if not isinstance(raw_prediction, Mapping):
                        continue
                    symbol = (
                        raw_prediction.get("symbol")
                        or raw_prediction.get("pair")
                        or raw_prediction.get("pair_name")
                        or raw_prediction.get("instrument")
                        or raw_prediction.get("coin")
                    )
                    yield symbol, raw_prediction

        pairs: Dict[str, Dict[str, object]] = {}

        for symbol_hint, raw_prediction in _prediction_items() or ():
            if not isinstance(raw_prediction, Mapping):
                continue

            symbol_value = symbol_hint
            if not symbol_value:
                symbol_value = (
                    raw_prediction.get("symbol")
                    or raw_prediction.get("pair")
                    or raw_prediction.get("pair_name")
                    or raw_prediction.get("instrument")
                    or raw_prediction.get("coin")
                )
            if not symbol_value:
                continue

            prediction_payload: Mapping[str, object] = raw_prediction
            additional_meta: Dict[str, object] = {}
            nested = raw_prediction.get("prediction")
            if isinstance(nested, Mapping):
                prediction_payload = nested
                additional_meta = {
                    key: value
                    for key, value in raw_prediction.items()
                    if key
                    not in {
                        "prediction",
                        "symbol",
                        "pair",
                        "pair_name",
                        "instrument",
                        "coin",
                    }
                }

            entry = self._normalise_prediction(prediction_payload)
            if not entry:
                continue

            canonical, display = _normalise_symbol(symbol_value)
            entry["symbol"] = display
            entry["canonical"] = canonical
            entry.setdefault("source", source)
            entry.setdefault("generated_at", generated_at)
            if horizon is not None and "horizon_minutes" not in entry:
                entry["horizon_minutes"] = horizon
            if window is not None and "window_minutes" not in entry:
                entry["window_minutes"] = window

            if additional_meta:
                serialised_meta = _serialise_meta(additional_meta)
                if serialised_meta:
                    existing_meta = entry.setdefault("meta", {})
                    if isinstance(existing_meta, Mapping):
                        # convert to dict in case meta from prediction is mapping-like
                        merged_meta = dict(existing_meta)
                        merged_meta.update(serialised_meta)
                        entry["meta"] = merged_meta
                    else:
                        entry["meta"] = serialised_meta

            pairs[canonical] = entry

        snapshot: Dict[str, Any] = {
            "generated_at": generated_at,
            "source": source,
            "horizon_minutes": horizon,
            "window_minutes": window,
            "pairs": pairs,
        }
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            snapshot["meta"] = _serialise_meta(meta)
        return snapshot

    def _normalise_prediction(self, prediction: Mapping[str, object]) -> Dict[str, object]:
        probability = _coerce_probability(
            prediction.get("probability") or prediction.get("prob") or prediction.get("p")
        )
        if probability is None:
            probability = _coerce_probability(prediction.get("probability_pct"))
        if probability is None:
            logit = _safe_float(prediction.get("logit"))
            if logit is not None:
                try:
                    odds = math.exp(logit)
                    probability = odds / (1.0 + odds)
                except OverflowError:
                    probability = 1.0 if logit > 0 else 0.0

        ev_bps = _safe_float(
            prediction.get("ev_bps")
            or prediction.get("expected_value_bps")
            or prediction.get("expected_value")
            or prediction.get("ev")
        )
        if ev_bps is None:
            ev_pct = _safe_float(prediction.get("expected_value_pct"))
            if ev_pct is not None:
                ev_bps = ev_pct * 100.0

        confidence = _safe_float(prediction.get("confidence"))
        if confidence is not None and confidence > 1.0:
            confidence = confidence / 100.0
        if confidence is not None:
            confidence = max(0.0, min(confidence, 1.0))

        score = _safe_float(prediction.get("score"))
        if score is None and probability is not None:
            score = probability * 100.0

        horizon = _safe_float(prediction.get("horizon_minutes") or prediction.get("horizon"))
        window = _safe_float(prediction.get("window_minutes") or prediction.get("window"))

        meta: Dict[str, object] = {}
        for key, value in prediction.items():
            if key in {
                "probability",
                "probability_pct",
                "prob",
                "p",
                "ev_bps",
                "expected_value_bps",
                "expected_value",
                "expected_value_pct",
                "ev",
                "score",
                "confidence",
                "horizon_minutes",
                "horizon",
                "window_minutes",
                "window",
                "logit",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                meta[key] = value
            elif isinstance(value, Mapping):
                meta[key] = _serialise_meta(value)

        entry: Dict[str, object] = {}
        if probability is not None:
            entry["probability"] = probability
            entry["probability_pct"] = probability * 100.0
        if ev_bps is not None:
            entry["ev_bps"] = ev_bps
            entry["ev_pct"] = ev_bps / 100.0
        if score is not None:
            entry["score"] = score
        if confidence is not None:
            entry["confidence"] = confidence
            entry["confidence_pct"] = confidence * 100.0
        if horizon is not None:
            entry["horizon_minutes"] = horizon
        if window is not None:
            entry["window_minutes"] = window
        if meta:
            entry["meta"] = meta
        generated = _safe_float(prediction.get("generated_at") or prediction.get("ts"))
        if generated is not None:
            entry["generated_at"] = generated
        return entry


_STORE_CACHE: Dict[Tuple[str, str], FreqAIPredictionStore] = {}


def get_prediction_store(
    data_dir: Path = DATA_DIR,
    *,
    prediction_path: Optional[Path | str] = None,
) -> FreqAIPredictionStore:
    resolved_dir = Path(data_dir).expanduser().resolve()
    if prediction_path:
        candidate = Path(prediction_path).expanduser()
        if not candidate.is_absolute():
            candidate = resolved_dir / candidate
    else:
        candidate = resolved_dir / "ai" / "freqai_predictions.json"
    resolved_path = candidate.resolve(strict=False)

    key = (str(resolved_dir), str(resolved_path))
    store = _STORE_CACHE.get(key)
    if store is None:
        store = FreqAIPredictionStore(resolved_dir, predictions_path=resolved_path)
        _STORE_CACHE[key] = store
    return store
