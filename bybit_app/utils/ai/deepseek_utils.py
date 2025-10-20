from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from ..paths import DATA_DIR

__all__ = [
    "extract_deepseek_snapshot",
    "evaluate_deepseek_guidance",
    "load_deepseek_status",
    "resolve_deepseek_watchlist",
    "resolve_deepseek_drawdown_limit",
]


_STATUS_PATH = DATA_DIR / "ai" / "status.json"


def _safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN check
        return None
    return numeric


def load_deepseek_status(path: Path | None = None) -> Dict[str, object]:
    """Load the latest DeepSeek status snapshot from disk."""

    target = Path(path) if path is not None else _STATUS_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _normalise_symbol(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def resolve_deepseek_watchlist(path: Path | None = None) -> list[str]:
    """Return the preferred DeepSeek watchlist (upper-cased symbols)."""

    status = load_deepseek_status(path)
    watchlist: list[str] = []

    raw_watchlist = status.get("watchlist")
    if isinstance(raw_watchlist, Sequence) and not isinstance(raw_watchlist, (str, bytes)):
        for entry in raw_watchlist:
            symbol = _normalise_symbol(entry)
            if symbol:
                watchlist.append(symbol)

    fallback_symbol = _normalise_symbol(status.get("symbol"))
    if fallback_symbol and fallback_symbol not in watchlist:
        watchlist.append(fallback_symbol)

    return watchlist


def resolve_deepseek_drawdown_limit(path: Path | None = None) -> Optional[float]:
    """Resolve the drawdown alert threshold suggested by DeepSeek, if any."""

    status = load_deepseek_status(path)

    risk_payload = status.get("risk")
    candidate = None
    if isinstance(risk_payload, Mapping):
        candidate = risk_payload.get("max_drawdown_alert_pct")
        if candidate is None:
            candidate = risk_payload.get("max_drawdown_pct")

    if candidate is None:
        candidate = status.get("max_drawdown_alert_pct")

    limit = _safe_float(candidate)
    if limit is None:
        return None
    return float(limit)


_DIRECTION_ALIASES = {
    "long": "buy",
    "buy": "buy",
    "bull": "buy",
    "bullish": "buy",
    "accumulate": "buy",
    "accumulation": "buy",
    "short": "sell",
    "sell": "sell",
    "bear": "sell",
    "bearish": "sell",
    "distribute": "sell",
    "distribution": "sell",
}

_OPPOSITION_SKIP_THRESHOLD = 0.55
_BOOST_MAX = 1.25
_NEUTRAL_BOOST_THRESHOLD = 0.7
_NEUTRAL_LOW_THRESHOLD = 0.3
_MIN_MULTIPLIER = 0.55


def _normalise_direction(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    for key, mapped in _DIRECTION_ALIASES.items():
        if cleaned == key or cleaned.startswith(f"{key} "):
            return mapped
    if cleaned in {"flat", "neutral", "sideline", "hold"}:
        return None
    return cleaned if cleaned in {"buy", "sell"} else None


def extract_deepseek_snapshot(source: Mapping[str, object] | None) -> Dict[str, object]:
    """Return a normalised DeepSeek payload extracted from *source*."""

    if not isinstance(source, Mapping):
        return {}

    snapshot: Dict[str, object] = {}

    def _assign_numeric(target: str, *keys: str) -> None:
        if target in snapshot:
            return
        for payload in candidates:
            for key in keys:
                if not isinstance(payload, Mapping):
                    continue
                value = payload.get(key)
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                snapshot[target] = numeric
                return

    def _assign_text(target: str, *keys: str) -> None:
        if target in snapshot:
            return
        for payload in candidates:
            for key in keys:
                if not isinstance(payload, Mapping):
                    continue
                value = payload.get(key)
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        snapshot[target] = cleaned
                        return

    candidates: list[Mapping[str, object]] = []
    direct = source.get("deepseek")
    if isinstance(direct, Mapping):
        candidates.append(direct)
    model_metrics = source.get("model_metrics")
    if isinstance(model_metrics, Mapping):
        candidates.append(model_metrics)
    features = source.get("features")
    if isinstance(features, Mapping):
        candidates.append(features)
    candidates.append(source)

    _assign_numeric("score", "score", "confidence", "deepseek_score", "deepseek_confidence")
    _assign_numeric("stop_loss", "stop_loss", "deepseek_stop_loss")
    _assign_numeric("take_profit", "take_profit", "deepseek_take_profit")
    _assign_numeric("entry", "entry", "deepseek_entry")
    _assign_text("direction", "direction", "deepseek_direction", "trend")
    _assign_text("summary", "summary", "deepseek_summary", "comment", "note")

    score_value = snapshot.get("score")
    if isinstance(score_value, (int, float)):
        score_clamped = max(0.0, min(float(score_value), 1.0))
        snapshot["score"] = score_clamped

    return snapshot


def evaluate_deepseek_guidance(
    payload: Mapping[str, object] | None,
    mode: str,
) -> Dict[str, object]:
    """Evaluate DeepSeek guidance for the given *mode*.

    Returns a dictionary with fields ``allow`` (bool), ``multiplier`` (float),
    ``alignment`` (str) and optional context such as ``reason``.
    """

    result: Dict[str, object] = {
        "allow": True,
        "multiplier": 1.0,
        "alignment": "neutral",
    }

    if not isinstance(payload, Mapping):
        return result

    score = _safe_float(payload.get("score"))
    if score is not None:
        score = max(0.0, min(score, 1.0))
        result["score"] = score
    direction_raw = payload.get("direction")
    if direction_raw is not None:
        result["direction"] = str(direction_raw).strip()
    summary_text = payload.get("summary")
    if isinstance(summary_text, str) and summary_text.strip():
        result["summary"] = summary_text.strip()
    stop_loss = _safe_float(payload.get("stop_loss"))
    if stop_loss is not None:
        result["stop_loss"] = stop_loss
    take_profit = _safe_float(payload.get("take_profit"))
    if take_profit is not None:
        result["take_profit"] = take_profit

    mode_clean = str(mode or "").strip().lower()
    if mode_clean:
        result["mode"] = mode_clean

    canonical_direction = _normalise_direction(direction_raw)
    if canonical_direction:
        result["canonical_direction"] = canonical_direction

    multiplier = 1.0
    influence = "neutral"

    if canonical_direction and mode_clean in {"buy", "sell"}:
        if canonical_direction == mode_clean:
            result["alignment"] = "aligned"
            if score is None:
                multiplier = 1.08
            else:
                boost = 0.05 + max(0.0, score - 0.5) * 0.35
                multiplier = min(_BOOST_MAX, 1.0 + boost)
            if multiplier > 1.0:
                influence = "boost"
        else:
            result["alignment"] = "opposed"
            if score is not None and score >= _OPPOSITION_SKIP_THRESHOLD:
                result["allow"] = False
                result["reason"] = "opposed_high_confidence"
                influence = "block"
                reduction = (score - _OPPOSITION_SKIP_THRESHOLD) * 0.5
                multiplier = max(_MIN_MULTIPLIER, 1.0 - reduction)
            else:
                if score is None:
                    multiplier = 0.8
                else:
                    reduction = (_OPPOSITION_SKIP_THRESHOLD - score)
                    multiplier = max(_MIN_MULTIPLIER, 1.0 - reduction * 0.8)
                influence = "reduce" if multiplier < 1.0 else "neutral"
    else:
        result["alignment"] = "neutral"
        if score is not None:
            if score < _NEUTRAL_LOW_THRESHOLD:
                reduction = (_NEUTRAL_LOW_THRESHOLD - score) * 0.8
                multiplier = max(_MIN_MULTIPLIER, 1.0 - reduction)
                if multiplier < 1.0:
                    influence = "reduce"
            elif score > _NEUTRAL_BOOST_THRESHOLD:
                boost = (score - _NEUTRAL_BOOST_THRESHOLD) * 0.3
                multiplier = min(_BOOST_MAX, 1.0 + boost)
                if multiplier > 1.0:
                    influence = "boost"

    multiplier = max(_MIN_MULTIPLIER, min(multiplier, _BOOST_MAX))
    result["multiplier"] = multiplier
    result["influence"] = influence
    return result
