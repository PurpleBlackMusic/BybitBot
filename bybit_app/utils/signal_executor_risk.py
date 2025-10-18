"""Risk scaling and allocation helpers for the signal executor."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ai_thresholds import resolve_min_ev_from_settings
from .envs import Settings
from .self_learning import TradePerformanceSnapshot
from .signal_executor_models import _safe_float


class SignalExecutorRiskMixin:
    """Derive volatility, risk, and sizing metrics for trades."""

    def _resolve_volatility_percent(
        self, summary: Mapping[str, object]
    ) -> Tuple[Optional[float], Optional[str]]:
        if not isinstance(summary, Mapping):
            return None, None

        candidates: List[Tuple[float, str]] = []

        def _add_candidate(value: object, source: str) -> None:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return
            if not math.isfinite(number):
                return
            number = abs(number)
            if number <= 0:
                return
            candidates.append((number, source))

        def _lookup_path(root: Mapping[str, object], path: Sequence[str]) -> None:
            current: object = root
            for part in path:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    current = None
                    break
            if current is not None:
                _add_candidate(current, ".".join(path))

        direct_paths: Tuple[Tuple[str, ...], ...] = (
            ("volatility_pct",),
            ("volatilityPercent",),
            ("volatility_percent",),
            ("volatility",),
            ("metrics", "volatility_pct"),
            ("features", "volatility_pct"),
            ("market_features", "volatility_pct"),
            ("summary", "volatility_pct"),
            ("meta", "volatility_pct"),
            ("stats", "volatility_pct"),
            ("technical", "volatility_pct"),
            ("volatility", "pct"),
            ("volatility", "percent"),
        )

        for path in direct_paths:
            _lookup_path(summary, path)

        window_paths: Tuple[Tuple[str, ...], ...] = (
            ("volatility_windows",),
            ("volatility", "windows"),
            ("market_features", "volatility_windows"),
        )

        for path in window_paths:
            current: object = summary
            for part in path:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    current = None
                    break
            if isinstance(current, Mapping):
                window_meta = current.get("meta")
                if isinstance(window_meta, Mapping):
                    default_pct = window_meta.get("default_pct")
                    _add_candidate(default_pct, ".".join((*path, "meta", "default_pct")))
                for key, value in current.items():
                    if key == "meta":
                        continue
                    _add_candidate(value, ".".join((*path, key)))

        if not candidates:
            return None, None

        candidates.sort(key=lambda entry: entry[0], reverse=True)
        return candidates[0]

    def _resolve_risk_per_trade_pct(
        self,
        settings: Settings,
        summary: Optional[Mapping[str, object]],
        performance: Optional[TradePerformanceSnapshot] = None,
    ) -> Tuple[float, Optional[Dict[str, object]]]:
        """Return the effective risk percent per trade with adaptive scaling."""

        try:
            base_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            base_pct = 0.0
        if not math.isfinite(base_pct) or base_pct < 0.0:
            base_pct = 0.0

        probability_value: Optional[float] = None
        probability_source: Optional[str] = None

        if isinstance(summary, Mapping):
            probability_paths: Tuple[Tuple[str, ...], ...] = (
                ("probability",),
                ("probability_pct",),
                ("primary_watch", "probability"),
                ("primary_watch", "probability_pct"),
                ("metrics", "probability"),
                ("metrics", "probability_pct"),
                ("stats", "probability"),
                ("stats", "probability_pct"),
                ("meta", "probability"),
                ("meta", "probability_pct"),
            )

            candidates: List[Tuple[float, str]] = []

            for path in probability_paths:
                current: object = summary
                for part in path:
                    if isinstance(current, Mapping):
                        current = current.get(part)
                    else:
                        current = None
                        break
                if current is None:
                    continue
                candidate = _safe_float(current)
                if candidate is None:
                    continue
                if not math.isfinite(candidate):
                    continue
                normalised = float(candidate)
                if normalised > 1.0:
                    normalised /= 100.0
                if normalised <= 0.0:
                    continue
                if normalised > 1.0:
                    normalised = 1.0
                candidates.append((normalised, ".".join(path)))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                probability_value, probability_source = candidates[0]

        snapshot = performance or getattr(self, "_performance_state", None)
        meta: Optional[Dict[str, object]] = None

        if probability_value is None:
            effective_pct = max(base_pct, 0.0)
            if effective_pct > 0.0:
                meta = {
                    "mode": "static",
                    "effective_pct": effective_pct,
                }
        else:
            probability_scale_pct = 1.5
            min_risk_pct = 0.5
            max_risk_pct = 2.0

            scaled_pct = probability_value * probability_scale_pct
            adaptive_pct = max(min_risk_pct, min(max_risk_pct, scaled_pct))
            effective_pct = adaptive_pct
            meta = {
                "mode": "adaptive",
                "probability": probability_value,
                "probability_source": probability_source,
                "scale_pct": probability_scale_pct,
                "adaptive_pct": scaled_pct,
                "effective_pct": adaptive_pct,
                "min_pct": min_risk_pct,
                "max_pct": max_risk_pct,
            }
            if adaptive_pct != scaled_pct:
                meta["clamped"] = True
            if adaptive_pct == min_risk_pct:
                meta["clamped_to_min"] = min_risk_pct
            elif adaptive_pct == max_risk_pct:
                meta["clamped_to_max"] = max_risk_pct

            if base_pct > 0.0:
                meta["base_pct"] = base_pct
                if base_pct > effective_pct:
                    effective_pct = base_pct
                    meta["effective_pct"] = effective_pct
                    meta["base_applied"] = True

        if snapshot is not None:
            adjustments: Dict[str, object] = {}
            if snapshot.loss_streak >= 5:
                effective_pct *= 0.5
                adjustments["loss_streak"] = snapshot.loss_streak
            elif snapshot.win_streak >= 3:
                effective_pct *= 1.1
                adjustments["win_streak"] = snapshot.win_streak

            if snapshot.sample_count >= 10 and snapshot.average_pnl < 0:
                effective_pct *= 0.85
                adjustments["average_pnl"] = snapshot.average_pnl

            if adjustments:
                meta = meta or {"mode": "adaptive"}
                meta.setdefault("adjustments", {}).update(adjustments)
                meta["effective_pct"] = effective_pct

        if meta is not None and probability_value is not None:
            meta["probability"] = probability_value
            if probability_source:
                meta["probability_source"] = probability_source

        return effective_pct, meta

    def _signal_sizing_factor(
        self, summary: Dict[str, object], settings: Settings
    ) -> float:
        override = _safe_float(summary.get("auto_sizing_factor"))
        if override is not None and override > 0:
            return max(0.05, min(override, 1.0))

        contributions: List[float] = []
        stability_components: List[float] = []

        mode = str(summary.get("mode") or "wait").lower()
        probability = _safe_float(summary.get("probability"))
        buy_threshold = _safe_float(getattr(settings, "ai_buy_threshold", None))
        sell_threshold = _safe_float(getattr(settings, "ai_sell_threshold", None))
        if buy_threshold is None or buy_threshold <= 0:
            buy_threshold = 0.52
        if sell_threshold is None or sell_threshold <= 0:
            sell_threshold = 0.42

        if probability is not None:
            span = 0.25
            if mode == "buy":
                alignment = probability - buy_threshold
            elif mode == "sell":
                alignment = sell_threshold - probability
            else:
                alignment = abs(probability - 0.5) - 0.02
            contributions.append(max(0.0, min(alignment / span, 1.0)))

        ev_bps = _safe_float(summary.get("ev_bps"))
        thresholds = summary.get("thresholds")
        min_ev = resolve_min_ev_from_settings(settings, default_bps=12.0)
        if isinstance(thresholds, dict):
            threshold_override = _safe_float(thresholds.get("min_ev_bps"))
            if threshold_override is not None:
                min_ev = max(threshold_override, 0.0)
        if ev_bps is not None:
            baseline = max(min_ev * 0.9, 4.0)
            span = max(baseline * 1.6, 16.0)
            margin = ev_bps - max(min_ev, baseline)
            contributions.append(max(0.0, min(margin / span, 1.0)))

        primary = summary.get("primary_watch")
        if isinstance(primary, dict):
            edge_score = _safe_float(primary.get("edge_score"))
            if edge_score is not None and edge_score > 0:
                contributions.append(math.tanh(edge_score / 6.0))
                stability_components.append(min(1.0, math.tanh(edge_score / 4.0)))

            def _normalise(value: Optional[float], lower: float, upper: float) -> Optional[float]:
                if value is None:
                    return None
                if value <= lower:
                    return 0.0
                if value >= upper:
                    return 1.0
                return (value - lower) / (upper - lower)

            win_rate = _safe_float(primary.get("win_rate_pct"))
            if win_rate is not None:
                normalised = _normalise(win_rate, 42.0, 65.0)
                if normalised is not None:
                    contributions.append(normalised)
                stability_win = _normalise(win_rate, 50.0, 70.0)
                if stability_win is not None:
                    stability_components.append(stability_win)

            realised = _safe_float(primary.get("realized_bps_avg"))
            if realised is not None:
                normalised = _normalise(realised, -10.0, 20.0)
                if normalised is not None:
                    contributions.append(normalised)
                stability_realised = _normalise(realised, 0.0, 30.0)
                if stability_realised is not None:
                    stability_components.append(stability_realised)

            median_hold = _safe_float(primary.get("median_hold_sec"))
            if median_hold is not None and median_hold > 0:
                fast_floor = 15.0 * 60.0
                slow_ceiling = 2.0 * 60.0 * 60.0
                if median_hold <= fast_floor:
                    contributions.append(1.0)
                elif median_hold >= slow_ceiling:
                    contributions.append(0.0)
                else:
                    span = slow_ceiling - fast_floor
                    contributions.append(1.0 - ((median_hold - fast_floor) / span))
                balanced_floor = 10.0 * 60.0
                balanced_ceiling = 90.0 * 60.0
                if median_hold <= balanced_floor:
                    stability_components.append(0.7)
                elif median_hold >= 3.0 * 60.0 * 60.0:
                    stability_components.append(0.0)
                elif median_hold <= balanced_ceiling:
                    stability_components.append(1.0)
                else:
                    span = (3.0 * 60.0 * 60.0) - balanced_ceiling
                    stability_components.append(
                        max(0.0, 1.0 - ((median_hold - balanced_ceiling) / span))
                    )

        confidence_score = _safe_float(summary.get("confidence_score"))
        if confidence_score is not None:
            contributions.append(max(0.0, min(confidence_score, 1.0)))
            stability_components.append(max(0.0, min(confidence_score, 1.0)))

        stability_score = _safe_float(summary.get("stability_score"))
        if stability_score is None:
            stability_payload = summary.get("automation_health")
            if isinstance(stability_payload, Mapping):
                stability_score = _safe_float(stability_payload.get("stability_score"))
        if stability_score is not None:
            stability_components.append(max(0.0, min(stability_score, 1.0)))

        if contributions:
            average = sum(contributions) / len(contributions)
            floor = 0.3 if summary.get("actionable") else 0.2

            risk_pct, _ = self._resolve_risk_per_trade_pct(
                settings,
                summary,
                performance=self._performance_state,
            )

            risk_dampen = 1.0
            if risk_pct > 0.0:
                risk_dampen = max(0.3, min(1.0, (3.0 - min(risk_pct, 5.0)) / 3.0))

            stability_strength = 0.0
            if stability_components:
                stability_strength = sum(stability_components) / len(stability_components)

            floor += stability_strength * 0.25 * risk_dampen
            floor = min(floor, 0.85)
            headroom = max(0.05, 1.0 - floor)
            strength = min(1.0, average * (0.7 + 0.3 * stability_strength))
            base_factor = floor + headroom * strength
            base_factor += headroom * 0.2 * stability_strength * risk_dampen * average
            base_factor = min(base_factor, 1.0)
        else:
            base_factor = 1.0

        staleness = summary.get("staleness")
        if isinstance(staleness, dict):
            state = str(staleness.get("state") or "").lower()
            if state == "warning":
                base_factor = min(base_factor, 0.6)
            elif state == "stale":
                base_factor = min(base_factor, 0.3)

        return max(0.2, min(base_factor, 1.0))
