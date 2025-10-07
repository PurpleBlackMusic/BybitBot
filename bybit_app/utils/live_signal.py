"""Generate live AI-like signals from real market data."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .envs import Settings, get_api_client, get_settings
from .market_scanner import scan_market_opportunities
from .paths import DATA_DIR


def _parse_symbol_list(raw: object) -> List[str]:
    """Normalise comma/sequence based symbol input."""

    if raw is None:
        return []

    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[object] = raw
    else:
        items = str(raw).split(",")

    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        symbol = text.upper()
        if symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class LiveSignal:
    """Container for the generated live status."""

    payload: Dict[str, object]


class LiveSignalFetcher:
    """Build a Guardian-compatible status snapshot from fresh market data."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        data_dir: Optional[Path] = None,
        cache_ttl: float = 30.0,
        stale_grace: Optional[float] = None,
    ) -> None:
        self._settings: Optional[Settings] = settings
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.live_only = bool(getattr(settings, "ai_live_only", False)) if settings else False
        base_ttl = max(float(cache_ttl), 0.0)
        if self.live_only:
            self.cache_ttl = 0.0
        else:
            self.cache_ttl = base_ttl
        if stale_grace is None:
            # allow a wider reuse window for previously good payloads in case the
            # scanner temporarily fails — never smaller than 15 seconds to avoid
            # aggressive churn when cache_ttl is tiny.
            if self.live_only:
                self.stale_grace = 0.0
            else:
                self.stale_grace = max(self.cache_ttl * 2.0, 15.0)
        else:
            grace = max(float(stale_grace), 0.0)
            self.stale_grace = 0.0 if self.live_only else grace
        self._cached_status: Optional[Dict[str, object]] = None
        self._cache_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # public API
    def fetch(self) -> Dict[str, object]:
        """Return a best-effort live status payload.

        If the market scanner cannot produce any opportunities the function
        returns an empty dict allowing callers to fallback to cached/demo data.
        """

        now = time.time()

        if self._cached_status is not None and self.cache_ttl > 0:
            if now - self._cache_timestamp <= self.cache_ttl:
                return copy.deepcopy(self._cached_status)

        settings = self._settings or get_settings()

        try:
            api = get_api_client()
        except Exception:
            api = None

        opportunities = self._scan_market(settings, api)
        if not opportunities:
            if (
                self._cached_status is not None
                and self.stale_grace > 0
                and not self.live_only
            ):
                if now - self._cache_timestamp <= self.stale_grace:
                    return copy.deepcopy(self._cached_status)
            return {}

        status = self._build_status_from_opportunities(opportunities, settings)
        if status:
            status.setdefault("status_source", "live")
            self._cached_status = copy.deepcopy(status)
            self._cache_timestamp = now
        return status

    # ------------------------------------------------------------------
    # helpers
    def _scan_market(self, settings: Settings, api) -> List[Dict[str, object]]:
        try:
            min_turnover = float(getattr(settings, "ai_min_turnover_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_turnover = 0.0

        try:
            max_spread = float(getattr(settings, "ai_max_spread_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_spread = 0.0

        try:
            min_change_pct = float(getattr(settings, "ai_min_ev_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_change_pct = 0.0

        if min_change_pct <= 0:
            min_change_pct = 0.5
        else:
            min_change_pct /= 100.0
            if min_change_pct < 0.05:
                min_change_pct = 0.05

        whitelist = _parse_symbol_list(getattr(settings, "ai_whitelist", "")) or None
        blacklist = _parse_symbol_list(getattr(settings, "ai_blacklist", "")) or None

        try:
            limit_hint = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        except Exception:
            limit_hint = 0

        if limit_hint <= 0:
            limit_hint = 10
        else:
            limit_hint = min(max(limit_hint * 2, 5), 50)

        try:
            opportunities = scan_market_opportunities(
                api,
                data_dir=self.data_dir,
                limit=limit_hint,
                min_turnover=min_turnover,
                min_change_pct=min_change_pct,
                max_spread_bps=max_spread if max_spread > 0 else 50.0,
                whitelist=whitelist or (),
                blacklist=blacklist or (),
                cache_ttl=0.0 if self.live_only else None,
            )
        except Exception:
            return []

        return opportunities

    def _build_status_from_opportunities(
        self, opportunities: Sequence[Dict[str, object]], settings: Settings
    ) -> Dict[str, object]:
        primary = copy.deepcopy(opportunities[0])
        watchlist = [copy.deepcopy(entry) for entry in opportunities]

        symbol = str(primary.get("symbol") or "BTCUSDT").upper()
        probability = _safe_float(primary.get("probability")) or 0.0
        ev_bps = _safe_float(primary.get("ev_bps")) or 0.0
        change_pct = _safe_float(primary.get("change_pct"))
        turnover = _safe_float(primary.get("turnover_usd"))
        spread_bps = _safe_float(primary.get("spread_bps"))

        trend = str(primary.get("trend") or primary.get("trend_hint") or "").lower()
        if trend not in {"buy", "sell"}:
            if probability >= 0.52:
                trend = "buy"
            elif probability <= 0.48:
                trend = "sell"
            elif change_pct is not None and change_pct < 0:
                trend = "sell"
            else:
                trend = "wait"

        summary_bits: List[str] = []
        if change_pct is not None:
            summary_bits.append(f"24ч изменение ≈ {change_pct:.2f}%")
        if turnover is not None and turnover > 0:
            millions = turnover / 1_000_000.0
            summary_bits.append(f"оборот ≈ ${millions:.2f}M")
        if spread_bps is not None:
            summary_bits.append(f"спред {spread_bps:.2f} б.п.")

        analysis: Optional[str]
        if summary_bits:
            analysis = ", ".join(summary_bits)
        else:
            analysis = None

        note = str(primary.get("note") or "").strip()
        if note:
            if analysis:
                analysis = f"{note}. {analysis}"
            else:
                analysis = note

        now = time.time()

        confidence_pct = probability * 100.0
        confidence_level: str
        if confidence_pct >= 70.0:
            confidence_level = "высокая"
        elif confidence_pct >= 55.0:
            confidence_level = "средняя"
        else:
            confidence_level = "низкая"

        status: Dict[str, object] = {
            "symbol": symbol,
            "probability": probability,
            "ev_bps": ev_bps,
            "side": trend,
            "last_tick_ts": now,
            "generated_ts": now,
            "watchlist": watchlist,
            "source": "live_scanner",
            "confidence_text": (
                f"Уверенность рынка: {confidence_pct:.1f}% — {confidence_level}."
            ),
            "ev_text": f"Оценка выгоды ≈ {ev_bps:.1f} б.п.",
        }

        if analysis:
            status["analysis"] = analysis

        min_ev = max(float(getattr(settings, "ai_min_ev_bps", 0.0) or 0.0), 0.0)
        if min_ev > 0:
            status["risk"] = {
                "min_ev_bps": min_ev,
                "comment": f"Сигнал учитывает минимальную выгоду {min_ev:.1f} б.п.",
            }

        return status

