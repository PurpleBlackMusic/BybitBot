from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .bybit_api import BybitAPI
from .paths import DATA_DIR

SNAPSHOT_FILENAME = "market_snapshot.json"
DEFAULT_CACHE_TTL = 300.0


def _snapshot_path(data_dir: Path) -> Path:
    return Path(data_dir) / "ai" / SNAPSHOT_FILENAME


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


def _probability_from_change(change_pct: Optional[float], trend: str) -> Optional[float]:
    if change_pct is None:
        return None
    strength = _strength_from_change(change_pct)
    if trend == "buy":
        return 0.5 + strength / 2.0
    if trend == "sell":
        return 0.5 - strength / 2.0
    return 0.5


def _score_turnover(turnover: Optional[float]) -> float:
    if turnover is None or turnover <= 0:
        return 0.0
    return math.log10(turnover + 1.0)


def _edge_score(
    turnover: Optional[float],
    change_pct: Optional[float],
    spread_bps: Optional[float],
    *,
    boost: bool = False,
) -> float:
    strength = _strength_from_change(change_pct)
    liquidity = _score_turnover(turnover)
    if spread_bps is not None and spread_bps > 0:
        penalty = math.sqrt(spread_bps)
    else:
        penalty = 1.0
    score = 0.0
    if penalty > 0:
        score = (liquidity * strength) / penalty
    if boost:
        score += 5.0
    return score


def load_market_snapshot(data_dir: Path = DATA_DIR) -> Optional[Dict[str, object]]:
    path = _snapshot_path(data_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_market_snapshot(snapshot: Dict[str, object], data_dir: Path = DATA_DIR) -> None:
    path = _snapshot_path(data_dir)
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
    min_turnover: float = 2_000_000.0,
    min_change_pct: float = 0.5,
    max_spread_bps: float = 35.0,
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
    cache_ttl: float = DEFAULT_CACHE_TTL,
) -> List[Dict[str, object]]:
    """Rank spot symbols by liquidity and momentum to surface opportunities."""

    min_turnover = max(0.0, float(min_turnover))
    effective_change = float(min_change_pct) if min_change_pct is not None else 0.5
    if effective_change < 0.05:
        effective_change = 0.05
    max_spread_bps = float(max_spread_bps)

    snapshot = load_market_snapshot(data_dir)
    now = time.time()
    if snapshot is not None and cache_ttl is not None and cache_ttl >= 0:
        ts = _safe_float(snapshot.get("ts"))
        if ts is not None and now - ts > cache_ttl:
            snapshot = None

    if snapshot is None and api is not None:
        try:
            snapshot = fetch_market_snapshot(api)
        except Exception:
            snapshot = None
        else:
            save_market_snapshot(snapshot, data_dir=data_dir)

    if snapshot is None:
        return []

    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        result = snapshot.get("result")
        if isinstance(result, dict):
            rows = result.get("list")  # type: ignore[assignment]
        else:
            rows = []

    entries: List[Dict[str, object]] = []
    wset = {str(symbol).strip().upper() for symbol in (whitelist or []) if str(symbol).strip()}
    bset = {str(symbol).strip().upper() for symbol in (blacklist or []) if str(symbol).strip()}

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if bset and symbol in bset:
            continue

        turnover = _safe_float(raw.get("turnover24h"))
        change_pct = _normalise_percent(raw.get("price24hPcnt"))
        volume = _safe_float(raw.get("volume24h"))
        bid = _safe_float(raw.get("bestBidPrice"))
        ask = _safe_float(raw.get("bestAskPrice"))
        spread_bps = _spread_bps(bid, ask)

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

        probability = _probability_from_change(change_pct, trend)
        ev_bps = change_pct * 100.0 if change_pct is not None else None
        score = _edge_score(turnover, change_pct, spread_bps, boost=force_include)

        note_parts: List[str] = []
        if change_pct is not None:
            note_parts.append(f"24ч {change_pct:+.2f}%")
        if turnover is not None and turnover > 0:
            note_parts.append(f"оборот ${turnover / 1_000_000:.2f}M")
        if spread_bps is not None:
            note_parts.append(f"спред {spread_bps:.1f} б.п.")

        entry = {
            "symbol": symbol,
            "trend": trend,
            "probability": probability,
            "ev_bps": ev_bps,
            "score": score,
            "note": ", ".join(note_parts) or None,
            "turnover_usd": turnover,
            "change_pct": change_pct,
            "spread_bps": spread_bps,
            "volume": volume,
            "source": "market_scanner",
            "actionable": actionable,
        }
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

