
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .envs import get_api_client
from .log import log

try:  # pragma: no cover - imported for typing/IDE support
    from .bybit_api import BybitAPI
except Exception:  # pragma: no cover - avoid circular imports at runtime
    BybitAPI = object  # type: ignore[assignment]

_DEFAULT_STATUS_TTL = 180.0
_STATUS_CACHE: Dict[Tuple[str, str], "ListingStatusSnapshot"] = {}
_MAINTENANCE_TOKENS = {
    "halt",
    "suspend",
    "maintenance",
    "pause",
    "downtime",
    "upgrade",
    "migration",
}
_DELIST_TOKENS = {
    "delist",
    "delisted",
    "delisting",
    "terminate",
    "terminated",
    "sunset",
    "end-of-life",
    "offline",
}


@dataclass(frozen=True)
class ListingStatusSnapshot:
    """Cached instrument status snapshot returned by :func:`classify_listing_rows`."""

    timestamp: float
    trading: frozenset[str]
    maintenance: frozenset[str]
    delisted: frozenset[str]
    statuses: Mapping[str, str]

    def status(self, symbol: object) -> Optional[str]:
        cleaned = _clean_symbol(symbol)
        if not cleaned:
            return None
        return self.statuses.get(cleaned)

    def is_tradeable(self, symbol: object) -> bool:
        cleaned = _clean_symbol(symbol)
        if not cleaned:
            return False
        if cleaned in self.delisted or cleaned in self.maintenance:
            return False
        if not self.trading:
            return True
        return cleaned in self.trading

    def maintenance_symbols(self) -> Set[str]:
        return set(self.maintenance)

    def delisted_symbols(self) -> Set[str]:
        return set(self.delisted)


def _cache_key(api: "BybitAPI", category: str) -> Tuple[str, str]:
    marker = "unknown"
    try:
        creds = getattr(api, "creds", None)
        if creds is not None and getattr(creds, "testnet", False):
            marker = "testnet"
        elif creds is not None:
            marker = "mainnet"
    except Exception:  # pragma: no cover - defensive against unexpected attrs
        marker = "unknown"
    return marker, category


def _extract_rows(payload: object) -> Sequence[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, Mapping):
            rows = result.get("list")
            if isinstance(rows, Sequence):
                return [row for row in rows if isinstance(row, Mapping)]
        rows = payload.get("list")
        if isinstance(rows, Sequence):
            return [row for row in rows if isinstance(row, Mapping)]
        return []
    if isinstance(payload, Sequence):
        return [row for row in payload if isinstance(row, Mapping)]
    return []


def _clean_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _classify_status(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered or lowered in {"trading", "listed", "active"}:
        return "trading"
    for token in _DELIST_TOKENS:
        if token in lowered:
            return "delisted"
    for token in _MAINTENANCE_TOKENS:
        if token in lowered:
            return "maintenance"
    # Treat unknown statuses as maintenance stops to keep the bot on the safe side.
    return "maintenance"


def classify_listing_rows(
    rows: Sequence[Mapping[str, object]], *, timestamp: Optional[float] = None
) -> ListingStatusSnapshot:
    trading: Set[str] = set()
    maintenance: Set[str] = set()
    delisted: Set[str] = set()
    statuses: Dict[str, str] = {}

    for row in rows:
        symbol = _clean_symbol(row.get("symbol"))
        if not symbol:
            continue
        status_text = str(row.get("status") or "").strip()
        statuses[symbol] = status_text
        category = _classify_status(status_text)
        if category == "delisted":
            delisted.add(symbol)
        elif category == "maintenance":
            maintenance.add(symbol)
        else:
            trading.add(symbol)

    snapshot = ListingStatusSnapshot(
        timestamp=time.time() if timestamp is None else float(timestamp),
        trading=frozenset(trading),
        maintenance=frozenset(maintenance),
        delisted=frozenset(delisted),
        statuses=dict(statuses),
    )
    return snapshot


def get_listing_status_snapshot(
    api: "BybitAPI",
    *,
    category: str = "spot",
    ttl: float = _DEFAULT_STATUS_TTL,
) -> ListingStatusSnapshot:
    ttl = max(float(ttl), 0.0)
    key = _cache_key(api, category)
    snapshot = _STATUS_CACHE.get(key)
    now = time.time()
    if snapshot is not None and ttl > 0 and now - snapshot.timestamp <= ttl:
        return snapshot

    try:
        response = api.instruments_info(category=category)
    except Exception as exc:  # pragma: no cover - defensive network guard
        log("listing_guard.status_fetch_failed", err=str(exc), category=category)
        if snapshot is not None:
            return snapshot
        raise

    rows = _extract_rows(response)
    snapshot = classify_listing_rows(rows, timestamp=now)
    _STATUS_CACHE[key] = snapshot
    return snapshot


def maintenance_symbols(
    api: "BybitAPI", *, category: str = "spot", ttl: float = _DEFAULT_STATUS_TTL
) -> Set[str]:
    snapshot = get_listing_status_snapshot(api, category=category, ttl=ttl)
    return snapshot.maintenance_symbols()


def delisted_symbols(
    api: "BybitAPI", *, category: str = "spot", ttl: float = _DEFAULT_STATUS_TTL
) -> Set[str]:
    snapshot = get_listing_status_snapshot(api, category=category, ttl=ttl)
    return snapshot.delisted_symbols()


def is_symbol_tradeable(
    api: "BybitAPI", symbol: object, *, category: str = "spot", ttl: float = _DEFAULT_STATUS_TTL
) -> bool:
    snapshot = get_listing_status_snapshot(api, category=category, ttl=ttl)
    return snapshot.is_tradeable(symbol)


def stop_reason(
    api: "BybitAPI", symbol: object, *, category: str = "spot", ttl: float = _DEFAULT_STATUS_TTL
) -> Optional[str]:
    snapshot = get_listing_status_snapshot(api, category=category, ttl=ttl)
    return snapshot.status(symbol)


def is_recently_listed(symbol: str, minutes: int = 5) -> bool:
    api = get_api_client()
    response = api._safe_req(
        "GET",
        "/v5/market/instruments-info",
        params={"category": "spot", "symbol": symbol},
    )
    rows = _extract_rows(response)
    if not rows:
        return False
    first = rows[0]
    ts_value = first.get("launchTime") or first.get("createdTime")
    try:
        launch_ms = int(ts_value)
    except (TypeError, ValueError):
        return False
    if launch_ms <= 0:
        return False
    return (int(time.time() * 1000) - launch_ms) < minutes * 60 * 1000
