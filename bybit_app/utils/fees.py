from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

from .log import log


_SUCCESS_CACHE_TTL_SECONDS = 90.0
_ERROR_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class FeeRateSnapshot:
    """Normalised fee rate information returned by the API."""

    maker_rate: Optional[float]
    taker_rate: Optional[float]
    symbol: Optional[str] = None
    base_coin: Optional[str] = None
    category: str = "spot"
    fetched_at: Optional[float] = None
    raw: Mapping[str, object] | None = None

    @property
    def maker_fee_bps(self) -> Optional[float]:
        if self.maker_rate is None:
            return None
        return self.maker_rate * 10_000.0

    @property
    def taker_fee_bps(self) -> Optional[float]:
        if self.taker_rate is None:
            return None
        return self.taker_rate * 10_000.0


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _normalise_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    if isinstance(symbol, str):
        stripped = symbol.strip().upper()
        return stripped or None
    return None


def _normalise_base(base: Optional[str]) -> Optional[str]:
    if not base:
        return None
    if isinstance(base, str):
        stripped = base.strip().upper()
        return stripped or None
    return None


def _extract_entries(payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        entries = result.get("list")
        if isinstance(entries, Sequence):
            return [entry for entry in entries if isinstance(entry, Mapping)]
        entries = result.get("rows")
        if isinstance(entries, Sequence):
            return [entry for entry in entries if isinstance(entry, Mapping)]
    entries = payload.get("list")
    if isinstance(entries, Sequence):
        return [entry for entry in entries if isinstance(entry, Mapping)]
    return []

@dataclass
class _CacheEntry:
    value: Optional[FeeRateSnapshot]
    expires_at: float


_CACHE: dict[Tuple[str, Optional[str], Optional[str]], _CacheEntry] = {}
_CACHE_LOCK = Lock()


def _get_cache_entry(key: Tuple[str, Optional[str], Optional[str]], now: float) -> tuple[bool, Optional[FeeRateSnapshot]]:
    entry = _CACHE.get(key)
    if entry is None:
        return False, None
    if entry.expires_at > now:
        return True, entry.value
    # expired
    del _CACHE[key]
    return False, None


def _set_cache_entry(
    key: Tuple[str, Optional[str], Optional[str]],
    value: Optional[FeeRateSnapshot],
    *,
    ttl: float,
) -> None:
    _CACHE[key] = _CacheEntry(value=value, expires_at=time.time() + ttl)


def _cached_fee_rate(category: str, symbol: Optional[str], base_coin: Optional[str]) -> Optional[FeeRateSnapshot]:
    from .envs import get_api_client  # local import to avoid cycles

    normalised_symbol = _normalise_symbol(symbol)
    normalised_base = _normalise_base(base_coin)

    cache_key = (category, normalised_symbol, normalised_base)
    now = time.time()

    with _CACHE_LOCK:
        found, cached_value = _get_cache_entry(cache_key, now)
    if found:
        return cached_value

    try:
        api = get_api_client()
    except Exception as exc:  # pragma: no cover - defensive logging
        log("fees.api.client_error", err=str(exc))
        with _CACHE_LOCK:
            _set_cache_entry(cache_key, None, ttl=_ERROR_CACHE_TTL_SECONDS)
        return None

    try:
        payload = api.fee_rate(category=category, symbol=normalised_symbol, baseCoin=normalised_base)
    except Exception as exc:  # pragma: no cover - defensive logging
        log(
            "fees.api.request_error",
            err=str(exc),
            category=category,
            symbol=normalised_symbol,
            base=normalised_base,
        )
        with _CACHE_LOCK:
            _set_cache_entry(cache_key, None, ttl=_ERROR_CACHE_TTL_SECONDS)
        return None

    if not isinstance(payload, Mapping):
        with _CACHE_LOCK:
            _set_cache_entry(cache_key, None, ttl=_ERROR_CACHE_TTL_SECONDS)
        return None

    entries = _extract_entries(payload)
    target_symbol = normalised_symbol
    target_base = normalised_base

    selected: Mapping[str, object] | None = None
    for entry in entries:
        entry_symbol = _normalise_symbol(entry.get("symbol"))
        entry_base = _normalise_base(entry.get("baseCoin") or entry.get("baseAsset"))

        if target_symbol and entry_symbol and entry_symbol != target_symbol:
            continue
        if target_base and entry_base and entry_base != target_base:
            continue
        selected = entry
        break

    if selected is None and entries:
        selected = entries[0]

    if selected is None:
        with _CACHE_LOCK:
            _set_cache_entry(cache_key, None, ttl=_ERROR_CACHE_TTL_SECONDS)
        return None

    maker_rate = _safe_float(selected.get("makerFeeRate"))
    taker_rate = _safe_float(selected.get("takerFeeRate"))

    fetched_at = _safe_float(payload.get("time")) or _safe_float(payload.get("ts"))
    if fetched_at is None:
        fetched_at = time.time()

    snapshot = FeeRateSnapshot(
        maker_rate=maker_rate,
        taker_rate=taker_rate,
        symbol=_normalise_symbol(selected.get("symbol")) or normalised_symbol,
        base_coin=_normalise_base(selected.get("baseCoin") or selected.get("baseAsset")) or normalised_base,
        category=category,
        fetched_at=fetched_at,
        raw=selected,
    )

    with _CACHE_LOCK:
        _set_cache_entry(cache_key, snapshot, ttl=_SUCCESS_CACHE_TTL_SECONDS)

    return snapshot


def fee_rate_for_symbol(
    *, category: str = "spot", symbol: Optional[str] = None, base_coin: Optional[str] = None
) -> Optional[FeeRateSnapshot]:
    """Return the cached fee rate snapshot for the given instrument."""

    return _cached_fee_rate(category, symbol, base_coin)


def resolve_fee_rate_bps(
    *,
    category: str = "spot",
    symbol: Optional[str] = None,
    base_coin: Optional[str] = None,
    default_maker_bps: float | None = None,
    default_taker_bps: float | None = None,
) -> tuple[Optional[float], Optional[float], str]:
    """Resolve maker/taker fee rates in basis points with fallbacks."""

    snapshot = fee_rate_for_symbol(category=category, symbol=symbol, base_coin=base_coin)
    if snapshot is not None:
        return snapshot.maker_fee_bps, snapshot.taker_fee_bps, "api"

    return default_maker_bps, default_taker_bps, "fallback"


def clear_fee_rate_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def update_fee_hint(
    original: MutableMapping[str, object],
    *,
    category: str = "spot",
    symbol: Optional[str] = None,
    base_coin: Optional[str] = None,
) -> None:
    """Populate a mapping with fee rate hints when available."""

    snapshot = fee_rate_for_symbol(category=category, symbol=symbol, base_coin=base_coin)
    if snapshot is None:
        return

    maker = snapshot.maker_fee_bps
    taker = snapshot.taker_fee_bps
    if maker is not None:
        original.setdefault("maker_fee_bps", maker)
    if taker is not None:
        original.setdefault("taker_fee_bps", taker)
    original.setdefault("fee_source", "api")
