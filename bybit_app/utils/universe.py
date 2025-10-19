
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, TypeVar, Generic, cast

from .bybit_api import BybitAPI
from .envs import get_settings, update_settings
from .instruments import filter_listed_spot_symbols
from .paths import DATA_DIR

UNIVERSE_FILE = DATA_DIR / "config" / "universe.json"

MAINNET_DEFAULT_MIN_TURNOVER = 1_000_000.0
MAINNET_TURNOVER_FLOOR = 500_000.0
MAINNET_DEFAULT_MAX_SPREAD_BPS = 45.0
MAINNET_MAX_SPREAD_CAP = 90.0

TESTNET_DEFAULT_MIN_TURNOVER = 250_000.0
TESTNET_TURNOVER_FLOOR = 50_000.0
TESTNET_DEFAULT_MAX_SPREAD_BPS = 80.0
TESTNET_MAX_SPREAD_CAP = 150.0

DEBUG_WHITELIST = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


@dataclass(frozen=True)
class LiquiditySnapshot:
    symbol: str
    turnover: float
    spread_bps: float
    age_days: Optional[float] = None
    mean_spread_bps: Optional[float] = None
    volatility_rank: Optional[str] = None

_BLACKLIST_PATTERNS = (
    re.compile(r"^BB"),
    re.compile(r"^BULL"),
    re.compile(r"^BEAR"),
    re.compile(r"^[0-9]+[LS]$"),
)


def _normalize_symbol(symbol: str | None) -> str:
    if isinstance(symbol, str):
        return symbol.strip().upper()
    return ""


def _extract_base_asset(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


_MS_PER_DAY = 86_400_000.0
_MIN_LISTING_AGE_DAYS = 90.0
_MEAN_SPREAD_BPS_THRESHOLD = 20.0
_VOLATILITY_LOW_THRESHOLD = 3.0
_VOLATILITY_HIGH_THRESHOLD = 12.0
_INSTRUMENT_META_CACHE: dict[str, tuple[float, Mapping[str, object]]] = {}
_INSTRUMENT_META_CACHE_TTL = 30.0 * 60.0
_INSTRUMENT_META_CACHE_LOCK = threading.RLock()

T = TypeVar("T")


class _SingleFlightSlot(Generic[T]):
    """Internal synchronisation primitive used to dedupe refreshes."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: T | None = None
        self.error: BaseException | None = None


class _SingleFlightGroup(Generic[T]):
    """Single-flight helper ensuring only one refresh operation runs at once."""

    __slots__ = ("_lock", "_inflight")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: Dict[object, _SingleFlightSlot[T]] = {}

    def run(self, key: object, fn: Callable[[], T]) -> T:
        with self._lock:
            slot = self._inflight.get(key)
            if slot is None:
                slot = _SingleFlightSlot[T]()
                self._inflight[key] = slot
                leader = True
            else:
                leader = False

        if not leader:
            slot.event.wait()
            if slot.error is not None:
                raise slot.error
            return cast(T, slot.result)

        try:
            slot.result = fn()
            return slot.result
        except BaseException as exc:  # pragma: no cover - defensive propagation
            slot.error = exc
            raise
        finally:
            slot.event.set()
            with self._lock:
                self._inflight.pop(key, None)


_INSTRUMENT_META_REFRESH = _SingleFlightGroup[Dict[str, Mapping[str, object]]]()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_preserve_order(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)
    return deduped


def _normalize_quote_assets(quotes: Iterable[str] | None) -> tuple[str, ...]:
    if not quotes:
        return ("USDT",)

    normalized = _dedupe_preserve_order(_normalize_symbol(raw) for raw in quotes)
    cleaned = [quote for quote in normalized if quote]
    return tuple(cleaned or ["USDT"])


def _resolve_cache_entries(symbols: set[str]) -> tuple[dict[str, Mapping[str, object]], set[str]]:
    now = time.time()
    cached: dict[str, Mapping[str, object]] = {}
    missing: set[str] = set()
    with _INSTRUMENT_META_CACHE_LOCK:
        for symbol in symbols:
            entry = _INSTRUMENT_META_CACHE.get(symbol)
            if not entry:
                missing.add(symbol)
                continue
            ts, meta = entry
            if _INSTRUMENT_META_CACHE_TTL and now - ts > _INSTRUMENT_META_CACHE_TTL:
                _INSTRUMENT_META_CACHE.pop(symbol, None)
                missing.add(symbol)
                continue
            cached[symbol] = meta
    return cached, missing


def _cache_instrument_metadata(records: Mapping[str, Mapping[str, object]]) -> None:
    now = time.time()
    if not records:
        return
    with _INSTRUMENT_META_CACHE_LOCK:
        for symbol, meta in records.items():
            _INSTRUMENT_META_CACHE[symbol] = (now, meta)


def _fetch_instrument_metadata(
    api: BybitAPI, symbols: Iterable[str]
) -> dict[str, Mapping[str, object]]:
    target = {_normalize_symbol(sym) for sym in symbols if _normalize_symbol(sym)}
    if not target:
        return {}

    cached, missing = _resolve_cache_entries(target)
    if not missing:
        return cached

    pending = set(missing)

    def _refresh() -> dict[str, Mapping[str, object]]:
        fresh: dict[str, Mapping[str, object]] = {}
        payload: Mapping[str, object]

        if hasattr(api, "instruments_info"):
            cursor_token: str | None = None
            while pending:
                try:
                    payload = api.instruments_info(
                        category="spot",
                        status="Trading",
                        limit=200,
                        cursor=cursor_token,
                    )
                except Exception:
                    break

                result = payload.get("result") if isinstance(payload, Mapping) else None
                rows = result.get("list") if isinstance(result, Mapping) else []
                if not isinstance(rows, list):
                    rows = []

                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    symbol = _normalize_symbol(row.get("symbol"))
                    if symbol and symbol in pending:
                        fresh[symbol] = row
                        pending.discard(symbol)

                next_cursor_raw = None
                if isinstance(result, Mapping):
                    next_cursor_raw = result.get("nextPageCursor") or result.get("nextPageToken")
                if next_cursor_raw is None and isinstance(payload, Mapping):
                    next_cursor_raw = payload.get("nextPageCursor") or payload.get("nextPageToken")
                next_cursor = str(next_cursor_raw).strip() if next_cursor_raw else ""
                if not pending or not next_cursor or next_cursor == cursor_token:
                    break
                cursor_token = next_cursor

            for symbol in list(pending):
                try:
                    detail = api.instruments_info(category="spot", symbol=symbol)
                except Exception:
                    continue
                result = detail.get("result") if isinstance(detail, Mapping) else None
                rows = result.get("list") if isinstance(result, Mapping) else []
                if isinstance(rows, list) and rows:
                    row = rows[0]
                    if isinstance(row, Mapping):
                        fresh[symbol] = row
                        pending.discard(symbol)
        else:
            try:
                payload = api._safe_req(
                    "GET",
                    "/v5/market/instruments-info",
                    params={"category": "spot"},
                )
            except Exception:
                rows: list[Mapping[str, object]] = []
            else:
                result = payload.get("result") if isinstance(payload, Mapping) else None
                rows = result.get("list") if isinstance(result, Mapping) else []
                if not isinstance(rows, list):
                    rows = []

            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                symbol = _normalize_symbol(row.get("symbol"))
                if symbol and symbol in pending:
                    fresh[symbol] = row

            remaining = pending - set(fresh.keys())

            for symbol in list(remaining):
                try:
                    detail = api._safe_req(
                        "GET",
                        "/v5/market/instruments-info",
                        params={"category": "spot", "symbol": symbol},
                    )
                except Exception:
                    continue
                result = detail.get("result") if isinstance(detail, Mapping) else None
                rows = result.get("list") if isinstance(result, Mapping) else []
                if isinstance(rows, list) and rows:
                    row = rows[0]
                    if isinstance(row, Mapping):
                        fresh[symbol] = row

        if fresh:
            _cache_instrument_metadata(fresh)

        refreshed, _ = _resolve_cache_entries(target)
        return dict(refreshed)

    refreshed = _INSTRUMENT_META_REFRESH.run("spot_instruments", _refresh)
    merged = dict(cached)
    merged.update(refreshed)
    return merged


def _has_allowed_quote(symbol: str, quotes: tuple[str, ...]) -> bool:
    if not quotes:
        return True
    return any(symbol.endswith(quote) for quote in quotes)


def _resolve_age_days(meta: Mapping[str, object] | None) -> Optional[float]:
    if not meta:
        return None

    direct_keys = ("age_days", "ageDays")
    for key in direct_keys:
        value = _to_float(meta.get(key)) if isinstance(meta, Mapping) else None
        if value is not None and value >= 0:
            return value

    timestamp_keys = (
        "launchTime",
        "createdTime",
        "listTime",
        "created_at",
        "listedAt",
    )
    now_ms = time.time() * 1000.0
    now_sec = time.time()
    for key in timestamp_keys:
        raw = meta.get(key)
        ts = _to_float(raw)
        if ts is None or ts <= 0:
            continue
        if ts > 1_000_000_000_000.0:
            age_ms = max(now_ms - ts, 0.0)
            return age_ms / _MS_PER_DAY
        age_sec = max(now_sec - ts, 0.0)
        return age_sec / 86_400.0
    return None


def _resolve_mean_spread(
    spread_bps: float, meta: Mapping[str, object] | None
) -> Optional[float]:
    if meta:
        for key in (
            "meanSpreadBps",
            "mean_spread_bps",
            "avgSpreadBps",
            "averageSpreadBps",
        ):
            value = meta.get(key)
            spread = _to_float(value)
            if spread is not None:
                return max(spread, 0.0)
    return max(spread_bps, 0.0) if spread_bps is not None else None


def _estimate_volatility_pct(row: Mapping[str, object]) -> Optional[float]:
    high = _to_float(row.get("highPrice24h") or row.get("highPrice"))
    low = _to_float(row.get("lowPrice24h") or row.get("lowPrice"))
    last = _to_float(row.get("lastPrice") or row.get("closePrice"))
    if (
        high is not None
        and low is not None
        and last is not None
        and last > 0.0
        and high >= low
    ):
        return max(((high - low) / last) * 100.0, 0.0)

    change_pct = _to_float(row.get("price24hPcnt"))
    if change_pct is not None:
        # Bybit reports percentage change as a decimal (0.05 for 5%).
        return abs(change_pct) * 100.0

    return None


def _resolve_volatility_rank(
    row: Mapping[str, object], meta: Mapping[str, object] | None
) -> Optional[str]:
    if meta:
        for key in ("volatilityRank", "volatility_rank", "volatilityLevel"):
            raw = meta.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                value = raw.strip().lower()
                if value in {"low", "medium", "mid", "medium_low"}:
                    return "medium" if value in {"medium", "mid"} else value
                if value in {"high", "low"}:
                    return value
            else:
                numeric = _to_float(raw)
                if numeric is not None:
                    if numeric < _VOLATILITY_LOW_THRESHOLD:
                        return "low"
                    if numeric < _VOLATILITY_HIGH_THRESHOLD:
                        return "medium"
                    return "high"

    volatility_pct = _estimate_volatility_pct(row)
    if volatility_pct is None:
        return None
    if volatility_pct < _VOLATILITY_LOW_THRESHOLD:
        return "low"
    if volatility_pct < _VOLATILITY_HIGH_THRESHOLD:
        return "medium"
    return "high"

def filter_quote_pairs(
    symbols: Iterable[str], quote_assets: Iterable[str] | None = None
) -> list[str]:
    quotes = _normalize_quote_assets(quote_assets)
    normalized = (_normalize_symbol(raw) for raw in symbols)
    quoted_only = (sym for sym in normalized if _has_allowed_quote(sym, quotes))
    return _dedupe_preserve_order(quoted_only)


def filter_usdt_pairs(symbols: Iterable[str]) -> list[str]:
    return filter_quote_pairs(symbols, ("USDT",))


def is_symbol_blacklisted(symbol: str) -> bool:
    """Return True when the pair should be excluded from the trading universe."""

    if not symbol:
        return False

    sym = _normalize_symbol(symbol)
    if not sym:
        return False

    base = _extract_base_asset(sym)

    if sym in DEBUG_WHITELIST:
        return False

    for pattern in _BLACKLIST_PATTERNS:
        if pattern.match(base):
            return True
    return False


def filter_blacklisted_symbols(symbols: Iterable[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = _normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        if symbol in DEBUG_WHITELIST or not is_symbol_blacklisted(symbol):
            filtered.append(symbol)
    return filtered


def _resolve_liquidity_filters(
    min_turnover: float | None,
    max_spread_bps: float | None,
):
    settings = get_settings()
    is_testnet = getattr(settings, "testnet", False)

    if is_testnet:
        default_turnover = TESTNET_DEFAULT_MIN_TURNOVER
        turnover_floor = TESTNET_TURNOVER_FLOOR
        default_spread = TESTNET_DEFAULT_MAX_SPREAD_BPS
        spread_cap = TESTNET_MAX_SPREAD_CAP
    else:
        default_turnover = MAINNET_DEFAULT_MIN_TURNOVER
        turnover_floor = MAINNET_TURNOVER_FLOOR
        default_spread = MAINNET_DEFAULT_MAX_SPREAD_BPS
        spread_cap = MAINNET_MAX_SPREAD_CAP

    if min_turnover is None:
        min_turnover = getattr(settings, "ai_min_turnover_usd", default_turnover)
    if max_spread_bps is None:
        max_spread_bps = getattr(settings, "ai_max_spread_bps", default_spread)

    min_turnover = max(float(min_turnover or 0.0), turnover_floor)
    spread_value = float(max_spread_bps or 0.0)
    if spread_value <= 0:
        spread_value = default_spread
    max_spread_bps = min(max(spread_value, 5.0), spread_cap)
    return min_turnover, max_spread_bps


def filter_available_spot_pairs(
    symbols: Iterable[str],
    *,
    quote_assets: Iterable[str] | None = None,
    as_of: float | int | None = None,
) -> list[str]:
    """Return tradable spot pairs filtered by quote asset and listing status."""

    quoted_only = filter_quote_pairs(symbols, quote_assets)
    if not quoted_only:
        return []

    settings = get_settings()
    is_testnet = getattr(settings, "testnet", True)

    listed = (
        _normalize_symbol(symbol)
        for symbol in filter_listed_spot_symbols(
            quoted_only, testnet=is_testnet, as_of=as_of
        )
    )
    filtered_listed = filter_blacklisted_symbols(listed)
    if filtered_listed:
        return filtered_listed

    return filter_blacklisted_symbols(quoted_only)

def build_universe(
    api: BybitAPI,
    size: int = 8,
    min_turnover: float | None = None,
    max_spread_bps: float | None = None,
    quote_assets: Iterable[str] | None = None,
    persist: bool | None = None,
) -> list[str]:
    min_turnover, max_spread_bps = _resolve_liquidity_filters(min_turnover, max_spread_bps)
    scored = build_universe_scored(
        api,
        size=0,
        min_turnover=min_turnover,
        max_spread_bps=max_spread_bps,
        quote_assets=quote_assets,
        score_fn=liquidity_score,
    )

    ordered_symbols = [symbol for symbol, _ in scored]
    filtered_symbols = filter_available_spot_pairs(
        ordered_symbols, quote_assets=quote_assets
    )
    top = filtered_symbols[: int(size)] if size else filtered_symbols

    if persist is None:
        persist = os.environ.get("PYTEST_CURRENT_TEST") is None

    if persist:
        payload = {"ts": int(time.time() * 1000), "symbols": top}
        UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        UNIVERSE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return top

def load_universe(*, quote_assets: Iterable[str] | None = None) -> list[str]:
    if not UNIVERSE_FILE.exists():
        return []
    try:
        data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
        return filter_available_spot_pairs(
            data.get("symbols") or [],
            quote_assets=quote_assets,
            as_of=data.get("ts"),
        )
    except Exception:
        return []

def apply_universe_to_settings(
    symbols: list[str], *, quote_assets: Iterable[str] | None = None
):
    filtered = filter_available_spot_pairs(symbols, quote_assets=quote_assets)
    update_settings(ai_symbols=",".join(filtered))


def liquidity_score(turnover24h: float, spread_bps: float) -> float:
    turnover = max(float(turnover24h), 0.0)
    spread = max(float(spread_bps), 0.0)
    turnover_component = math.log1p(turnover)
    spread_penalty = 1.0 / (1.0 + (spread / 10.0))
    return turnover_component * spread_penalty

def build_universe_scored(
    api: BybitAPI,
    size: int = 8,
    min_turnover: float | None = None,
    max_spread_bps: float | None = None,
    quote_assets: Iterable[str] | None = None,
    whitelist: list[str] | None = None,
    blacklist: list[str] | None = None,
    score_fn: Callable[[float, float], float] | None = None,
) -> list[tuple[str, float]]:
    min_turnover, max_spread_bps = _resolve_liquidity_filters(min_turnover, max_spread_bps)
    response = api._safe_req("GET", "/v5/market/tickers", params={"category": "spot"})
    rows = (response.get("result") or {}).get("list") or []
    score_fn = score_fn or liquidity_score

    quotes = _normalize_quote_assets(quote_assets)

    whitelist_clean = filter_blacklisted_symbols(
        filter_quote_pairs(whitelist or [], quotes)
    )
    whitelist_set = set(whitelist_clean)
    blacklist_set = {
        symbol
        for symbol in (_normalize_symbol(item) for item in (blacklist or []))
        if symbol
    }

    seen: set[str] = set()
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        sym = _normalize_symbol(item.get("symbol"))
        if not sym or sym in seen:
            continue
        seen.add(sym)

        if not _has_allowed_quote(sym, quotes):
            continue
        if sym in blacklist_set:
            continue
        if is_symbol_blacklisted(sym):
            continue

        candidates.append((sym, item))

    instrument_meta = _fetch_instrument_metadata(api, (sym for sym, _ in candidates))

    snapshots: list[LiquiditySnapshot] = []
    for sym, item in candidates:
        turnover = _safe_float(item.get("turnover24h"))
        bid = _safe_float(item.get("bestBidPrice"))
        ask = _safe_float(item.get("bestAskPrice"))
        if ask <= 0 or bid <= 0 or ask < bid:
            continue

        spread_bps = max(((ask - bid) / ask) * 10_000.0, 0.0)

        meta = instrument_meta.get(sym)
        age_days = _resolve_age_days(meta)
        mean_spread = _resolve_mean_spread(spread_bps, meta)
        volatility_rank = _resolve_volatility_rank(item, meta)

        is_whitelisted = sym in whitelist_set

        if not is_whitelisted:
            if age_days is None or age_days <= _MIN_LISTING_AGE_DAYS:
                continue
            if mean_spread is None or mean_spread >= _MEAN_SPREAD_BPS_THRESHOLD:
                continue
            if volatility_rank != "medium":
                continue

        if turnover >= float(min_turnover) and spread_bps <= float(max_spread_bps):
            snapshots.append(
                LiquiditySnapshot(
                    sym,
                    turnover,
                    spread_bps,
                    age_days=age_days,
                    mean_spread_bps=mean_spread,
                    volatility_rank=volatility_rank,
                )
            )

    scored = [
        (snapshot.symbol, float(score_fn(snapshot.turnover, snapshot.spread_bps)))
        for snapshot in snapshots
    ]

    existing_symbols = {symbol for symbol, _ in scored}
    for symbol in whitelist_set - existing_symbols:
        scored.append((symbol, float("inf")))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: int(size)] if size else scored

def auto_rotate_universe(
    api: BybitAPI,
    size: int,
    min_turnover: float,
    max_spread_bps: float,
    whitelist: list[str],
    blacklist: list[str],
    *,
    quote_assets: Iterable[str] | None = None,
) -> list[str] | None:
    from .cache_kv import TTLKV

    kv = TTLKV(DATA_DIR / "config" / "universe_kv.json")
    last = kv.get("last_rotate_ts", ttl_sec=None, default=0) or 0
    if time.time() - float(last) < 22 * 3600:  # не чаще раза в ~сутки
        return None
    top = build_universe_scored(
        api,
        size=size,
        min_turnover=min_turnover,
        max_spread_bps=max_spread_bps,
        quote_assets=quote_assets,
        whitelist=whitelist,
        blacklist=blacklist,
    )
    syms = filter_available_spot_pairs(
        [symbol for symbol, _ in top], quote_assets=quote_assets
    )
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(
        json.dumps({"ts": int(time.time() * 1000), "symbols": syms}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    kv.set("last_rotate_ts", time.time())
    return syms
