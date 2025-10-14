
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable

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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _has_allowed_quote(symbol: str, quotes: tuple[str, ...]) -> bool:
    if not quotes:
        return True
    return any(symbol.endswith(quote) for quote in quotes)


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
    symbols: Iterable[str], *, quote_assets: Iterable[str] | None = None
) -> list[str]:
    """Return tradable spot pairs filtered by quote asset and listing status."""

    quoted_only = filter_quote_pairs(symbols, quote_assets)
    if not quoted_only:
        return []

    settings = get_settings()
    is_testnet = getattr(settings, "testnet", True)

    listed = (
        _normalize_symbol(symbol)
        for symbol in filter_listed_spot_symbols(quoted_only, testnet=is_testnet)
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
            data.get("symbols") or [], quote_assets=quote_assets
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
    snapshots: list[LiquiditySnapshot] = []
    for item in rows:
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

        turnover = _safe_float(item.get("turnover24h"))
        bid = _safe_float(item.get("bestBidPrice"))
        ask = _safe_float(item.get("bestAskPrice"))
        if ask <= 0 or bid <= 0 or ask < bid:
            continue

        spread_bps = max(((ask - bid) / ask) * 10_000.0, 0.0)
        if turnover >= float(min_turnover) and spread_bps <= float(max_spread_bps):
            snapshots.append(LiquiditySnapshot(sym, turnover, spread_bps))

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
