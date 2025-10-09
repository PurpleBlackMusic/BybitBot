
from __future__ import annotations

import json
import re
import time
from typing import Iterable

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


def filter_usdt_pairs(symbols: Iterable[str]) -> list[str]:
    return [sym for sym in (_normalize_symbol(raw) for raw in symbols) if sym.endswith("USDT")]


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
    for raw in symbols:
        symbol = _normalize_symbol(raw)
        if not symbol:
            continue
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


def filter_available_spot_pairs(symbols: Iterable[str]) -> list[str]:
    """Return USDT pairs that are currently listed on the exchange."""

    usdt_only = filter_usdt_pairs(symbols)
    if not usdt_only:
        return []

    settings = get_settings()
    is_testnet = getattr(settings, "testnet", True)

    listed = [
        sym
        for sym in (
            _normalize_symbol(symbol)
            for symbol in filter_listed_spot_symbols(usdt_only, testnet=is_testnet)
        )
        if sym
    ]

    filtered_listed = filter_blacklisted_symbols(listed)
    if filtered_listed:
        return filtered_listed

    return filter_blacklisted_symbols(usdt_only)

def build_universe(
    api: BybitAPI,
    size: int = 8,
    min_turnover: float | None = None,
    max_spread_bps: float | None = None,
) -> list[str]:
    min_turnover, max_spread_bps = _resolve_liquidity_filters(min_turnover, max_spread_bps)
    response = api._safe_req("GET", "/v5/market/tickers", params={"category": "spot"})
    rows = (response.get("result") or {}).get("list") or []

    scored: list[tuple[float, str]] = []
    for item in rows:
        sym = _normalize_symbol(item.get("symbol"))
        if not sym.endswith("USDT"):
            continue
        if is_symbol_blacklisted(sym):
            continue

        turnover = _safe_float(item.get("turnover24h"))
        bid = _safe_float(item.get("bestBidPrice"))
        ask = _safe_float(item.get("bestAskPrice"))
        if ask <= 0:
            continue

        spread_bps = ((ask - bid) / ask) * 10_000.0
        if turnover >= float(min_turnover) and spread_bps <= float(max_spread_bps):
            scored.append((turnover, sym))

    scored.sort(key=lambda entry: entry[0], reverse=True)

    ordered_symbols = [symbol for _, symbol in scored]
    filtered_symbols = filter_available_spot_pairs(ordered_symbols)
    top = filtered_symbols[: int(size)] if size else filtered_symbols

    payload = {"ts": int(time.time() * 1000), "symbols": top}
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return top

def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        return []
    try:
        data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
        return filter_available_spot_pairs(data.get("symbols") or [])
    except Exception:
        return []

def apply_universe_to_settings(symbols: list[str]):
    filtered = filter_available_spot_pairs(symbols)
    update_settings(ai_symbols=",".join(filtered))


def liquidity_score(turnover24h: float, spread_bps: float) -> float:
    return float(turnover24h) / max(1.0, (spread_bps + 1.0))

def build_universe_scored(
    api: BybitAPI,
    size: int = 8,
    min_turnover: float | None = None,
    max_spread_bps: float | None = None,
    whitelist: list[str] | None = None,
    blacklist: list[str] | None = None,
) -> list[tuple[str, float]]:
    min_turnover, max_spread_bps = _resolve_liquidity_filters(min_turnover, max_spread_bps)
    response = api._safe_req("GET", "/v5/market/tickers", params={"category": "spot"})
    rows = (response.get("result") or {}).get("list") or []
    scored: list[tuple[str, float]] = []
    whitelist_set = set(filter_usdt_pairs(whitelist or []))
    blacklist_set = {
        symbol
        for symbol in (_normalize_symbol(item) for item in (blacklist or []))
        if symbol
    }

    for item in rows:
        sym = _normalize_symbol(item.get("symbol"))
        if not sym.endswith("USDT"):
            continue
        if sym in blacklist_set:
            continue
        if is_symbol_blacklisted(sym):
            continue

        turnover = _safe_float(item.get("turnover24h"))
        bid = _safe_float(item.get("bestBidPrice"))
        ask = _safe_float(item.get("bestAskPrice"))
        if ask <= 0:
            continue

        spread_bps = ((ask - bid) / ask) * 10_000.0
        if turnover >= float(min_turnover) and spread_bps <= float(max_spread_bps):
            score = liquidity_score(turnover, spread_bps)
            scored.append((sym, score))

    existing_symbols = {symbol for symbol, _ in scored}
    for symbol in whitelist_set - existing_symbols:
        scored.append((symbol, float("inf")))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: int(size)] if size else scored

def auto_rotate_universe(api: BybitAPI, size: int, min_turnover: float, max_spread_bps: float, whitelist: list[str], blacklist: list[str]):
    from .cache_kv import TTLKV
    kv = TTLKV(DATA_DIR / "config" / "universe_kv.json")
    last = kv.get("last_rotate_ts", ttl_sec=None, default=0) or 0
    if time.time() - float(last) < 22*3600:  # не чаще раза в ~сутки
        return None
    top = build_universe_scored(api, size=size, min_turnover=min_turnover, max_spread_bps=max_spread_bps, whitelist=whitelist, blacklist=blacklist)
    syms = filter_available_spot_pairs([s for s, _ in top])
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(json.dumps({"ts": int(time.time()*1000), "symbols": syms}, ensure_ascii=False, indent=2), encoding="utf-8")
    kv.set("last_rotate_ts", time.time())
    return syms
