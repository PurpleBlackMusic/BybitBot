from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation
from math import ceil
import time
from threading import RLock
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar
import re

from .bybit_api import BybitAPI
from .log import log
from . import validators
from .envs import Settings
from .precision import ceil_qty_to_min_notional, format_to_step

_MIN_QUOTE = Decimal("5")
_PRICE_CACHE_TTL = 5.0
_BALANCE_CACHE_TTL = 5.0
_INSTRUMENT_CACHE_TTL = 600.0
_INSTRUMENT_DYNAMIC_TTL = 30.0
_SYMBOL_CACHE_TTL = 300.0
_ORDERBOOK_LIMIT = 200
_DEFAULT_MARK_DEVIATION = Decimal("0.01")
_TWAP_DEFAULT_MAX_SLICES = 10
_TOLERANCE_MARGIN = Decimal("0.00000001")

_BYBIT_ERROR = re.compile(r"Bybit error (?P<code>-?\d+): (?P<message>.+)")
_PRICE_LIMIT_FIELDS = re.compile(
    r"(?P<key>price[\s_-]?(?:cap|floor)|max[\s_-]?price|min[\s_-]?price|"
    r"upper[\s_-]?(?:limit|price)|lower[\s_-]?(?:limit|price)|priceLimit(?:Upper|Lower)?)"
    r"\s*(?:[:=]|is|<=|>=)?\s*\(?\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

_PRICE_LIMIT_COMPARISONS = re.compile(
    r"""
    (?:\b(?:buy|sell|your|the)\b\s*)?
    (?:order\s+)?
    price
    \s+cannot\s+be\s+
    (?P<comparison>higher|greater|lower|less)
    \s+than\s+
    (?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \s*(?P<currency>[A-Za-z]{2,10})?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRICE_LIMIT_KEY_ALIASES = {
    "price_cap": {"pricecap", "maxprice", "upperlimit", "upperprice", "pricelimitupper"},
    "price_floor": {"pricefloor", "minprice", "lowerlimit", "lowerprice", "pricelimitlower"},
}


T = TypeVar("T")


class TTLCache(Generic[T]):
    """A minimal, thread-safe TTL cache for repeated API lookups."""

    __slots__ = ("_ttl", "_store", "_lock")

    def __init__(self, ttl: float):
        self._ttl = max(float(ttl), 0.0)
        self._store: Dict[str, Tuple[float, T]] = {}
        self._lock = RLock()

    def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            ts, value = entry
            if self._ttl and now - ts > self._ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_INSTRUMENT_CACHE: TTLCache[Dict[str, object]] = TTLCache(_INSTRUMENT_CACHE_TTL)
_PRICE_CACHE: TTLCache[Decimal] = TTLCache(_PRICE_CACHE_TTL)
_BALANCE_CACHE: TTLCache[Dict[str, Decimal]] = TTLCache(_BALANCE_CACHE_TTL)
_SYMBOL_CACHE: TTLCache[Dict[str, object]] = TTLCache(_SYMBOL_CACHE_TTL)


def _network_label(api: BybitAPI) -> str:
    """Return the active network label for the supplied API instance."""

    creds = getattr(api, "creds", None)
    if creds is None:
        return "unknown"

    testnet_flag = getattr(creds, "testnet", None)
    if testnet_flag is None:
        return "unknown"

    return "testnet" if bool(testnet_flag) else "mainnet"


def _balance_cache_key(api: BybitAPI, account_type: str) -> str:
    network = _network_label(api)
    account = (account_type or "UNIFIED").upper() or "UNIFIED"
    return f"{network}:{account}"


def _balance_payload_cache_key(api: BybitAPI, account_type: str) -> str:
    return f"{_balance_cache_key(api, account_type)}#payload:{id(api)}"


def _symbol_cache_key(api: BybitAPI) -> str:
    """Return a cache key that incorporates the API's active network."""

    network = _network_label(api)

    return f"spot_usdt:{network}"


def _price_cache_key(api: BybitAPI, symbol: str) -> str:
    """Return a network-aware cache key for latest price lookups."""

    network = _network_label(api)
    normalised_symbol = (symbol or "").upper()
    return f"{network}:{normalised_symbol}"


def _instrument_cache_key(api: BybitAPI, symbol: str) -> str:
    """Return a cache key for instrument metadata scoped to the API network."""

    network = _network_label(api)
    return f"{network}:{(symbol or '').upper()}"


class OrderValidationError(RuntimeError):
    """Raised when a trade request violates exchange constraints."""

    def __init__(self, message: str, *, code: str, details: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"message": str(self), "code": self.code}
        if self.details:
            payload["details"] = self.details
        return payload


def _extract_bybit_error_code_from_message(message: str) -> Optional[str]:
    match = _BYBIT_ERROR.search(message)
    if match:
        return match.group("code")
    return None


def _normalise_price_limit_key(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^a-z]", "", raw.lower())
    if not cleaned:
        return None
    for target, aliases in _PRICE_LIMIT_KEY_ALIASES.items():
        if cleaned in aliases:
            return target
    if cleaned in ("pricecap", "pricelimit"):
        return "price_cap"
    if cleaned == "pricefloor":
        return "price_floor"
    return None


def parse_price_limit_error_details(message: str) -> Dict[str, str]:
    """Extract price limit hints from a Bybit error message."""

    hints: Dict[str, tuple[int, str]] = {}
    if not message:
        return {}

    for match in _PRICE_LIMIT_FIELDS.finditer(message):
        raw_key = match.group("key")
        value = match.group("value")
        if not raw_key or not value:
            continue
        key = _normalise_price_limit_key(raw_key)
        if not key:
            continue
        hints[key] = (0, value)

    for match in _PRICE_LIMIT_COMPARISONS.finditer(message):
        comparison = match.group("comparison")
        value = match.group("value")
        if not comparison or not value:
            continue
        comparison_lower = comparison.lower()
        if comparison_lower in ("higher", "greater"):
            key = "price_cap"
        elif comparison_lower in ("lower", "less"):
            key = "price_floor"
        else:
            continue

        previous = hints.get(key)
        if previous is None or previous[0] < 1:
            hints[key] = (1, value)

    return {key: value for key, (_priority, value) in hints.items()}

_WALLET_AVAILABLE_FIELDS = (
    "totalAvailableBalance",
    "availableToWithdraw",
    "availableBalance",
    "available",
    "availableMargin",
    "free",
    "transferBalance",
    "cashBalance",
    "availableFunds",
)
_WALLET_FALLBACK_FIELDS = ("walletBalance",)
_WALLET_SYMBOL_FIELDS = ("coin", "asset", "currency")
_WALLET_RESERVED_FIELDS = (
    "orderMargin",
    "order_margin",
    "totalOrderIM",
    "totalOrderMargin",
    "locked",
    "lockedBalance",
    "frozenBalance",
    "frozen",
    "serviceCash",
    "commission",
    "pendingCommission",
    "pendingFee",
)
_KNOWN_QUOTES = (
    "USDT",
    "USDC",
    "USDD",
    "BUSD",
    "DAI",
    "USD",
    "EUR",
    "BTC",
    "ETH",
    "JPY",
)


def _to_decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _round_up(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    multiplier = (value / step).to_integral_value(rounding=ROUND_UP)
    return multiplier * step


def _split_symbol(symbol: str) -> Tuple[str, str]:
    upper = (symbol or "").upper()
    for quote in _KNOWN_QUOTES:
        if upper.endswith(quote):
            base = upper[: -len(quote)] or upper
            return base, quote
    return upper, ""


def _normalise_symbol_input(symbol: object) -> Optional[str]:
    if isinstance(symbol, str):
        cleaned = symbol.strip().upper()
        if not cleaned:
            return None
        for separator in (" ", "-", "_", "/", ":"):
            if separator in cleaned:
                cleaned = cleaned.replace(separator, "")
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
    return None


_LEVERAGE_SUFFIXES = (
    "3L",
    "3S",
    "4L",
    "4S",
    "5L",
    "5S",
    "8L",
    "8S",
    "10L",
    "10S",
)


def _symbol_rank(symbol: str, base_coin: str) -> Tuple[int, int, int, str]:
    normalised = symbol.upper()
    canonical = f"{base_coin}USDT" if base_coin else normalised
    exact_match = 0 if normalised == canonical else 1
    core = normalised[:-4] if normalised.endswith("USDT") else normalised
    leverage_penalty = 0
    for suffix in _LEVERAGE_SUFFIXES:
        if core.endswith(suffix):
            leverage_penalty = 1
            break
    length = len(normalised)
    return exact_match, leverage_penalty, length, normalised


def _fetch_spot_instruments(api: BybitAPI, *, limit: int = 500) -> List[Dict[str, object]]:
    cursor: Optional[str] = None
    instruments: List[Dict[str, object]] = []
    seen_cursors: set[str] = set()

    while True:
        try:
            response = api.instruments_info(
                category="spot",
                limit=limit,
                cursor=cursor,
            )
        except Exception as exc:  # pragma: no cover - network/runtime errors
            raise RuntimeError(f"Не удалось получить список инструментов: {exc}") from exc

        result = response.get("result") if isinstance(response, dict) else None
        rows = result.get("list") if isinstance(result, dict) else None
        page = [row for row in rows or [] if isinstance(row, dict)]
        instruments.extend(page)

        next_cursor = result.get("nextPageCursor") if isinstance(result, dict) else None
        if next_cursor is None:
            break
        cursor = str(next_cursor).strip()
        if not cursor:
            break
        if cursor in seen_cursors:  # defensive guard against API loops
            break
        seen_cursors.add(cursor)

    return instruments


def _tradable_spot_usdt_universe(
    api: BybitAPI, *, force_refresh: bool = False
) -> Dict[str, object]:
    cache_key = _symbol_cache_key(api)
    cached = None if force_refresh else _SYMBOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    entries = _fetch_spot_instruments(api)
    tradable: Dict[str, Dict[str, object]] = {}
    by_base: Dict[str, List[Tuple[Tuple[int, int, int, str], str]]] = {}
    alias_sources: Dict[str, set[str]] = {}
    alias_to_base: Dict[str, str] = {}

    for entry in entries:
        symbol = _normalise_symbol_input(entry.get("symbol"))
        if not symbol:
            continue
        quote_coin = str(entry.get("quoteCoin") or "").upper()
        status = str(entry.get("status") or "").strip().lower()
        if quote_coin != "USDT":
            continue
        if status and status != "trading":
            continue
        tradable[symbol] = entry
        base_coin = str(
            entry.get("baseCoin")
            or entry.get("baseAsset")
            or entry.get("baseCurrency")
            or ""
        ).upper()
        split_base, _ = _split_symbol(symbol)
        primary_base = base_coin or split_base

        if primary_base:
            rank = _symbol_rank(symbol, primary_base)
            by_base.setdefault(primary_base, []).append((rank, symbol))

        alias_keys: set[str] = set()
        if base_coin:
            alias_keys.add(base_coin)
        if split_base:
            alias_keys.add(split_base)
        alias_field = entry.get("alias")
        if isinstance(alias_field, str) and alias_field.strip():
            alias_cleaned = _normalise_symbol_input(alias_field) or alias_field.strip().upper()
            if alias_cleaned:
                alias_keys.add(alias_cleaned)

        for alias_key in alias_keys:
            alias_to_base.setdefault(alias_key, primary_base or alias_key)
            alias_sources.setdefault(alias_key, set()).add(symbol)

    ordered_by_base: Dict[str, List[str]] = {}
    for base_key, options in by_base.items():
        options.sort(key=lambda item: item[0])
        seen: set[str] = set()
        ordered: List[str] = []
        for _, candidate in options:
            if candidate in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate)
        ordered_by_base[base_key] = ordered

    alias_map: Dict[str, List[str]] = {}
    for alias_key, candidates in alias_sources.items():
        base_key = alias_to_base.get(alias_key)
        ordered: List[str] = []
        if base_key and base_key in ordered_by_base:
            for candidate in ordered_by_base[base_key]:
                if candidate in candidates:
                    ordered.append(candidate)
        if not ordered:
            ordered = sorted(candidates)
        alias_map[alias_key] = ordered

    for base_key, ordered in ordered_by_base.items():
        alias_map.setdefault(base_key, list(ordered))

    payload: Dict[str, object] = {
        "symbols": tradable,
        "by_base": ordered_by_base,
        "aliases": alias_map,
        "ts": time.time(),
    }
    _SYMBOL_CACHE.set(cache_key, payload)
    return payload


def resolve_trade_symbol(
    symbol: object,
    *,
    api: BybitAPI,
    allow_nearest: bool = True,
    force_refresh: bool = False,
) -> Tuple[Optional[str], Dict[str, object]]:
    cleaned = _normalise_symbol_input(symbol)
    if not cleaned:
        return None, {"reason": "empty"}

    def _attempt(
        universe_data: Mapping[str, object] | None,
        *,
        cache_state: str,
    ) -> Tuple[Optional[str], Dict[str, object]]:
        tradable = universe_data.get("symbols") if isinstance(universe_data, Mapping) else {}
        alias_map = universe_data.get("aliases") if isinstance(universe_data, Mapping) else {}
        base_map = universe_data.get("by_base") if isinstance(universe_data, Mapping) else {}

        if isinstance(tradable, Mapping) and cleaned in tradable:
            return cleaned, {"reason": "exact", "symbol": cleaned, "cache_state": cache_state}

        def _match_alias(alias_key: str, *, requested_quote: str | None = None):
            if not alias_key or not isinstance(alias_map, Mapping):
                return None
            options = alias_map.get(alias_key)
            if not isinstance(options, list) or not options:
                return None
            resolved_symbol = options[0]
            meta: Dict[str, object] = {
                "reason": "alias_match",
                "requested": cleaned,
                "resolved": resolved_symbol,
                "alias": alias_key,
                "cache_state": cache_state,
            }
            if requested_quote and requested_quote != "USDT":
                meta["requested_quote"] = requested_quote
            return resolved_symbol, meta

        alias_direct = _match_alias(cleaned)
        if alias_direct:
            return alias_direct

        if not allow_nearest:
            return None, {"reason": "not_listed", "requested": cleaned, "cache_state": cache_state}

        base, quote = _split_symbol(cleaned)
        alias_base = _match_alias(base, requested_quote=quote) if base else None
        if alias_base:
            return alias_base

        if not isinstance(base_map, Mapping) or not base:
            return None, {"reason": "not_listed", "requested": cleaned, "cache_state": cache_state}

        candidates = base_map.get(base)
        if isinstance(candidates, list) and candidates:
            resolved = candidates[0]
            meta: Dict[str, object] = {
                "reason": "base_match",
                "requested": cleaned,
                "resolved": resolved,
                "cache_state": cache_state,
            }
            if quote and quote != "USDT":
                meta["requested_quote"] = quote
            return resolved, meta

        return None, {"reason": "not_listed", "requested": cleaned, "cache_state": cache_state}

    universe = _tradable_spot_usdt_universe(api, force_refresh=force_refresh)
    cache_state = "refreshed" if force_refresh else "cached"
    resolved, meta = _attempt(universe if isinstance(universe, Mapping) else None, cache_state=cache_state)
    if resolved or force_refresh:
        return resolved, meta

    refreshed_universe = _tradable_spot_usdt_universe(api, force_refresh=True)
    resolved, meta = _attempt(refreshed_universe if isinstance(refreshed_universe, Mapping) else None, cache_state="refreshed")
    return resolved, meta


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    multiplier = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return multiplier * step


def _format_step_decimal(value: Decimal, step: Decimal) -> str:
    return format_to_step(value, step, rounding=ROUND_DOWN)


def _normalise_orderbook_levels(
    levels: Sequence[Sequence[object]], *, descending: bool
) -> list[tuple[Decimal, Decimal]]:
    normalised: list[tuple[Decimal, Decimal]] = []
    for entry in levels or []:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        price = _to_decimal(entry[0])
        qty = _to_decimal(entry[1])
        if price <= 0 or qty <= 0:
            continue
        normalised.append((price, qty))
    normalised.sort(key=lambda level: level[0], reverse=descending)
    return normalised


def _apply_tick(
    price: Decimal,
    tick_size: Decimal,
    side: str,
    *,
    max_price_allowed: Decimal | None = None,
    min_price_allowed: Decimal | None = None,
) -> Decimal:
    if tick_size <= 0:
        adjusted = price
    else:
        side_normalised = (side or "").lower()
        rounding = ROUND_UP if side_normalised == "buy" else ROUND_DOWN
        multiplier = (price / tick_size).to_integral_value(rounding=rounding)
        adjusted = multiplier * tick_size
        if adjusted <= 0:
            adjusted = tick_size

    if max_price_allowed is not None:
        if adjusted > max_price_allowed:
            if tick_size > 0:
                multiplier = (max_price_allowed / tick_size).to_integral_value(rounding=ROUND_DOWN)
                adjusted = multiplier * tick_size
                if adjusted <= 0:
                    adjusted = tick_size
            else:
                adjusted = max_price_allowed
        if adjusted > max_price_allowed:
            adjusted = max_price_allowed

    if min_price_allowed is not None:
        if adjusted < min_price_allowed:
            if tick_size > 0:
                multiplier = (min_price_allowed / tick_size).to_integral_value(rounding=ROUND_UP)
                adjusted = multiplier * tick_size
                if adjusted <= 0:
                    adjusted = tick_size
            else:
                adjusted = min_price_allowed
        if adjusted < min_price_allowed:
            adjusted = min_price_allowed

    return adjusted


def _orderbook_snapshot(api: BybitAPI, symbol: str) -> tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]:
    try:
        response = api.orderbook(category="spot", symbol=symbol, limit=_ORDERBOOK_LIMIT)
    except Exception as exc:  # pragma: no cover - network/runtime errors
        raise RuntimeError(f"Не удалось получить стакан по {symbol}: {exc}") from exc

    result = response.get("result") if isinstance(response, Mapping) else None
    asks_raw = []
    bids_raw = []
    if isinstance(result, Mapping):
        asks_raw = result.get("a") or []
        bids_raw = result.get("b") or []

    asks = _normalise_orderbook_levels(asks_raw, descending=False)
    bids = _normalise_orderbook_levels(bids_raw, descending=True)
    return asks, bids


def _synthetic_orderbook_levels(
    price: Decimal,
    *,
    min_amount: Decimal,
    min_qty: Decimal,
    qty_step: Decimal,
    target_quote: Decimal | None,
    target_base: Decimal | None,
) -> list[tuple[Decimal, Decimal]]:
    if price <= 0:
        return []

    qty_candidate = Decimal("1")
    if target_base is not None and target_base > 0:
        qty_candidate = max(qty_candidate, target_base)
    if target_quote is not None and target_quote > 0 and price > 0:
        qty_from_quote = target_quote / price
        if qty_from_quote > qty_candidate:
            qty_candidate = qty_from_quote
    if min_amount > 0 and price > 0:
        qty_from_min_notional = min_amount / price
        if qty_from_min_notional > qty_candidate:
            qty_candidate = qty_from_min_notional
    if min_qty > 0 and min_qty > qty_candidate:
        qty_candidate = min_qty

    if qty_step > 0:
        qty_candidate = _round_up(qty_candidate, qty_step)
    if qty_candidate <= 0:
        qty_candidate = qty_step if qty_step > 0 else Decimal("0.00000001")

    return [(price, qty_candidate)]


def _plan_limit_ioc_order(
    *,
    asks: Sequence[tuple[Decimal, Decimal]],
    bids: Sequence[tuple[Decimal, Decimal]],
    side: str,
    target_quote: Decimal | None,
    target_base: Decimal | None,
    qty_step: Decimal,
    min_qty: Decimal,
    max_price: Decimal | None = None,
    min_price: Decimal | None = None,
) -> tuple[Decimal, Decimal, Decimal, list[tuple[Decimal, Decimal]]]:
    side_normalised = side
    levels = asks if side_normalised == "buy" else bids
    if not levels:
        raise OrderValidationError(
            "Стакан пуст — нет доступной ликвидности.",
            code="orderbook_empty",
            details={"side": side_normalised},
        )

    consumed: list[tuple[Decimal, Decimal]] = []
    accumulated_base = Decimal("0")
    accumulated_quote = Decimal("0")
    worst_price = levels[0][0]

    if target_base is not None and target_base <= 0:
        raise OrderValidationError(
            "Количество должно быть положительным.",
            code="qty_invalid",
        )

    if target_quote is not None and target_quote <= 0:
        raise OrderValidationError(
            "Количество должно быть положительным.",
            code="qty_invalid",
        )

    price_limit_hit = False

    for price, qty in levels:
        if side_normalised == "buy" and max_price is not None:
            if price - max_price > _TOLERANCE_MARGIN:
                price_limit_hit = True
                break
            if max_price - price <= _TOLERANCE_MARGIN:
                price_limit_hit = True
        if side_normalised == "sell" and min_price is not None:
            if min_price - price > _TOLERANCE_MARGIN:
                price_limit_hit = True
                break
            if price - min_price <= _TOLERANCE_MARGIN:
                price_limit_hit = True

        worst_price = price
        if target_base is not None:
            remaining_base = target_base - accumulated_base
            if remaining_base <= 0:
                break
            take_base = qty if qty <= remaining_base else remaining_base
            accumulated_base += take_base
            accumulated_quote += take_base * price
            consumed.append((price, take_base))
            if accumulated_base >= target_base:
                break
        else:
            remaining_quote = target_quote - accumulated_quote if target_quote is not None else Decimal("0")
            if remaining_quote <= 0:
                break
            level_quote = qty * price
            if level_quote >= remaining_quote:
                take_base = remaining_quote / price
                accumulated_base += take_base
                accumulated_quote += remaining_quote
                consumed.append((price, take_base))
                break
            accumulated_base += qty
            accumulated_quote += level_quote
            consumed.append((price, qty))

    if target_base is not None and accumulated_base < target_base:
        raise OrderValidationError(
            "Недостаточная глубина стакана для заданного количества.",
            code="insufficient_liquidity",
            details={
                "requested_base": _format_decimal(target_base),
                "available_base": _format_decimal(accumulated_base),
                "side": side_normalised,
                **(
                    {
                        "price_cap": _format_decimal(max_price),
                        "price_limit_hit": True,
                    }
                    if price_limit_hit and side_normalised == "buy" and max_price is not None
                    else {}
                ),
                **(
                    {
                        "price_floor": _format_decimal(min_price),
                        "price_limit_hit": True,
                    }
                    if price_limit_hit and side_normalised == "sell" and min_price is not None
                    else {}
                ),
            },
        )

    if target_quote is not None and accumulated_quote < target_quote:
        raise OrderValidationError(
            "Недостаточная глубина стакана для заданного объёма в котировочной валюте.",
            code="insufficient_liquidity",
            details={
                "requested_quote": _format_decimal(target_quote),
                "available_quote": _format_decimal(accumulated_quote),
                "side": side_normalised,
                **(
                    {
                        "price_cap": _format_decimal(max_price),
                        "price_limit_hit": True,
                    }
                    if price_limit_hit and side_normalised == "buy" and max_price is not None
                    else {}
                ),
                **(
                    {
                        "price_floor": _format_decimal(min_price),
                        "price_limit_hit": True,
                    }
                    if price_limit_hit and side_normalised == "sell" and min_price is not None
                    else {}
                ),
            },
        )

    rounding_fn = _round_up
    if side_normalised != "buy" or target_quote is not None:
        rounding_fn = _round_down

    qty_rounded = rounding_fn(accumulated_base, qty_step)

    if qty_rounded <= 0:
        raise OrderValidationError(
            "Количество меньше минимального шага для базовой валюты.",
            code="qty_step",
            details={"requested": _format_decimal(accumulated_base), "step": _format_decimal(qty_step)},
        )

    if min_qty > 0 and qty_rounded < min_qty:
        raise OrderValidationError(
            "Количество меньше минимального лота для базовой валюты.",
            code="min_qty",
            details={
                "requested": _format_decimal(accumulated_base),
                "rounded": _format_decimal(qty_rounded),
                "min_qty": _format_decimal(min_qty),
                "step": _format_decimal(qty_step),
            },
        )

    quote_total = qty_rounded * worst_price
    return worst_price, qty_rounded, quote_total, consumed


def _max_mark_price(
    price: Decimal | None,
    *,
    side: str,
    deviation: Decimal | None = None,
) -> tuple[Decimal | None, Decimal | None]:
    if price is None or price <= 0:
        return None, None

    effective_deviation = deviation if deviation and deviation > 0 else _DEFAULT_MARK_DEVIATION

    if side == "buy":
        return price * (Decimal("1") + effective_deviation), None
    return None, price * (Decimal("1") - effective_deviation)


def _extract_order_fields(payload: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("result"), Mapping):
        return payload["result"]  # type: ignore[index]
    if isinstance(payload.get("body"), Mapping):
        return payload["body"]  # type: ignore[index]
    return payload


def _execution_stats(
    response: Mapping[str, object] | None,
    *,
    expected_qty: Decimal,
    limit_price: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    if not isinstance(response, Mapping):
        return Decimal("0"), Decimal("0"), expected_qty

    fields = _extract_order_fields(response)
    order_qty = _to_decimal(fields.get("orderQty") or fields.get("qty") or expected_qty)
    leaves_qty = _to_decimal(fields.get("leavesQty"))
    cum_exec_qty = _to_decimal(fields.get("cumExecQty"))
    cum_exec_value = _to_decimal(fields.get("cumExecValue"))
    avg_price = _to_decimal(fields.get("avgPrice"))

    executed_qty = Decimal("0")
    if cum_exec_qty > 0:
        executed_qty = cum_exec_qty
    elif order_qty > 0 and leaves_qty >= 0:
        executed_qty = order_qty - leaves_qty

    executed_quote = Decimal("0")
    if cum_exec_value > 0:
        executed_quote = cum_exec_value
    elif executed_qty > 0:
        ref_price = avg_price if avg_price > 0 else limit_price
        executed_quote = executed_qty * ref_price

    remaining_qty = leaves_qty if leaves_qty > 0 else max(order_qty - executed_qty, Decimal("0"))

    return executed_qty, executed_quote, remaining_qty


def _instrument_limits(api: BybitAPI, symbol: str) -> Dict[str, object]:
    key = symbol.upper()
    cache_key = _instrument_cache_key(api, key)
    cached = _INSTRUMENT_CACHE.get(cache_key)
    now = time.time()
    if cached is not None:
        dynamic_ts = cached.get("_dynamic_ts") if isinstance(cached, dict) else None
        try:
            ts_value = float(dynamic_ts) if dynamic_ts is not None else None
        except (TypeError, ValueError):
            ts_value = None
        if ts_value is not None and now - ts_value < _INSTRUMENT_DYNAMIC_TTL:
            return cached

    try:
        response = api.instruments_info(category="spot", symbol=key)
    except Exception as exc:  # pragma: no cover - network/runtime errors
        raise RuntimeError(f"Не удалось получить правила для {key}: {exc}") from exc

    result = (response or {}).get("result") if isinstance(response, dict) else None
    entries = []
    if isinstance(result, dict):
        entries = result.get("list") or []
    elif isinstance(result, list):  # pragma: no cover - defensive
        entries = result

    instrument = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("symbol") or "").upper() == key:
            instrument = entry
            break
    if instrument is None and entries:
        instrument = entries[0]

    if instrument is None:
        raise OrderValidationError(
            f"Не найдены данные об инструменте {key}",
            code="instrument_missing",
            details={"symbol": key},
        )

    status = str(instrument.get("status") or "").strip().lower()
    if status and status != "trading":
        raise OrderValidationError(
            f"Инструмент {key} недоступен для торговли (status={status}).",
            code="instrument_inactive",
            details={"status": status, "symbol": key},
        )

    lot = instrument.get("lotSizeFilter") or {}
    min_amount = _to_decimal(lot.get("minOrderAmt") or lot.get("minNotional") or lot.get("minOrderAmtValue"))
    quote_step = _to_decimal(lot.get("minOrderAmtIncrement") or lot.get("quotePrecision") or "0.01", Decimal("0.01"))
    if quote_step <= 0:
        quote_step = Decimal("0.01")

    min_qty = _to_decimal(
        lot.get("minOrderQty")
        or lot.get("basePrecision")
        or lot.get("minOrderQtyValue")
        or "0"
    )
    qty_step = _to_decimal(
        lot.get("qtyStep")
        or lot.get("stepSize")
        or lot.get("minOrderQtyIncrement")
        or lot.get("basePrecision")
        or "0.00000001",
        Decimal("0.00000001"),
    )
    if qty_step <= 0:
        qty_step = Decimal("0.00000001")

    base_coin = str(
        instrument.get("baseCoin")
        or instrument.get("baseAsset")
        or instrument.get("baseCurrency")
        or ""
    ).upper()
    quote_coin = str(
        instrument.get("quoteCoin")
        or instrument.get("quoteAsset")
        or instrument.get("settleCoin")
        or ""
    ).upper()

    if not quote_coin:
        guessed_base, guessed_quote = _split_symbol(key)
        if guessed_quote:
            quote_coin = guessed_quote
        if not base_coin and guessed_base:
            base_coin = guessed_base

    price_filter = instrument.get("priceFilter") or {}
    tick_size = _to_decimal(price_filter.get("tickSize") or price_filter.get("tick_size") or "0")
    if tick_size <= 0:
        tick_size = Decimal("0")
    min_price = _to_decimal(price_filter.get("minPrice") or price_filter.get("min_price") or "0")
    max_price = _to_decimal(price_filter.get("maxPrice") or price_filter.get("max_price") or "0")

    limits: Dict[str, object] = {
        "min_order_amt": max(min_amount, _MIN_QUOTE),
        "quote_step": quote_step,
        "min_order_qty": min_qty,
        "qty_step": qty_step,
        "base_coin": base_coin,
        "quote_coin": quote_coin,
        "tick_size": tick_size,
        "min_price": min_price,
        "max_price": max_price,
        "status": status,
    }
    limits["_instrument"] = instrument
    limits["_dynamic_ts"] = now
    _INSTRUMENT_CACHE.set(cache_key, limits)
    return limits


def _latest_price(api: BybitAPI, symbol: str) -> Decimal:
    key = (symbol or "").upper()
    cache_key = _price_cache_key(api, key)
    cached = _PRICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        response = api.tickers(category="spot", symbol=key)
    except Exception as exc:  # pragma: no cover - network/runtime errors
        raise RuntimeError(f"Не удалось получить котировку для {key}: {exc}") from exc

    rows = []
    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, dict):
            rows = result.get("list") or []  # type: ignore[assignment]
        elif isinstance(response.get("list"), list):
            rows = response.get("list")  # type: ignore[assignment]

    price = Decimal("0")
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        entry_symbol = str(entry.get("symbol") or key).upper()
        if entry_symbol and entry_symbol != key:
            continue
        for field in ("markPrice", "bestAskPrice", "bestBidPrice", "lastPrice"):
            candidate = _to_decimal(entry.get(field))
            if candidate > 0:
                price = candidate
                break
        if price > 0:
            break

    if price <= 0:
        raise OrderValidationError(
            f"Биржа не вернула котировку для {key}",
            code="price_missing",
            details={"symbol": key},
        )

    _PRICE_CACHE.set(cache_key, price)
    return price


def _collect_error_metadata(payload: object, *, _seen: set[int] | None = None) -> Tuple[set[str], set[str]]:
    if _seen is None:
        _seen = set()
    if payload is None:
        return set(), set()

    payload_id = id(payload)
    if payload_id in _seen:
        return set(), set()
    _seen.add(payload_id)

    codes: set[str] = set()
    messages: set[str] = set()

    if isinstance(payload, BaseException):
        messages.add(str(payload))
        for name in ("retCode", "ret_code", "code", "status_code"):
            value = getattr(payload, name, None)
            if value is not None:
                codes.add(str(value))
        for name in ("retMsg", "ret_msg", "msg", "message", "error", "error_msg"):
            value = getattr(payload, name, None)
            if value:
                messages.add(str(value))
        args = getattr(payload, "args", ())
        for arg in args:
            sub_codes, sub_messages = _collect_error_metadata(arg, _seen=_seen)
            codes.update(sub_codes)
            messages.update(sub_messages)
        return codes, messages

    if isinstance(payload, Mapping):
        for name in ("retCode", "ret_code", "code", "status_code"):
            if name in payload and payload[name] is not None:
                codes.add(str(payload[name]))
        for name in ("retMsg", "ret_msg", "msg", "message", "error", "error_msg"):
            if name in payload and payload[name]:
                messages.add(str(payload[name]))
        for value in payload.values():
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
                sub_codes, sub_messages = _collect_error_metadata(value, _seen=_seen)
                codes.update(sub_codes)
                messages.update(sub_messages)
        return codes, messages

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            sub_codes, sub_messages = _collect_error_metadata(item, _seen=_seen)
            codes.update(sub_codes)
            messages.update(sub_messages)
        return codes, messages

    messages.add(str(payload))
    return codes, messages


def _has_account_type_only_support_unified_marker(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    compact = lowered.replace(" ", "")
    if "accounttype" not in compact and "account type" not in lowered:
        return False

    if "only support unified" in lowered or "only supports unified" in lowered:
        return True

    return "onlysupportunified" in compact or "onlysupportsunified" in compact


def _looks_like_http_status_code(value: str) -> bool:
    stripped = value.strip()
    if not stripped.isdigit():
        return False
    if len(stripped) != 3:
        return False
    number = int(stripped)
    return 100 <= number <= 599


def _is_unsupported_wallet_account_type_error(error: object) -> bool:
    codes, messages = _collect_error_metadata(error)

    normalised_codes = {str(code).strip() for code in codes if str(code).strip()}
    has_unsupported_code = any(code == "10001" for code in normalised_codes)
    has_marker = False
    for message in messages:
        if not message:
            continue
        normalised = message.lower()
        if "10001" in normalised:
            has_unsupported_code = True
        if _has_account_type_only_support_unified_marker(message):
            has_marker = True
            if has_unsupported_code:
                return True

    if not has_marker:
        return False

    if has_unsupported_code:
        return True

    if not normalised_codes:
        return True

    if all(_looks_like_http_status_code(code) for code in normalised_codes):
        return True

    return False


def _extract_available_amount(row: Mapping[str, object]) -> Decimal | None:
    """Pick the most tradable balance value from a wallet row."""

    available_positive: Decimal | None = None
    available_non_positive: Decimal | None = None
    fallback_positive: Decimal | None = None
    fallback_non_positive: Decimal | None = None

    for field in _WALLET_AVAILABLE_FIELDS:
        candidate = row.get(field)
        if candidate is None:
            continue
        amount = _to_decimal(candidate)
        if amount > 0:
            if available_positive is None or amount > available_positive:
                available_positive = amount
        elif available_non_positive is None or amount > available_non_positive:
            available_non_positive = amount

    for field in _WALLET_FALLBACK_FIELDS:
        candidate = row.get(field)
        if candidate is None:
            continue
        amount = _to_decimal(candidate)
        if amount > 0:
            if fallback_positive is None or amount > fallback_positive:
                fallback_positive = amount
        elif fallback_non_positive is None or amount > fallback_non_positive:
            fallback_non_positive = amount

    reserved_total = Decimal("0")
    for field in _WALLET_RESERVED_FIELDS:
        candidate = row.get(field)
        if candidate is None:
            continue
        amount = _to_decimal(candidate)
        if amount > 0:
            reserved_total += amount

    fallback_net_positive: Decimal | None = None
    fallback_net_non_positive: Decimal | None = fallback_non_positive

    if fallback_positive is not None:
        net = fallback_positive - reserved_total
        if net > 0:
            fallback_net_positive = net
        else:
            fallback_net_non_positive = net
    elif fallback_non_positive is not None and reserved_total > 0:
        net = fallback_non_positive - reserved_total
        if fallback_net_non_positive is None or net > fallback_net_non_positive:
            fallback_net_non_positive = net

    positive_candidates = [
        amount
        for amount in (available_positive, fallback_net_positive)
        if amount is not None and amount > 0
    ]
    if positive_candidates:
        return max(positive_candidates)

    non_positive_candidates = [
        amount
        for amount in (available_non_positive, fallback_net_non_positive)
        if amount is not None
    ]
    if non_positive_candidates:
        return max(non_positive_candidates)

    return None


def _iter_wallet_rows(payload: object, *, _seen: Optional[set[int]] = None) -> List[Mapping[str, object]]:
    if _seen is None:
        _seen = set()

    rows: List[Mapping[str, object]] = []

    if isinstance(payload, Mapping):
        payload_id = id(payload)
        if payload_id in _seen:
            return rows
        _seen.add(payload_id)

        symbol_value: Optional[object] = None
        for field in _WALLET_SYMBOL_FIELDS:
            symbol_value = payload.get(field)
            if isinstance(symbol_value, str) and symbol_value.strip():
                rows.append(payload)  # payload already contains symbol + balances
                break

        containers: List[object] = []
        for key in ("coin", "coins", "details", "assets", "list"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                containers.extend(value)

        for value in payload.values():
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
                containers.append(value)

        for item in containers:
            rows.extend(_iter_wallet_rows(item, _seen=_seen))
        return rows

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            rows.extend(_iter_wallet_rows(item, _seen=_seen))

    return rows


def _parse_wallet_balances(payload: object) -> Dict[str, Decimal]:
    balances: Dict[str, Decimal] = {}

    for row in _iter_wallet_rows(payload):
        symbol = None
        for field in _WALLET_SYMBOL_FIELDS:
            raw_symbol = row.get(field)
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                symbol = raw_symbol.strip().upper()
                break
        if not symbol:
            continue

        available = _extract_available_amount(row)
        if available is None:
            continue

        balances[symbol] = balances.get(symbol, Decimal("0")) + available

    return balances


def _load_spot_exchange_balances(api: BybitAPI) -> Dict[str, Decimal]:
    loader = getattr(api, "asset_exchange_query_asset_info", None)
    try:
        if callable(loader):
            payload = loader()
        else:
            requester = getattr(api, "_safe_req", None)
            if not callable(requester):  # pragma: no cover - defensive
                return {}
            payload = requester(
                "GET",
                "/v5/asset/exchange/query-asset-info",
                params=None,
                body=None,
                signed=True,
            )
    except Exception as exc:  # pragma: no cover - network/runtime errors
        log(
            "wallet_balance_spot_exchange_fallback_error",
            error=str(exc),
        )
        return {}

    return _parse_wallet_balances(payload)


def _load_wallet_balances(api: BybitAPI, account_type: str) -> Dict[str, Decimal]:
    try:
        payload = api.wallet_balance(accountType=account_type)
    except Exception as exc:  # pragma: no cover - network/runtime errors
        # Bybit v5 no longer supports non-unified wallet lookups and returns
        # error 10001 when ``accountType`` is anything other than ``UNIFIED``.
        # Treat this as an empty response instead of surfacing an exception so
        # that guard flows gracefully fall back to the unified wallet balances.
        if account_type and account_type.upper() != "UNIFIED":
            message = str(exc)
            if _is_unsupported_wallet_account_type_error(exc):
                log(
                    "wallet_balance_unsupported_account_type",
                    account_type=account_type,
                    error=message,
                )
                fallback_balances = _load_spot_exchange_balances(api)
                if fallback_balances:
                    log(
                        "wallet_balance_spot_exchange_fallback_used",
                        account_type=account_type,
                        coins=len(fallback_balances),
                    )
                return fallback_balances
        raise RuntimeError(f"Не удалось получить баланс кошелька: {exc}") from exc

    return _parse_wallet_balances(payload)


def _wallet_available_balances(
    api: BybitAPI,
    account_type: str = "UNIFIED",
    *,
    required_asset: str | None = None,
) -> Dict[str, Decimal]:
    account_key = (account_type or "UNIFIED").upper() or "UNIFIED"
    required_normalised = (required_asset or "").strip().upper()
    cache_key = _balance_cache_key(api, account_key)

    cached = _BALANCE_CACHE.get(cache_key)
    if cached is not None:
        if required_normalised:
            cached_amount = cached.get(required_normalised, Decimal("0"))
            if cached_amount is None or cached_amount <= 0:
                if cached:
                    _BALANCE_CACHE.invalidate(cache_key)
                    _BALANCE_CACHE.invalidate(_balance_payload_cache_key(api, account_key))
                else:
                    return cached
            else:
                return cached
        else:
            return cached

    primary_balances = _load_wallet_balances(api, account_type=account_key)

    combined = dict(primary_balances)
    # Users often keep spot funds on a dedicated SPOT account while trading
    # through a unified account.  When the unified wallet response does not
    # expose these assets we perform a transparent fallback to the SPOT
    # account and merge the balances so the guard sees the available funds.
    needs_fallback = False
    if account_key in {"UNIFIED", "TRADE"}:
        if not primary_balances:
            needs_fallback = True
        else:
            needs_fallback = all(amount <= 0 for amount in primary_balances.values())
            if not needs_fallback and required_normalised:
                amount = primary_balances.get(required_normalised)
                if amount is None or amount <= 0:
                    needs_fallback = True
    if needs_fallback:
        spot_cache_key = _balance_cache_key(api, "SPOT")
        spot_cached = _BALANCE_CACHE.get(spot_cache_key)
        if spot_cached is None:
            spot_cached = _load_wallet_balances(api, account_type="SPOT")
            if not spot_cached:
                spot_cached = _load_spot_exchange_balances(api)
            _BALANCE_CACHE.set(spot_cache_key, dict(spot_cached))
        elif not spot_cached:
            exchange_balances = _load_spot_exchange_balances(api)
            if exchange_balances:
                spot_cached = exchange_balances
                _BALANCE_CACHE.set(spot_cache_key, dict(exchange_balances))
        for asset, amount in (spot_cached or {}).items():
            combined[asset] = combined.get(asset, Decimal("0")) + amount

    _BALANCE_CACHE.set(cache_key, combined)
    return combined


def _normalise_balances(balances: Mapping[str, object] | None) -> Dict[str, Decimal] | None:
    if balances is None:
        return None

    normalised: Dict[str, Decimal] = {}
    for asset, amount in balances.items():
        if not isinstance(asset, str):
            continue
        symbol = asset.strip().upper()
        if not symbol:
            continue
        normalised[symbol] = _to_decimal(amount)
    return normalised


def _format_decimal(value: Decimal) -> str:
    """Render Decimal without scientific notation for logging/messages."""

    try:
        normalised = value.normalize()
    except InvalidOperation:  # pragma: no cover - defensive branch
        normalised = Decimal("0")

    text = format(normalised, "f")
    return text if text else "0"


def _build_wallet_payload_from_balances(
    account_type: str, balances: Mapping[str, Decimal]
) -> Dict[str, object]:
    coins: List[Dict[str, object]] = []
    for asset, amount in sorted(balances.items()):
        value = _format_decimal(amount)
        coins.append(
            {
                "coin": asset,
                "walletBalance": value,
                "equity": value,
                "balance": value,
                "availableBalance": value,
                "availableToWithdraw": value,
                "available": value,
                "availableMargin": value,
                "cashBalance": value,
                "transferBalance": value,
                "totalAvailableBalance": value,
            }
        )

    return {
        "result": {
            "list": [
                {
                    "accountType": account_type,
                    "account": account_type,
                    "coin": coins,
                }
            ]
        }
    }


@dataclass(frozen=True)
class SpotTradeSnapshot:
    """Reusable container for cached spot trading resources."""

    symbol: str
    price: Decimal | None = None
    balances: Dict[str, Decimal] | None = None
    limits: Mapping[str, object] | None = None

    def as_kwargs(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if self.price is not None:
            payload["price_snapshot"] = self.price
        if self.balances is not None:
            payload["balances"] = self.balances
        if self.limits is not None:
            payload["limits"] = self.limits
        return payload


@dataclass(frozen=True)
class PreparedSpotMarketOrder:
    """Normalised market order payload accompanied by validation metadata."""

    payload: Dict[str, object]
    audit: Dict[str, object]


@dataclass(frozen=True)
class TWAPRuntimeConfig:
    """Runtime configuration extracted from :class:`Settings`."""

    enabled: bool
    base_slices: int
    max_slices: int
    interval_range: Tuple[int, int]
    aggressiveness_bps: float


def _twap_runtime_config(settings: Settings | None) -> TWAPRuntimeConfig:
    if not isinstance(settings, Settings):
        return TWAPRuntimeConfig(False, 1, 1, (0, 0), 0.0)

    enabled = bool(getattr(settings, "twap_enabled", False))
    base_slices_raw = getattr(settings, "twap_slices", 0) or 0
    try:
        base_slices = int(base_slices_raw)
    except Exception:  # pragma: no cover - defensive
        base_slices = 0
    if base_slices <= 0:
        base_slices = 1

    max_slices_raw = getattr(settings, "twap_slices_max", None)
    try:
        max_slices_candidate = int(max_slices_raw) if max_slices_raw is not None else 0
    except Exception:  # pragma: no cover - defensive
        max_slices_candidate = 0
    if max_slices_candidate <= 0:
        max_slices_candidate = _TWAP_DEFAULT_MAX_SLICES
    max_slices = max(base_slices, max_slices_candidate)

    interval_base_raw = getattr(settings, "twap_interval_sec", 0) or 0
    try:
        interval_min = int(interval_base_raw)
    except Exception:  # pragma: no cover - defensive
        interval_min = 0
    if interval_min <= 0:
        child_raw = getattr(settings, "twap_child_secs", 0) or 0
        try:
            interval_min = int(child_raw)
        except Exception:  # pragma: no cover - defensive
            interval_min = 0
    if interval_min <= 0:
        interval_min = 5

    interval_max_raw = getattr(settings, "twap_interval_max_sec", None)
    try:
        interval_max_candidate = int(interval_max_raw) if interval_max_raw is not None else 0
    except Exception:  # pragma: no cover - defensive
        interval_max_candidate = 0
    if interval_max_candidate <= 0:
        interval_max_candidate = max(interval_min, 10)
    interval_max = max(interval_min, interval_max_candidate)

    aggressiveness_raw = getattr(settings, "twap_aggressiveness_bps", 0.0) or 0.0
    try:
        aggressiveness = float(aggressiveness_raw)
    except Exception:  # pragma: no cover - defensive
        aggressiveness = 0.0
    if aggressiveness <= 0:
        aggressiveness = 20.0

    return TWAPRuntimeConfig(enabled, base_slices, max_slices, (interval_min, interval_max), aggressiveness)


def _twap_price_deviation_ratio(details: Mapping[str, object] | None) -> Decimal | None:
    if not isinstance(details, Mapping):
        return None

    limit_value = details.get("limit_price")
    if limit_value is None:
        return None

    limit_price = _to_decimal(limit_value)
    if limit_price <= 0:
        return None

    max_allowed_value = details.get("max_allowed")
    if max_allowed_value is not None:
        max_allowed = _to_decimal(max_allowed_value)
        if max_allowed > 0:
            ratio = limit_price / max_allowed
            return ratio if ratio > 1 else None

    price_cap_value = details.get("price_cap")
    if price_cap_value is not None:
        price_cap = _to_decimal(price_cap_value)
        if price_cap > 0:
            ratio = limit_price / price_cap
            return ratio if ratio > 1 else None

    min_allowed_value = details.get("min_allowed")
    if min_allowed_value is not None:
        min_allowed = _to_decimal(min_allowed_value)
        if min_allowed > 0:
            ratio = min_allowed / limit_price
            return ratio if ratio > 1 else None

    price_floor_value = details.get("price_floor")
    if price_floor_value is not None:
        price_floor = _to_decimal(price_floor_value)
        if price_floor > 0:
            ratio = price_floor / limit_price
            return ratio if ratio > 1 else None

    return None


def _twap_scaled_slices(current: int, ratio: Decimal | None, max_slices: int) -> int:
    if not isinstance(ratio, Decimal) or ratio <= 1:
        return current

    candidate = current + 1
    scaled = Decimal(current) * ratio
    try:
        candidate = max(candidate, int(scaled.to_integral_value(rounding=ROUND_UP)))
    except InvalidOperation:  # pragma: no cover - defensive
        candidate = max(candidate, int(ceil(float(scaled))))
    candidate = max(current, candidate)
    return min(max_slices, candidate)


def prepare_spot_trade_snapshot(
    api: BybitAPI,
    symbol: str,
    *,
    account_type: str = "UNIFIED",
    include_limits: bool = True,
    include_price: bool = True,
    include_balances: bool = True,
    force_refresh: bool = False,
) -> SpotTradeSnapshot:
    """Fetch reusable inputs for a subsequent spot market order."""

    key = symbol.upper()
    instrument_cache_key = _instrument_cache_key(api, key)

    limits: Mapping[str, object] | None = None
    if include_limits:
        if force_refresh:
            _INSTRUMENT_CACHE.invalidate(instrument_cache_key)
        limits = _instrument_limits(api, key)

    price: Decimal | None = None
    if include_price:
        if force_refresh:
            _PRICE_CACHE.invalidate(_price_cache_key(api, key))
        price = _latest_price(api, key)

    balances: Dict[str, Decimal] | None = None
    if include_balances:
        account_key = (account_type or "UNIFIED").upper() or "UNIFIED"
        if force_refresh:
            cache_key = _balance_cache_key(api, account_key)
            _BALANCE_CACHE.invalidate(cache_key)
            _BALANCE_CACHE.invalidate(_balance_payload_cache_key(api, account_key))
        balances = _wallet_available_balances(api, account_type=account_type)

    return SpotTradeSnapshot(symbol=key, price=price, balances=balances, limits=limits)


def wallet_available_balances(
    api: BybitAPI,
    account_type: str = "UNIFIED",
    *,
    required_asset: str | None = None,
) -> Dict[str, Decimal]:
    """Public wrapper for cached wallet balance lookups with graceful fallbacks."""

    return _wallet_available_balances(
        api,
        account_type=account_type,
        required_asset=required_asset,
    )


def wallet_balance_payload(api: BybitAPI, account_type: str = "UNIFIED") -> Dict[str, object]:
    """Return the raw wallet payload, applying spot fallbacks for unsupported accounts."""

    account_key = (account_type or "UNIFIED").upper() or "UNIFIED"
    cache_key = _balance_payload_cache_key(api, account_key)
    cached = _BALANCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        payload = api.wallet_balance(accountType=account_key)
    except TypeError as exc:
        message = str(exc)
        if "accountType" in message or "unexpected" in message:
            payload = api.wallet_balance()
        else:  # pragma: no cover - propagate unrelated type errors
            raise
    except Exception as exc:  # pragma: no cover - network/runtime errors
        if account_key != "UNIFIED" and _is_unsupported_wallet_account_type_error(exc):
            message = str(exc)
            log(
                "wallet_balance_unsupported_account_type",
                account_type=account_key,
                error=message,
            )
            fallback_balances = _load_spot_exchange_balances(api)
            if fallback_balances:
                log(
                    "wallet_balance_spot_exchange_fallback_used",
                    account_type=account_key,
                    coins=len(fallback_balances),
                )
            payload_map = _build_wallet_payload_from_balances(account_key, fallback_balances)
            _BALANCE_CACHE.set(cache_key, dict(payload_map))
            return payload_map
        if account_key == "UNIFIED":
            raise
        raise RuntimeError(f"Не удалось получить баланс кошелька: {exc}") from exc

    if isinstance(payload, Mapping):
        payload_map = dict(payload)
    elif payload is None:
        payload_map = {}
    else:
        payload_map = {"result": payload}

    _BALANCE_CACHE.set(cache_key, dict(payload_map))
    return payload_map


_MIN_PERCENT_TOLERANCE = Decimal("0.05")
_MAX_PERCENT_TOLERANCE = Decimal("5.0")
_MIN_BPS_TOLERANCE = Decimal("5")
_MAX_BPS_TOLERANCE = Decimal("500")


def _resolve_slippage_tolerance(
    tol_type: str | None,
    tol_value: object,
) -> tuple[Decimal, str, str, Decimal]:
    """Normalise slippage tolerance inputs for request and balance guards."""

    auto_type: str | None = None
    raw_value = tol_value
    if isinstance(tol_value, str):
        text = tol_value.strip()
        lowered = text.lower()
        if lowered.endswith("%"):
            auto_type = "percent"
            text = text[:-1]
        elif lowered.endswith("bps"):
            auto_type = "bps"
            text = text[:-3]
        elif lowered.endswith("bp"):
            auto_type = "bps"
            text = text[:-2]
        raw_value = text

    tolerance_decimal = _to_decimal(raw_value, Decimal("0"))
    if tolerance_decimal < 0:
        tolerance_decimal = Decimal("0")

    tolerance_kind = (tol_type or auto_type or "percent").strip().lower()

    multiplier = Decimal("1")
    request_type = "Percent"
    if tolerance_kind in {"percent", "percentage"}:
        if tolerance_decimal > 0:
            if tolerance_decimal < _MIN_PERCENT_TOLERANCE:
                tolerance_decimal = _MIN_PERCENT_TOLERANCE
            elif tolerance_decimal > _MAX_PERCENT_TOLERANCE:
                tolerance_decimal = _MAX_PERCENT_TOLERANCE
        multiplier += tolerance_decimal / Decimal("100")
        request_type = "Percent"
    elif tolerance_kind in {"bps", "basispoints", "basis_points"}:
        if tolerance_decimal > 0:
            if tolerance_decimal < _MIN_BPS_TOLERANCE:
                tolerance_decimal = _MIN_BPS_TOLERANCE
            elif tolerance_decimal > _MAX_BPS_TOLERANCE:
                tolerance_decimal = _MAX_BPS_TOLERANCE
        multiplier += tolerance_decimal / Decimal("10000")
        request_type = "Bps"
    else:
        multiplier += tolerance_decimal
        request_type = "Value"

    if multiplier <= 0:
        multiplier = Decimal("1")

    request_value = format(
        tolerance_decimal.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
        "f",
    )

    return multiplier, request_type, request_value, tolerance_decimal


def prepare_spot_market_order(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: Decimal | float | int | str,
    unit: str = "quoteCoin",
    tol_type: str = "Percent",
    tol_value: float = 0.5,
    max_quote: object | None = None,
    *,
    price_snapshot: object | None = None,
    balances: Mapping[str, object] | None = None,
    limits: Mapping[str, object] | None = None,
    settings: Settings | None = None,
):
    """Проверить параметры маркет-ордера и подготовить запрос к REST API."""

    limit_map = limits if limits is not None else _instrument_limits(api, symbol)
    instrument_raw = limit_map.get("_instrument") if isinstance(limit_map, Mapping) else None
    if instrument_raw is None and isinstance(limit_map, Mapping):
        lot_filter = {
            "qtyStep": str(limit_map.get("qty_step") or limit_map.get("qtyStep") or "0"),
            "minOrderQty": str(limit_map.get("min_order_qty") or limit_map.get("minQty") or "0"),
            "minNotional": str(limit_map.get("min_order_amt") or limit_map.get("minNotional") or "0"),
            "minOrderAmt": str(limit_map.get("min_order_amt") or limit_map.get("minOrderAmt") or "0"),
        }
        price_filter = {
            "tickSize": str(limit_map.get("tick_size") or limit_map.get("tickSize") or "0"),
        }
        instrument_raw = {"priceFilter": price_filter, "lotSizeFilter": lot_filter}

    min_amount = _to_decimal(limit_map.get("min_order_amt") or _MIN_QUOTE)
    if min_amount <= 0:
        min_amount = _MIN_QUOTE

    quote_step = _to_decimal(limit_map.get("quote_step") or Decimal("0.01"), Decimal("0.01"))
    if quote_step <= 0:
        quote_step = Decimal("0.01")

    min_qty = _to_decimal(limit_map.get("min_order_qty") or Decimal("0"))
    if min_qty < 0:
        min_qty = Decimal("0")

    qty_step = _to_decimal(limit_map.get("qty_step") or Decimal("0.00000001"), Decimal("0.00000001"))
    if qty_step <= 0:
        qty_step = Decimal("0.00000001")

    quote_coin = str(limit_map.get("quote_coin") or "").upper()
    base_coin = str(limit_map.get("base_coin") or "").upper()
    if not quote_coin or not base_coin:
        guessed_base, guessed_quote = _split_symbol(symbol)
        if not quote_coin and guessed_quote:
            quote_coin = guessed_quote
        if not base_coin and guessed_base:
            base_coin = guessed_base

    unit_normalised = (unit or "quoteCoin").strip().lower()
    if unit_normalised not in {"basecoin", "quotecoin"}:
        unit_normalised = "quotecoin"

    try:
        requested_qty = _to_decimal(qty)
    except Exception as exc:  # pragma: no cover - defensive
        raise OrderValidationError(
            f"Не удалось интерпретировать объём сделки: {exc}",
            code="qty_invalid",
        ) from exc

    if requested_qty <= 0:
        raise OrderValidationError(
            "Количество должно быть положительным.",
            code="qty_invalid",
            details={"requested": str(qty)},
        )

    side_normalised = (side or "").strip().lower()
    if side_normalised not in {"buy", "sell"}:
        raise OrderValidationError(
            "Сторона ордера должна быть 'buy' или 'sell'.",
            code="side_invalid",
            details={"side": side},
        )

    (tolerance_multiplier, tolerance_type, tolerance_value, tolerance_decimal) = _resolve_slippage_tolerance(
        tol_type,
        tol_value,
    )

    price_deviation: Decimal | None = None
    if tolerance_decimal > 0 and isinstance(tolerance_type, str):
        tolerance_kind = tolerance_type.strip().lower()
        if tolerance_kind in {"percent", "percentage"}:
            price_deviation = tolerance_decimal / Decimal("100")
        elif tolerance_kind in {"bps", "basispoints", "basis_points"}:
            price_deviation = tolerance_decimal / Decimal("10000")

    max_available: Optional[Decimal] = None
    if max_quote is not None:
        max_available = _to_decimal(max_quote)
        if max_available < 0:
            max_available = Decimal("0")

    price_hint: Optional[Decimal] = None
    if price_snapshot is not None:
        candidate = _to_decimal(price_snapshot)
        if candidate > 0:
            price_hint = candidate

    balance_map: Dict[str, Decimal] | None = _normalise_balances(balances)
    balances_checked: List[Tuple[str, Decimal]] = []

    def _ensure_balance(asset: str, required: Decimal) -> None:
        nonlocal balance_map

        asset_normalised = asset.strip().upper()
        if not asset_normalised or required <= 0:
            return

        fetched_with_requirement = False
        if balance_map is None:
            balance_map = _wallet_available_balances(api, required_asset=asset_normalised)
            fetched_with_requirement = True

        def _current_available() -> Decimal:
            if balance_map:
                return balance_map.get(asset_normalised, Decimal("0"))
            return Decimal("0")

        available = _current_available()

        margin = _TOLERANCE_MARGIN
        if available + margin >= required:
            balances_checked.append((asset_normalised, required))
            return

        if available + margin < required and not fetched_with_requirement:
            refreshed = _wallet_available_balances(api, required_asset=asset_normalised)
            fetched_with_requirement = True
            merged: Dict[str, Decimal] = dict(balance_map or {})
            if refreshed:
                merged.update(refreshed)
            balance_map = merged if merged else balance_map
            available = _current_available()
            if available + margin >= required:
                balances_checked.append((asset_normalised, required))
                return

        if available + margin < required:
            spot_refreshed = _wallet_available_balances(
                api,
                account_type="SPOT",
                required_asset=asset_normalised,
            )
            merged: Dict[str, Decimal] = dict(balance_map or {})
            if spot_refreshed:
                merged.update(spot_refreshed)
            if merged:
                balance_map = merged
                available = _current_available()
                if available + margin >= required:
                    balances_checked.append((asset_normalised, required))
                    return

        details: Dict[str, object] = {
            "asset": asset_normalised,
            "required": _format_decimal(required),
            "available": _format_decimal(available),
        }
        if asset_normalised != "USDT":
            alt_balance = balance_map.get("USDT") if balance_map else None
            if alt_balance and alt_balance > 0:
                details["alt_usdt"] = _format_decimal(alt_balance)

        raise OrderValidationError(
            "Недостаточно средств для выполнения сделки.",
            code="insufficient_balance",
            details=details,
        )

    market_unit = "quoteCoin"
    target_quote: Optional[Decimal] = None
    target_base: Optional[Decimal] = None

    if unit_normalised == "quotecoin":
        rounded = _round_down(requested_qty, quote_step)
        if rounded <= 0:
            raise OrderValidationError(
                "Объём меньше минимального шага для валюты котировки.",
                code="qty_step",
                details={
                    "requested": _format_decimal(requested_qty),
                    "step": _format_decimal(quote_step),
                    "unit": "quote",
                },
            )
        if min_amount > 0 and rounded < min_amount:
            raise OrderValidationError(
                "Минимальный объём ордера не достигнут.",
                code="min_notional",
                details={
                    "requested": _format_decimal(requested_qty),
                    "rounded": _format_decimal(rounded),
                    "min_notional": _format_decimal(min_amount),
                    "step": _format_decimal(quote_step),
                    "unit": "quote",
                },
            )
        target_quote = rounded
    else:
        rounded = _round_down(requested_qty, qty_step)
        if rounded <= 0:
            raise OrderValidationError(
                "Количество меньше минимального шага для базовой валюты.",
                code="qty_step",
                details={
                    "requested": _format_decimal(requested_qty),
                    "step": _format_decimal(qty_step),
                    "unit": "base",
                },
            )
        if min_qty > 0 and rounded < min_qty:
            raise OrderValidationError(
                "Количество меньше минимального лота для базовой валюты.",
                code="min_qty",
                details={
                    "requested": _format_decimal(requested_qty),
                    "rounded": _format_decimal(rounded),
                    "min_qty": _format_decimal(min_qty),
                    "step": _format_decimal(qty_step),
                    "unit": "base",
                },
            )
        target_base = rounded
        market_unit = "baseCoin"

    if price_hint is None:
        price_hint = _latest_price(api, symbol)

    max_price_allowed: Decimal | None
    min_price_allowed: Decimal | None
    max_price_allowed, min_price_allowed = _max_mark_price(
        price_hint,
        side=side_normalised,
        deviation=price_deviation,
    )

    instrument_price_cap = _to_decimal(limit_map.get("max_price") or Decimal("0"))
    if instrument_price_cap <= 0:
        instrument_price_cap = None
    instrument_price_floor = _to_decimal(limit_map.get("min_price") or Decimal("0"))
    if instrument_price_floor <= 0:
        instrument_price_floor = None

    instrument_ceiling_enforced = False
    instrument_floor_enforced = False

    if instrument_price_cap is not None:
        if max_price_allowed is None or instrument_price_cap < max_price_allowed:
            max_price_allowed = instrument_price_cap
            instrument_ceiling_enforced = True
        elif (
            max_price_allowed is not None
            and abs(instrument_price_cap - max_price_allowed) <= _TOLERANCE_MARGIN
        ):
            instrument_ceiling_enforced = True

    if instrument_price_floor is not None:
        if min_price_allowed is None or instrument_price_floor > min_price_allowed:
            min_price_allowed = instrument_price_floor
            instrument_floor_enforced = True
        elif (
            min_price_allowed is not None
            and abs(instrument_price_floor - min_price_allowed) <= _TOLERANCE_MARGIN
        ):
            instrument_floor_enforced = True

    asks, bids = _orderbook_snapshot(api, symbol)

    def _with_synthetic_depth(levels: list[tuple[Decimal, Decimal]]) -> list[tuple[Decimal, Decimal]]:
        if levels:
            return levels
        fallback_price = price_hint if isinstance(price_hint, Decimal) else _to_decimal(price_hint)
        if fallback_price <= 0:
            fallback_price = _latest_price(api, symbol)
        synthetic = _synthetic_orderbook_levels(
            fallback_price,
            min_amount=min_amount,
            min_qty=min_qty,
            qty_step=qty_step,
            target_quote=target_quote,
            target_base=target_base,
        )
        return synthetic if synthetic else levels

    if side_normalised == "buy" and not asks:
        asks = _with_synthetic_depth(asks)
    if side_normalised == "sell" and not bids:
        bids = _with_synthetic_depth(bids)

    worst_price, qty_base, quote_total, consumed_levels = _plan_limit_ioc_order(
        asks=asks,
        bids=bids,
        side=side_normalised,
        target_quote=target_quote,
        target_base=target_base,
        qty_step=qty_step,
        min_qty=min_qty,
        max_price=max_price_allowed,
        min_price=min_price_allowed,
    )


    limit_map_tick = _to_decimal(limit_map.get("tick_size") or Decimal("0"))
    limit_price = _apply_tick(
        worst_price,
        limit_map_tick,
        side_normalised,
        max_price_allowed=max_price_allowed,
        min_price_allowed=min_price_allowed,
    )

    if max_price_allowed is not None and limit_price - max_price_allowed > _TOLERANCE_MARGIN:
        limit_price = max_price_allowed
    if min_price_allowed is not None and min_price_allowed - limit_price > _TOLERANCE_MARGIN:
        limit_price = min_price_allowed
    qty_base = _round_down(qty_base, qty_step) if qty_step > 0 else qty_base
    limit_notional = qty_base * limit_price

    planned_quote_reference: Optional[Decimal] = None
    quote_total_positive = quote_total if quote_total > 0 else None
    target_quote_positive = target_quote if target_quote is not None and target_quote > 0 else None
    if quote_total_positive is not None and target_quote_positive is not None:
        planned_quote_reference = quote_total_positive
        if planned_quote_reference > target_quote_positive:
            planned_quote_reference = target_quote_positive
    elif quote_total_positive is not None:
        planned_quote_reference = quote_total_positive
    elif target_quote_positive is not None:
        planned_quote_reference = target_quote_positive

    effective_notional: Decimal = limit_notional
    if planned_quote_reference is not None:
        effective_notional = planned_quote_reference

    def _recalculate_effective_notional() -> None:
        nonlocal effective_notional

        if planned_quote_reference is not None:
            effective_notional = planned_quote_reference
        else:
            effective_notional = limit_notional

    price_used = limit_price
    qty_base_raw = qty_base
    if target_quote is not None and price_used > 0:
        qty_base_raw = target_quote / price_used
    elif target_base is not None:
        qty_base_raw = target_base

    def _price_band_context_values() -> Dict[str, Decimal]:
        """Resolve requested/available trade amounts for price-band errors."""

        matched_quote: Optional[Decimal] = None
        if quote_total > 0:
            matched_quote = quote_total
        if qty_base > 0 and worst_price > 0:
            recalculated = qty_base * worst_price
            if matched_quote is None:
                matched_quote = recalculated
            else:
                matched_quote = min(matched_quote, recalculated)

        matched_base: Optional[Decimal] = None
        if qty_base > 0:
            matched_base = qty_base
        elif matched_quote is not None and worst_price > 0:
            matched_base = matched_quote / worst_price

        context: Dict[str, Decimal] = {}

        price_band_blocked = False
        if side_normalised == "buy" and max_price_allowed is not None:
            price_band_blocked = limit_price - max_price_allowed > _TOLERANCE_MARGIN
        elif side_normalised == "sell" and min_price_allowed is not None:
            price_band_blocked = min_price_allowed - limit_price > _TOLERANCE_MARGIN

        if price_band_blocked and not consumed_levels:
            context["available_quote"] = Decimal("0")
            context["available_base"] = Decimal("0")
        else:
            if matched_quote is not None:
                context["available_quote"] = matched_quote
            if matched_base is not None:
                context["available_base"] = matched_base

        requested_quote_value: Optional[Decimal] = None
        if target_quote is not None and target_quote > 0:
            requested_quote_value = target_quote
        elif target_base is not None and target_base > 0 and limit_price > 0:
            requested_quote_value = target_base * limit_price
        elif matched_quote is not None:
            requested_quote_value = matched_quote
        if requested_quote_value is not None:
            context["requested_quote"] = requested_quote_value

        requested_base_value: Optional[Decimal] = None
        if target_base is not None and target_base > 0:
            requested_base_value = target_base
        elif matched_base is not None:
            requested_base_value = matched_base
        if requested_base_value is not None:
            context["requested_base"] = requested_base_value

        return context

    def _enforce_price_band() -> None:
        if limit_price <= 0:
            return

        context_values = _price_band_context_values()

        def _populate_details(details: Dict[str, object]) -> None:
            for key, value in context_values.items():
                details.setdefault(key, _format_decimal(value))

        if (
            side_normalised == "buy"
            and max_price_allowed is not None
            and limit_price - max_price_allowed > _TOLERANCE_MARGIN
        ):
            details: Dict[str, object] = {
                "limit_price": _format_decimal(limit_price),
                "side": side_normalised,
            }
            if price_hint is not None and not instrument_ceiling_enforced:
                details["mark_price"] = _format_decimal(price_hint)
                details["max_allowed"] = _format_decimal(max_price_allowed)
            else:
                details["price_cap"] = _format_decimal(max_price_allowed)
                details["price_limit_hit"] = True

            _populate_details(details)

            raise OrderValidationError(
                "Ожидаемая цена превышает допустимый предел для инструмента.",
                code="price_deviation",
                details=details,
            )

        if (
            side_normalised == "sell"
            and min_price_allowed is not None
            and min_price_allowed - limit_price > _TOLERANCE_MARGIN
        ):
            details = {
                "limit_price": _format_decimal(limit_price),
                "side": side_normalised,
            }
            if price_hint is not None and not instrument_floor_enforced:
                details["mark_price"] = _format_decimal(price_hint)
                details["min_allowed"] = _format_decimal(min_price_allowed)
            else:
                details["price_floor"] = _format_decimal(min_price_allowed)
                details["price_limit_hit"] = True

            _populate_details(details)

            raise OrderValidationError(
                "Ожидаемая цена ниже допустимого предела для инструмента.",
                code="price_deviation",
                details=details,
            )

    def _apply_validation_result(result: validators.SpotValidationResult) -> None:
        nonlocal validated, limit_price, qty_base, limit_notional

        price_candidate = result.price
        qty_candidate = result.qty
        if price_candidate <= 0 or qty_candidate <= 0:
            raise OrderValidationError(
                "Валидация объёма вернула некорректные значения.",
                code="validation_failed",
                details={
                    "price": str(price_candidate),
                    "qty": str(qty_candidate),
                },
            )

        validated = result
        limit_price = price_candidate
        qty_base = qty_candidate
        limit_notional = result.notional

        _recalculate_effective_notional()

        _enforce_price_band()

    validated = validators.validate_spot_rules(
        instrument=instrument_raw,
        price=price_used,
        qty=qty_base_raw,
        side=side_normalised,
    )

    _apply_validation_result(validated)

    if (
        target_quote is not None
        and limit_price > 0
        and min_amount > 0
        and any(reason.startswith("notional") for reason in validated.reasons)
    ):
        adjusted_qty_text = ceil_qty_to_min_notional(
            qty_base,
            limit_price,
            min_amount,
            qty_step,
        )
        adjusted_qty = _to_decimal(adjusted_qty_text)
        if adjusted_qty > qty_base:
            validated_adjusted = validators.validate_spot_rules(
                instrument=instrument_raw,
                price=limit_price,
                qty=adjusted_qty,
                side=side_normalised,
            )
            _apply_validation_result(validated_adjusted)

    if min_amount > 0 and limit_notional < min_amount and limit_price > 0:
        qty_needed = min_amount / limit_price
        if qty_step > 0:
            qty_needed = _round_up(qty_needed, qty_step)
        validated_min = validators.validate_spot_rules(
            instrument=instrument_raw,
            price=limit_price,
            qty=qty_needed,
            side=side_normalised,
        )
        _apply_validation_result(validated_min)

    if unit_normalised == "quotecoin":
        market_unit = "quoteCoin"
    else:
        market_unit = "baseCoin"

    if side_normalised == "buy" and target_quote is not None and limit_price > 0:
        affordable_qty = _round_down(target_quote / limit_price, qty_step)
        if affordable_qty <= 0:
            raise OrderValidationError(
                "Расчётное количество меньше шага инструмента.",
                code="qty_step",
                details={
                    "target_quote": _format_decimal(target_quote),
                    "limit_price": _format_decimal(limit_price),
                    "qty_step": _format_decimal(qty_step),
                },
            )
        if min_qty > 0 and affordable_qty < min_qty:
            raise OrderValidationError(
                "Количество меньше минимального лота для базовой валюты.",
                code="min_qty",
                details={
                    "requested": _format_decimal(affordable_qty),
                    "rounded": _format_decimal(affordable_qty),
                    "min_qty": _format_decimal(min_qty),
                    "step": _format_decimal(qty_step),
                    "unit": "base",
                },
            )
        if affordable_qty < qty_base:
            if min_amount > 0 and (affordable_qty * limit_price) < min_amount:
                pass
            else:
                qty_base = affordable_qty
                limit_notional = qty_base * limit_price
                _recalculate_effective_notional()

    tolerance_margin = _TOLERANCE_MARGIN
    tolerance_guard_reduction: Decimal | None = None

    def _shrink_order_to_ceiling(max_quote_allowed: Decimal) -> bool:
        nonlocal qty_base, limit_notional, tolerance_guard_reduction, planned_quote_reference, effective_notional

        if limit_price <= 0 or max_quote_allowed <= 0:
            return False

        max_qty_allowed = max_quote_allowed / limit_price
        if qty_step > 0:
            max_qty_allowed = _round_down(max_qty_allowed, qty_step)
        if max_qty_allowed <= 0:
            return False

        qty_candidate = qty_base if qty_base <= max_qty_allowed else max_qty_allowed
        if qty_candidate <= 0:
            return False

        step_fallback = qty_step if qty_step > 0 else _TOLERANCE_MARGIN
        last_qty: Decimal | None = None
        attempts = 0

        while qty_candidate > 0 and attempts < 32:
            attempts += 1
            validated_loop = validators.validate_spot_rules(
                instrument=instrument_raw,
                price=limit_price,
                qty=qty_candidate,
                side=side_normalised,
            )
            candidate_qty = validated_loop.qty
            candidate_notional = validated_loop.notional

            if min_amount > 0 and candidate_notional < min_amount:
                raise OrderValidationError(
                    "Минимальный объём ордера не достигнут.",
                    code="min_notional",
                    details={
                        "requested": _format_decimal(target_quote or requested_qty),
                        "rounded": _format_decimal(candidate_notional),
                        "min_notional": _format_decimal(min_amount),
                        "unit": "quote",
                    },
                )

            if min_qty > 0 and candidate_qty < min_qty:
                raise OrderValidationError(
                    "Количество меньше минимального лота для базовой валюты.",
                    code="min_qty",
                    details={
                        "requested": _format_decimal(target_quote or requested_qty),
                        "rounded": _format_decimal(candidate_qty),
                        "min_qty": _format_decimal(min_qty),
                        "unit": "base",
                    },
                )

            if candidate_notional - max_quote_allowed <= _TOLERANCE_MARGIN:
                reduction = qty_base - candidate_qty
                tolerance_guard_reduction = reduction if reduction > 0 else None
                qty_base = candidate_qty
                limit_notional = candidate_notional
                if planned_quote_reference is not None and candidate_notional > 0:
                    if candidate_notional < planned_quote_reference:
                        planned_quote_reference = candidate_notional
                elif (
                    planned_quote_reference is None
                    and target_quote is not None
                    and target_quote > 0
                    and candidate_notional > 0
                ):
                    planned_quote_reference = min(target_quote, candidate_notional)
                _recalculate_effective_notional()
                return True

            overshoot = candidate_notional - max_quote_allowed
            if overshoot <= 0:
                break

            if qty_step > 0:
                steps_to_trim = (overshoot / (limit_price * qty_step)).to_integral_value(rounding=ROUND_UP)
                if steps_to_trim <= 0:
                    steps_to_trim = 1
                next_qty = _round_down(candidate_qty - (qty_step * steps_to_trim), qty_step)
            else:
                decrement = overshoot / limit_price
                if decrement <= 0:
                    decrement = step_fallback
                elif decrement < step_fallback:
                    decrement = step_fallback
                next_qty = candidate_qty - decrement

            if next_qty <= 0:
                break
            if last_qty is not None and next_qty >= last_qty:
                break

            last_qty = candidate_qty
            qty_candidate = next_qty

        return False

    if target_quote is not None:
        tolerance_target = target_quote * tolerance_multiplier
        if tolerance_target > 0 and limit_notional - tolerance_target > tolerance_margin:
            if not _shrink_order_to_ceiling(tolerance_target):
                raise OrderValidationError(
                    "Расчётный объём превышает допустимый предел с учётом толеранса.",
                    code="tolerance_exceeded",
                    details={
                        "requested_quote": _format_decimal(target_quote),
                        "effective_notional": _format_decimal(effective_notional),
                        "tolerance_ceiling": _format_decimal(tolerance_target),
                        "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                        "tolerance_type": tolerance_type,
                        "tolerance_value": tolerance_value,
                    },
                )

    _recalculate_effective_notional()
    tolerance_adjusted = effective_notional * tolerance_multiplier

    quote_ceiling: Decimal | None = None
    quote_ceiling_raw: Decimal | None = None
    rounding_allowance = Decimal("0")
    tick_gap = Decimal("0")
    if target_quote is not None:
        quote_ceiling_raw = target_quote * tolerance_multiplier
        effective_quote = limit_notional if market_unit != "quoteCoin" else effective_notional
        if market_unit != "quoteCoin" and qty_step > 0:
            rounding_allowance = (qty_step * limit_price).copy_abs()
        if market_unit != "quoteCoin" and limit_price > worst_price:
            tick_gap = (limit_price - worst_price) * qty_base
        tick_allowance = tick_gap if (tolerance_decimal > 0 and market_unit != "quoteCoin") else Decimal("0")
        quote_ceiling = quote_ceiling_raw + rounding_allowance + tick_allowance
        if effective_quote - quote_ceiling > tolerance_margin:
            if not _shrink_order_to_ceiling(quote_ceiling_raw or tolerance_target):
                raise OrderValidationError(
                    "Расчётный объём превышает допустимый предел с учётом толеранса.",
                    code="tolerance_exceeded",
                    details={
                        "requested_quote": _format_decimal(target_quote),
                        "effective_notional": _format_decimal(effective_quote),
                        "tolerance_ceiling": _format_decimal(quote_ceiling_raw),
                        "rounding_allowance": _format_decimal(rounding_allowance),
                        "tick_gap": _format_decimal(tick_gap),
                        "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                        "tolerance_type": tolerance_type,
                        "tolerance_value": tolerance_value,
                    },
                )

            _recalculate_effective_notional()
            if market_unit != "quoteCoin":
                effective_quote = limit_notional
            else:
                effective_quote = effective_notional

            tick_gap = Decimal("0")
            if market_unit != "quoteCoin" and limit_price > worst_price:
                tick_gap = (limit_price - worst_price) * qty_base
            tick_allowance = tick_gap if (tolerance_decimal > 0 and market_unit != "quoteCoin") else Decimal("0")
            quote_ceiling = (quote_ceiling_raw or Decimal("0")) + rounding_allowance + tick_allowance
            if effective_quote - quote_ceiling > tolerance_margin:
                raise OrderValidationError(
                    "Расчётный объём превышает допустимый предел с учётом толеранса.",
                    code="tolerance_exceeded",
                    details={
                        "requested_quote": _format_decimal(target_quote),
                        "effective_notional": _format_decimal(effective_quote),
                        "tolerance_ceiling": _format_decimal(quote_ceiling_raw),
                        "rounding_allowance": _format_decimal(rounding_allowance),
                        "tick_gap": _format_decimal(tick_gap),
                        "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                        "tolerance_type": tolerance_type,
                        "tolerance_value": tolerance_value,
                    },
                )

    if max_available is not None:
        if max_available <= 0 or tolerance_adjusted - max_available > tolerance_margin:
            raise OrderValidationError(
                "Недостаточно свободного капитала с учётом допуска по проскальзыванию.",
                code="max_quote",
                details={
                    "required": _format_decimal(tolerance_adjusted),
                    "available": _format_decimal(max_available if max_available > 0 else Decimal("0")),
                    "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                },
            )

    _enforce_price_band()

    if min_amount > 0 and limit_notional < min_amount:
        raise OrderValidationError(
            "Минимальный объём ордера не достигнут.",
            code="min_notional",
            details={
                "requested": _format_decimal(target_quote or requested_qty),
                "rounded": _format_decimal(limit_notional),
                "min_notional": _format_decimal(min_amount),
                "unit": "quote",
            },
        )

    if side_normalised == "buy":
        _ensure_balance(quote_coin or "", tolerance_adjusted)
    else:
        _ensure_balance(base_coin or "", qty_base)

    qty_value = qty_base
    qty_step_for_payload = qty_step

    if qty_step_for_payload > 0:
        qty_value = _round_down(_to_decimal(qty_value), qty_step_for_payload)
    qty_text = _format_step_decimal(qty_value, qty_step_for_payload)
    price_text = _format_step_decimal(limit_price, limit_map_tick)
    if not qty_text:
        qty_text = "0"
    if not price_text:
        price_text = "0"

    time_in_force = "GTC"
    allow_partial_fills = True
    reprice_after_sec: int | None = None
    max_amendments_setting: int | None = None
    if isinstance(settings, Settings):
        tif_candidate = getattr(settings, "order_time_in_force", None)
        if not tif_candidate:
            tif_candidate = getattr(settings, "spot_limit_tif", None)
        if isinstance(tif_candidate, str) and tif_candidate.strip():
            tif_upper = tif_candidate.strip().upper()
            mapping = {"POSTONLY": "PostOnly", "IOC": "IOC", "FOK": "FOK", "GTC": "GTC"}
            time_in_force = mapping.get(tif_upper, tif_upper)
        allow_partial_fills = bool(getattr(settings, "allow_partial_fills", True))
        reprice_after_raw = getattr(settings, "reprice_unfilled_after_sec", None)
        if isinstance(reprice_after_raw, (int, float)) and reprice_after_raw >= 0:
            reprice_after_sec = int(reprice_after_raw)
        max_amendments_raw = getattr(settings, "max_amendments", None)
        if isinstance(max_amendments_raw, int) and max_amendments_raw >= 0:
            max_amendments_setting = max_amendments_raw

    payload: Dict[str, object] = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": qty_text,
        "price": price_text,
        "timeInForce": time_in_force,
        "orderFilter": "Order",
        "accountType": "UNIFIED",
    }

    audit: Dict[str, object] = {
        "symbol": symbol.upper(),
        "side": side.capitalize(),
        "unit": market_unit,
        "requested_qty": _format_decimal(requested_qty),
        "rounded_qty": _format_decimal(qty_base if unit_normalised == "basecoin" else limit_notional),
        "requested_unit": "quote" if unit_normalised == "quotecoin" else "base",
        "min_order_amt": _format_decimal(min_amount),
        "min_order_qty": _format_decimal(min_qty),
        "quote_step": _format_decimal(quote_step),
        "qty_step": _format_decimal(qty_step),
        "tolerance_multiplier": _format_decimal(tolerance_multiplier),
        "tolerance_value": tolerance_value,
        "tolerance_type": tolerance_type,
        "effective_notional": _format_decimal(effective_notional),
        "tolerance_adjusted_notional": _format_decimal(tolerance_adjusted),
        "quote_coin": quote_coin or None,
        "base_coin": base_coin or None,
        "time_in_force": time_in_force,
        "allow_partial_fills": allow_partial_fills,
        "qty_payload": qty_text,
        "price_payload": price_text,
    }

    audit["validator_ok"] = validated.ok
    if validated.reasons:
        audit["validator_reasons"] = list(validated.reasons)

    audit["limit_notional"] = _format_decimal(limit_notional)
    if reprice_after_sec is not None:
        audit["reprice_unfilled_after_sec"] = reprice_after_sec
    if max_amendments_setting is not None:
        audit["max_amendments"] = max_amendments_setting

    if max_available is not None:
        audit["max_available"] = _format_decimal(max_available)
    if price_hint is not None:
        audit["price_used"] = _format_decimal(price_hint)
    if max_price_allowed is not None:
        audit["price_ceiling"] = _format_decimal(max_price_allowed)
    if min_price_allowed is not None:
        audit["price_floor"] = _format_decimal(min_price_allowed)
    if balances_checked:
        audit["balances_checked"] = [
            {"asset": asset, "required": _format_decimal(amount)} for asset, amount in balances_checked
        ]

    audit["limit_price"] = _format_decimal(limit_price)
    audit["order_qty_base"] = _format_decimal(qty_base)
    audit["order_notional"] = _format_decimal(limit_notional)
    if tolerance_guard_reduction is not None:
        audit["tolerance_guard_reduction"] = _format_decimal(tolerance_guard_reduction)
    if target_quote is not None:
        audit["requested_quote_notional"] = _format_decimal(target_quote)
        if quote_ceiling_raw is not None:
            audit["quote_tolerance_core"] = _format_decimal(quote_ceiling_raw)
        if rounding_allowance > 0:
            audit["quote_rounding_allowance"] = _format_decimal(rounding_allowance)
        if tick_gap > 0:
            audit["quote_tick_gap"] = _format_decimal(tick_gap)
        if quote_ceiling is not None:
            audit["quote_tolerance_ceiling"] = _format_decimal(quote_ceiling)
    audit["consumed_levels"] = [
        {"price": _format_decimal(price), "qty": _format_decimal(qty)} for price, qty in consumed_levels
    ]

    for key, value in _price_band_context_values().items():
        audit[f"price_band_{key}"] = _format_decimal(value)

    return PreparedSpotMarketOrder(payload=payload, audit=audit)


def place_spot_market_with_tolerance(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: Decimal | float | int | str,
    unit: str = "quoteCoin",
    tol_type: str = "Percent",
    tol_value: float = 0.5,
    max_quote: object | None = None,
    *,
    price_snapshot: object | None = None,
    balances: Mapping[str, object] | None = None,
    limits: Mapping[str, object] | None = None,
    settings: Settings | None = None,
):
    """Создать маркет-ордер со строгой валидацией объёма и балансов."""
    unit_normalised = (unit or "quoteCoin").strip().lower()
    original_qty = _to_decimal(qty)
    if original_qty <= 0:
        raise OrderValidationError(
            "Количество должно быть положительным.",
            code="qty_invalid",
            details={"requested": str(qty)},
        )

    remaining_qty = original_qty
    remaining_cap = _to_decimal(max_quote) if max_quote is not None else None
    min_order_amt: Decimal | None = None
    min_order_qty: Decimal | None = None
    attempt_payloads: list[Dict[str, object]] = []
    attempt_audits: list[Dict[str, object]] = []
    attempt_logs: list[Dict[str, object]] = []
    executed_quote_total = Decimal("0")
    executed_base_total = Decimal("0")
    last_response_raw: object | None = None

    twap_cfg = _twap_runtime_config(settings)
    twap_active = False
    target_slices = max(1, twap_cfg.base_slices if twap_cfg.enabled else 1)
    max_slices = max(1, twap_cfg.max_slices if twap_cfg.enabled else 1)
    twap_orders_sent = 0
    twap_adjustments: list[Dict[str, object]] = []

    max_attempts = max(3, max_slices * 5 if twap_cfg.enabled else 3)
    if isinstance(settings, Settings):
        override_raw = getattr(settings, "twap_adjustment_max_attempts", None)
        try:
            override_value = int(override_raw) if override_raw is not None else 0
        except Exception:  # pragma: no cover - defensive
            override_value = 0
        if override_value > 0:
            max_attempts = max(1, min(max_attempts, override_value))
    attempt = 0
    adjustment_attempts = 0

    def _register_adjustment_retry(reason: str, error: OrderValidationError) -> None:
        nonlocal attempt, adjustment_attempts
        attempt += 1
        adjustment_attempts += 1
        if attempt >= max_attempts:
            details = {
                "reason": reason,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "adjustment_attempts": adjustment_attempts,
                "twap_active": twap_active,
                "twap_target_slices": target_slices,
                "twap_adjustments": list(twap_adjustments),
                "last_error": getattr(error, "details", {}) or {},
            }
            log(
                "spot.market.twap_adjustment_exhausted",
                symbol=symbol,
                side=side,
                **{k: v for k, v in details.items() if k != "twap_adjustments"},
            )
            raise OrderValidationError(
                "Не удалось выполнить ордер в пределах допустимого количества попыток.",
                code="twap_adjustment_exhausted",
                details=details,
            ) from error

    while attempt < max_attempts:
        if remaining_qty <= 0:
            break

        if min_order_amt is not None or min_order_qty is not None:
            threshold: Decimal | None = None
            skip_tail = False
            if unit_normalised == "quotecoin" and min_order_amt is not None:
                threshold = min_order_amt
                if threshold - remaining_qty > _TOLERANCE_MARGIN:
                    skip_tail = True
            elif unit_normalised != "quotecoin" and min_order_qty is not None:
                threshold = min_order_qty
                if threshold - remaining_qty > _TOLERANCE_MARGIN:
                    skip_tail = True

            if skip_tail:
                log(
                    "spot.market.twap_tail_skipped",
                    symbol=symbol,
                    side=side,
                    remaining_qty=_format_decimal(remaining_qty),
                    min_threshold=_format_decimal(threshold) if threshold is not None else None,
                    unit="quote" if unit_normalised == "quotecoin" else "base",
                    attempts=attempt,
                    twap_active=twap_active,
                )
                break

        if twap_active:
            slices_left = max(1, target_slices - twap_orders_sent)
            chunk_qty = remaining_qty / Decimal(slices_left)
        else:
            chunk_qty = remaining_qty

        if chunk_qty <= 0:
            break

        try:
            prepared = prepare_spot_market_order(
                api,
                symbol,
                side,
                chunk_qty,
                unit=unit,
                tol_type=tol_type,
                tol_value=tol_value,
                max_quote=remaining_cap,
                price_snapshot=price_snapshot if attempt == 0 else None,
                balances=balances,
                limits=limits,
                settings=settings,
            )
        except OrderValidationError as exc:
            if twap_cfg.enabled and exc.code == "price_deviation":
                adjustment: Dict[str, object] = {
                    "code": exc.code,
                    "message": str(exc),
                    "remaining_qty": _format_decimal(remaining_qty),
                    "target_slices": target_slices,
                }
                ratio = _twap_price_deviation_ratio(getattr(exc, "details", {}) or {})
                scaled_target = target_slices
                if isinstance(ratio, Decimal):
                    adjustment["ratio"] = _format_decimal(ratio)
                    scaled_target = _twap_scaled_slices(target_slices, ratio, max_slices)
                if not twap_active:
                    twap_active = True
                    if scaled_target != target_slices:
                        target_slices = scaled_target
                    adjustment["action"] = "activate"
                    adjustment["target_slices"] = target_slices
                    twap_adjustments.append(adjustment)
                    _register_adjustment_retry(exc.code or "price_deviation", exc)
                    continue
                if scaled_target > target_slices:
                    target_slices = scaled_target
                    adjustment["action"] = "increase"
                    adjustment["target_slices"] = target_slices
                    twap_adjustments.append(adjustment)
                    _register_adjustment_retry(exc.code or "price_deviation", exc)
                    continue
            if twap_cfg.enabled and exc.code == "insufficient_liquidity":
                details = getattr(exc, "details", {}) or {}
                price_limit_hit = bool(details.get("price_limit_hit"))
                if price_limit_hit:
                    adjustment = {
                        "code": exc.code,
                        "message": str(exc),
                        "remaining_qty": _format_decimal(remaining_qty),
                        "target_slices": target_slices,
                    }
                    requested_value = details.get("requested_quote") or details.get("requested_base")
                    available_value = details.get("available_quote") or details.get("available_base")
                    requested_amount = _to_decimal(requested_value) if requested_value is not None else None
                    available_amount = _to_decimal(available_value) if available_value is not None else None
                    ratio: Decimal | None = None
                    if isinstance(requested_amount, Decimal) and requested_amount > 0 and isinstance(
                        available_amount, Decimal
                    ):
                        ratio = max(Decimal("0"), available_amount / requested_amount)
                    scaled_target = target_slices
                    if isinstance(ratio, Decimal):
                        adjustment["ratio"] = _format_decimal(ratio)
                        if ratio > 0:
                            try:
                                desired = (Decimal("1") / ratio).to_integral_value(rounding=ROUND_UP)
                            except (InvalidOperation, ZeroDivisionError):  # pragma: no cover - defensive
                                desired = Decimal(max_slices)
                            try:
                                desired_int = int(desired)
                            except (TypeError, ValueError):  # pragma: no cover - defensive
                                desired_int = max_slices
                            if desired_int <= 0:
                                desired_int = 1
                            scaled_target = min(max_slices, max(target_slices, desired_int))
                        else:
                            scaled_target = max_slices
                    else:
                        scaled_target = max_slices
                    if not twap_active:
                        twap_active = True
                        if scaled_target != target_slices:
                            target_slices = scaled_target
                        adjustment["action"] = "activate"
                        adjustment["target_slices"] = target_slices
                        twap_adjustments.append(adjustment)
                        _register_adjustment_retry(exc.code or "insufficient_liquidity", exc)
                        continue
                    if scaled_target > target_slices:
                        target_slices = scaled_target
                        adjustment["action"] = "increase"
                        adjustment["target_slices"] = target_slices
                        twap_adjustments.append(adjustment)
                        _register_adjustment_retry(exc.code or "insufficient_liquidity", exc)
                        continue
                    adjustment["action"] = "retry"
                    adjustment["target_slices"] = target_slices
                    twap_adjustments.append(adjustment)
                    _register_adjustment_retry(exc.code or "insufficient_liquidity", exc)
                    continue
            raise

        min_order_amt_candidate = _to_decimal(prepared.audit.get("min_order_amt"))
        if min_order_amt_candidate > 0:
            min_order_amt = min_order_amt_candidate
        min_order_qty_candidate = _to_decimal(prepared.audit.get("min_order_qty"))
        if min_order_qty_candidate > 0:
            min_order_qty = min_order_qty_candidate

        attempt += 1

        if twap_active:
            twap_orders_sent += 1
            prepared.audit["twap_active"] = True
            prepared.audit["twap_target_slices"] = target_slices
            prepared.audit["twap_order_index"] = twap_orders_sent

        try:
            response = api.place_order(**prepared.payload)
        except RuntimeError as exc:
            error_text = str(exc)
            error_code = _extract_bybit_error_code_from_message(error_text)
            if error_code is None:
                if "170193" in error_text:
                    error_code = "170193"
                elif "170194" in error_text:
                    error_code = "170194"

            if error_code not in {"170193", "170194"}:
                raise

            details = parse_price_limit_error_details(error_text)
            side_normalised = str(side or "").strip().lower()
            if "price_cap" not in details:
                audit_cap = prepared.audit.get("price_ceiling")
                if audit_cap:
                    details["price_cap"] = str(audit_cap)
            if "price_floor" not in details:
                audit_floor = prepared.audit.get("price_floor")
                if audit_floor:
                    details["price_floor"] = str(audit_floor)

            limit_price = prepared.audit.get("limit_price")
            if limit_price and "limit_price" not in details:
                details["limit_price"] = str(limit_price)

            if side_normalised == "buy":
                requested_quote = (
                    prepared.audit.get("requested_quote_notional")
                    or prepared.audit.get("limit_notional")
                )
                if requested_quote and "requested_quote" not in details:
                    details["requested_quote"] = str(requested_quote)
            else:
                requested_base = prepared.audit.get("order_qty_base")
                if requested_base and "requested_base" not in details:
                    details["requested_base"] = str(requested_base)

            for key in ("requested_quote", "available_quote", "requested_base", "available_base"):
                audit_value = prepared.audit.get(f"price_band_{key}")
                if audit_value is not None and key not in details:
                    details[key] = str(audit_value)

            details["price_limit_hit"] = True
            if side_normalised:
                details.setdefault("side", side_normalised)

            message = "Ожидаемая цена выходит за пределы допустимого значения."
            if "price_cap" in details and details["price_cap"]:
                message = "Ожидаемая цена превышает допустимый предел для инструмента."
            elif "price_floor" in details and details["price_floor"]:
                message = "Ожидаемая цена ниже допустимого предела для инструмента."

            raise OrderValidationError(
                message,
                code="price_deviation",
                details=details,
            ) from exc
        last_response_raw = response

        ret_code = None
        ret_msg = None
        if isinstance(response, Mapping):
            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg")

        log(
            "spot.market.order_rest",
            symbol=symbol,
            side=side,
            attempt=attempt,
            ret_code=ret_code,
            ret_msg=ret_msg,
            request=prepared.payload,
            audit=prepared.audit,
        )

        qty_decimal = _to_decimal(prepared.payload.get("qty"))
        price_decimal = _to_decimal(prepared.payload.get("price"))
        exec_base, exec_quote, leaves_base = _execution_stats(
            response if isinstance(response, Mapping) else None,
            expected_qty=qty_decimal,
            limit_price=price_decimal,
        )

        executed_base_total += exec_base
        executed_quote_total += exec_quote
        if remaining_cap is not None:
            remaining_cap = max(Decimal("0"), remaining_cap - exec_quote)

        attempt_payloads.append(prepared.payload)
        attempt_audits.append(prepared.audit)
        log_entry = {
            "executed_base": _format_decimal(exec_base),
            "executed_quote": _format_decimal(exec_quote),
            "leaves_base": _format_decimal(leaves_base),
        }
        if twap_active:
            log_entry["twap_slices"] = target_slices
            log_entry["twap_order_index"] = twap_orders_sent
        attempt_logs.append(log_entry)

        step_value = _to_decimal(
            prepared.audit.get("quote_step") if unit_normalised == "quotecoin" else prepared.audit.get("qty_step")
        )
        if step_value <= 0:
            step_value = Decimal("0.00000001")

        if unit_normalised == "quotecoin":
            remaining_qty = max(Decimal("0"), original_qty - executed_quote_total)
        else:
            remaining_qty = max(Decimal("0"), original_qty - executed_base_total)

        if remaining_qty <= step_value:
            break
        if leaves_base <= 0 and not twap_active:
            break

        if twap_active:
            continue

        chunk_qty = min(remaining_qty, chunk_qty)
        if exec_base <= 0 and exec_quote <= 0:
            chunk_qty = max(step_value, chunk_qty / Decimal("2"))
        else:
            chunk_qty = remaining_qty

    final_response = last_response_raw

    if isinstance(final_response, dict):
        local = final_response.get("_local")
        combined = dict(local) if isinstance(local, dict) else {}
        combined["order_audit"] = attempt_audits[-1] if attempt_audits else {}
        combined["order_payload"] = attempt_payloads[-1] if attempt_payloads else {}
        combined["attempts"] = [
            {"audit": audit, "payload": payload, **log_info}
            for audit, payload, log_info in zip(attempt_audits, attempt_payloads, attempt_logs)
        ]
        if twap_cfg.enabled:
            combined["twap"] = {
                "active": twap_active,
                "target_slices": target_slices,
                "orders_sent": twap_orders_sent,
                "max_slices": max_slices,
                "adjustments": twap_adjustments,
                "interval_range": twap_cfg.interval_range,
                "aggressiveness_bps": twap_cfg.aggressiveness_bps,
            }
        final_response["_local"] = combined

    return final_response
