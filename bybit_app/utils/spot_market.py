from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation
from math import ceil
import time
from threading import RLock
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

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


def _normalise_orderbook_levels(levels: Sequence[Sequence[object]]) -> list[tuple[Decimal, Decimal]]:
    normalised: list[tuple[Decimal, Decimal]] = []
    for entry in levels or []:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        price = _to_decimal(entry[0])
        qty = _to_decimal(entry[1])
        if price <= 0 or qty <= 0:
            continue
        normalised.append((price, qty))
    return normalised


def _apply_tick(price: Decimal, tick_size: Decimal, side: str) -> Decimal:
    if tick_size <= 0:
        return price
    side_normalised = (side or "").lower()
    rounding = ROUND_UP if side_normalised == "buy" else ROUND_DOWN
    multiplier = (price / tick_size).to_integral_value(rounding=rounding)
    adjusted = multiplier * tick_size
    if adjusted <= 0:
        adjusted = tick_size
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

    asks = _normalise_orderbook_levels(asks_raw)
    bids = _normalise_orderbook_levels(bids_raw)
    return asks, bids


def _plan_limit_ioc_order(
    *,
    asks: Sequence[tuple[Decimal, Decimal]],
    bids: Sequence[tuple[Decimal, Decimal]],
    side: str,
    target_quote: Decimal | None,
    target_base: Decimal | None,
    qty_step: Decimal,
    min_qty: Decimal,
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

    for price, qty in levels:
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
    best_positive: Decimal | None = None
    best_non_positive: Decimal | None = None

    for field in _WALLET_AVAILABLE_FIELDS:
        candidate = row.get(field)
        if candidate is None:
            continue
        amount = _to_decimal(candidate)
        if amount > 0:
            if best_positive is None or amount > best_positive:
                best_positive = amount
        elif best_non_positive is None:
            best_non_positive = amount

    if best_positive is not None:
        return best_positive

    for field in _WALLET_FALLBACK_FIELDS:
        candidate = row.get(field)
        if candidate is None:
            continue
        amount = _to_decimal(candidate)
        if amount > 0:
            if best_positive is None or amount > best_positive:
                best_positive = amount
        elif best_non_positive is None:
            best_non_positive = amount

    if best_positive is not None:
        return best_positive

    return best_non_positive


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


def _wallet_available_balances(api: BybitAPI, account_type: str = "UNIFIED") -> Dict[str, Decimal]:
    account_key = (account_type or "UNIFIED").upper() or "UNIFIED"
    cache_key = _balance_cache_key(api, account_key)

    cached = _BALANCE_CACHE.get(cache_key)
    if cached is not None:
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

    min_allowed_value = details.get("min_allowed")
    if min_allowed_value is not None:
        min_allowed = _to_decimal(min_allowed_value)
        if min_allowed > 0:
            ratio = min_allowed / limit_price
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
            _BALANCE_CACHE.invalidate(_balance_cache_key(api, account_key))
        balances = _wallet_available_balances(api, account_type=account_type)

    return SpotTradeSnapshot(symbol=key, price=price, balances=balances, limits=limits)


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

        if balance_map is None:
            balance_map = _wallet_available_balances(api)

        available = balance_map.get(asset_normalised, Decimal("0"))
        margin = Decimal("0.00000001")
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

    asks, bids = _orderbook_snapshot(api, symbol)
    worst_price, qty_base, quote_total, consumed_levels = _plan_limit_ioc_order(
        asks=asks,
        bids=bids,
        side=side_normalised,
        target_quote=target_quote,
        target_base=target_base,
        qty_step=qty_step,
        min_qty=min_qty,
    )


    limit_map_tick = _to_decimal(limit_map.get("tick_size") or Decimal("0"))
    limit_price = _apply_tick(worst_price, limit_map_tick, side_normalised)
    qty_base = _round_down(qty_base, qty_step) if qty_step > 0 else qty_base
    limit_notional = qty_base * limit_price

    price_used = limit_price
    qty_base_raw = qty_base
    if target_quote is not None and price_used > 0:
        qty_base_raw = target_quote / price_used
    elif target_base is not None:
        qty_base_raw = target_base

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

    tolerance_margin = _TOLERANCE_MARGIN
    tolerance_guard_reduction: Decimal | None = None

    def _shrink_order_to_ceiling(max_quote_allowed: Decimal) -> bool:
        nonlocal qty_base, limit_notional, tolerance_guard_reduction

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
                        "effective_notional": _format_decimal(limit_notional),
                        "tolerance_ceiling": _format_decimal(tolerance_target),
                        "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                        "tolerance_type": tolerance_type,
                        "tolerance_value": tolerance_value,
                    },
                )

    effective_notional = limit_notional

    tolerance_margin = _TOLERANCE_MARGIN
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

            effective_notional = limit_notional
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

    max_price, min_price = _max_mark_price(price_hint, side=side_normalised, deviation=price_deviation)
    if max_price is not None and limit_price > max_price:
        raise OrderValidationError(
            "Ожидаемая цена превышает допустимое отклонение от mark.",
            code="price_deviation",
            details={
                "limit_price": _format_decimal(limit_price),
                "mark_price": _format_decimal(price_hint),
                "max_allowed": _format_decimal(max_price),
                "side": side_normalised,
            },
        )
    if min_price is not None and limit_price < min_price:
        raise OrderValidationError(
            "Ожидаемая цена ниже допустимого отклонения от mark.",
            code="price_deviation",
            details={
                "limit_price": _format_decimal(limit_price),
                "mark_price": _format_decimal(price_hint),
                "min_allowed": _format_decimal(min_price),
                "side": side_normalised,
            },
        )

    if min_amount > 0 and effective_notional < min_amount:
        raise OrderValidationError(
            "Минимальный объём ордера не достигнут.",
            code="min_notional",
            details={
                "requested": _format_decimal(target_quote or requested_qty),
                "rounded": _format_decimal(effective_notional),
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
    attempt = 0

    while attempt < max_attempts:
        if remaining_qty <= 0:
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
                if isinstance(ratio, Decimal):
                    adjustment["ratio"] = _format_decimal(ratio)
                if not twap_active:
                    twap_active = True
                    if target_slices < max_slices:
                        new_target = _twap_scaled_slices(target_slices, ratio, max_slices)
                        if new_target != target_slices:
                            target_slices = new_target
                    adjustment["action"] = "activate"
                    adjustment["target_slices"] = target_slices
                    twap_adjustments.append(adjustment)
                    continue
                if target_slices < max_slices:
                    new_target = _twap_scaled_slices(target_slices, ratio, max_slices)
                    if new_target > target_slices:
                        target_slices = new_target
                        adjustment["action"] = "increase"
                        adjustment["target_slices"] = target_slices
                        twap_adjustments.append(adjustment)
                        continue
            raise

        attempt += 1

        if twap_active:
            twap_orders_sent += 1
            prepared.audit["twap_active"] = True
            prepared.audit["twap_target_slices"] = target_slices
            prepared.audit["twap_order_index"] = twap_orders_sent

        response = api.place_order(**prepared.payload)
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
