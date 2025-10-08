from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation
import time
from threading import RLock
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

from .bybit_api import BybitAPI
from .log import log

_MIN_QUOTE = Decimal("5")
_PRICE_CACHE_TTL = 5.0
_BALANCE_CACHE_TTL = 5.0
_INSTRUMENT_CACHE_TTL = 600.0
_SYMBOL_CACHE_TTL = 300.0


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
    cache_key = "spot_usdt"
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


def _instrument_limits(api: BybitAPI, symbol: str) -> Dict[str, object]:
    key = symbol.upper()
    cached = _INSTRUMENT_CACHE.get(key)
    if cached is not None:
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
    _INSTRUMENT_CACHE.set(key, limits)
    return limits


def _latest_price(api: BybitAPI, symbol: str) -> Decimal:
    key = symbol.upper()
    cached = _PRICE_CACHE.get(key)
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

    _PRICE_CACHE.set(key, price)
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
                return {}
        raise RuntimeError(f"Не удалось получить баланс кошелька: {exc}") from exc

    balances: Dict[str, Decimal] = {}
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict):
            accounts = result.get("list")
            if isinstance(accounts, (list, tuple)):
                for account in accounts:
                    if not isinstance(account, dict):
                        continue
                    coins = account.get("coin") or account.get("coins")
                    if not isinstance(coins, (list, tuple)):
                        continue
                    for row in coins:
                        if not isinstance(row, dict):
                            continue
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


def _wallet_available_balances(api: BybitAPI, account_type: str = "UNIFIED") -> Dict[str, Decimal]:
    key = account_type.upper() or "UNIFIED"
    cached = _BALANCE_CACHE.get(key)
    if cached is not None:
        return cached

    primary_balances = _load_wallet_balances(api, account_type=key)

    combined = dict(primary_balances)
    # Users often keep spot funds on a dedicated SPOT account while trading
    # through a unified account.  When the unified wallet response does not
    # expose these assets we perform a transparent fallback to the SPOT
    # account and merge the balances so the guard sees the available funds.
    needs_fallback = False
    if key in {"UNIFIED", "TRADE"}:
        if not primary_balances:
            needs_fallback = True
        else:
            needs_fallback = all(amount <= 0 for amount in primary_balances.values())
    if needs_fallback:
        spot_key = "SPOT"
        spot_cached = _BALANCE_CACHE.get(spot_key)
        if spot_cached is None:
            spot_cached = _load_wallet_balances(api, account_type=spot_key)
            _BALANCE_CACHE.set(spot_key, dict(spot_cached))
        for asset, amount in (spot_cached or {}).items():
            combined[asset] = combined.get(asset, Decimal("0")) + amount

    _BALANCE_CACHE.set(key, combined)
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

    limits: Mapping[str, object] | None = None
    if include_limits:
        if force_refresh:
            _INSTRUMENT_CACHE.invalidate(key)
        limits = _instrument_limits(api, key)

    price: Decimal | None = None
    if include_price:
        if force_refresh:
            _PRICE_CACHE.invalidate(key)
        price = _latest_price(api, key)

    balances: Dict[str, Decimal] | None = None
    if include_balances:
        account_key = account_type.upper() or "UNIFIED"
        if force_refresh:
            _BALANCE_CACHE.invalidate(account_key)
        balances = _wallet_available_balances(api, account_type=account_type)

    return SpotTradeSnapshot(symbol=key, price=price, balances=balances, limits=limits)


_MIN_PERCENT_TOLERANCE = Decimal("0.05")
_MAX_PERCENT_TOLERANCE = Decimal("1.0")
_MIN_BPS_TOLERANCE = Decimal("5")
_MAX_BPS_TOLERANCE = Decimal("100")


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
    qty: float,
    unit: str = "quoteCoin",
    tol_type: str = "Percent",
    tol_value: float = 0.5,
    max_quote: object | None = None,
    *,
    price_snapshot: object | None = None,
    balances: Mapping[str, object] | None = None,
    limits: Mapping[str, object] | None = None,
):
    """Проверить параметры маркет-ордера и подготовить запрос к REST API."""

    limit_map = limits if limits is not None else _instrument_limits(api, symbol)

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
    qty_value = Decimal("0")
    effective_notional = Decimal("0")
    price_used: Optional[Decimal] = None

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
        qty_value = rounded
        effective_notional = qty_value
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

        needs_price = min_amount > 0 or max_available is not None
        if needs_price and price_hint is None:
            price_hint = _latest_price(api, symbol)

        qty_value = rounded
        market_unit = "baseCoin"
        if price_hint is not None and price_hint > 0:
            price_used = price_hint
            effective_notional = qty_value * price_hint
            if min_amount > 0 and effective_notional < min_amount:
                raise OrderValidationError(
                    "Минимальный объём ордера не достигнут.",
                    code="min_notional",
                    details={
                        "requested": _format_decimal(requested_qty),
                        "rounded": _format_decimal(qty_value),
                        "min_notional": _format_decimal(min_amount),
                        "price": _format_decimal(price_hint),
                        "unit": "base",
                    },
                )
        else:
            effective_notional = qty_value

    projected_spend = effective_notional * tolerance_multiplier

    if max_available is not None:
        tolerance_margin = Decimal("0.00000001")
        if max_available <= 0 or projected_spend - max_available > tolerance_margin:
            raise OrderValidationError(
                "Недостаточно свободного капитала с учётом допуска по проскальзыванию.",
                code="max_quote",
                details={
                    "required": _format_decimal(projected_spend),
                    "available": _format_decimal(max_available if max_available > 0 else Decimal("0")),
                    "tolerance_multiplier": _format_decimal(tolerance_multiplier),
                },
            )

    tolerance_adjusted = projected_spend

    if side_normalised == "buy":
        _ensure_balance(quote_coin or "", tolerance_adjusted)
    else:  # sell
        if market_unit == "baseCoin":
            _ensure_balance(base_coin or "", qty_value)
        elif price_used is not None and price_used > 0:
            required_base = (effective_notional / price_used).copy_abs()
            _ensure_balance(base_coin or "", required_base)

    qty_text = format(qty_value.normalize(), "f") if qty_value != 0 else "0"

    payload: Dict[str, object] = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty_text,
        "marketUnit": market_unit,
        "accountType": "UNIFIED",
    }

    if tolerance_multiplier > Decimal("1") and tolerance_decimal > Decimal("0"):
        payload["slippageToleranceType"] = tolerance_type
        payload["slippageTolerance"] = tolerance_value

    audit: Dict[str, object] = {
        "symbol": symbol.upper(),
        "side": side.capitalize(),
        "unit": market_unit,
        "requested_qty": _format_decimal(requested_qty),
        "rounded_qty": _format_decimal(qty_value),
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
    }

    if max_available is not None:
        audit["max_available"] = _format_decimal(max_available)
    if price_used is not None:
        audit["price_used"] = _format_decimal(price_used)
    if balances_checked:
        audit["balances_checked"] = [
            {"asset": asset, "required": _format_decimal(amount)} for asset, amount in balances_checked
        ]

    return PreparedSpotMarketOrder(payload=payload, audit=audit)


def place_spot_market_with_tolerance(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: float,
    unit: str = "quoteCoin",
    tol_type: str = "Percent",
    tol_value: float = 0.5,
    max_quote: object | None = None,
    *,
    price_snapshot: object | None = None,
    balances: Mapping[str, object] | None = None,
    limits: Mapping[str, object] | None = None,
):
    """Создать маркет-ордер со строгой валидацией объёма и балансов."""

    prepared = prepare_spot_market_order(
        api,
        symbol,
        side,
        qty,
        unit=unit,
        tol_type=tol_type,
        tol_value=tol_value,
        max_quote=max_quote,
        price_snapshot=price_snapshot,
        balances=balances,
        limits=limits,
    )

    response = api.place_order(**prepared.payload)

    ret_code = None
    ret_msg = None
    if isinstance(response, Mapping):
        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg")

    log(
        "spot.market.order_rest",
        symbol=symbol,
        side=side,
        ret_code=ret_code,
        ret_msg=ret_msg,
        request=prepared.payload,
        audit=prepared.audit,
    )

    if isinstance(response, dict):
        local = response.get("_local")
        if isinstance(local, dict):
            combined = dict(local)
        else:
            combined = {}
        combined["order_audit"] = prepared.audit
        combined["order_payload"] = prepared.payload
        response["_local"] = combined

    return response
