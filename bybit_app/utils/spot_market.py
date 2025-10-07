from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_UP, InvalidOperation
import time
from typing import Dict, Generic, Mapping, Sequence, Tuple, TypeVar

from .bybit_api import BybitAPI
from .log import log

_MIN_QUOTE = Decimal("5")
_PRICE_CACHE_TTL = 5.0
_BALANCE_CACHE_TTL = 5.0
_INSTRUMENT_CACHE_TTL = 600.0


T = TypeVar("T")


class TTLCache(Generic[T]):
    """A minimal TTL cache for repeated API lookups."""

    __slots__ = ("_ttl", "_store")

    def __init__(self, ttl: float):
        self._ttl = max(float(ttl), 0.0)
        self._store: Dict[str, Tuple[float, T]] = {}

    def get(self, key: str) -> T | None:
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if self._ttl and time.time() - ts > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: T) -> None:
        self._store[key] = (time.time(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


_INSTRUMENT_CACHE: TTLCache[Dict[str, object]] = TTLCache(_INSTRUMENT_CACHE_TTL)
_PRICE_CACHE: TTLCache[Decimal] = TTLCache(_PRICE_CACHE_TTL)
_BALANCE_CACHE: TTLCache[Dict[str, Decimal]] = TTLCache(_BALANCE_CACHE_TTL)

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


def _first_decimal(payload: object, fields: Tuple[str, ...]) -> Decimal | None:
    if not isinstance(payload, dict):
        return None

    for field in fields:
        if field not in payload:
            continue
        candidate = payload[field]
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        try:
            return _to_decimal(candidate)
        except Exception:  # pragma: no cover - defensive
            continue
    return None


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
        raise RuntimeError(f"Не найдены данные об инструменте {key}")

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

    limits: Dict[str, object] = {
        "min_order_amt": max(min_amount, _MIN_QUOTE),
        "quote_step": quote_step,
        "min_order_qty": min_qty,
        "qty_step": qty_step,
        "base_coin": base_coin,
        "quote_coin": quote_coin,
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
        raise RuntimeError(f"Биржа не вернула котировку для {key}")

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
                        available = _first_decimal(row, _WALLET_AVAILABLE_FIELDS)
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
    """Создать маркет-ордер со slippageTolerance под подпись пользователя."""

    limit_map = limits if limits is not None else _instrument_limits(api, symbol)
    min_amount = limit_map["min_order_amt"]  # type: ignore[index]
    quote_step = limit_map["quote_step"]  # type: ignore[index]
    min_qty = limit_map.get("min_order_qty", Decimal("0"))  # type: ignore[assignment]
    qty_step = limit_map.get("qty_step", Decimal("0"))  # type: ignore[assignment]
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

    max_available: Decimal | None = None
    if max_quote is not None:
        max_available = _to_decimal(max_quote)
        if max_available < 0:
            max_available = Decimal("0")

    price_hint: Decimal | None = None
    if price_snapshot is not None:
        price_hint = _to_decimal(price_snapshot)
        if price_hint <= 0:
            price_hint = None

    if unit_normalised == "quotecoin":
        quote_amount = _to_decimal(qty)
        adjusted = max(quote_amount, min_amount)
        adjusted = _round_up(adjusted, quote_step)
        qty_value = adjusted
        market_unit = "quoteCoin"
        effective_notional = qty_value
    else:
        base_qty = _to_decimal(qty)
        if base_qty <= 0:
            raise RuntimeError("Количество для покупки должно быть положительным")
        if min_qty > 0 and base_qty < min_qty:
            base_qty = min_qty
        base_qty = _round_up(base_qty, qty_step)

        needs_price = min_amount > 0 or max_available is not None
        if needs_price and price_hint is None:
            price_hint = _latest_price(api, symbol)

        if price_hint is not None:
            notional = base_qty * price_hint
            if min_amount > 0 and notional < min_amount:
                required = _round_up(min_amount / price_hint, qty_step)
                base_qty = max(base_qty, required)
                notional = base_qty * price_hint
            effective_notional = notional
        else:
            effective_notional = base_qty

        qty_value = base_qty
        market_unit = "baseCoin"

    tolerance = max(float(tol_value), 1.0)
    tolerance_multiplier = _to_decimal(tolerance)
    if tolerance_multiplier <= 0:
        tolerance_multiplier = Decimal("1")

    projected_spend: Decimal | None = None
    balance_map: Dict[str, Decimal] | None = _normalise_balances(balances)

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
            return

        available_text = _format_decimal(available)
        required_text = _format_decimal(required)

        alt_message = ""
        if asset_normalised != "USDT":
            alt_balance = balance_map.get("USDT")
            if alt_balance and alt_balance > 0:
                alt_message = (
                    " На счету есть "
                    f"{_format_decimal(alt_balance)} USDT, но биржа не конвертирует автоматически для спот-ордеров."
                )

        raise RuntimeError(
            "Недостаточно средств: "
            f"{asset_normalised} доступно ~{available_text}, требуется минимум ~{required_text}.{alt_message}"
        )

    if max_available is not None:
        tolerance_margin = Decimal("0.00000001")
        projected_spend = effective_notional * tolerance_multiplier
        if max_available <= 0 or projected_spend - max_available > tolerance_margin:
            required = _format_decimal(projected_spend)
            available = _format_decimal(max_available if max_available > 0 else Decimal("0"))
            raise RuntimeError(
                "Недостаточно свободного баланса для сделки: "
                f"доступно ~{available}, требуется минимум ~{required}."
            )

    tolerance_adjusted = effective_notional * tolerance_multiplier

    side_normalised = side.strip().lower()
    if side_normalised == "buy":
        if market_unit == "quoteCoin":
            _ensure_balance(quote_coin or "", tolerance_adjusted)
        elif price_hint is not None:
            _ensure_balance(quote_coin or "", tolerance_adjusted)
    elif side_normalised == "sell":
        if market_unit == "baseCoin":
            _ensure_balance(base_coin or "", qty_value)
        elif price_hint is not None and price_hint > 0:
            required_base = (effective_notional / price_hint).copy_abs()
            _ensure_balance(base_coin or "", required_base)

    qty_text = format(qty_value.normalize(), "f") if qty_value != 0 else "0"

    body = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty_text,
        "marketUnit": market_unit,  # "baseCoin" или "quoteCoin"
        "slippageToleranceType": tol_type,
        "slippageTolerance": f"{tolerance:.4f}",
    }

    response = api.place_order(**body)
    log(
        "spot.market.slip",
        symbol=symbol,
        side=side,
        body=body,
        resp=response,
        min_notional=str(min_amount),
        min_qty=str(min_qty),
        effective_notional=str(effective_notional),
        price_snapshot=str(price_hint) if price_hint is not None else None,
        projected_spend=str(projected_spend) if max_available is not None else None,
        max_available=str(max_available) if max_available is not None else None,
    )
    return response
