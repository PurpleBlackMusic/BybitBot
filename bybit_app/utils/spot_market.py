from __future__ import annotations

from decimal import Decimal, ROUND_UP, InvalidOperation
import time
from typing import Dict, Tuple

from .bybit_api import BybitAPI
from .log import log

_MIN_QUOTE = Decimal("5")
_INSTRUMENT_CACHE: Dict[str, Tuple[float, Dict[str, Decimal]]] = {}
_PRICE_CACHE: Dict[str, Tuple[float, Decimal]] = {}
_PRICE_CACHE_TTL = 5.0


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


def _instrument_limits(api: BybitAPI, symbol: str) -> Dict[str, Decimal]:
    key = symbol.upper()
    now = time.time()
    cached = _INSTRUMENT_CACHE.get(key)
    if cached and now - cached[0] < 600:
        return cached[1]

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

    limits = {
        "min_order_amt": max(min_amount, _MIN_QUOTE),
        "quote_step": quote_step,
        "min_order_qty": min_qty,
        "qty_step": qty_step,
    }
    _INSTRUMENT_CACHE[key] = (now, limits)
    return limits


def _latest_price(api: BybitAPI, symbol: str) -> Decimal:
    key = symbol.upper()
    now = time.time()
    cached = _PRICE_CACHE.get(key)
    if cached and now - cached[0] < _PRICE_CACHE_TTL:
        return cached[1]

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

    _PRICE_CACHE[key] = (now, price)
    return price


def _format_decimal(value: Decimal) -> str:
    """Render Decimal without scientific notation for logging/messages."""

    try:
        normalised = value.normalize()
    except InvalidOperation:  # pragma: no cover - defensive branch
        normalised = Decimal("0")

    text = format(normalised, "f")
    return text if text else "0"


def place_spot_market_with_tolerance(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: float,
    unit: str = "quoteCoin",
    tol_type: str = "Percent",
    tol_value: float = 0.5,
    max_quote: object | None = None,
):
    """Создать маркет-ордер со slippageTolerance под подпись пользователя."""

    limits = _instrument_limits(api, symbol)
    min_amount = limits["min_order_amt"]
    quote_step = limits["quote_step"]
    min_qty = limits.get("min_order_qty", Decimal("0"))
    qty_step = limits.get("qty_step", Decimal("0"))

    unit_normalised = (unit or "quoteCoin").strip().lower()
    if unit_normalised not in {"basecoin", "quotecoin"}:
        unit_normalised = "quotecoin"

    max_available: Decimal | None = None
    if max_quote is not None:
        max_available = _to_decimal(max_quote)
        if max_available < 0:
            max_available = Decimal("0")

    price_snapshot: Decimal | None = None

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
        if needs_price:
            price_snapshot = _latest_price(api, symbol)

        if price_snapshot is not None:
            notional = base_qty * price_snapshot
            if min_amount > 0 and notional < min_amount:
                required = _round_up(min_amount / price_snapshot, qty_step)
                base_qty = max(base_qty, required)
                notional = base_qty * price_snapshot
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
        price_snapshot=str(price_snapshot) if price_snapshot is not None else None,
        projected_spend=str(projected_spend) if max_available is not None else None,
        max_available=str(max_available) if max_available is not None else None,
    )
    return response
