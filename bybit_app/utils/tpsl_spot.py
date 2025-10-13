
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP

from .helpers import ensure_link_id
from .bybit_api import BybitAPI
from .log import log
from .spot_rules import (
    SpotInstrumentNotFound,
    format_decimal,
    load_spot_instrument,
    quantize_price_only,
    quantize_spot_order,
)

def place_spot_limit_with_tpsl(api: BybitAPI, symbol: str, side: str, qty: float, price: float, tp: float | None, sl: float | None, tp_order_type: str = "Market", sl_order_type: str = "Market", tp_limit: float | None = None, sl_limit: float | None = None, link_id: str | None = None, tif: str = "PostOnly"):
    """Создать СПОТ лимит ордер с серверным TP/SL для UTA (v5 create order).
    Требует: category='spot'. Если tp/sl заданы и тип=Limit — должны быть tp_limit/sl_limit.
    Возвращает ответ API.
    """
    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc
    validated = quantize_spot_order(
        instrument=instrument,
        price=price,
        qty=qty,
        side=side,
    )

    if not validated.ok:
        raise ValueError(
            "Не удалось привести цену/количество к требованиям биржи",
            validated.reasons,
        )

    if validated.price <= 0 or validated.qty <= 0:
        raise ValueError("Количество или цена после квантизации невалидны")

    qty_text = format_decimal(validated.qty)
    price_text = format_decimal(validated.price)

    entry_side = side.capitalize()
    exit_side = "Sell" if entry_side == "Buy" else "Buy"
    tick_size: Decimal = validated.tick_size
    exit_rounding = ROUND_UP if exit_side == "Buy" else ROUND_DOWN

    def _quantize_optional(value: float | None) -> str | None:
        if value is None:
            return None
        quantized = quantize_price_only(value, tick_size=tick_size, rounding=exit_rounding)
        return format_decimal(quantized)

    body = {
        "category": "spot",
        "symbol": symbol,
        "side": entry_side,
        "orderType": "Limit",
        "qty": qty_text,
        "price": price_text,
        "timeInForce": tif,
    }
    if link_id:
        body["orderLinkId"] = ensure_link_id(link_id)
    if tp is not None:
        tp_text = _quantize_optional(tp)
        if tp_text is None:
            raise ValueError("Некорректное значение takeProfit")
        body["takeProfit"] = tp_text
        body["tpOrderType"] = tp_order_type
        if tp_order_type == "Limit":
            assert tp_limit is not None, "tp_limit required for Limit tp"
            tp_limit_text = _quantize_optional(tp_limit)
            if tp_limit_text is None:
                raise ValueError("Некорректное значение tp_limit")
            body["tpLimitPrice"] = tp_limit_text
    if sl is not None:
        sl_text = _quantize_optional(sl)
        if sl_text is None:
            raise ValueError("Некорректное значение stopLoss")
        body["stopLoss"] = sl_text
        body["slOrderType"] = sl_order_type
        if sl_order_type == "Limit":
            assert sl_limit is not None, "sl_limit required for Limit sl"
            sl_limit_text = _quantize_optional(sl_limit)
            if sl_limit_text is None:
                raise ValueError("Некорректное значение sl_limit")
            body["slLimitPrice"] = sl_limit_text
    r = api.place_order(**body)
    log("spot.tpsl.create", symbol=symbol, side=side, body=body, resp=r)
    return r
