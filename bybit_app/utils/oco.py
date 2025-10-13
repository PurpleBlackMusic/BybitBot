
from __future__ import annotations

import uuid
from decimal import ROUND_DOWN, ROUND_UP

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log
from .oco_guard import register_group
from .spot_rules import (
    SpotInstrumentNotFound,
    format_decimal,
    load_spot_instrument,
    quantize_price_only,
    quantize_spot_order,
)
from .telegram_notify import enqueue_telegram_message


def place_spot_oco(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: str,
    price: str,
    take_profit: str,
    stop_loss: str,
    group: str | None = None,
):
    group = group or ("OCO-" + uuid.uuid4().hex[:10])
    primary_link = group+"-PRIMARY"
    tp_link = group+"-TP"
    sl_link = group+"-SL"

    side_normalised = side.capitalize()
    if side_normalised not in {"Buy", "Sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc

    validated = quantize_spot_order(
        instrument=instrument,
        price=price,
        qty=qty,
        side=side_normalised,
    )
    if not validated.ok or validated.qty <= 0 or validated.price <= 0:
        raise ValueError(
            "Не удалось привести цену/количество к требованиям биржи",
            list(validated.reasons),
        )

    qty_text = format_decimal(validated.qty)
    price_text = format_decimal(validated.price)

    exit_side = "Sell" if side_normalised == "Buy" else "Buy"
    exit_rounding = ROUND_UP if exit_side == "Buy" else ROUND_DOWN

    tp_price = quantize_price_only(take_profit, tick_size=validated.tick_size, rounding=exit_rounding)
    sl_price = quantize_price_only(stop_loss, tick_size=validated.tick_size, rounding=exit_rounding)

    tp_text = format_decimal(tp_price)
    sl_text = format_decimal(sl_price)

    primary = api.place_order(
        category="spot",
        symbol=symbol,
        side=side_normalised,
        orderType="Limit",
        qty=qty_text,
        price=price_text,
        timeInForce="GTC",
        orderLinkId=ensure_link_id(primary_link),
    )
    log(
        "oco.primary",
        symbol=symbol,
        side=side_normalised,
        qty=qty_text,
        price=price_text,
        resp=primary,
        group=group,
    )
    if side_normalised == "Buy":
        tp = api.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            orderType="Limit",
            qty=qty_text,
            price=tp_text,
            timeInForce="GTC",
            orderLinkId=ensure_link_id(tp_link),
        )
        sl = api.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=qty_text,
            triggerDirection=2,
            triggerPrice=sl_text,
            orderFilter="tpslOrder",
            orderLinkId=ensure_link_id(sl_link),
        )
    else:
        tp = api.place_order(
            category="spot",
            symbol=symbol,
            side="Buy",
            orderType="Limit",
            qty=qty_text,
            price=tp_text,
            timeInForce="GTC",
            orderLinkId=ensure_link_id(tp_link),
        )
        sl = api.place_order(
            category="spot",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty_text,
            triggerDirection=1,
            triggerPrice=sl_text,
            orderFilter="tpslOrder",
            orderLinkId=ensure_link_id(sl_link),
        )
    log("oco.exits", tp=tp, sl=sl, group=group)
    register_group(group, symbol=symbol, category="spot", primary=primary_link, tp=tp_link, sl=sl_link)
    enqueue_telegram_message(
        f"✅ OCO создан [{group}] {symbol} {side_normalised} qty={qty_text} entry={price_text} TP={tp_text} SL={sl_text}"
    )
    return {"group": group, "primary": primary, "tp": tp, "sl": sl}
