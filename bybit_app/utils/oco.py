
from __future__ import annotations
from .helpers import ensure_link_id
import uuid
from .bybit_api import BybitAPI
from .log import log
from .telegram_notify import send_telegram
from .oco_guard import register_group

def place_spot_oco(api: BybitAPI, symbol: str, side: str, qty: str, price: str, take_profit: str, stop_loss: str, group: str | None = None):
    group = group or ("OCO-" + uuid.uuid4().hex[:10])
    primary_link = group+"-PRIMARY"
    tp_link = group+"-TP"
    sl_link = group+"-SL"
    primary = api.place_order(category="spot", symbol=symbol, side=side, orderType="Limit", qty=qty, price=price, timeInForce="GTC", orderLinkId=ensure_link_id(primary_link))
    log("oco.primary", symbol=symbol, side=side, qty=qty, price=price, resp=primary, group=group)
    if side.lower() == "buy":
        tp = api.place_order(category="spot", symbol=symbol, side="Sell", orderType="Limit", qty=qty, price=take_profit, timeInForce="GTC", orderLinkId=ensure_link_id(tp_link))
        sl = api.place_order(category="spot", symbol=symbol, side="Sell", orderType="Market", qty=qty, triggerDirection=2, triggerPrice=stop_loss, orderFilter="tpslOrder", orderLinkId=ensure_link_id(sl_link))
    else:
        tp = api.place_order(category="spot", symbol=symbol, side="Buy", orderType="Limit", qty=qty, price=take_profit, timeInForce="GTC", orderLinkId=ensure_link_id(tp_link))
        sl = api.place_order(category="spot", symbol=symbol, side="Buy", orderType="Market", qty=qty, triggerDirection=1, triggerPrice=stop_loss, orderFilter="tpslOrder", orderLinkId=ensure_link_id(sl_link))
    log("oco.exits", tp=tp, sl=sl, group=group)
    register_group(group, symbol=symbol, category="spot", primary=primary_link, tp=tp_link, sl=sl_link)
    send_telegram(f"✅ OCO создан [{group}] {symbol} {side} qty={qty} entry={price} TP={take_profit} SL={stop_loss}")
    return {"group": group, "primary": primary, "tp": tp, "sl": sl}
