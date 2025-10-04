
from __future__ import annotations
from .helpers import ensure_link_id
import uuid
from .bybit_api import BybitAPI
from .log import log
from .telegram_notify import send_telegram
from .oco_guard import register_group

def place_linear_oco(api: BybitAPI, symbol: str, side: str, qty: str, price: str, take_profit: str, stop_loss: str, reduce_only: bool = False, group: str | None = None):
    group = group or ("FOCO-" + uuid.uuid4().hex[:10])
    primary_link = group+"-PRIMARY"
    tp_link = group+"-TP"
    sl_link = group+"-SL"
    primary = api.place_order(category="linear", symbol=symbol, side=side, orderType="Limit", qty=qty, price=price, timeInForce="GTC", reduceOnly=reduce_only, orderLinkId=ensure_link_id(primary_link))
    log("foco.primary", symbol=symbol, side=side, qty=qty, price=price, resp=primary, group=group)
    # Для фьючерсов размещаем TP/SL как tpslOrder с triggerPrice (рыночные)
    trig_dir_tp = 2 if side.lower()=="buy" else 1
    trig_dir_sl = 1 if side.lower()=="buy" else 2
    tp = api.place_order(category="linear", symbol=symbol, side=("Sell" if side.lower()=="buy" else "Buy"), orderType="Market", qty=qty, triggerDirection=trig_dir_tp, triggerPrice=take_profit, orderFilter="tpslOrder", reduceOnly=True, orderLinkId=ensure_link_id(tp_link))
    sl = api.place_order(category="linear", symbol=symbol, side=("Sell" if side.lower()=="buy" else "Buy"), orderType="Market", qty=qty, triggerDirection=trig_dir_sl, triggerPrice=stop_loss, orderFilter="tpslOrder", reduceOnly=True, orderLinkId=ensure_link_id(sl_link))
    log("foco.exits", tp=tp, sl=sl, group=group)
    register_group(group, symbol=symbol, category="linear", primary=primary_link, tp=tp_link, sl=sl_link)
    send_telegram(f"✅ Futures OCO создан [{group}] {symbol} {side} qty={qty} entry={price} TP={take_profit} SL={stop_loss}")
    return {"group": group, "primary": primary, "tp": tp, "sl": sl}
