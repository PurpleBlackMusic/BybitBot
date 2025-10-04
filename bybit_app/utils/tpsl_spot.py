
from __future__ import annotations
from .helpers import ensure_link_id
from .bybit_api import BybitAPI
from .log import log

def place_spot_limit_with_tpsl(api: BybitAPI, symbol: str, side: str, qty: float, price: float, tp: float | None, sl: float | None, tp_order_type: str = "Market", sl_order_type: str = "Market", tp_limit: float | None = None, sl_limit: float | None = None, link_id: str | None = None, tif: str = "PostOnly"):
    """Создать СПОТ лимит ордер с серверным TP/SL для UTA (v5 create order).
    Требует: category='spot'. Если tp/sl заданы и тип=Limit — должны быть tp_limit/sl_limit.
    Возвращает ответ API.
    """
    body = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": f"{qty:.10f}",
        "price": f"{price:.10f}",
        "timeInForce": tif
    }
    if link_id:
        body["orderLinkId"] = link_id
    if tp is not None:
        body["takeProfit"] = f"{tp:.10f}"
        body["tpOrderType"] = tp_order_type
        if tp_order_type == "Limit":
            assert tp_limit is not None, "tp_limit required for Limit tp"
            body["tpLimitPrice"] = f"{tp_limit:.10f}"
    if sl is not None:
        body["stopLoss"] = f"{sl:.10f}"
        body["slOrderType"] = sl_order_type
        if sl_order_type == "Limit":
            assert sl_limit is not None, "sl_limit required for Limit sl"
            body["slLimitPrice"] = f"{sl_limit:.10f}"
    r = api.place_order(**body)
    log("spot.tpsl.create", symbol=symbol, side=side, body=body, resp=r)
    return r
