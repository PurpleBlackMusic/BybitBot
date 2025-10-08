
from __future__ import annotations
from .helpers import ensure_link_id
import time, math
from .bybit_api import BybitAPI
from .log import log

def _bps(x): return float(x)/10000.0

def twap_spot(api: BybitAPI, symbol: str, side: str, total_qty: float, slices: int = 5, child_secs: int = 10, aggressiveness_bps: float = 2.0):
    """Клиентский TWAP для спота: разбиваем объём на равные части и отправляем IOC-лимитки около лучшей цены.
    - side: 'Buy' или 'Sell'
    - aggressiveness_bps: на сколько б.п. от худшей стороны готовы отступить, чтобы быстрее исполниться (для Buy — от best_ask вверх).
    Возвращает список ответов на размещение.
    """
    replies = []
    q_child = max(total_qty / max(1, int(slices)), 0.0)
    for i in range(int(slices)):
        ob = api.orderbook(category="spot", symbol=symbol, limit=5)
        bids = ((ob.get("result") or {}).get("b") or [])
        asks = ((ob.get("result") or {}).get("a") or [])
        if not bids or not asks:
            break
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        if side.lower() == "buy":
            px = best_ask * (1 + _bps(aggressiveness_bps))
        else:
            px = best_bid * (1 - _bps(aggressiveness_bps))
        qty = f"{q_child:.10f}"
        try:
            r = api.place_order(
                category="spot",
                symbol=symbol,
                side=side.upper(),
                orderType="Limit",
                price=f"{px:.10f}",
                qty=qty,
                timeInForce="IOC",
                orderFilter="Order",
                orderLinkId=ensure_link_id(f"TWAP-{i}-{int(time.time())}"),
            )
            replies.append(r)
            log("twap.child", i=i, price=px, qty=qty, side=side, symbol=symbol, resp=r)
        except Exception as e:
            log("twap.error", i=i, error=str(e))
        time.sleep(max(0, int(child_secs)))
    return replies