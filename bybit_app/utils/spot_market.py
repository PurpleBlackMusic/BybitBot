
from __future__ import annotations
from .bybit_api import BybitAPI
from .log import log

def place_spot_market_with_tolerance(api: BybitAPI, symbol: str, side: str, qty: float, unit: str = "baseCoin", tol_type: str = "Percent", tol_value: float = 0.5):
    """СПОТ маркет ордер с ограничением слиппеджа (Bybit v5: slippageTolerance*).
    Примечание: при использовании толеранса слиппеджа TP/SL не поддерживаются в том же запросе.
    """
    body = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": f"{qty:.10f}",
        "marketUnit": unit,  # 'baseCoin' или 'quoteCoin'
        "slippageToleranceType": tol_type,
        "slippageTolerance": f"{tol_value}",
    }
    r = api._req("POST", "/v5/order/create", json=body)
    log("spot.market.slip", symbol=symbol, side=side, body=body, resp=r)
    return r
