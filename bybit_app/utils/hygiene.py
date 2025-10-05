from __future__ import annotations

import time

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log


def cancel_stale_orders(
    api: BybitAPI,
    category: str = "spot",
    symbol: str | None = None,
    older_than_sec: int = 900,
    batch_size: int = 10,
):
    """Отменить застоявшиеся ордера с подписью пользователя."""

    response = api.open_orders(category=category, symbol=symbol, openOnly=1)
    rows = ((response.get("result") or {}).get("list") or []) if response else []
    now_ms = int(time.time() * 1000)

    to_cancel: list[dict[str, object]] = []
    for item in rows:
        created = int(item.get("createdTime") or item.get("updatedTime") or 0)
        if now_ms - created < older_than_sec * 1000:
            continue
        to_cancel.append(
            {
                "category": category,
                "symbol": item.get("symbol"),
                "orderId": item.get("orderId"),
                "orderLinkId": item.get("orderLinkId"),
            }
        )

    result = {"total": len(to_cancel), "batches": []}
    for idx in range(0, len(to_cancel), batch_size):
        chunk = to_cancel[idx : idx + batch_size]
        payload = [
            {
                "symbol": entry["symbol"],
                "orderId": entry["orderId"],
                "orderLinkId": ensure_link_id(entry.get("orderLinkId")),
            }
            for entry in chunk
        ]
        resp = api.cancel_batch(category=category, request=payload)
        result["batches"].append(resp)
        log("order.hygiene.cancel_batch", count=len(chunk), resp=resp)

    return result
