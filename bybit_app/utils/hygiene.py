from __future__ import annotations

import time

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log


def _normalise_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return str(value).strip()
    except Exception:
        return ""


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


def cancel_twap_leftovers(
    api: BybitAPI,
    *,
    category: str = "spot",
    symbol: str | None = None,
) -> dict[str, object]:
    """Cancel lingering TWAP limit orders that may remain after restarts."""

    try:
        response = api.open_orders(category=category, symbol=symbol, openOnly=1)
    except Exception as exc:
        log("order.hygiene.twap.open_orders.error", err=str(exc), category=category, symbol=symbol)
        return {"total": 0, "batches": [], "error": str(exc)}

    rows = ((response.get("result") or {}).get("list") or []) if isinstance(response, dict) else []
    to_cancel: list[dict[str, object]] = []

    for item in rows:
        if not isinstance(item, dict):
            continue
        tif = _normalise_text(
            item.get("timeInForce")
            or item.get("timeInForceValue")
            or item.get("tif")
            or item.get("time_in_force")
        ).upper()
        if tif and tif != "GTC":
            continue
        order_type = _normalise_text(item.get("orderType") or item.get("orderTypeV2")).lower()
        if order_type and order_type != "limit":
            continue
        link = _normalise_text(item.get("orderLinkId") or item.get("orderLinkID"))
        if not link or not link.upper().startswith("TWAP"):
            continue
        status = _normalise_text(item.get("orderStatus") or item.get("status")).lower()
        if status and status.startswith("cancel"):
            continue
        payload = {
            "category": category,
            "symbol": item.get("symbol"),
            "orderId": item.get("orderId"),
            "orderLinkId": ensure_link_id(link),
        }
        if payload["orderId"] is None and payload["orderLinkId"] is None:
            continue
        to_cancel.append(payload)

    result = {"total": len(to_cancel), "batches": []}
    for idx in range(0, len(to_cancel), 10):
        chunk = to_cancel[idx : idx + 10]
        request = [
            {
                "symbol": entry.get("symbol"),
                "orderId": entry.get("orderId"),
                "orderLinkId": ensure_link_id(entry.get("orderLinkId")),
            }
            for entry in chunk
        ]
        try:
            resp = api.cancel_batch(category=category, request=request)
        except Exception as exc:
            log(
                "order.hygiene.twap.cancel.error",
                err=str(exc),
                count=len(chunk),
                category=category,
                symbol=symbol,
            )
            result.setdefault("errors", []).append(str(exc))
            continue
        result["batches"].append(resp)
        log(
            "order.hygiene.twap.cancelled",
            count=len(chunk),
            category=category,
            symbol=symbol,
            resp=resp,
        )

    return result
