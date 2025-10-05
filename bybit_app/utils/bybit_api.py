from __future__ import annotations
import time, hmac, hashlib, json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode
import requests
from functools import lru_cache
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .envs import Settings

API_MAIN = "https://api.bybit.com"
API_TEST = "https://api-testnet.bybit.com"

from .helpers import ensure_link_id
from .log import log

@dataclass
class BybitCreds:
    key: str
    secret: str
    testnet: bool = True

class BybitAPI:
    def __init__(self, creds: BybitCreds, recv_window: int = 5000, timeout: int = 10000, verify_ssl: bool = True):
        self.creds = creds
        self.recv_window = int(recv_window)
        self.timeout = int(timeout)
        self.verify_ssl = bool(verify_ssl)
        self.session = requests.Session()

    @property
    def base(self) -> str:
        return API_TEST if self.creds.testnet else API_MAIN

    # --- signing helpers ---
    def _headers(self, ts: str, sign: str):
        return {
            "X-BAPI-API-KEY": self.creds.key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

    def _sign(self, ts: str, query_or_body: str):
        payload = f"{ts}{self.creds.key}{self.recv_window}{query_or_body}"
        return hmac.new(self.creds.secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _req(self, method: str, path: str, params: dict | None = None, body: dict | None = None, signed: bool = False):
        url = self.base + path
        if not signed:
            if method.upper() == "GET":
                r = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
            else:
                r = self.session.request(
                    method.upper(),
                    url,
                    params=None,
                    data=json.dumps(body or {}),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers={"Content-Type": "application/json"},
                )
            r.raise_for_status()
            return r.json()

        # signed
        ts = str(int(time.time() * 1000))
        if method.upper() == "GET":
            ordered_params = sorted((params or {}).items())
            q = urlencode(ordered_params)
            sign = self._sign(ts, q)
            headers = self._headers(ts, sign)
            r = self.session.get(
                url,
                params=ordered_params,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        else:
            q = json.dumps(body or {}, separators=(',', ':'), ensure_ascii=False, sort_keys=True)
            sign = self._sign(ts, q)
            headers = self._headers(ts, sign)
            r = self.session.request(method.upper(), url, params=None, data=q, headers=headers,
                                     timeout=self.timeout, verify=self.verify_ssl)
        r.raise_for_status()
        return r.json()

    def _safe_req(self, method: str, path: str, params=None, body=None, signed=False):
        resp = self._req(method, path, params=params, body=body, signed=signed)
        # bybit v5 формат: {retCode, retMsg, result, ...}
        if isinstance(resp, dict):
            ret_code = resp.get("retCode", 0)
            if isinstance(ret_code, str):
                try:
                    ret_code = int(ret_code)
                except ValueError:
                    pass
            if ret_code != 0:
                raise RuntimeError(
                    f"Bybit error {resp.get('retCode')}: {resp.get('retMsg')} ({path})"
                )
        return resp

    @staticmethod
    def _normalise_numeric_fields(payload: dict[str, object], numeric_fields: set[str]) -> None:
        for key in numeric_fields & payload.keys():
            value = payload[key]
            if value is None or isinstance(value, bool):
                continue

            try:
                if isinstance(value, Decimal):
                    dec_value = value
                elif isinstance(value, int):
                    dec_value = Decimal(value)
                elif isinstance(value, float):
                    dec_value = Decimal(str(value))
                elif isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    dec_value = Decimal(stripped)
                else:
                    continue
            except (InvalidOperation, ValueError, TypeError):
                continue

            normalised = format(dec_value.normalize(), "f")
            if "." in normalised:
                normalised = normalised.rstrip("0").rstrip(".")

            payload[key] = normalised or "0"

    @staticmethod
    def _sanitise_order_link_id(payload: dict[str, object], key: str = "orderLinkId") -> None:
        if key not in payload:
            return

        sanitised = ensure_link_id(payload.get(key))
        if sanitised is None:
            payload.pop(key, None)
        else:
            payload[key] = sanitised

    # --- public market ---
    def server_time(self):
        return self._safe_req("GET", "/v5/market/time")

    def instruments_info(self, category: str = "spot", symbol: str | None = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._safe_req("GET", "/v5/market/instruments-info", params=params)

    def tickers(self, category: str = "spot", symbol: str | None = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._safe_req("GET", "/v5/market/tickers", params=params)

    def orderbook(self, category: str, symbol: str, limit: int = 50):
        params = {"category": category, "symbol": symbol, "limit": int(limit)}
        return self._safe_req("GET", "/v5/market/orderbook", params=params)

    def kline(self, category: str, symbol: str, interval: int = 1, limit: int = 200):
        params = {"category": category, "symbol": symbol, "interval": str(interval), "limit": int(limit)}
        return self._safe_req("GET", "/v5/market/kline", params=params)

    # --- private ---
    def wallet_balance(self, accountType: str = "UNIFIED"):
        params = {"accountType": accountType}
        return self._safe_req("GET", "/v5/account/wallet-balance", params=params, signed=True)

    def open_orders(
        self,
        category: str = "spot",
        symbol: str | None = None,
        openOnly: int = 1,
        cursor: str | None = None,
    ):
        params = {"category": category, "openOnly": int(openOnly)}
        if symbol:
            params["symbol"] = symbol
        if cursor:
            stripped = cursor.strip()
            if stripped:
                params["cursor"] = stripped
        return self._safe_req("GET", "/v5/order/realtime", params=params, signed=True)

    def execution_list(
        self,
        category: str = "spot",
        symbol: str | None = None,
        start: int | None = None,
        end: int | None = None,
        limit: int = 50,
    ):
        """Fetch recent trade executions for the authenticated account.

        Mirrors the ``GET /v5/execution/list`` endpoint and is used by health
        checks to confirm that real trades are flowing through the account.
        """

        params: dict[str, object] = {"category": category, "limit": int(limit)}
        if symbol:
            params["symbol"] = symbol
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        return self._safe_req("GET", "/v5/execution/list", params=params, signed=True)

    def fee_rate(
        self,
        category: str = "spot",
        symbol: str | None = None,
        baseCoin: str | None = None,
    ):
        """Fetch taker/maker fee rates for the given instrument.

        Mirrors the ``GET /v5/account/fee-rate`` endpoint documented by Bybit.
        The endpoint is private and therefore requires an authenticated request.
        """

        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        if baseCoin:
            params["baseCoin"] = baseCoin
        return self._safe_req("GET", "/v5/account/fee-rate", params=params, signed=True)

    def place_order(self, **kwargs):
        """Submit a single signed order with basic validation and normalisation."""

        if not kwargs:
            raise ValueError("place_order requires parameters")

        payload = dict(kwargs)

        required_fields = {"category", "symbol", "side", "orderType"}
        missing = [field for field in required_fields if not payload.get(field)]
        if missing:
            raise ValueError(
                "place_order requires fields: " + ", ".join(sorted(missing))
            )

        quantity_keys = {"qty", "orderQty", "orderValue", "notional"}
        if not any(key in payload and payload[key] is not None for key in quantity_keys):
            raise ValueError(
                "place_order requires a quantity such as `qty` or `orderValue`"
            )

        if isinstance(payload["side"], str):
            normalised_side = payload["side"].capitalize()
            if normalised_side not in {"Buy", "Sell"}:
                raise ValueError("place_order side must be 'buy' or 'sell'")
            payload["side"] = normalised_side

        numeric_fields = {
            "qty",
            "orderQty",
            "price",
            "triggerPrice",
            "takeProfit",
            "stopLoss",
            "basePrice",
            "tpTriggerPrice",
            "slTriggerPrice",
            "notional",
            "orderValue",
        }

        self._normalise_numeric_fields(payload, numeric_fields)
        self._sanitise_order_link_id(payload)

        response = self._safe_req(
            "POST", "/v5/order/create", body=payload, signed=True
        )

        self._self_check_order_action(
            action="place",
            payload=payload,
            response=response,
        )

        return response

    def cancel_order(self, **kwargs):
        """Cancel an active order through the signed v5 endpoint."""

        if not kwargs:
            raise ValueError("cancel_order requires parameters")

        payload = dict(kwargs)

        if not payload.get("category"):
            raise ValueError("cancel_order requires `category`")

        if not (payload.get("orderId") or payload.get("orderLinkId")):
            raise ValueError("cancel_order requires `orderId` or `orderLinkId`")

        self._sanitise_order_link_id(payload)

        response = self._safe_req("POST", "/v5/order/cancel", body=payload, signed=True)

        self._self_check_order_action(
            action="cancel",
            payload=payload,
            response=response,
        )

        return response

    def _self_check_order_action(
        self,
        *,
        action: str,
        payload: Mapping[str, object],
        response: Mapping[str, object] | None,
    ) -> None:
        """Verify that order placement/cancellation affected Bybit state."""

        try:
            if action not in {"place", "cancel"}:
                return

            category_raw = payload.get("category")
            category = str(category_raw).strip() if isinstance(category_raw, str) else None
            if not category:
                return

            result = response.get("result") if isinstance(response, Mapping) else None
            if not isinstance(result, Mapping):
                result = {}

            status_value: object = result.get("orderStatus")
            if status_value is None:
                status_value = payload.get("orderStatus")

            status_str: str | None
            if isinstance(status_value, str):
                status_str = status_value.strip() or None
            elif status_value is None:
                status_str = None
            else:
                status_str = str(status_value).strip() or None

            status_key: str | None
            if status_str:
                status_key = status_str.lower().replace("_", "").replace(" ", "")
            else:
                status_key = None

            link_value: object = payload.get("orderLinkId")
            if not isinstance(link_value, str) or not link_value.strip():
                link_value = result.get("orderLinkId")

            order_link_id = ensure_link_id(link_value) if isinstance(link_value, str) else None

            order_id_value: object = payload.get("orderId")
            if not isinstance(order_id_value, str) or not order_id_value.strip():
                order_id_value = result.get("orderId")

            order_id = str(order_id_value).strip() if isinstance(order_id_value, (str, int)) else None
            if order_id == "":
                order_id = None

            if not order_link_id and not order_id:
                return

            symbol_value: object = payload.get("symbol")
            if not isinstance(symbol_value, str) or not symbol_value.strip():
                symbol_value = result.get("symbol")

            symbol = str(symbol_value).strip().upper() if isinstance(symbol_value, str) else None

            params = {"category": category, "openOnly": 1}
            if symbol:
                params["symbol"] = symbol

            found = False
            cursor: str | None = None
            seen_cursors: set[str] = set()
            max_pages = 5

            for _ in range(max_pages):
                if cursor:
                    params["cursor"] = cursor
                elif "cursor" in params:
                    params.pop("cursor", None)

                try:
                    orders_payload = self.open_orders(**params)
                except Exception as exc:  # pragma: no cover - network/runtime errors
                    log(
                        f"order.self_check.{action}.error",
                        category=category,
                        symbol=symbol,
                        orderLinkId=order_link_id,
                        orderId=order_id,
                        err=str(exc),
                    )
                    return

                orders_source = (orders_payload or {}).get("result") if isinstance(orders_payload, Mapping) else None
                orders = orders_source.get("list") if isinstance(orders_source, Mapping) else []

                if isinstance(orders, Iterable):
                    for row in orders:
                        if not isinstance(row, Mapping):
                            continue

                        if order_link_id:
                            candidate_link = row.get("orderLinkId")
                            if (
                                isinstance(candidate_link, str)
                                and ensure_link_id(candidate_link) == order_link_id
                            ):
                                found = True
                                break

                        if order_id and not found:
                            candidate_id = row.get("orderId")
                            if (
                                isinstance(candidate_id, (str, int))
                                and str(candidate_id).strip() == order_id
                            ):
                                found = True
                                break

                if found:
                    break

                next_cursor_raw = orders_source.get("nextPageCursor") if isinstance(orders_source, Mapping) else None
                if not isinstance(next_cursor_raw, str):
                    break

                next_cursor = next_cursor_raw.strip()
                if not next_cursor:
                    break

                if next_cursor in seen_cursors:
                    break

                seen_cursors.add(next_cursor)
                cursor = next_cursor

            expect_present = action == "place"

            CLOSED_STATUS_KEYS = {
                "filled",
                "cancelled",
                "canceled",
                "rejected",
                "deactivated",
                "expired",
                "closed",
                "terminated",
                "completed",
            }

            OPEN_STATUS_KEYS = {
                "created",
                "new",
                "pendingnew",
                "partiallyfilled",
                "partiallyfilledcanceled",
                "partiallyfilledcancelled",
                "active",
                "pendingcancel",
                "triggered",
                "untriggered",
                "live",
                "open",
                "accepted",
            }

            if status_key:
                if status_key in CLOSED_STATUS_KEYS:
                    expect_present = False
                elif action == "place" and status_key in OPEN_STATUS_KEYS:
                    expect_present = True

            ok = found if expect_present else not found

            log(
                f"order.self_check.{action}",
                category=category,
                symbol=symbol,
                orderLinkId=order_link_id,
                orderId=order_id,
                found=found,
                ok=ok,
                status=status_str,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log(
                "order.self_check.exception",
                action=action,
                err=str(exc),
            )

    def cancel_all(self, category: str | None = None, **kwargs):
        """Cancel multiple orders in bulk through the signed v5 endpoint."""

        if not category:
            raise ValueError("cancel_all requires `category`")

        payload: dict[str, object] = {"category": category}

        for key, value in kwargs.items():
            if value is None:
                continue

            if key == "orderLinkId":
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    value = stripped
                sanitised = ensure_link_id(value)
                if sanitised is None:
                    continue
                payload[key] = sanitised
                continue

            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                payload[key] = stripped
            else:
                payload[key] = value

        return self._safe_req(
            "POST", "/v5/order/cancel-all", body=payload, signed=True
        )

    def amend_order(self, **kwargs):
        """Amend an existing active order via the signed v5 endpoint."""

        if not kwargs:
            raise ValueError("amend_order requires parameters")

        payload = dict(kwargs)

        if not payload.get("category"):
            raise ValueError("amend_order requires `category`")

        if not (payload.get("orderId") or payload.get("orderLinkId")):
            raise ValueError("amend_order requires `orderId` or `orderLinkId`")

        numeric_fields = {
            "qty",
            "price",
            "triggerPrice",
            "takeProfit",
            "stopLoss",
            "basePrice",
            "tpTriggerPrice",
            "slTriggerPrice",
        }

        self._normalise_numeric_fields(payload, numeric_fields)
        self._sanitise_order_link_id(payload)

        return self._safe_req("POST", "/v5/order/amend", body=payload, signed=True)

    def cancel_batch(
        self,
        *,
        category: str | None = None,
        request: Iterable[Mapping[str, object]] | None = None,
        requests: Iterable[Mapping[str, object]] | None = None,
        **kwargs,
    ):
        if not category:
            raise ValueError("cancel_batch requires `category`")

        payload_iterable = request if request is not None else requests
        if payload_iterable is None:
            raise ValueError("cancel_batch requires `request` or `requests`")

        if isinstance(payload_iterable, list):
            payload_list = payload_iterable
        else:
            payload_list = list(payload_iterable)

        if not payload_list:
            raise ValueError("cancel_batch requires a non-empty request list")

        normalised_requests: list[dict[str, object]] = []
        for index, entry in enumerate(payload_list):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    "cancel_batch entries must be mappings; "
                    f"got {type(entry).__name__} at index {index}"
                )

            request_payload = dict(entry)
            if not (
                request_payload.get("orderId") or request_payload.get("orderLinkId")
            ):
                raise ValueError(
                    "cancel_batch entries require `orderId` or `orderLinkId`"
                )

            self._sanitise_order_link_id(request_payload)
            normalised_requests.append(request_payload)

        body: dict[str, object] = {
            "category": category,
            "request": normalised_requests,
        }

        for key, value in kwargs.items():
            if value is None:
                continue

            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                body[key] = stripped
            else:
                body[key] = value

        return self._safe_req(
            "POST", "/v5/order/cancel-batch", body=body, signed=True
        )

    def batch_cancel(
        self,
        category: str,
        requests: Iterable[Mapping[str, object]] | None = None,
        request: Iterable[Mapping[str, object]] | None = None,
        **kwargs,
    ):
        """Backwards compatible alias for :meth:`cancel_batch`.

        Legacy callers historically passed either ``requests`` or ``request``
        as the payload key. Accept both so that unsynchronised deployments keep
        working while still funnelling the request through the signed helper.
        The payload may be any iterable of request dictionaries; it will be
        materialised into a list before dispatch to guard against generators
        being consumed twice downstream. Any extra keyword arguments (for
        example ``symbol``) are forwarded unchanged to :meth:`cancel_batch`.
        """

        payload_iterable = request if request is not None else requests
        if payload_iterable is None:
            raise ValueError("batch_cancel requires `requests` or `request`")

        return self.cancel_batch(
            category=category,
            request=payload_iterable,
            **kwargs,
        )

    def batch_place(
        self,
        category: str,
        orders: Iterable[Mapping[str, object]] | None = None,
    ):
        """Place up to 10 orders in a single signed request."""

        if not category:
            raise ValueError("batch_place requires `category`")

        if orders is None:
            raise ValueError("batch_place requires `orders`")

        if isinstance(orders, list):
            order_list = orders
        else:
            order_list = list(orders)

        if not order_list:
            raise ValueError("batch_place requires a non-empty order list")

        if len(order_list) > 10:
            raise ValueError("batch_place supports up to 10 orders per request")

        numeric_fields = {
            "qty",
            "price",
            "triggerPrice",
            "takeProfit",
            "stopLoss",
            "basePrice",
            "tpTriggerPrice",
            "slTriggerPrice",
            "notional",
            "orderValue",
        }

        required_fields = {"symbol", "side", "orderType"}
        quantity_keys = {"qty", "orderQty", "orderValue", "notional"}

        normalised_orders: list[dict[str, object]] = []
        for index, entry in enumerate(order_list):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    "batch_place entries must be mappings; "
                    f"got {type(entry).__name__} at index {index}"
                )

            payload = dict(entry)

            missing = [field for field in required_fields if not payload.get(field)]
            if missing:
                raise ValueError(
                    "batch_place entries require fields: " + ", ".join(sorted(missing))
                )

            if not any(key in payload and payload[key] is not None for key in quantity_keys):
                raise ValueError(
                    "batch_place entries require a quantity such as `qty` or `orderValue`"
                )

            if isinstance(payload["side"], str):
                normalised_side = payload["side"].capitalize()
                if normalised_side not in {"Buy", "Sell"}:
                    raise ValueError("batch_place side must be 'buy' or 'sell'")
                payload["side"] = normalised_side

            self._normalise_numeric_fields(payload, numeric_fields)
            self._sanitise_order_link_id(payload)

            normalised_orders.append(payload)

        request_body = {"category": category, "request": normalised_orders}
        return self._safe_req(
            "POST", "/v5/order/create-batch", body=request_body, signed=True
        )


@lru_cache(maxsize=16)
def _build_api(key: str, secret: str, testnet: bool, recv_window: int, timeout: int, verify_ssl: bool) -> BybitAPI:
    creds = BybitCreds(key=key, secret=secret, testnet=testnet)
    return BybitAPI(creds, recv_window=recv_window, timeout=timeout, verify_ssl=verify_ssl)


def get_api(
    creds: BybitCreds,
    recv_window: int = 5000,
    timeout: int = 10000,
    verify_ssl: bool = True,
) -> BybitAPI:
    """Return a cached ``BybitAPI`` instance for the given credentials.

    Reusing the underlying :class:`requests.Session` keeps connection pools warm
    and avoids the overhead of creating short-lived clients on every call.
    """

    return _build_api(
        creds.key or "",
        creds.secret or "",
        bool(creds.testnet),
        int(recv_window),
        int(timeout),
        bool(verify_ssl),
    )


def clear_api_cache() -> None:
    """Reset the cached API clients (useful in tests)."""

    _build_api.cache_clear()


def creds_from_settings(settings: "Settings") -> BybitCreds:
    """Build :class:`BybitCreds` from a ``Settings`` instance."""

    return BybitCreds(
        key=getattr(settings, "api_key", "") or "",
        secret=getattr(settings, "api_secret", "") or "",
        testnet=bool(getattr(settings, "testnet", True)),
    )


def api_from_settings(settings: "Settings") -> BybitAPI:
    """Shortcut that returns a cached API client using settings defaults."""

    return get_api(
        creds_from_settings(settings),
        recv_window=int(getattr(settings, "recv_window_ms", 5000)),
        timeout=int(getattr(settings, "http_timeout_ms", 10000)),
        verify_ssl=bool(getattr(settings, "verify_ssl", True)),
    )

# --- metadata used by KillSwitch & API Nanny ---
API_CALLS = {
    "server_time": {"method": "GET", "path": "/v5/market/time"},
    "instruments_info": {"method": "GET", "path": "/v5/market/instruments-info"},
    "tickers": {"method": "GET", "path": "/v5/market/tickers"},
    "orderbook": {"method": "GET", "path": "/v5/market/orderbook"},
    "kline": {"method": "GET", "path": "/v5/market/kline"},
    "wallet_balance": {"method": "GET", "path": "/v5/account/wallet-balance"},
    "open_orders": {"method": "GET", "path": "/v5/order/realtime"},
    "place_order": {"method": "POST", "path": "/v5/order/create"},
    "cancel_order": {"method": "POST", "path": "/v5/order/cancel"},
    "cancel_all": {"method": "POST", "path": "/v5/order/cancel-all"},
    "amend_order": {"method": "POST", "path": "/v5/order/amend"},
    "cancel_batch": {"method": "POST", "path": "/v5/order/cancel-batch"},
    "batch_cancel": {"method": "POST", "path": "/v5/order/cancel-batch"},
    "fee_rate": {"method": "GET", "path": "/v5/account/fee-rate"},
    "batch_place": {"method": "POST", "path": "/v5/order/create-batch"},
}
