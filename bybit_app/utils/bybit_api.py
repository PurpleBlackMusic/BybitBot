from __future__ import annotations
import hmac, hashlib, json, time, uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode
import threading
import requests
from functools import lru_cache
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any
import time

from tenacity import (
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

if TYPE_CHECKING:
    from .envs import Settings

API_MAIN = "https://api.bybit.com"
API_TEST = "https://api-testnet.bybit.com"

from .helpers import ensure_link_id
from .http_client import create_http_session
from .bybit_errors import (
    BybitErrorPolicy,
    extract_ret_message,
    normalise_ret_code,
    resolve_error_policy,
)
from .log import log
from .time_sync import SyncedTimestamp, invalidate_synced_clock, synced_timestamp


class _RetryableRequestError(RuntimeError):
    """Internal marker for retryable network or exchange failures."""

    def __init__(self, message: str, *, meta: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.meta = dict(meta or {})

@dataclass
class BybitCreds:
    key: str
    secret: str
    testnet: bool = True

@dataclass
class _OrderRegistryEntry:
    signature: str
    payload: dict[str, object]
    response: Mapping[str, object] | None
    timestamp: float


class _LocalOrderRegistry:
    """Keep a bounded record of recently placed orders for idempotency."""

    def __init__(self, *, max_entries: int = 256) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, _OrderRegistryEntry] = OrderedDict()

    def _prune(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def remember(
        self,
        link_id: str,
        *,
        signature: str,
        payload: dict[str, object],
        response: Mapping[str, object] | None,
        timestamp: float,
    ) -> None:
        self._entries[link_id] = _OrderRegistryEntry(
            signature=signature,
            payload=dict(payload),
            response=deepcopy(response) if isinstance(response, Mapping) else response,
            timestamp=timestamp,
        )
        self._entries.move_to_end(link_id)
        self._prune()

    def lookup(self, link_id: str) -> _OrderRegistryEntry | None:
        entry = self._entries.get(link_id)
        if entry is not None:
            self._entries.move_to_end(link_id)
        return entry


class BybitAPI:
    def __init__(self, creds: BybitCreds, recv_window: int = 15000, timeout: int = 10000, verify_ssl: bool = True):
        self.creds = creds
        self.recv_window = int(recv_window)
        self.timeout = int(timeout)
        self.verify_ssl = bool(verify_ssl)
        self._http_local: threading.local = threading.local()
        self._order_registry = _LocalOrderRegistry()
        self._clock_offset_ms: float = 0.0
        self._clock_latency_ms: float = 0.0
        self._quota_lock = threading.Lock()
        self._last_quota: dict[str, object] = {}

    def _thread_local_session(self) -> requests.Session:
        session = getattr(self._http_local, "session", None)
        if session is None:
            session = create_http_session()
            self._http_local.session = session
        return session

    @property
    def session(self) -> requests.Session:
        """Return the thread-local HTTP client used for REST calls."""

        return self._thread_local_session()

    @session.setter
    def session(self, value: requests.Session | None) -> None:
        if value is None:
            if hasattr(self._http_local, "session"):
                delattr(self._http_local, "session")
            return
        self._http_local.session = value

    @property
    def base(self) -> str:
        return API_TEST if self.creds.testnet else API_MAIN

    @property
    def clock_offset_ms(self) -> float:
        """Return the most recent server/local clock offset in milliseconds."""

        return self._clock_offset_ms

    @property
    def clock_latency_ms(self) -> float:
        """Return the last measured half round-trip latency in milliseconds."""

        return self._clock_latency_ms

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

    def _timestamp_ms(self, *, force_refresh: bool = False) -> int:
        timeout_seconds = max(self.timeout / 1000.0, 1.0)
        snapshot: SyncedTimestamp = synced_timestamp(
            self.base,
            session=self.session,
            timeout=timeout_seconds,
            verify=self.verify_ssl,
            force_refresh=force_refresh,
        )
        self._clock_offset_ms = snapshot.offset_ms
        self._clock_latency_ms = snapshot.latency_ms
        return snapshot.as_int()

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
        ts = str(self._timestamp_ms())
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
        self._record_quota_headers(r)
        return r.json()

    def _safe_req(self, method: str, path: str, params=None, body=None, signed=False):
        max_attempts = 5 if signed else 3
        wait_strategy = wait_exponential(multiplier=0.5, max=5.0) + wait_random(0.0, 0.5)

        def _log_retry(retry_state: RetryCallState) -> None:
            if retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                if isinstance(exc, _RetryableRequestError):
                    meta = dict(exc.meta)
                    meta.update(
                        {
                            "attempt": retry_state.attempt_number,
                            "maxAttempts": max_attempts,
                            "method": method,
                            "path": path,
                        }
                    )
                    log("bybit.request.retry", **meta)

        @retry(
            reraise=True,
            wait=wait_strategy,
            stop=stop_after_attempt(max_attempts),
            retry=retry_if_exception_type(_RetryableRequestError),
            after=_log_retry,
        )
        def _execute():
            try:
                resp = self._req(method, path, params=params, body=body, signed=signed)
            except requests.exceptions.HTTPError as exc:
                response = exc.response
                status_code = response.status_code if response is not None else None
                detail = ""
                if response is not None:
                    try:
                        payload = response.json()
                        if isinstance(payload, Mapping):
                            detail = str(payload.get("retMsg") or payload.get("ret_message") or "").strip()
                            if not detail:
                                detail = str(payload.get("message", "")).strip()
                    except ValueError:
                        detail = response.text.strip()
                    if not detail:
                        detail = response.text.strip()
                message = detail or str(exc)
                if status_code == 401:
                    raise RuntimeError(
                        "Bybit authentication failed: please verify API key/secret, permissions, and network selection."
                        f" ({message})"
                    ) from exc

                meta = {"status": status_code, "message": message}
                log("bybit.http_error", method=method, path=path, **meta)

                if status_code in {429, 503}:
                    raise _RetryableRequestError(
                        f"HTTP error {status_code or 'unknown'} while calling {path}: {message}",
                        meta=meta,
                    ) from exc

                raise RuntimeError(
                    f"HTTP error {status_code or 'unknown'} while calling {path}: {message}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                meta = {"err": str(exc)}
                log("bybit.network_error", method=method, path=path, **meta)
                raise _RetryableRequestError(str(exc), meta=meta) from exc

            if not isinstance(resp, dict):
                return resp

            ret_code_value = resp.get("retCode", 0)
            numeric_code, ret_code_text = normalise_ret_code(ret_code_value)

            if numeric_code == 0 or ret_code_text == "0":
                return resp

            ret_msg = extract_ret_message(resp)
            error_meta = {"retCode": ret_code_value, "retMsg": ret_msg or None, "path": path}
            log("bybit.exchange_error", method=method, **error_meta)

            code_for_message = ret_code_text or str(ret_code_value)
            message = f"Bybit error {code_for_message}: {ret_msg or 'unknown error'} ({path})"

            policy = resolve_error_policy(numeric_code)
            if policy.requires_signed and not signed:
                policy = BybitErrorPolicy()

            if policy.invalidate_clock and signed:
                invalidate_synced_clock(self.base)
                self._timestamp_ms(force_refresh=True)
            if policy.retryable:
                raise _RetryableRequestError(message, meta=error_meta)

            raise RuntimeError(message)

        try:
            return _execute()
        except RetryError as exc:
            final = exc.last_attempt.exception()
            if isinstance(final, BaseException):
                raise final
            raise

    def _record_quota_headers(self, response: requests.Response | None) -> None:
        if response is None:
            return
        headers = getattr(response, "headers", None)
        if not headers:
            return

        quota_fields: dict[str, object] = {}
        for key, value in headers.items():
            lower_key = key.lower()
            if "quota" not in lower_key and "ratelimit" not in lower_key:
                continue
            if value is None:
                continue
            parsed: object = value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    try:
                        if "." in stripped:
                            parsed = float(stripped)
                        else:
                            parsed = int(stripped)
                    except ValueError:
                        parsed = stripped
                else:
                    parsed = stripped
            quota_fields[key] = parsed

        if not quota_fields:
            return

        quota_fields["updated_at"] = time.time()
        with self._quota_lock:
            self._last_quota = quota_fields

    @property
    def quota_snapshot(self) -> dict[str, object]:
        with self._quota_lock:
            return dict(self._last_quota)

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

    @staticmethod
    def _is_successful_response(response: Mapping[str, object] | None) -> bool:
        if not isinstance(response, Mapping):
            return False
        ret_code = response.get("retCode")
        if ret_code is None:
            return False
        try:
            return int(ret_code) == 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, (list, tuple, set)):
            return [BybitAPI._json_safe(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): BybitAPI._json_safe(val) for key, val in value.items()}
        return value

    @staticmethod
    def _order_log_meta(data: Mapping[str, Any]) -> dict[str, Any]:
        category = data.get("category")
        symbol = data.get("symbol")
        side = data.get("side")
        order_type = data.get("orderType")
        order_link = data.get("orderLinkId")
        return {
            "category": str(category) if category is not None else None,
            "symbol": str(symbol).upper() if symbol is not None else None,
            "side": str(side).capitalize() if isinstance(side, str) else side,
            "orderType": str(order_type) if order_type is not None else None,
            "orderLinkId": str(order_link) if order_link is not None else None,
        }

    # --- public market ---
    def server_time(self):
        return self._safe_req("GET", "/v5/market/time")

    def instruments_info(
        self,
        category: str = "spot",
        symbol: str | None = None,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        if limit is not None:
            try:
                params["limit"] = int(limit)
            except (TypeError, ValueError):
                pass
        if cursor:
            stripped = cursor.strip()
            if stripped:
                params["cursor"] = stripped
        return self._safe_req("GET", "/v5/market/instruments-info", params=params)

    def tickers(self, category: str = "spot", symbol: str | None = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._safe_req("GET", "/v5/market/tickers", params=params)

    def orderbook(self, category: str, symbol: str, limit: int = 50):
        params = {"category": category, "symbol": symbol, "limit": int(limit)}
        return self._safe_req("GET", "/v5/market/orderbook", params=params)

    def kline(
        self,
        category: str,
        symbol: str,
        interval: int = 1,
        limit: int = 200,
        start: int | None = None,
        end: int | None = None,
    ):
        params = {"category": category, "symbol": symbol, "interval": str(interval), "limit": int(limit)}
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
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

        existing_link = payload.get("orderLinkId")
        if not isinstance(existing_link, str) or not existing_link.strip():
            payload["orderLinkId"] = str(uuid.uuid4())

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

        pre_normalised = {key: payload.get(key) for key in numeric_fields if key in payload}

        self._normalise_numeric_fields(payload, numeric_fields)
        self._sanitise_order_link_id(payload)

        order_link_id: str | None
        link_candidate = payload.get("orderLinkId")
        if isinstance(link_candidate, str) and link_candidate.strip():
            order_link_id = link_candidate.strip()
        else:
            order_link_id = None

        json_payload = self._json_safe(payload)
        if isinstance(json_payload, Mapping):
            json_payload = dict(json_payload)
        payload_signature = json.dumps(
            json_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        log_meta = self._order_log_meta(payload)

        if order_link_id:
            existing = self._order_registry.lookup(order_link_id)
            if existing is not None:
                if existing.signature != payload_signature:
                    log(
                        "order.idempotent.conflict",
                        **log_meta,
                        existingPayload=existing.payload,
                    )
                    raise ValueError(
                        "orderLinkId reused with different parameters; "
                        "generate a new link id to submit a new order"
                    )
                log("order.idempotent.skip", **log_meta)
                return deepcopy(existing.response) if isinstance(existing.response, Mapping) else existing.response

        normalised_changes = {}
        for key, before in pre_normalised.items():
            after = payload.get(key)
            if before != after:
                normalised_changes[key] = {
                    "before": self._json_safe(before),
                    "after": self._json_safe(after),
                }

        if normalised_changes:
            log("order.normalise", **log_meta, fields=normalised_changes)

        request_payload = self._json_safe(payload)
        log("order.request", **log_meta, payload=request_payload)

        try:
            response = self._safe_req(
                "POST", "/v5/order/create", body=payload, signed=True
            )
        except Exception as exc:
            log("order.error", **log_meta, err=str(exc))
            raise

        self._self_check_order_action(
            action="place",
            payload=payload,
            response=response,
        )

        response_payload = self._json_safe(response)
        log("order.response", **log_meta, payload=response_payload)

        if order_link_id and self._is_successful_response(response):
            json_response = (
                dict(response_payload)
                if isinstance(response_payload, Mapping)
                else response
            )
            self._order_registry.remember(
                order_link_id,
                signature=payload_signature,
                payload=json_payload if isinstance(json_payload, dict) else {},
                response=json_response if isinstance(json_response, Mapping) else None,
                timestamp=time.time(),
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

            category_for_check = "spot" if action == "place" else category
            if not category_for_check:
                category_for_check = "spot" if action == "place" else None
            if not category_for_check:
                return

            max_attempts = 5 if action == "place" else 1
            delay_seconds = 0.3

            def _search_realtime_orders() -> tuple[bool, bool]:
                params: dict[str, object] = {"category": category_for_check}
                if symbol:
                    params["symbol"] = symbol
                if order_link_id:
                    params["orderLinkId"] = order_link_id
                if order_id:
                    params["orderId"] = order_id

                try:
                    orders_payload = self._safe_req(
                        "GET",
                        "/v5/order/realtime",
                        params=params,
                        signed=True,
                    )
                except Exception as exc:  # pragma: no cover - network/runtime errors
                    log(
                        f"order.self_check.{action}.error",
                        category=category_for_check,
                        symbol=symbol,
                        orderLinkId=order_link_id,
                        orderId=order_id,
                        err=str(exc),
                    )
                    return False, True

                orders_source: object
                if isinstance(orders_payload, Mapping):
                    orders_source = orders_payload.get("result")
                else:
                    orders_source = None

                if isinstance(orders_source, Mapping):
                    orders = orders_source.get("list")
                else:
                    orders = orders_source

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
                                return True, False

                        if order_id:
                            candidate_id = row.get("orderId")
                            if (
                                isinstance(candidate_id, (str, int))
                                and str(candidate_id).strip() == order_id
                            ):
                                return True, False

                return False, False

            def _search_open_orders() -> tuple[bool, bool]:
                params: dict[str, object] = {"category": category_for_check, "openOnly": 1}
                if symbol:
                    params["symbol"] = symbol
                if order_link_id:
                    params["orderLinkId"] = order_link_id
                if order_id:
                    params["orderId"] = order_id

                cursor: str | None = None
                seen_cursors: set[str] = set()
                max_pages = 5

                for _ in range(max_pages):
                    if cursor:
                        params["cursor"] = cursor
                    elif "cursor" in params:
                        params.pop("cursor", None)

                    try:
                        orders_payload = self._safe_req(
                            "GET",
                            "/v5/order/realtime",
                            params=params,
                            signed=True,
                        )
                    except Exception as exc:  # pragma: no cover - network/runtime errors
                        log(
                            f"order.self_check.{action}.error",
                            category=category_for_check,
                            symbol=symbol,
                            orderLinkId=order_link_id,
                            orderId=order_id,
                            err=str(exc),
                        )
                        return False, True

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
                                    return True, False

                            if order_id:
                                candidate_id = row.get("orderId")
                                if (
                                    isinstance(candidate_id, (str, int))
                                    and str(candidate_id).strip() == order_id
                                ):
                                    return True, False

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

                return False, False

            search_fn = _search_realtime_orders if action == "place" else _search_open_orders

            found = False
            aborted = False
            for attempt in range(max_attempts):
                found, aborted = search_fn()
                if found or aborted:
                    break
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)

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
                checkCategory=category_for_check,
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
    recv_window: int = 15000,
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
        creds.testnet,
        int(recv_window),
        int(timeout),
        bool(verify_ssl),
    )


def clear_api_cache() -> None:
    """Reset the cached API clients (useful in tests)."""

    _build_api.cache_clear()
    invalidate_synced_clock(API_MAIN)
    invalidate_synced_clock(API_TEST)


def creds_from_settings(settings: "Settings") -> BybitCreds:
    """Build :class:`BybitCreds` from a ``Settings`` instance."""

    if hasattr(settings, "get_api_key") and hasattr(settings, "get_api_secret"):
        key = settings.get_api_key()
        secret = settings.get_api_secret()
    else:
        key = getattr(settings, "api_key", "") or ""
        secret = getattr(settings, "api_secret", "") or ""

    if not key:
        key = getattr(settings, "api_key", "") or ""
    if not secret:
        secret = getattr(settings, "api_secret", "") or ""

    return BybitCreds(
        key=key or "",
        secret=secret or "",
        testnet=settings.testnet,
    )


def api_from_settings(settings: "Settings") -> BybitAPI:
    """Shortcut that returns a cached API client using settings defaults."""

    return get_api(
        creds_from_settings(settings),
        recv_window=int(getattr(settings, "recv_window_ms", 15000)),
        timeout=int(getattr(settings, "http_timeout_ms", 10000)),
        verify_ssl=settings.verify_ssl,
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
