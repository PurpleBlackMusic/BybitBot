from __future__ import annotations
import time, hmac, hashlib, json
from dataclasses import dataclass
from urllib.parse import urlencode
import requests

API_MAIN = "https://api.bybit.com"
API_TEST = "https://api-testnet.bybit.com"

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
                r = self.session.get(url, params=params, timeout=self.timeout, verify=self.verify_ssl)
            else:
                r = self.session.request(method.upper(), url, params=None, data=json.dumps(body or {}),
                                         timeout=self.timeout, verify=self.verify_ssl, headers={"Content-Type":"application/json"})
            r.raise_for_status()
            return r.json()

        # signed
        ts = str(int(time.time() * 1000))
        if method.upper() == "GET":
            q = urlencode(sorted((params or {}).items()))
            sign = self._sign(ts, q)
            headers = self._headers(ts, sign)
            r = self.session.get(url, params=params, headers=headers, timeout=self.timeout, verify=self.verify_ssl)
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
        if isinstance(resp, dict) and resp.get("retCode", 0) != 0:
            raise RuntimeError(f"Bybit error {resp.get('retCode')}: {resp.get('retMsg')} ({path})")
        return resp

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

    def open_orders(self, category: str = "spot", symbol: str | None = None, openOnly: int = 1):
        params = {"category": category, "openOnly": int(openOnly)}
        if symbol:
            params["symbol"] = symbol
        return self._safe_req("GET", "/v5/order/realtime", params=params, signed=True)

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
        # required: category, symbol, side, orderType, qty or (notional), [price for Limit]
        return self._safe_req("POST", "/v5/order/create", body=kwargs, signed=True)

    def cancel_order(self, **kwargs):
        return self._safe_req("POST", "/v5/order/cancel", body=kwargs, signed=True)

    def cancel_batch(self, **kwargs):
        return self._safe_req("POST", "/v5/order/cancel-batch", body=kwargs, signed=True)

    def batch_place(self, category: str, orders: list[dict]):
        """Place up to 10 orders in a single request via the batch endpoint."""

        payload = {"category": category, "request": orders}
        return self._safe_req("POST", "/v5/order/create-batch", body=payload, signed=True)

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
    "cancel_batch": {"method": "POST", "path": "/v5/order/cancel-batch"},
    "fee_rate": {"method": "GET", "path": "/v5/account/fee-rate"},
    "batch_place": {"method": "POST", "path": "/v5/order/create-batch"},
}
