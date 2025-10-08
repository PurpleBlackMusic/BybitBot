from __future__ import annotations

import json
import threading
import time
import ssl
import random
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Iterable, Optional

import websocket  # websocket-client

from copy import deepcopy

from .envs import get_settings, get_api_client
from .helpers import ensure_link_id
from .paths import DATA_DIR
from .store import JLStore
from .log import log
from .ws_private_v5 import WSPrivateV5
from .realtime_cache import get_realtime_cache
from .spot_pnl import spot_inventory_and_pnl
from .spot_market import _instrument_limits


class WSManager:
    """Минимальный, но стабильный менеджер WS с авто‑пингом и переподключением."""

    def __init__(self):
        self.s = get_settings()

        # public
        self._pub_running: bool = False
        self._pub_thread: Optional[threading.Thread] = None
        self._pub_ws: Optional[websocket.WebSocketApp] = None
        self._pub_subs: tuple[str, ...] = tuple()
        self._pub_ping_thread: Optional[threading.Thread] = None

        # private
        self._priv: Optional[WSPrivateV5] = None
        self._priv_url: Optional[str] = None

        # stores
        self.pub_store = JLStore(DATA_DIR / "ws" / "public.jsonl", max_lines=2000)
        self.priv_store = JLStore(DATA_DIR / "ws" / "private.jsonl", max_lines=2000)

        # heartbeats
        self.last_beat: float = 0.0
        self.last_public_beat: float = 0.0
        self.last_private_beat: float = 0.0

        # cached payloads for UI/diagnostics
        self._state_lock = threading.Lock()
        self._last_order_update: Optional[dict] = None
        self._last_execution: Optional[dict] = None
        self._realtime_cache = get_realtime_cache()
        self._fill_lock = threading.Lock()

    # ----------------------- Public -----------------------
    def _refresh_settings(self) -> None:
        """Reload settings so WS endpoints respect latest configuration."""
        try:
            self.s = get_settings(force_reload=True)
        except Exception as e:  # pragma: no cover - defensive, rare
            log("ws.settings.refresh.error", err=str(e))

    def _handle_private_beat(self) -> None:
        now = time.time()
        self.last_beat = now
        self.last_private_beat = now

    def _public_url(self) -> str:
        self._refresh_settings()
        return (
            "wss://stream-testnet.bybit.com/v5/public/spot"
            if self.s.testnet
            else "wss://stream.bybit.com/v5/public/spot"
        )

    def start_public(self, subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        subs = tuple(subs)
        self._pub_subs = subs

        # Если уже запущен — просто убедимся в подписке (если соединение активно)
        if self._pub_running and self._pub_ws is not None:
            if self._is_socket_connected(self._pub_ws):
                try:
                    for t in subs:
                        self._pub_ws.send(json.dumps({"op": "subscribe", "args": [t]}))
                except Exception as e:
                    log("ws.public.resub.error", err=str(e))
            return True

        url = self._public_url()
        self._pub_running = True

        def on_open(ws):
            log("ws.public.open", url=url)
            # стартуем лёгкий пинг‑луп (каждые ~20с)
            def _ping_loop():
                try:
                    while self._pub_running and self._pub_ws is ws:
                        try:
                            req = {"op": "ping", "req_id": str(int(time.time() * 1000))}
                            ws.send(json.dumps(req))
                        except Exception as e:
                            log("ws.public.ping.error", err=str(e))
                            break
                        time.sleep(20)
                except Exception as e:
                    log("ws.public.ping.exit", err=str(e))

            try:
                if not (self._pub_ping_thread and self._pub_ping_thread.is_alive()):
                    self._pub_ping_thread = threading.Thread(target=_ping_loop, daemon=True)
                    self._pub_ping_thread.start()
            except Exception as e:
                log("ws.public.ping.start.error", err=str(e))

            # первичная подписка
            try:
                current_subs = tuple(self._pub_subs)
                if current_subs:
                    req = {"op": "subscribe", "args": list(current_subs)}
                    ws.send(json.dumps(req))
            except Exception as e:
                log("ws.public.sub.error", err=str(e))

        def on_message(ws, message: str):
            now = time.time()
            self.last_beat = now
            self.last_public_beat = now
            try:
                obj = json.loads(message)
            except Exception:
                obj = {"raw": message}
            self.pub_store.append(obj)
            self._record_public_payload(obj)
            if isinstance(obj, dict) and obj.get("op") == "pong":
                log("ws.public.pong")

        def on_error(ws, error):
            log("ws.public.error", err=str(error))

        def on_close(ws, code, msg):
            log("ws.public.close", code=code, msg=msg)

        def run():
            backoff = 1.0
            while self._pub_running:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                self._pub_ws = ws
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                if not self._pub_running:
                    break
                # экспоненциальный backoff (cap 60s) + джиттер
                sleep_for = min(backoff, 60.0) + random.uniform(0, 0.5)
                log("ws.public.reconnect.wait", seconds=round(sleep_for, 2))
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 60.0)

                # попытка восстановить подписки при следующем on_open

        self._pub_thread = threading.Thread(target=run, daemon=True)
        self._pub_thread.start()
        return True

    def stop_public(self):
        self._pub_running = False
        try:
            if self._pub_ws:
                self._pub_ws.close()
        except Exception:
            pass
        self._pub_ws = None

    # ----------------------- Private ----------------------
    def _private_url(self) -> str:
        self._refresh_settings()
        return (
            "wss://stream-testnet.bybit.com/v5/private"
            if self.s.testnet
            else "wss://stream.bybit.com/v5/private"
        )

    def start_private(self) -> bool:
        try:
            url = self._private_url()

            if self._priv is not None and self._priv_url != url:
                try:
                    self._priv.stop()
                except Exception as e:  # pragma: no cover - defensive, rare
                    log("ws.private.stop.error", err=str(e))
                finally:
                    self._priv = None

            if self._priv is None:
                def _on_private_msg(message):
                    self.priv_store.append(message)
                    self._handle_private_beat()
                    self._process_private_payload(message)

                self._priv = WSPrivateV5(
                    url=url,
                    on_msg=_on_private_msg,
                )
                self._priv_url = url
            if getattr(self._priv, "is_running", None) and self._priv.is_running():
                return True
            started = self._priv.start()
            if not started:
                self._priv = None
                self._priv_url = None
            return bool(started)
        except Exception as e:
            log("ws.private.start.error", err=str(e))
            self._priv = None
            self._priv_url = None
            return False

    def _is_socket_connected(self, ws: Optional[websocket.WebSocketApp]) -> bool:
        """Return True if the given websocket has an active socket."""

        if ws is None:
            return False

        sock = getattr(ws, "sock", None)
        if sock is None:
            return False

        return bool(getattr(sock, "connected", False))

    def start(self, subs: Iterable[str] | None = None, include_private: bool = True) -> bool:
        """Start public and (optionally) private WebSocket channels."""
        subs_to_use: Iterable[str]
        if subs is None:
            subs_to_use = self._pub_subs or ("tickers.BTCUSDT",)
        else:
            subs_to_use = subs

        ok_public = self.start_public(subs_to_use)
        ok_private = True
        if include_private:
            ok_private = self.start_private()
        return bool(ok_public and ok_private)

    def stop_private(self):
        try:
            if self._priv:
                self._priv.stop()
        except Exception:
            pass
        finally:
            self._priv = None
            self._priv_url = None

    # ----------------------- Utils ------------------------
    def stop_all(self):
        self.stop_public()
        self.stop_private()

    def status(self):
        last_beat = self.last_beat if self.last_beat else None
        now = time.time()
        pub_last = self.last_public_beat or 0.0
        priv_last = self.last_private_beat or 0.0

        pub_age = max(0.0, now - pub_last) if pub_last else None
        priv_age = max(0.0, now - priv_last) if priv_last else None

        private_ws = getattr(self._priv, "_ws", None)

        public_ws = self._pub_ws
        public_socket_connected = False
        if public_ws is not None:
            sock = getattr(public_ws, "sock", None)
            if sock is not None:
                public_socket_connected = bool(getattr(sock, "connected", False))

        public_thread_alive = bool(
            self._pub_thread and getattr(self._pub_thread, "is_alive", lambda: False)()
        )

        public_running = bool(
            self._pub_running
            and (
                (public_thread_alive and public_ws)
                or public_socket_connected
            )
        )
        if not public_running and pub_age is not None and pub_age < 60.0:
            public_running = True

        priv = self._priv
        private_running = False
        if priv is not None:
            is_running = getattr(priv, "is_running", None)
            if callable(is_running):
                try:
                    private_running = bool(is_running())
                except Exception:  # pragma: no cover - defensive
                    private_running = False
            if not private_running:
                priv_thread = getattr(priv, "_thread", None)
                if priv_thread is not None:
                    is_alive = getattr(priv_thread, "is_alive", None)
                    if callable(is_alive):
                        try:
                            private_running = bool(is_alive())
                        except Exception:  # pragma: no cover - defensive
                            private_running = False
                if not private_running and private_ws is not None:
                    private_running = True
        if not private_running and priv_age is not None and priv_age < 90.0:
            private_running = True

        return {
            "public": {
                "running": public_running,
                "subscriptions": list(self._pub_subs),
                "last_beat": self.last_public_beat or None,
                "age_seconds": pub_age,
            },
            "private": {
                "running": private_running,
                "connected": bool(private_ws),
                "last_beat": self.last_private_beat or None,
                "age_seconds": priv_age,
            },
        }

    def _process_private_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return

        topic = str(payload.get("topic") or payload.get("topicName") or "").lower()
        if not topic:
            return

        data = payload.get("data")
        if isinstance(data, dict):
            rows = [data]
        elif isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        else:
            rows = []

        if not rows:
            return

        self._realtime_cache.update_private(topic, {"rows": rows})

        if "execution" in topic:
            try:
                from .pnl import add_execution
            except Exception as exc:  # pragma: no cover - import errors rare
                log("ws.private.execution.import_error", err=str(exc))
                return

            for row in rows:
                try:
                    add_execution(row)
                except Exception as exc:  # pragma: no cover - ledger write failures
                    log("ws.private.execution.persist.error", err=str(exc))
                else:
                    self._record_execution(row)
            try:
                self._handle_execution_fill(rows)
            except Exception as exc:  # pragma: no cover - defensive
                log("ws.private.execution.fill.error", err=str(exc))
        elif "order" in topic:
            for row in rows:
                self._record_order_update(row)

    def _record_public_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return

        topic = payload.get("topic") or payload.get("topicName") or payload.get("requestTopic")
        if isinstance(topic, str):
            topic = topic.strip()
        else:
            topic = ""

        if not topic:
            return

        if not payload.get("data") and not payload.get("result"):
            return

        self._realtime_cache.update_public(str(topic), payload)

    def _record_order_update(self, row: dict) -> None:
        if not isinstance(row, dict):
            return

        update = {
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "status": row.get("orderStatus") or row.get("status"),
            "orderLinkId": row.get("orderLinkId") or row.get("orderId"),
            "cancelType": row.get("cancelType") or row.get("cancelTypeV2"),
            "rejectReason": row.get("rejectReason")
            or row.get("triggerRejectReason")
            or row.get("rejectReasonV2"),
            "updatedTime": row.get("updatedTime")
            or row.get("updateTime")
            or row.get("transactTime"),
            "category": row.get("category"),
        }

        for key in ("orderLinkId", "cancelType", "rejectReason", "updatedTime"):
            value = update.get(key)
            if value is not None:
                update[key] = str(value)

        update["raw"] = deepcopy(row)

        with self._state_lock:
            self._last_order_update = update
        self._realtime_cache.update_private("orders", update)

    def _record_execution(self, row: dict) -> None:
        if not isinstance(row, dict):
            return

        update = {
            "symbol": row.get("symbol"),
            "side": row.get("side"),
            "orderLinkId": row.get("orderLinkId") or row.get("orderId"),
            "execType": row.get("execType"),
            "orderStatus": row.get("orderStatus") or row.get("status"),
            "execQty": row.get("execQty"),
            "execPrice": row.get("execPrice"),
            "execTime": row.get("execTime")
            or row.get("tradeTime")
            or row.get("transactionTime"),
            "category": row.get("category"),
        }

        for key in ("orderLinkId", "execQty", "execPrice", "execTime"):
            value = update.get(key)
            if value is not None:
                update[key] = str(value)

        update["raw"] = deepcopy(row)

        with self._state_lock:
            self._last_execution = update
        self._realtime_cache.update_private("executions", update)

    def latest_order_update(self) -> Optional[dict]:
        with self._state_lock:
            if self._last_order_update is None:
                return None
            return deepcopy(self._last_order_update)

    def latest_execution(self) -> Optional[dict]:
        with self._state_lock:
            if self._last_execution is None:
                return None
            return deepcopy(self._last_execution)

    def _handle_execution_fill(self, rows: list[dict]) -> None:
        fills = [row for row in rows if self._is_fill_row(row)]
        if not fills:
            return

        with self._fill_lock:
            inventory: dict[str, dict[str, object]] = {}
            try:
                inventory = spot_inventory_and_pnl()
            except Exception as exc:
                log("ws.private.inventory.error", err=str(exc))
            else:
                snapshot = {
                    "ts": int(time.time() * 1000),
                    "positions": inventory,
                }
                try:
                    self._realtime_cache.update_private("inventory", snapshot)
                except Exception as exc:
                    log("ws.private.inventory.cache.error", err=str(exc))

            buy_fill_rows: dict[str, dict[str, object]] = {}
            for row in fills:
                raw_symbol = row.get("symbol")
                symbol = str(raw_symbol or "").strip().upper()
                if not symbol:
                    continue
                side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
                if side != "buy":
                    continue
                buy_fill_rows[symbol] = row

            if not buy_fill_rows:
                return

            settings = get_settings()
            config = self._resolve_tp_config(settings)
            if not config:
                return

            try:
                api = get_api_client()
            except Exception as exc:
                log(
                    "ws.private.tp_ladder.api.error",
                    err=str(exc),
                    symbols=sorted(buy_fill_rows),
                )
                return

            limits_cache: dict[str, dict[str, object]] = {}
            for symbol, row in buy_fill_rows.items():
                try:
                    self._regenerate_tp_ladder(
                        row,
                        inventory,
                        config=config,
                        api=api,
                        limits_cache=limits_cache,
                    )
                except Exception as exc:
                    log(
                        "ws.private.tp_ladder.error",
                        err=str(exc),
                        symbol=symbol,
                    )

    def _is_fill_row(self, row: dict) -> bool:
        if not isinstance(row, dict):
            return False
        qty = self._decimal_from(row.get("execQty"))
        if qty > 0:
            return True
        qty = self._decimal_from(row.get("lastExecQty"))
        if qty > 0:
            return True
        exec_type = str(row.get("execType") or "").strip().lower()
        if exec_type in {"trade", "fill"}:
            return True
        status = str(row.get("orderStatus") or row.get("status") or "").strip().lower()
        if status in {"filled", "partiallyfilled", "partially_filled"}:
            return True
        return False

    def _regenerate_tp_ladder(
        self,
        row: dict,
        inventory: dict[str, dict[str, object]] | None,
        *,
        config: list[tuple[Decimal, Decimal]] | None = None,
        api=None,
        limits_cache: dict[str, dict[str, object]] | None = None,
    ) -> None:
        if not isinstance(row, dict):
            return
        raw_symbol = row.get("symbol")
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            return
        side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
        if side != "buy":
            return

        symbol_inventory = None
        if isinstance(inventory, dict):
            symbol_inventory = inventory.get(symbol)
            if symbol_inventory is None and raw_symbol not in (None, symbol):
                symbol_inventory = inventory.get(raw_symbol)
        if not isinstance(symbol_inventory, dict):
            return

        qty = self._decimal_from(symbol_inventory.get("position_qty"))
        avg_cost = self._decimal_from(symbol_inventory.get("avg_cost"))
        if qty <= 0 or avg_cost <= 0:
            return

        if config is None:
            settings = get_settings()
            config = self._resolve_tp_config(settings)
        if not config:
            return

        if api is None:
            try:
                api = get_api_client()
            except Exception as exc:
                log("ws.private.tp_ladder.api.error", err=str(exc), symbol=symbol)
                return

        try:
            if limits_cache is not None and symbol in limits_cache:
                limits = limits_cache[symbol]
            else:
                limits = _instrument_limits(api, symbol)
                if limits_cache is not None and isinstance(limits, dict):
                    limits_cache[symbol] = limits
        except Exception as exc:
            log("ws.private.tp_ladder.instrument.error", err=str(exc), symbol=symbol)
            return

        if not isinstance(limits, dict):
            log("ws.private.tp_ladder.instrument.error", err="invalid limits", symbol=symbol)
            return

        qty_step = self._decimal_from(limits.get("qty_step"), Decimal("0.00000001"))
        if qty_step <= 0:
            qty_step = Decimal("0.00000001")
        price_step = self._decimal_from(limits.get("tick_size"), Decimal("0.00000001"))
        if price_step <= 0:
            price_step = Decimal("0.00000001")
        min_qty = self._decimal_from(limits.get("min_order_qty"))
        if min_qty < 0:
            min_qty = Decimal("0")

        self._cancel_existing_tp_orders(api, symbol)
        self._place_tp_orders(
            api,
            symbol,
            qty,
            avg_cost,
            config,
            qty_step,
            price_step,
            min_qty,
        )

    def _cancel_existing_tp_orders(self, api, symbol: str) -> None:
        try:
            response = api.open_orders(category="spot", symbol=symbol, openOnly=1)
        except Exception as exc:
            log("ws.private.tp_ladder.open_orders.error", err=str(exc), symbol=symbol)
            return

        rows = ((response.get("result") or {}).get("list") or []) if isinstance(response, dict) else []
        requests: list[dict[str, object]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            link = str(item.get("orderLinkId") or item.get("orderLinkID") or "").strip()
            if not link.upper().startswith("AI-TP-"):
                continue
            payload = {
                "symbol": item.get("symbol") or symbol,
                "orderId": item.get("orderId"),
                "orderLinkId": ensure_link_id(link),
            }
            if payload["orderId"] is None and payload["orderLinkId"] is None:
                continue
            requests.append(payload)

        for idx in range(0, len(requests), 10):
            chunk = requests[idx : idx + 10]
            try:
                api.cancel_batch(category="spot", request=chunk)
            except Exception as exc:
                log(
                    "ws.private.tp_ladder.cancel.error",
                    err=str(exc),
                    symbol=symbol,
                    count=len(chunk),
                )
            else:
                log(
                    "ws.private.tp_ladder.cancelled",
                    symbol=symbol,
                    count=len(chunk),
                )

    def _place_tp_orders(
        self,
        api,
        symbol: str,
        total_qty: Decimal,
        avg_cost: Decimal,
        config: list[tuple[Decimal, Decimal]],
        qty_step: Decimal,
        price_step: Decimal,
        min_qty: Decimal,
    ) -> None:
        total_qty = max(total_qty, Decimal("0"))
        if total_qty <= 0:
            return

        allocations: list[tuple[Decimal, Decimal]] = []
        remaining = total_qty

        for idx, (profit_bps, fraction) in enumerate(config):
            if idx == len(config) - 1:
                target = remaining
            else:
                target = total_qty * fraction
            qty = self._round_to_step(target, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if qty > remaining:
                qty = self._round_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if min_qty > 0 and qty < min_qty:
                continue
            remaining -= qty
            allocations.append((profit_bps, qty))

        if remaining > Decimal("0") and allocations:
            extra = self._round_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if extra > 0:
                profit_bps, qty = allocations[-1]
                new_qty = qty + extra
                if min_qty <= 0 or new_qty >= min_qty:
                    allocations[-1] = (profit_bps, new_qty)
                    remaining -= extra

        if not allocations:
            return

        timestamp = int(time.time() * 1000)
        rung_index = 0

        for profit_bps, qty in allocations:
            rung_index += 1
            price = avg_cost * (Decimal("1") + profit_bps / Decimal("10000"))
            price = self._round_to_step(price, price_step, rounding=ROUND_UP)
            if price <= 0:
                continue
            qty_text = self._format_decimal_step(qty, qty_step)
            price_text = self._format_decimal_step(price, price_step)
            link_seed = f"AI-TP-{symbol}-{timestamp}-{rung_index}"
            link_id = ensure_link_id(link_seed) or link_seed
            payload = {
                "category": "spot",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Limit",
                "qty": qty_text,
                "price": price_text,
                "timeInForce": "GTC",
                "orderLinkId": link_id,
                "orderFilter": "Order",
            }
            try:
                api.place_order(**payload)
            except Exception as exc:
                log(
                    "ws.private.tp_ladder.place.error",
                    err=str(exc),
                    symbol=symbol,
                    rung=rung_index,
                )
                continue
            log(
                "ws.private.tp_ladder.place",
                symbol=symbol,
                rung=rung_index,
                qty=qty_text,
                price=price_text,
                profit_bps=str(profit_bps),
            )

    def _resolve_tp_config(self, settings) -> list[tuple[Decimal, Decimal]]:
        levels_raw = getattr(settings, "spot_tp_ladder_bps", "") or ""
        splits_raw = getattr(settings, "spot_tp_ladder_split_pct", "") or ""

        levels: list[Decimal] = []
        for chunk in str(levels_raw).replace(";", ",").split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                levels.append(Decimal(text))
            except (InvalidOperation, ValueError):
                continue

        fractions: list[Decimal] = []
        for chunk in str(splits_raw).replace(";", ",").split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                fractions.append(Decimal(text) / Decimal("100"))
            except (InvalidOperation, ValueError):
                continue

        if not levels or not fractions:
            return []

        while len(fractions) < len(levels):
            fractions.append(fractions[-1] if fractions else Decimal("0"))
        if len(fractions) > len(levels):
            fractions = fractions[: len(levels)]

        total_fraction = sum(fractions)
        if total_fraction <= 0:
            return []

        normalised = [fraction / total_fraction for fraction in fractions]
        return list(zip(levels[: len(normalised)], normalised))

    @staticmethod
    def _decimal_from(value: object, default: Decimal = Decimal("0")) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return default

    @staticmethod
    def _round_to_step(value: Decimal, step: Decimal, *, rounding: str) -> Decimal:
        if step <= 0:
            return value
        multiplier = (value / step).to_integral_value(rounding=rounding)
        return multiplier * step

    @staticmethod
    def _format_decimal_step(value: Decimal, step: Decimal) -> str:
        if step > 0:
            exponent = step.normalize().as_tuple().exponent
            places = abs(exponent) if exponent < 0 else 0
        else:
            exponent = value.normalize().as_tuple().exponent
            places = abs(exponent) if exponent < 0 else 0
        if places > 0:
            text = f"{value:.{places}f}"
        else:
            text = format(
                value.quantize(Decimal("1")) if value == value.to_integral_value() else value.normalize(),
                "f",
            )
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"


manager = WSManager()


# --- Backward‑compat helpers for legacy pages ---
def start(subs: Iterable[str] | None = None, include_private: bool = True) -> bool:
    """Module-level proxy mirroring :meth:`WSManager.start`."""
    return manager.start(subs=subs, include_private=include_private)


def stop(include_private: bool = True) -> bool:
    """Stop WebSocket channels started via :func:`start`."""
    manager.stop_public()
    if include_private:
        manager.stop_private()
    return True
