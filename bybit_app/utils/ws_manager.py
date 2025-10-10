from __future__ import annotations

import json
import re
import threading
import time
import ssl
import random
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
from typing import Iterable, Mapping, Optional, Sequence

import websocket  # websocket-client

from copy import deepcopy

from .envs import get_settings, get_api_client, creds_ok
from .helpers import ensure_link_id
from .paths import DATA_DIR
from .pnl import read_ledger
from .store import JLStore
from .log import log
from .ws_private_v5 import WSPrivateV5
from .realtime_cache import get_realtime_cache
from .spot_pnl import spot_inventory_and_pnl
from .spot_market import _instrument_limits
from .precision import format_to_step, quantize_to_step
from .telegram_notify import enqueue_telegram_message
from .trade_notifications import format_sell_close_message


_BYBIT_ERROR = re.compile(r"Bybit error (?P<code>-?\d+): (?P<message>.+)")
_TP_LADDER_SKIP_CODES = {"170194", "170131"}


def _extract_error_code(exc: Exception) -> Optional[str]:
    match = _BYBIT_ERROR.search(str(exc))
    if match:
        return match.group("code")
    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class WSManager:
    """Минимальный, но стабильный менеджер WS с авто‑пингом и переподключением."""

    _LEDGER_RECOVERY_LIMIT = 800

    def __init__(self):
        self.s = get_settings()

        # public
        self._pub_running: bool = False
        self._pub_thread: Optional[threading.Thread] = None
        self._pub_ws: Optional[websocket.WebSocketApp] = None
        self._pub_subs: tuple[str, ...] = tuple()
        self._pub_ping_thread: Optional[threading.Thread] = None
        self._pub_url_override: Optional[str] = None
        self._pub_current_url: Optional[str] = None

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
        self._tp_ladder_plan: dict[str, dict[str, object]] = {}
        self._inventory_snapshot: dict[str, dict[str, Decimal]] = {}
        self._inventory_baseline: dict[str, dict[str, Decimal]] = {}
        self._inventory_last_exec_id: Optional[str] = None

    # ----------------------- Public -----------------------
    def _refresh_settings(self) -> None:
        """Reload settings so WS endpoints respect latest configuration."""
        try:
            self.s = get_settings(force_reload=True)
        except TypeError:
            # Older call signatures may not accept the keyword argument.
            self.s = get_settings()
        except Exception as e:  # pragma: no cover - defensive, rare
            log("ws.settings.refresh.error", err=str(e))

    def _handle_private_beat(self) -> None:
        now = time.time()
        self.last_beat = now
        self.last_private_beat = now

    def _public_url(self) -> str:
        self._refresh_settings()
        base_url = (
            "wss://stream-testnet.bybit.com/v5/public/spot"
            if self.s.testnet
            else "wss://stream.bybit.com/v5/public/spot"
        )
        if not self.s.testnet and self._pub_url_override is not None:
            self._pub_url_override = None
        return self._pub_url_override or base_url

    def _fallback_public_to_mainnet(self, reason: str) -> None:
        if not self.s.testnet:
            return
        self._pub_url_override = None
        log("ws.public.testnet.network_error", reason=reason)

    @staticmethod
    def _is_network_error(error: object) -> bool:
        if isinstance(error, OSError):
            return True
        text = str(error or "").lower()
        network_signals = (
            "name or service not known",
            "temporary failure in name resolution",
            "getaddrinfo failed",
            "timed out",
            "connection refused",
            "cannot assign requested address",
        )
        return any(token in text for token in network_signals)

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

        self._pub_running = True

        def on_open(ws):
            log("ws.public.open", url=self._pub_current_url)
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
            log("ws.public.error", err=str(error), url=self._pub_current_url)
            if self._is_network_error(error):
                self._fallback_public_to_mainnet(str(error))

        def on_close(ws, code, msg):
            log("ws.public.close", code=code, msg=msg, url=self._pub_current_url)

        def run():
            backoff = 1.0
            while self._pub_running:
                current_url = self._public_url()
                self._pub_current_url = current_url
                verify_ssl = bool(getattr(self.s, "verify_ssl", True))
                cert_reqs = ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE
                sslopt = {"cert_reqs": cert_reqs}
                ws = websocket.WebSocketApp(
                    current_url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                self._pub_ws = ws
                ws.run_forever(sslopt=sslopt)
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
        finally:
            self._pub_current_url = None
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
            settings = get_settings()
        except Exception:
            settings = None
        if settings is not None:
            self.s = settings
        self._prime_inventory_snapshot()
        if settings is not None and not creds_ok(settings):
            log("ws.private.disabled", reason="missing credentials")
            self._priv = None
            self._priv_url = None
            return False
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

    def autostart(
        self,
        include_private: bool = True,
        *,
        subs: Iterable[str] | None = None,
    ) -> tuple[bool, bool]:
        self._refresh_settings()
        settings = self.s
        if not getattr(settings, "ws_autostart", False):
            return False, False

        status_snapshot: Optional[dict[str, object]]
        try:
            status_snapshot = self.status()
        except Exception:  # pragma: no cover - defensive
            status_snapshot = None

        threshold = _safe_float(getattr(settings, "ws_watchdog_max_age_sec", None))
        if threshold is None or threshold <= 0:
            threshold = None

        def _age(info: object) -> Optional[float]:
            if isinstance(info, dict):
                return _safe_float(info.get("age_seconds"))
            return None

        if threshold is not None and status_snapshot:
            public_info = status_snapshot.get("public")  # type: ignore[assignment]
            private_info = status_snapshot.get("private")  # type: ignore[assignment]
            public_age = _age(public_info)
            private_age = _age(private_info)
            if public_age is not None and public_age > threshold:
                log(
                    "ws.autostart.public.restart",
                    age=round(public_age, 2),
                    threshold=threshold,
                )
                self.stop_public()
                status_snapshot = None
            if include_private and private_age is not None and private_age > threshold:
                log(
                    "ws.autostart.private.restart",
                    age=round(private_age, 2),
                    threshold=threshold,
                )
                self.stop_private()
                status_snapshot = None

        if status_snapshot is None:
            try:
                status_snapshot = self.status()
            except Exception:  # pragma: no cover - defensive
                status_snapshot = None

        def _running(info: object) -> bool:
            if isinstance(info, dict):
                return bool(info.get("running"))
            return False

        public_info = status_snapshot.get("public") if isinstance(status_snapshot, dict) else None
        private_info = status_snapshot.get("private") if isinstance(status_snapshot, dict) else None
        public_running = _running(public_info)
        private_running = _running(private_info)

        started_public = False
        if not public_running:
            subscriptions = tuple(subs) if subs is not None else self._pub_subs or ("tickers.BTCUSDT",)
            try:
                self.start_public(subs=subscriptions)
            except Exception as exc:  # pragma: no cover - defensive network guard
                log(
                    "ws.autostart.public.error",
                    err=str(exc),
                    subs=list(subscriptions),
                )
            else:
                started_public = True

        started_private = False
        if include_private and creds_ok(settings):
            if not private_running:
                try:
                    started_private = bool(self.start_private())
                except Exception as exc:  # pragma: no cover - defensive network guard
                    log("ws.autostart.private.error", err=str(exc))

        return started_public, started_private

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
            previous_inventory = {
                symbol: dict(stats)
                for symbol, stats in self._inventory_snapshot.items()
                if isinstance(stats, Mapping)
            }
            if not self._inventory_baseline and previous_inventory:
                self._inventory_baseline = {
                    symbol: dict(stats)
                    for symbol, stats in previous_inventory.items()
                }

            inventory: dict[str, dict[str, object]] = {}
            normalized_inventory = previous_inventory
            inventory_updated = False
            try:
                inventory = spot_inventory_and_pnl(settings=self.s)
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
                normalized_inventory = self._normalise_inventory_snapshot(inventory)
                inventory_updated = True

            buy_fill_rows: dict[str, dict[str, object]] = {}
            sell_fill_rows: dict[str, list[dict[str, object]]] = {}
            for row in fills:
                raw_symbol = row.get("symbol")
                symbol = str(raw_symbol or "").strip().upper()
                if not symbol:
                    continue
                side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
                if side == "buy":
                    buy_fill_rows[symbol] = row
                elif side == "sell":
                    sell_fill_rows.setdefault(symbol, []).append(row)

            if sell_fill_rows and inventory_updated:
                self._notify_sell_fills(
                    sell_fill_rows,
                    normalized_inventory,
                    previous_inventory,
                )

            last_exec_id = self._inventory_last_exec_id
            for row in rows:
                marker = self._ledger_entry_id(row)
                if marker:
                    last_exec_id = marker

            self._inventory_snapshot = normalized_inventory
            self._inventory_last_exec_id = last_exec_id

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
                        settings=settings,
                    )
                except Exception as exc:
                    log(
                        "ws.private.tp_ladder.error",
                        err=str(exc),
                        symbol=symbol,
                    )

    def _notify_sell_fills(
        self,
        fills_by_symbol: Mapping[str, Sequence[Mapping[str, object]]],
        inventory_snapshot: Mapping[str, Mapping[str, Decimal]],
        previous_snapshot: Mapping[str, Mapping[str, Decimal]],
    ) -> None:
        settings = self.s
        notify_enabled = bool(
            getattr(settings, "telegram_notify", False)
            or getattr(settings, "tg_trade_notifs", False)
        )
        if not notify_enabled:
            log("telegram.trade.skip", reason="notifications_disabled")
            return

        for symbol, rows in fills_by_symbol.items():
            if not rows:
                continue
            symbol_upper = str(symbol or "").upper()
            current_stats = inventory_snapshot.get(symbol_upper) or inventory_snapshot.get(symbol)
            if not isinstance(current_stats, Mapping):
                log(
                    "telegram.trade.skip",
                    symbol=symbol_upper,
                    reason="inventory_missing",
                )
                continue
            previous_stats = previous_snapshot.get(symbol_upper) or previous_snapshot.get(symbol)
            if not isinstance(previous_stats, Mapping):
                previous_stats = self._recover_previous_stats(
                    symbol_upper,
                    rows,
                )
            if not isinstance(previous_stats, Mapping):
                log(
                    "telegram.trade.skip",
                    symbol=symbol_upper,
                    reason="missing_previous_inventory",
                )
                continue

            total_qty = Decimal("0")
            total_quote = Decimal("0")
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                qty = self._decimal_from(row.get("execQty"))
                if qty <= 0:
                    qty = self._decimal_from(row.get("lastExecQty"))
                price = self._decimal_from(row.get("execPrice"))
                if qty <= 0:
                    continue
                total_qty += qty
                if price > 0:
                    total_quote += price * qty

            if total_qty <= 0:
                log(
                    "telegram.trade.skip",
                    symbol=symbol_upper,
                    reason="no_sell_qty",
                )
                continue

            avg_price = Decimal("0")
            if total_quote > 0:
                avg_price = total_quote / total_qty
            else:
                for row in rows:
                    price = self._decimal_from(row.get("execPrice"))
                    if price > 0:
                        avg_price = price
                        break

            if avg_price <= 0:
                log(
                    "telegram.trade.skip",
                    symbol=symbol_upper,
                    reason="invalid_avg_price",
                )
                continue

            current_realized = self._decimal_from(current_stats.get("realized_pnl"))
            previous_realized = self._decimal_from(previous_stats.get("realized_pnl"))
            trade_realized = current_realized - previous_realized
            remaining_qty = self._decimal_from(current_stats.get("position_qty"))

            base_asset = symbol_upper[:-4] if symbol_upper.endswith("USDT") else symbol_upper
            qty_step = self._infer_step_from_rows(rows, "execQty")
            price_step = self._infer_step_from_rows(rows, "execPrice")
            qty_text = self._format_decimal_step(total_qty, qty_step)
            price_text = self._format_decimal_step(avg_price, price_step)
            pnl_display = trade_realized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pnl_text = f"{pnl_display:+.2f} USDT"

            remainder_text = None
            position_closed = remaining_qty <= Decimal("0")
            if not position_closed:
                remainder_qty_text = self._format_decimal_step(remaining_qty, qty_step)
                remainder_text = f"{remainder_qty_text} {base_asset}"

            message = format_sell_close_message(
                symbol=symbol_upper,
                qty_text=qty_text,
                base_asset=base_asset,
                price_text=price_text,
                pnl_text=pnl_text,
                remainder_text=remainder_text,
                position_closed=position_closed,
            )

            try:
                enqueue_telegram_message(message)
            except Exception as exc:  # pragma: no cover - defensive guard
                log(
                    "telegram.trade.error",
                    symbol=symbol_upper,
                    error=str(exc),
                )
            else:
                log(
                    "telegram.trade.notify",
                    symbol=symbol_upper,
                    side="sell",
                    qty=str(total_qty),
                    price=str(avg_price),
                    pnl=str(trade_realized),
                )

    def _prime_inventory_snapshot(self) -> None:
        with self._fill_lock:
            needs_bootstrap = not self._inventory_snapshot or not self._inventory_baseline
        if not needs_bootstrap:
            return
        try:
            inventory = spot_inventory_and_pnl(settings=self.s)
        except Exception as exc:
            log("ws.private.inventory.bootstrap.error", err=str(exc))
            return
        normalized = self._normalise_inventory_snapshot(inventory)
        with self._fill_lock:
            if not self._inventory_snapshot:
                self._inventory_snapshot = normalized
            if not self._inventory_baseline:
                self._inventory_baseline = {
                    symbol: dict(stats)
                    for symbol, stats in normalized.items()
                }

    def _recover_previous_stats(
        self,
        symbol: str,
        rows: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Decimal] | None:
        recovered = self._previous_stats_from_ledger(symbol, rows)
        if isinstance(recovered, Mapping):
            return recovered
        baseline = self._inventory_baseline.get(symbol)
        if isinstance(baseline, Mapping):
            return dict(baseline)
        return None

    def _previous_stats_from_ledger(
        self,
        symbol: str,
        rows: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Decimal] | None:
        limit = self._LEDGER_RECOVERY_LIMIT
        last_exec_id = self._inventory_last_exec_id
        read_kwargs: dict[str, object] = {"settings": self.s}
        limited_read = False
        if last_exec_id:
            read_kwargs["last_exec_id"] = last_exec_id
            if limit:
                read_kwargs["n"] = limit
        else:
            if limit:
                read_kwargs["n"] = limit
                limited_read = True
        try:
            ledger_rows = read_ledger(**read_kwargs)
        except Exception as exc:
            log("ws.private.inventory.ledger.error", err=str(exc))
            return None
        if not ledger_rows:
            return None

        signature_counts: Counter[tuple[object, ...]] = Counter()
        for row in rows:
            signature = self._row_signature(row)
            if signature is not None:
                signature_counts[signature] += 1

        if not signature_counts:
            return None

        def _filter_rows(
            candidates: Sequence[Mapping[str, object]]
        ) -> tuple[list[Mapping[str, object]], bool]:
            counts = Counter(signature_counts)
            filtered: list[Mapping[str, object]] = []
            removed_any = False
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                signature = self._row_signature(candidate)
                if signature is not None and counts.get(signature, 0) > 0:
                    counts[signature] -= 1
                    removed_any = True
                    continue
                filtered.append(candidate)
            return filtered, removed_any

        filtered_rows, removed = _filter_rows(ledger_rows)

        if not removed and limited_read:
            try:
                ledger_rows = read_ledger(None, settings=self.s)
            except Exception as exc:
                log("ws.private.inventory.ledger.error", err=str(exc))
                return None
            filtered_rows, removed = _filter_rows(ledger_rows)

        if not removed:
            return None

        try:
            inventory = spot_inventory_and_pnl(events=filtered_rows, settings=self.s)
        except Exception as exc:
            log("ws.private.inventory.recover.error", err=str(exc))
            return None

        normalized = self._normalise_inventory_snapshot(inventory)
        stats = normalized.get(symbol)
        if not isinstance(stats, Mapping):
            return None
        return dict(stats)

    @staticmethod
    def _row_signature(row: Mapping[str, object] | None) -> tuple[object, ...] | None:
        if not isinstance(row, Mapping):
            return None
        entry_id = WSManager._ledger_entry_id(row)
        if entry_id:
            return ("exec_id", entry_id)
        symbol = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or row.get("orderSide") or "").lower()
        qty = str(row.get("execQty") or row.get("lastExecQty") or "")
        price = str(row.get("execPrice") or "")
        ts = str(
            row.get("execTime")
            or row.get("transactTime")
            or row.get("time")
            or ""
        )
        if not symbol and not qty and not price:
            return None
        return ("fields", symbol, side, qty, price, ts)

    @staticmethod
    def _ledger_entry_id(row: Mapping[str, object] | None) -> Optional[str]:
        if not isinstance(row, Mapping):
            return None
        for key in (
            "execId",
            "executionId",
            "execID",
            "execKey",
            "tradeId",
            "fillId",
            "matchId",
        ):
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _normalise_inventory_snapshot(
        inventory: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, dict[str, Decimal]]:
        snapshot: dict[str, dict[str, Decimal]] = {}
        if not isinstance(inventory, Mapping):
            return snapshot

        for key, stats in inventory.items():
            if not isinstance(stats, Mapping):
                continue
            symbol = str(key or "").upper()
            if not symbol:
                continue
            snapshot[symbol] = {
                "position_qty": WSManager._decimal_from(stats.get("position_qty")),
                "avg_cost": WSManager._decimal_from(stats.get("avg_cost")),
                "realized_pnl": WSManager._decimal_from(stats.get("realized_pnl")),
            }

        return snapshot

    @staticmethod
    def _infer_step_from_rows(
        rows: Sequence[Mapping[str, object]],
        key: str,
        default: Decimal = Decimal("0.00000001"),
    ) -> Decimal:
        step: Decimal | None = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            value = WSManager._decimal_from(row.get(key))
            if value <= 0:
                continue
            exponent = value.normalize().as_tuple().exponent
            candidate = Decimal("1")
            if exponent < 0:
                candidate = Decimal("1").scaleb(exponent)
            if step is None or candidate < step:
                step = candidate
        if step is None:
            return default
        return step

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
        settings=None,
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

        settings_obj = settings
        if config is None:
            if settings_obj is None:
                try:
                    settings_obj = get_settings()
                except Exception:
                    settings_obj = None
            config = self._resolve_tp_config(settings_obj) if settings_obj else []
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
        min_notional = self._decimal_from(limits.get("min_order_amt"))
        price_band_min = self._decimal_from(limits.get("min_price"))
        price_band_max = self._decimal_from(limits.get("max_price"))

        reserved = self._reserved_sell_qty(symbol)
        available_qty = qty - reserved
        if reserved > 0 and qty_step > 0 and available_qty > qty_step:
            available_qty -= qty_step
        if available_qty <= 0:
            return

        plan = self._build_tp_plan(
            total_qty=available_qty,
            avg_cost=avg_cost,
            config=config,
            qty_step=qty_step,
            price_step=price_step,
            min_qty=min_qty,
            min_notional=min_notional,
            min_price=price_band_min,
            max_price=price_band_max,
        )
        if not plan:
            return

        signature = tuple((entry["price_text"], entry["qty_text"]) for entry in plan)
        previous_meta = self._tp_ladder_plan.get(symbol) or {}
        previous_signature = None
        previous_avg_cost = Decimal("0")
        previous_qty = Decimal("0")
        if isinstance(previous_meta, Mapping):
            existing_sig = previous_meta.get("signature")
            if isinstance(existing_sig, tuple):
                previous_signature = existing_sig
            previous_avg_cost = self._decimal_from(previous_meta.get("avg_cost"))
            previous_qty = self._decimal_from(previous_meta.get("qty"))
        if previous_signature == signature:
            log("ws.private.tp_ladder.skip", symbol=symbol, reason="unchanged")
            return

        threshold_bps = Decimal("0")
        qty_threshold = qty_step
        if settings_obj is None:
            try:
                settings_obj = get_settings()
            except Exception:
                settings_obj = None
        if settings_obj is not None:
            threshold_raw = getattr(settings_obj, "spot_tp_reprice_threshold_bps", 0) or 0
            threshold_bps = self._decimal_from(threshold_raw)
            qty_threshold_raw = getattr(settings_obj, "spot_tp_reprice_qty_buffer", None)
            if qty_threshold_raw is not None:
                candidate = self._decimal_from(qty_threshold_raw)
                if candidate > qty_threshold:
                    qty_threshold = candidate

        price_change = abs(avg_cost - previous_avg_cost)
        allowed_delta = Decimal("0")
        if threshold_bps > 0 and avg_cost > 0:
            allowed_delta = (avg_cost * threshold_bps) / Decimal("10000")
        qty_change = abs(available_qty - previous_qty)
        if (
            previous_signature is not None
            and allowed_delta > 0
            and price_change < allowed_delta
            and qty_threshold > 0
            and qty_change <= qty_threshold
        ):
            log(
                "ws.private.tp_ladder.skip",
                symbol=symbol,
                reason="below_threshold",
                price_change=str(price_change.normalize()),
                qty_change=str(qty_change.normalize()),
            )
            return

        self._cancel_existing_tp_orders(api, symbol)
        self._execute_tp_plan(api, symbol, plan)
        self._tp_ladder_plan[symbol] = {
            "signature": signature,
            "avg_cost": avg_cost,
            "qty": available_qty,
            "updated_ts": time.time(),
        }

    def private_snapshot(self) -> Mapping[str, object] | None:
        cache = self._realtime_cache
        if cache is None or not hasattr(cache, "snapshot"):
            return None
        try:
            snapshot = cache.snapshot(private_ttl=None)
        except Exception:
            return None
        if isinstance(snapshot, Mapping):
            return snapshot
        return None

    def realtime_private_rows(
        self,
        topic_keyword: str,
        *,
        snapshot: Mapping[str, object] | None = None,
    ) -> list[Mapping[str, object]]:
        if snapshot is None:
            snapshot = self.private_snapshot()
        if not isinstance(snapshot, Mapping):
            return []

        private = snapshot.get("private") if isinstance(snapshot, Mapping) else None
        if not isinstance(private, Mapping):
            return []

        rows: list[Mapping[str, object]] = []
        keyword = topic_keyword.lower()
        for topic, record in private.items():
            topic_key = str(topic).lower()
            if keyword not in topic_key:
                continue
            if not isinstance(record, Mapping):
                continue
            payload = record.get("payload")
            candidates: Sequence[object] | None = None
            if isinstance(payload, Mapping):
                maybe_rows = payload.get("rows")
                if isinstance(maybe_rows, Sequence):
                    candidates = maybe_rows
            elif isinstance(payload, Sequence):
                candidates = payload
            if not candidates:
                continue
            for entry in candidates:
                if isinstance(entry, Mapping):
                    rows.append(entry)
        return rows

    # Backwards compatibility for legacy callers that still reference the
    # private helper name.
    def _realtime_private_rows(
        self,
        topic_keyword: str,
        *,
        snapshot: Mapping[str, object] | None = None,
    ) -> list[Mapping[str, object]]:
        return self.realtime_private_rows(topic_keyword, snapshot=snapshot)

    def _reserved_sell_qty(self, symbol: str) -> Decimal:
        rows = self.realtime_private_rows("order")
        if not rows:
            return Decimal("0")

        symbol_upper = symbol.upper()
        reserved: dict[str, Decimal] = {}
        total = Decimal("0")

        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").strip().upper() != symbol_upper:
                continue
            side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
            if side != "sell":
                continue
            order_type = str(row.get("orderType") or row.get("orderTypeV2") or "").strip().lower()
            if order_type and order_type != "limit":
                continue
            status = str(row.get("orderStatus") or row.get("status") or "").strip().lower()
            if status:
                closed_prefixes = (
                    "cancel",
                    "reject",
                    "filled",
                    "trigger",
                    "inactive",
                    "deactivate",
                    "expire",
                )
                if any(status.startswith(prefix) for prefix in closed_prefixes):
                    continue

            qty = self._decimal_from(row.get("leavesQty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("qty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("orderQty"))
            if qty <= 0:
                continue

            key: Optional[str] = None
            for candidate_key in ("orderId", "orderID", "orderLinkId", "orderLinkID"):
                candidate = row.get(candidate_key)
                if isinstance(candidate, str) and candidate.strip():
                    key = candidate.strip()
                    break
            if key is None:
                key = f"anon-{id(row)}"

            previous = reserved.get(key)
            if previous is not None:
                total -= previous
            reserved[key] = qty
            total += qty

        return total

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

    def _apply_price_band(
        self,
        price: Decimal,
        *,
        price_step: Decimal,
        min_price: Decimal,
        max_price: Decimal,
    ) -> Decimal:
        adjusted = price
        if min_price > 0 and adjusted < min_price:
            adjusted = min_price
        if max_price > 0 and adjusted > max_price:
            adjusted = max_price
        adjusted = quantize_to_step(adjusted, price_step, rounding=ROUND_UP)
        if min_price > 0 and adjusted < min_price:
            adjusted = quantize_to_step(min_price, price_step, rounding=ROUND_UP)
        if max_price > 0 and adjusted > max_price:
            adjusted = quantize_to_step(max_price, price_step, rounding=ROUND_DOWN)
        return adjusted

    def _build_tp_plan(
        self,
        *,
        total_qty: Decimal,
        avg_cost: Decimal,
        config: list[tuple[Decimal, Decimal]],
        qty_step: Decimal,
        price_step: Decimal,
        min_qty: Decimal,
        min_notional: Decimal,
        min_price: Decimal,
        max_price: Decimal,
    ) -> list[dict[str, object]]:
        total_qty = max(total_qty, Decimal("0"))
        if total_qty <= 0 or not config:
            return []

        if min_notional > 0 and avg_cost > 0:
            min_qty_from_notional = quantize_to_step(
                min_notional / avg_cost, qty_step, rounding=ROUND_UP
            )
            if min_qty_from_notional > min_qty:
                min_qty = min_qty_from_notional

        allocations: list[tuple[Decimal, Decimal]] = []
        remaining = total_qty

        for idx, (profit_bps, fraction) in enumerate(config):
            target = remaining if idx == len(config) - 1 else total_qty * fraction
            qty = quantize_to_step(target, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if qty > remaining:
                qty = quantize_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if min_qty > 0 and qty < min_qty:
                continue
            remaining -= qty
            allocations.append((profit_bps, qty))

        if remaining > Decimal("0") and allocations:
            extra = quantize_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if extra > 0:
                profit_bps, qty = allocations[-1]
                new_qty = qty + extra
                if min_qty <= 0 or new_qty >= min_qty:
                    allocations[-1] = (profit_bps, new_qty)

        if not allocations:
            return []

        aggregated: list[dict[str, object]] = []
        for profit_bps, qty in allocations:
            price_raw = avg_cost * (Decimal("1") + profit_bps / Decimal("10000"))
            price = quantize_to_step(price_raw, price_step, rounding=ROUND_UP)
            price = self._apply_price_band(
                price,
                price_step=price_step,
                min_price=min_price,
                max_price=max_price,
            )
            if price <= 0:
                continue
            qty_payload = quantize_to_step(qty, qty_step, rounding=ROUND_DOWN)
            if qty_payload <= 0:
                continue
            if min_qty > 0 and qty_payload < min_qty:
                continue
            merged = None
            for entry in aggregated:
                if entry["price"] == price:
                    merged = entry
                    break
            if merged:
                merged["qty"] += qty_payload
                merged["steps"].append(profit_bps)
            else:
                aggregated.append({"price": price, "qty": qty_payload, "steps": [profit_bps]})

        plan: list[dict[str, object]] = []
        for entry in aggregated:
            qty_payload = quantize_to_step(entry["qty"], qty_step, rounding=ROUND_DOWN)
            if qty_payload <= 0:
                continue
            if min_qty > 0 and qty_payload < min_qty:
                continue
            qty_text = format_to_step(qty_payload, qty_step, rounding=ROUND_DOWN)
            price = self._apply_price_band(
                entry["price"],
                price_step=price_step,
                min_price=min_price,
                max_price=max_price,
            )
            if price <= 0:
                continue
            if min_notional > 0 and price * qty_payload < min_notional:
                continue
            price_text = format_to_step(price, price_step, rounding=ROUND_UP)
            if not qty_text or qty_text == "0":
                continue
            if not price_text or price_text == "0":
                continue
            profit_labels = [str(step.normalize()) for step in entry["steps"]]
            plan.append(
                {
                    "qty": qty_payload,
                    "qty_text": qty_text,
                    "price": entry["price"],
                    "price_text": price_text,
                    "profit_labels": profit_labels,
                }
            )

        return plan

    def _execute_tp_plan(
        self,
        api,
        symbol: str,
        plan: list[dict[str, object]],
    ) -> None:
        if not plan:
            return

        timestamp = int(time.time() * 1000)
        for rung_index, entry in enumerate(plan, start=1):
            qty_text = str(entry.get("qty_text") or "0")
            price_text = str(entry.get("price_text") or "0")
            if qty_text == "0" or price_text == "0":
                continue
            profit_labels = entry.get("profit_labels") or []
            profit_text = ",".join(profit_labels) if profit_labels else "-"
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
            final_payload = payload
            try:
                api.place_order(**payload)
            except Exception as exc:
                error_code = _extract_error_code(exc)
                if error_code in _TP_LADDER_SKIP_CODES:
                    log(
                        "ws.private.tp_ladder.skip",
                        symbol=symbol,
                        rung=rung_index,
                        qty=qty_text,
                        price=price_text,
                        code=error_code,
                    )
                    continue
                if error_code == "170372":
                    retry_payload = self._reprice_tp_rung_to_ceiling(
                        api,
                        symbol,
                        final_payload,
                        qty_text=qty_text,
                        rung_index=rung_index,
                    )
                    if retry_payload is None:
                        continue
                    final_payload = retry_payload
                    price_text = str(final_payload.get("price") or price_text)
                else:
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
                profit_bps=profit_text,
            )

    def _reprice_tp_rung_to_ceiling(
        self,
        api,
        symbol: str,
        payload: Mapping[str, object],
        *,
        qty_text: str,
        rung_index: int,
    ) -> Optional[dict[str, object]]:
        try:
            limits = _instrument_limits(api, symbol)
        except Exception as exc:
            log(
                "ws.private.tp_ladder.reprice.limits_error",
                symbol=symbol,
                rung=rung_index,
                err=str(exc),
            )
            enqueue_telegram_message(
                f"⚠️ Не удалось обновить TP-рог {symbol} #{rung_index}: ошибка получения лимитов ({exc})."
            )
            return None

        max_price = self._decimal_from(limits.get("max_price"))
        tick_size = self._decimal_from(limits.get("tick_size"))
        min_notional = self._decimal_from(limits.get("min_order_amt"))
        qty_decimal = self._decimal_from(qty_text)
        price_decimal = self._decimal_from(payload.get("price"))

        if qty_decimal <= 0 or max_price <= 0:
            log(
                "ws.private.tp_ladder.reprice.skip",
                symbol=symbol,
                rung=rung_index,
                reason="invalid_limits",
                qty=str(qty_decimal),
                max_price=str(max_price),
            )
            enqueue_telegram_message(
                f"⚠️ TP-рог {symbol} #{rung_index} не скорректирован: некорректные лимиты (qty={qty_decimal}, max={max_price})."
            )
            return None

        capped_price = min(price_decimal, max_price)
        capped_price = quantize_to_step(capped_price, tick_size, rounding=ROUND_DOWN)
        if capped_price <= 0:
            log(
                "ws.private.tp_ladder.reprice.skip",
                symbol=symbol,
                rung=rung_index,
                reason="non_positive_price",
                price=str(capped_price),
            )
            enqueue_telegram_message(
                f"⚠️ TP-рог {symbol} #{rung_index} не скорректирован: цена после округления неположительна ({capped_price})."
            )
            return None

        notional = capped_price * qty_decimal
        if min_notional > 0 and notional < min_notional:
            log(
                "ws.private.tp_ladder.reprice.skip",
                symbol=symbol,
                rung=rung_index,
                reason="min_notional",
                notional=str(notional),
                min_notional=str(min_notional),
            )
            enqueue_telegram_message(
                f"⚠️ TP-рог {symbol} #{rung_index} не скорректирован: объём {notional.normalize()} "
                f"ниже минимального {min_notional.normalize()}"
            )
            return None

        adjusted_price_text = format_to_step(capped_price, tick_size, rounding=ROUND_DOWN)
        updated_payload = dict(payload)
        updated_payload["price"] = adjusted_price_text

        try:
            api.place_order(**updated_payload)
        except Exception as exc:
            log(
                "ws.private.tp_ladder.reprice.error",
                symbol=symbol,
                rung=rung_index,
                err=str(exc),
                price=adjusted_price_text,
            )
            enqueue_telegram_message(
                f"⚠️ TP-рог {symbol} #{rung_index} не скорректирован: повторная отправка не удалась ({exc})."
            )
            return None

        log(
            "ws.private.tp_ladder.reprice.applied",
            symbol=symbol,
            rung=rung_index,
            original_price=str(price_decimal),
            adjusted_price=adjusted_price_text,
            max_price=str(max_price),
        )
        return updated_payload

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
        if not isinstance(step, Decimal):
            try:
                step = Decimal(str(step))
            except Exception:
                step = Decimal("0")

        if step > 0:
            value = WSManager._round_to_step(value, step, rounding=ROUND_DOWN)

        exponent = step.normalize().as_tuple().exponent if step > 0 else value.normalize().as_tuple().exponent
        places = abs(exponent) if exponent < 0 else 0

        if places > 0:
            text = f"{value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN):.{places}f}"
        else:
            quantized = value.quantize(Decimal("1"), rounding=ROUND_DOWN) if value == value.to_integral_value() else value.normalize()
            text = format(quantized, "f")

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
