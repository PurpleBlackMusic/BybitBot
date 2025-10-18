"""Market data, liquidity, and symbol selection helpers for the executor."""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .bybit_api import BybitAPI
from .envs import Settings
from .log import log
from .signal_executor_models import _IMPULSE_SIGNAL_THRESHOLD, _safe_float, _safe_symbol
from .ws_orderbook import LiveOrderbook


class SignalExecutorMarketMixin:
    """Encapsulate websocket, orderbook, and symbol resolution helpers."""

    def _ensure_ws_activity(self, settings: Settings) -> None:
        """Ensure private websocket streams are started when requested."""

        if not getattr(settings, "ws_autostart", False):
            return

        manager = self._ws_manager()
        autostart = getattr(manager, "autostart", None) if manager is not None else None
        if not callable(autostart):
            return

        try:
            autostart(include_private=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("guardian.auto.ws.autostart.error", err=str(exc))

    def _live_orderbook_for_api(self, api: Optional[object]) -> Optional[LiveOrderbook]:
        if api is None or not isinstance(api, BybitAPI):
            return None
        instance = getattr(self, "_live_orderbook", None)
        if instance is None or instance.api is not api:
            instance = LiveOrderbook(api, category="spot")
            self._live_orderbook = instance  # type: ignore[attr-defined]
        return instance

    def _resolve_orderbook_top(
        self,
        api: Optional[object],
        symbol: Optional[str],
        *,
        limit: int = 1,
        spread_window_sec: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        if api is None or not symbol:
            return None

        cleaned_symbol = symbol.strip().upper()
        if not cleaned_symbol:
            return None

        snapshot: Optional[Dict[str, float]] = None

        live_orderbook = self._live_orderbook_for_api(api)
        if live_orderbook is not None:
            try:
                live_orderbook.start_ws([cleaned_symbol])
            except Exception as exc:  # pragma: no cover - defensive guard
                log(
                    "guardian.auto.liquidity_guard.orderbook_ws_start_error",
                    symbol=cleaned_symbol,
                    err=str(exc),
                )
            else:
                ws_top = live_orderbook.get_top_ws(cleaned_symbol, max_age_ms=1500)
                if ws_top:
                    snapshot = {}
                    for key in (
                        "best_ask",
                        "best_bid",
                        "best_ask_qty",
                        "best_bid_qty",
                        "spread_bps",
                    ):
                        value = ws_top.get(key)
                        if isinstance(value, (int, float)):
                            snapshot[key] = float(value)
                    ts_value = ws_top.get("ts")
                    if isinstance(ts_value, (int, float)):
                        snapshot["ts"] = float(ts_value)
                    else:
                        snapshot["ts"] = time.time() * 1000.0
                    if snapshot:
                        best_ask_val = snapshot.get("best_ask")
                        best_bid_val = snapshot.get("best_bid")
                        if (
                            "spread_bps" not in snapshot
                            and best_ask_val
                            and best_bid_val
                            and best_ask_val > 0
                        ):
                            snapshot["spread_bps"] = max(
                                ((best_ask_val - best_bid_val) / best_ask_val) * 10_000.0,
                                0.0,
                            )
                    if (
                        snapshot
                        and spread_window_sec is not None
                        and spread_window_sec > 0
                    ):
                        stats = live_orderbook.spread_window_stats(
                            cleaned_symbol, float(spread_window_sec)
                        )
                        if stats:
                            snapshot["spread_window_stats"] = stats
        if snapshot:
            return snapshot

        try:
            payload = api.orderbook(category="spot", symbol=cleaned_symbol, limit=max(limit, 1))
        except Exception as exc:  # pragma: no cover - defensive guard
            log(
                "guardian.auto.liquidity_guard.orderbook_error",
                symbol=cleaned_symbol,
                err=str(exc),
            )
            return None

        result = payload.get("result") if isinstance(payload, Mapping) else None
        asks_raw = result.get("a") if isinstance(result, Mapping) else None
        bids_raw = result.get("b") if isinstance(result, Mapping) else None

        best_ask = best_ask_qty = best_bid = best_bid_qty = None

        if isinstance(asks_raw, Sequence):
            for entry in asks_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = self._decimal_from(entry[0])
                qty = self._decimal_from(entry[1])
                if price > 0 and qty > 0:
                    best_ask = price
                    best_ask_qty = qty
                    break

        if isinstance(bids_raw, Sequence):
            for entry in bids_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = self._decimal_from(entry[0])
                qty = self._decimal_from(entry[1])
                if price > 0 and qty > 0:
                    best_bid = price
                    best_bid_qty = qty
                    break

        best_ask_float = self._decimal_to_float(best_ask)
        best_bid_float = self._decimal_to_float(best_bid)
        best_ask_qty_float = self._decimal_to_float(best_ask_qty)
        best_bid_qty_float = self._decimal_to_float(best_bid_qty)

        if not any(
            value is not None
            for value in (best_ask_float, best_bid_float, best_ask_qty_float, best_bid_qty_float)
        ):
            return None

        spread_bps: Optional[float] = None
        if best_ask_float and best_bid_float and best_ask_float > 0:
            spread_bps = max(((best_ask_float - best_bid_float) / best_ask_float) * 10_000.0, 0.0)

        snapshot = {}
        if best_ask_float:
            snapshot["best_ask"] = best_ask_float
        if best_bid_float:
            snapshot["best_bid"] = best_bid_float
        if best_ask_qty_float:
            snapshot["best_ask_qty"] = best_ask_qty_float
        if best_bid_qty_float:
            snapshot["best_bid_qty"] = best_bid_qty_float
        if spread_bps is not None:
            snapshot["spread_bps"] = spread_bps

        if live_orderbook is not None and snapshot:
            try:
                live_orderbook.record_top(cleaned_symbol, snapshot, source="rest")
            except Exception:  # pragma: no cover - defensive guard
                pass
            if spread_window_sec is not None and spread_window_sec > 0:
                stats = live_orderbook.spread_window_stats(
                    cleaned_symbol, float(spread_window_sec)
                )
                if stats:
                    snapshot["spread_window_stats"] = stats

        return snapshot

    def _apply_liquidity_guard(
        self,
        side: str,
        notional_quote: float,
        orderbook_top: Mapping[str, float],
        *,
        settings: Settings,
        price_hint: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        if notional_quote <= 0:
            return None

        context: Dict[str, object] = {"side": side}

        best_ask = _safe_float(orderbook_top.get("best_ask"))
        best_bid = _safe_float(orderbook_top.get("best_bid"))
        best_ask_qty = _safe_float(orderbook_top.get("best_ask_qty"))
        best_bid_qty = _safe_float(orderbook_top.get("best_bid_qty"))
        spread_bps = _safe_float(orderbook_top.get("spread_bps"))

        try:
            max_spread = float(getattr(settings, "ai_max_spread_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_spread = 0.0

        try:
            spread_window_sec = float(
                getattr(settings, "ai_spread_compression_window_sec", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            spread_window_sec = 0.0

        spread_window_sec = max(spread_window_sec, 0.0)

        if spread_bps is not None:
            context["spread_bps"] = spread_bps
            if max_spread > 0 and spread_bps > max_spread:
                context["max_spread_bps"] = max_spread
                reason = (
                    "ждём восстановления ликвидности — спред {spread:.1f} б.п. превышает лимит {limit:.1f} б.п."
                ).format(spread=spread_bps, limit=max_spread)
                return {"decision": "skip", "reason": reason, "context": context}

        if max_spread > 0 and spread_window_sec > 0:
            window_stats = orderbook_top.get("spread_window_stats")
            stats_mapping = window_stats if isinstance(window_stats, Mapping) else None
            if stats_mapping is None and spread_bps is not None:
                ts_value = _safe_float(orderbook_top.get("ts"))
                age_ms = 0.0
                if ts_value is not None:
                    age_ms = max(time.time() * 1000.0 - ts_value, 0.0)
                if age_ms <= spread_window_sec * 1000.0:
                    stats_mapping = {
                        "window_sec": float(spread_window_sec),
                        "observations": 1,
                        "max_bps": spread_bps,
                        "min_bps": spread_bps,
                        "avg_bps": spread_bps,
                        "latest_bps": spread_bps,
                        "age_ms": age_ms,
                    }
            if stats_mapping is None:
                context["spread_window_required_sec"] = spread_window_sec
                reason = (
                    "ждём сжатия спреда — нет данных за последние {window:.0f} с."
                ).format(window=spread_window_sec)
                return {"decision": "skip", "reason": reason, "context": context}

            context["spread_window_stats"] = dict(stats_mapping)
            max_recent = _safe_float(stats_mapping.get("max_bps"))
            if max_recent is not None and max_recent > max_spread:
                context["max_spread_bps"] = max_spread
                reason = (
                    "ждём сжатия спреда — максимум за {window:.0f} с. {max_recent:.1f} б.п. превышает лимит {limit:.1f} б.п."
                ).format(window=spread_window_sec, max_recent=max_recent, limit=max_spread)
                return {"decision": "skip", "reason": reason, "context": context}

        top_price = best_ask if side == "Buy" else best_bid
        top_qty = best_ask_qty if side == "Buy" else best_bid_qty

        if top_price is None or top_price <= 0 or top_qty is None or top_qty <= 0:
            reason = "ждём восстановления ликвидности — первый уровень стакана пуст."
            if top_price is not None and top_price > 0:
                context["top_price"] = top_price
            if top_qty is not None and top_qty > 0:
                context["top_qty"] = top_qty
            return {"decision": "skip", "reason": reason, "context": context}

        context["top_price"] = top_price
        context["top_qty"] = top_qty

        required_quote = max(float(notional_quote), 0.0)
        available_quote = top_price * top_qty
        context["required_quote"] = required_quote
        context["available_quote"] = available_quote

        if price_hint is not None and price_hint > 0 and top_price > 0:
            deviation_bps = (top_price / price_hint - 1.0) * 10_000.0
            context["price_deviation_bps"] = deviation_bps

        coverage_ratio = available_quote / required_quote if required_quote > 0 else 1.0
        context["coverage_ratio"] = coverage_ratio

        try:
            coverage_threshold = float(
                getattr(settings, "ai_top_depth_coverage", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            coverage_threshold = 0.0
        coverage_threshold = max(min(coverage_threshold, 0.95), 0.0)

        shortfall_quote = required_quote - available_quote
        context["shortfall_quote"] = shortfall_quote

        try:
            shortfall_limit = float(
                getattr(settings, "ai_top_depth_shortfall_usd", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            shortfall_limit = 0.0
        shortfall_limit = max(shortfall_limit, 0.0)
        if shortfall_limit > 0:
            context["shortfall_limit_quote"] = shortfall_limit

        partial_execution_allowed = bool(getattr(settings, "allow_partial_fills", True))
        slicing_fallback_enabled = bool(
            getattr(settings, "twap_enabled", False)
            or getattr(settings, "twap_spot_enabled", False)
            or getattr(settings, "iceberg_enabled", False)
            or getattr(settings, "spot_iceberg_enabled", False)
        )
        fallback_available = partial_execution_allowed or slicing_fallback_enabled

        coverage_issue = (
            required_quote > 0
            and coverage_threshold > 0
            and coverage_ratio + 1e-9 < coverage_threshold
        )
        shortfall_issue = (
            required_quote > 0
            and shortfall_limit > 0
            and shortfall_quote > shortfall_limit
        )

        if not coverage_issue and not shortfall_issue:
            return None

        reasons: List[str] = []
        reason_messages: List[str] = []

        if coverage_issue:
            context["coverage_threshold"] = coverage_threshold
            reasons.append("coverage")
            reason_messages.append(
                (
                    "ждём восстановления ликвидности — на первом уровне доступно ≈{available:.2f} USDT,"
                    " требуется ≈{required:.2f} USDT."
                ).format(available=available_quote, required=required_quote)
            )

        if shortfall_issue:
            reasons.append("shortfall")
            reason_messages.append(
                (
                    "ждём восстановления ликвидности — дефицит ≈{shortfall:.2f} USDT"
                    " превышает лимит ≈{limit:.2f} USDT."
                ).format(shortfall=shortfall_quote, limit=shortfall_limit)
            )

        if not fallback_available:
            reason_messages.append(
                "Нет доступных фолбэков (partial / twap), поэтому ордер пропущен."
            )

        context["reasons"] = reasons
        context["liquidity_guard_reasons"] = list(reasons)
        context["fallback_available"] = fallback_available
        if fallback_available:
            context["guard_relaxed"] = True

        decision = "skip"
        if fallback_available:
            decision = "relaxed"

        return {
            "decision": decision,
            "reason": " ".join(reason_messages),
            "context": context,
        }

    def _candidate_symbol_stream(
        self, summary: Mapping[str, object]
    ) -> List[Tuple[str, Optional[Dict[str, object]]]]:
        ordered: List[Tuple[str, Optional[Dict[str, object]]]] = []
        seen: Set[str] = set()

        def _append(symbol: object, meta: Optional[Dict[str, object]] = None) -> None:
            cleaned = _safe_symbol(symbol)
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            ordered.append((cleaned, dict(meta) if meta else None))

        trade_candidates = summary.get("trade_candidates")
        if isinstance(trade_candidates, Sequence):
            actionable_new: List[Tuple[str, Dict[str, object]]] = []
            ready_new: List[Tuple[str, Dict[str, object]]] = []
            actionable_existing: List[Tuple[str, Dict[str, object]]] = []
            backlog: List[Tuple[str, Dict[str, object]]] = []
            for idx, entry in enumerate(trade_candidates):
                if not isinstance(entry, Mapping):
                    continue
                symbol = _safe_symbol(entry.get("symbol"))
                if not symbol:
                    continue
                meta: Dict[str, object] = {
                    "source": "trade_candidates",
                    "rank": idx + 1,
                    "symbol": symbol,
                }
                priority = entry.get("priority")
                if priority is not None:
                    meta["priority"] = priority
                actionable = bool(entry.get("actionable"))
                ready = bool(entry.get("ready"))
                holding = bool(entry.get("holding"))
                meta["actionable"] = actionable
                meta["ready"] = ready
                meta["holding"] = holding
                for key in (
                    "probability",
                    "probability_pct",
                    "ev_bps",
                    "edge_score",
                    "score",
                    "watchlist_rank",
                    "note",
                    "trend",
                    "exposure_pct",
                    "position_qty",
                    "position_notional",
                ):
                    if key in entry:
                        meta[key] = entry[key]
                if actionable and not holding:
                    actionable_new.append((symbol, meta))
                elif ready and not holding:
                    ready_new.append((symbol, meta))
                elif actionable:
                    actionable_existing.append((symbol, meta))
                else:
                    backlog.append((symbol, meta))
            for group in (actionable_new, ready_new, actionable_existing, backlog):
                for item in group:
                    _append(*item)

        summary_symbol = summary.get("symbol")
        if summary_symbol:
            _append(summary_symbol, {"source": "summary.symbol"})

        watchlist = summary.get("watchlist")
        if isinstance(watchlist, Mapping):
            for symbol, entry in watchlist.items():
                _append(symbol, {"source": "watchlist", "entry": dict(entry)})
        elif isinstance(watchlist, Sequence):
            for entry in watchlist:
                if isinstance(entry, Mapping):
                    _append(entry.get("symbol"), {"source": "watchlist", "entry": dict(entry)})
                else:
                    _append(entry, {"source": "watchlist"})

        holdings = summary.get("holdings")
        if isinstance(holdings, Mapping):
            for symbol, entry in holdings.items():
                _append(symbol, {"source": "holdings", "entry": dict(entry)})

        symbol_plan = summary.get("symbol_plan")
        if isinstance(symbol_plan, Mapping):
            for key in ("priority_table", "priorityTable", "positions", "combined"):
                value = symbol_plan.get(key)
                if isinstance(value, Mapping):
                    for symbol, entry in value.items():
                        _append(symbol, {"source": f"symbol_plan.{key}", "entry": dict(entry)})
                elif isinstance(value, Sequence):
                    for entry in value:
                        if isinstance(entry, Mapping):
                            _append(
                                entry.get("symbol"),
                                {"source": f"symbol_plan.{key}", "entry": dict(entry)},
                            )
                        else:
                            _append(entry, {"source": f"symbol_plan.{key}"})

        if not ordered:
            _append(summary.get("fallback_symbol") or summary.get("symbol_requested"))

        if not ordered:
            return []

        return self._filter_quarantined_candidates(ordered)

    def _resolve_impulse_signal(
        self, summary: Mapping[str, object], symbol: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, object]]]:
        context: Dict[str, object] = {}

        entry = summary
        strength_hint = _safe_float(entry.get("impulse_strength"))
        if strength_hint is not None:
            context["impulse_strength_hint"] = strength_hint
        volume_impulse = entry.get("volume_impulse")
        if isinstance(volume_impulse, Mapping):
            context["volume_impulse"] = dict(volume_impulse)
        if bool(entry.get("impulse_signal")):
            context.setdefault("trigger", "flagged")
            return True, context

        best_impulse: Optional[float] = None
        if isinstance(volume_impulse, Mapping):
            for value in volume_impulse.values():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                if best_impulse is None or float(numeric) > best_impulse:
                    best_impulse = float(numeric)

        if best_impulse is None:
            best_impulse = strength_hint

        if best_impulse is not None and best_impulse >= _IMPULSE_SIGNAL_THRESHOLD:
            context.setdefault("trigger", "volume_impulse")
            context["impulse_strength"] = best_impulse
            return True, context

        return False, context or None

    @staticmethod
    def _impulse_stop_loss_bps(settings: Settings) -> float:
        fallback = _safe_float(getattr(settings, "spot_impulse_stop_loss_bps", None))
        if fallback is None or fallback <= 0:
            fallback = 80.0
        return float(fallback)

    @staticmethod
    def _merge_symbol_meta(
        candidate_meta: Optional[Mapping[str, object]],
        resolved_meta: Optional[Mapping[str, object]],
    ) -> Optional[Dict[str, object]]:
        combined: Dict[str, object] = {}
        if isinstance(resolved_meta, Mapping):
            combined.update(resolved_meta)
        if candidate_meta:
            combined["candidate"] = dict(candidate_meta)
        return combined or None

    def _select_symbol(
        self, summary: Dict[str, object], settings: Settings
    ) -> tuple[Optional[str], Optional[Dict[str, object]]]:
        fallback_meta: Optional[Dict[str, object]] = None
        api: Optional[object] = None
        api_error: Optional[str] = None

        for candidate, candidate_meta in self._candidate_symbol_stream(summary):
            resolved, meta, api, api_error = self._map_symbol(
                candidate,
                settings=settings,
                api=api,
                api_error=api_error,
            )
            combined_meta = self._merge_symbol_meta(candidate_meta, meta)
            if resolved:
                return resolved, combined_meta
            if combined_meta:
                fallback_meta = combined_meta

        return None, fallback_meta

    def _map_symbol(
        self,
        symbol: str,
        *,
        settings: Settings,
        api: Optional[object],
        api_error: Optional[str],
    ) -> tuple[Optional[str], Optional[Dict[str, object]], Optional[object], Optional[str]]:
        cleaned = _safe_symbol(symbol)
        if not cleaned:
            return None, None, api, api_error

        normalised, quote_source = self._ensure_usdt_symbol(cleaned)
        if not normalised:
            meta: Dict[str, object] = {"reason": "unsupported_quote", "requested": cleaned}
            if quote_source:
                meta["quote"] = quote_source
            return None, meta, api, api_error

        quote_meta: Optional[Dict[str, object]] = None
        if quote_source:
            quote_meta = {
                "requested": cleaned,
                "normalised": normalised,
                "from_quote": quote_source,
                "to_quote": "USDT",
            }

        cleaned = normalised

        if not settings.testnet:
            meta_payload: Dict[str, object] = {}
            if quote_meta:
                meta_payload["quote_conversion"] = quote_meta
            return cleaned, meta_payload or None, api, api_error

        if api_error is not None:
            failure_meta: Dict[str, object] = {
                "reason": "api_unavailable",
                "error": api_error,
                "requested": cleaned,
            }
            if quote_meta:
                failure_meta["quote_conversion"] = quote_meta
            return None, failure_meta, api, api_error

        if api is None:
            try:
                api = self._get_api_client()
            except Exception as exc:  # pragma: no cover - defensive
                error_text = str(exc)
                failure_meta = {
                    "reason": "api_unavailable",
                    "error": error_text,
                    "requested": cleaned,
                }
                if quote_meta:
                    failure_meta["quote_conversion"] = quote_meta
                return None, failure_meta, None, error_text

        resolved, meta = self._resolve_trade_symbol(cleaned, api=api, allow_nearest=True)
        if resolved is None:
            if quote_meta:
                extra: Dict[str, object] = {}
                if isinstance(meta, Mapping):
                    extra.update(meta)
                extra["quote_conversion"] = quote_meta
                return None, extra, api, api_error
            return None, meta, api, api_error

        final_meta: Dict[str, object] = {}
        if isinstance(meta, Mapping):
            final_meta.update(meta)
        if quote_meta:
            final_meta.setdefault("quote_conversion", quote_meta)
        return resolved, final_meta or None, api, api_error
