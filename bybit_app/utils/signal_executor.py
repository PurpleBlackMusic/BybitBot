"""Automate execution of actionable Guardian signals."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
import threading
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .envs import Settings, get_api_client, get_settings, creds_ok
from .live_checks import extract_wallet_totals
from .log import log
from .spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    resolve_trade_symbol,
)
from .symbols import ensure_usdt_symbol
from .ws_manager import manager as ws_manager

_PERCENT_TOLERANCE_MIN = 0.05
_PERCENT_TOLERANCE_MAX = 1.0


def _normalise_slippage_percent(value: float) -> float:
    """Clamp Bybit percent tolerance to the exchange supported range."""

    if value <= 0.0:
        return 0.0
    return max(_PERCENT_TOLERANCE_MIN, min(value, _PERCENT_TOLERANCE_MAX))


def _safe_symbol(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class ExecutionResult:
    """Outcome of the automatic executor."""

    status: str
    reason: Optional[str] = None
    order: Optional[Dict[str, object]] = None
    response: Optional[Dict[str, object]] = None
    context: Optional[Dict[str, object]] = None


_BYBIT_ERROR = re.compile(r"Bybit error (?P<code>-?\d+): (?P<message>.+)")


def _format_bybit_error(exc: Exception) -> str:
    text = str(exc)
    match = _BYBIT_ERROR.search(text)
    if match:
        code = match.group("code")
        message = match.group("message").strip()
        return f"Bybit отказал ({code}): {message}"
    return f"Не удалось отправить ордер: {text}"


class SignalExecutor:
    """Translate Guardian summaries into real trading actions."""

    def __init__(self, bot, settings: Optional[Settings] = None) -> None:
        self.bot = bot
        self._settings_override = settings

    def _decision(
        self,
        status: str,
        *,
        reason: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
        order: Optional[Dict[str, object]] = None,
    ) -> ExecutionResult:
        payload: Dict[str, object] = {"status": status}
        if reason:
            payload["reason"] = reason
        if order:
            payload["order"] = copy.deepcopy(order)
        if context:
            payload["context"] = copy.deepcopy(context)
        log("guardian.auto.decision", **payload)
        return ExecutionResult(
            status=status,
            reason=reason,
            order=copy.deepcopy(order) if order else None,
            context=copy.deepcopy(context) if context else None,
        )

    # ------------------------------------------------------------------
    # public API
    def execute_once(self) -> ExecutionResult:
        summary = self._fetch_summary()
        if not summary.get("actionable"):
            return self._decision(
                "skipped",
                reason="Signal is not actionable according to current thresholds.",
            )

        settings = self._resolve_settings()
        if not getattr(settings, "ai_enabled", False):
            return self._decision(
                "disabled",
                reason="Автоматизация выключена — включите AI сигналы в настройках.",
            )

        guard_result = self._apply_runtime_guards(settings)
        if guard_result is not None:
            return guard_result

        mode = str(summary.get("mode") or "").lower()
        if mode not in {"buy", "sell"}:
            return self._decision(
                "skipped",
                reason=f"Режим {mode or 'wait'} не предполагает немедленного исполнения.",
            )

        symbol, symbol_meta = self._select_symbol(summary, settings)
        if symbol is None:
            reason = "Не удалось определить инструмент для сделки."
            if symbol_meta and isinstance(symbol_meta, Mapping):
                meta_reason = str(symbol_meta.get("reason") or "")
                if meta_reason == "unsupported_quote":
                    requested = symbol_meta.get("requested") or summary.get("symbol")
                    reason = (
                        f"Инструмент {requested or ''} недоступен — поддерживаются только спотовые пары USDT."
                    )
                elif meta_reason:
                    reason = f"Инструмент недоступен для сделки ({meta_reason})."
            return self._decision(
                "skipped",
                reason=reason,
                context={"symbol_meta": symbol_meta} if symbol_meta else None,
            )

        side = "Buy" if mode == "buy" else "Sell"

        try:
            api, wallet_totals = self._resolve_wallet(
                require_success=not settings.dry_run
            )
        except Exception as exc:
            reason_text = f"Не удалось получить баланс: {exc}"
            if not creds_ok(settings) and not getattr(settings, "dry_run", True):
                reason_text = (
                    "API ключи не настроены — сохраните ключ и секрет перед отправкой ордеров."
                )
            return self._decision(
                "error",
                reason=reason_text,
            )
        total_equity, available_equity = wallet_totals

        sizing_factor = self._signal_sizing_factor(summary, settings)
        notional, usable_after_reserve = self._compute_notional(
            settings, total_equity, available_equity, sizing_factor
        )

        min_notional = 5.0

        order_context = {
            "symbol": symbol,
            "side": side,
            "notional_quote": notional,
            "available_equity": available_equity,
            "usable_after_reserve": usable_after_reserve,
            "total_equity": total_equity,
        }
        if symbol_meta:
            order_context["symbol_meta"] = symbol_meta

        raw_slippage_bps = getattr(settings, "ai_max_slippage_bps", 25)
        slippage_pct = max(float(raw_slippage_bps or 0.0) / 100.0, 0.0)
        slippage_pct = _normalise_slippage_percent(slippage_pct)

        if notional <= 0 or notional < min_notional:
            order_context["min_notional"] = min_notional
            if getattr(settings, "dry_run", True):
                order = copy.deepcopy(order_context)
                order["slippage_percent"] = slippage_pct
                order["note"] = "preview"
                log("guardian.auto.preview", order=order)
                return ExecutionResult(
                    status="dry_run", order=order, context=order_context
                )
            return self._decision(
                "skipped",
                reason="Недостаточно свободного капитала для безопасной сделки.",
                context=order_context,
            )

        if getattr(settings, "dry_run", True):
            order = copy.deepcopy(order_context)
            order["slippage_percent"] = slippage_pct
            log("guardian.auto.preview", order=order)
            return self._decision(
                "dry_run",
                reason="Режим dry-run активен — ордер не отправлен.",
                context=order_context,
                order=order,
            )

        if api is None:
            return self._decision(
                "error",
                reason="API клиент недоступен для выполнения сделки.",
                context=order_context,
            )

        try:
            response = place_spot_market_with_tolerance(
                api,
                symbol=symbol,
                side=side,
                qty=float(notional),
                unit="quoteCoin",
                tol_type="Percent",
                tol_value=slippage_pct,
                max_quote=usable_after_reserve,
            )
        except OrderValidationError as exc:
            validation_context = dict(order_context)
            validation_context["validation_code"] = exc.code
            if exc.details:
                validation_context["validation_details"] = exc.details
            log(
                "guardian.auto.order.validation_failed",
                error=exc.to_dict(),
                context=validation_context,
            )

            skip_codes = {"min_notional", "min_qty", "qty_step", "price_deviation"}
            if exc.code in skip_codes:
                return self._decision(
                    "skipped",
                    reason=f"Ордер пропущен ({exc.code}): {exc}",
                    context=validation_context,
                )

            return ExecutionResult(
                status="rejected",
                reason=f"Ордер отклонён биржей ({exc.code}): {exc}",
                context=validation_context,
            )
        except Exception as exc:  # pragma: no cover - network/HTTP errors
            formatted_error = _format_bybit_error(exc)
            lowered = formatted_error.lower()
            if "priceboundrate" in lowered or "price bound" in lowered:
                return self._decision(
                    "skipped",
                    reason=formatted_error,
                    context=order_context,
                )
            return self._decision(
                "error",
                reason=formatted_error,
                context=order_context,
            )

        order = copy.deepcopy(order_context)
        order["slippage_percent"] = slippage_pct
        log("guardian.auto.execute", order=order, response=response)
        audit = None
        if isinstance(response, Mapping):
            local = response.get("_local")
            if isinstance(local, Mapping):
                audit = local.get("order_audit")
        if audit:
            order["order_audit"] = audit
        return ExecutionResult(status="filled", order=order, response=response)

    # ------------------------------------------------------------------
    # helpers
    def current_signature(self) -> Optional[str]:
        """Return a stable identifier for the currently cached signal."""

        fingerprint = getattr(self.bot, "status_fingerprint", None)
        if callable(fingerprint):
            try:
                value = fingerprint()
            except Exception:
                return None
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            return str(value)
        return None

    def settings_marker(self) -> Tuple[bool, bool, bool]:
        """Return a tuple describing current automation toggles."""

        settings = self._resolve_settings()
        dry_run = bool(getattr(settings, "dry_run", True))
        ai_enabled = bool(getattr(settings, "ai_enabled", False))
        return dry_run, creds_ok(settings), ai_enabled

    def _fetch_summary(self) -> Dict[str, object]:
        summary = self.bot.status_summary()
        if isinstance(summary, dict):
            return copy.deepcopy(summary)
        return {}

    def _resolve_settings(self) -> Settings:
        if isinstance(self._settings_override, Settings):
            return self._settings_override

        candidate = getattr(self.bot, "settings", None)
        if isinstance(candidate, Settings):
            return candidate

        return get_settings(force_reload=True)

    def _apply_runtime_guards(self, settings: Settings) -> Optional[ExecutionResult]:
        guard = self._private_ws_guard(settings)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)
        return None

    def _private_ws_guard(
        self, settings: Settings
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        if not getattr(settings, "ws_watchdog_enabled", False):
            return None

        threshold = _safe_float(getattr(settings, "ws_watchdog_max_age_sec", None))
        if threshold is None or threshold <= 0:
            return None

        try:
            status = ws_manager.status()
        except Exception:  # pragma: no cover - defensive guard
            return None

        private_info = status.get("private") if isinstance(status, Mapping) else None
        if not isinstance(private_info, Mapping):
            return None

        age = _safe_float(private_info.get("age_seconds"))
        running = bool(private_info.get("running"))
        connected = bool(private_info.get("connected"))

        if age is None:
            return None

        if age <= threshold:
            return None

        context: Dict[str, object] = {
            "guard": "private_ws_stale",
            "age_seconds": age,
            "threshold_seconds": threshold,
            "running": running,
            "connected": connected,
        }

        log(
            "guardian.auto.guard.private_ws",
            age=round(age, 2),
            threshold=threshold,
            running=running,
            connected=connected,
        )

        pretty_age = self._format_seconds(age)
        pretty_threshold = self._format_seconds(threshold)
        message = (
            "Приватный WebSocket не присылал событий {age} — автоматика приостановлена до восстановления соединения"
        ).format(age=pretty_age)
        context["message"] = message + f" (порог {pretty_threshold})"
        return message, context

    @staticmethod
    def _format_seconds(value: float) -> str:
        if value < 60:
            return f"{value:.0f} сек"
        minutes = value / 60.0
        if minutes < 60:
            return f"{minutes:.1f} мин"
        hours = minutes / 60.0
        return f"{hours:.1f} ч"

    def _resolve_wallet(
        self, *, require_success: bool
    ) -> Tuple[Optional[object], Tuple[float, float]]:
        try:
            api = get_api_client()
        except Exception:
            if require_success:
                raise
            return None, (0.0, 0.0)

        try:
            payload = api.wallet_balance()
        except Exception:
            if require_success:
                raise
            return api, (0.0, 0.0)

        totals = extract_wallet_totals(payload)
        return api, totals

    def _compute_notional(
        self,
        settings: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
    ) -> Tuple[float, float]:
        try:
            reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            reserve_pct = 0.0

        # Минимальный страховой буфер 2% помогает избежать отмен из-за комиссий
        # и мелких движений цены даже если пользователь указал более низкое значение
        reserve_pct = max(reserve_pct, 2.0)

        try:
            risk_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            risk_pct = 0.0

        try:
            cap_pct = float(getattr(settings, "spot_max_cap_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            cap_pct = 0.0

        usable_after_reserve = max(
            available_equity - total_equity * reserve_pct / 100.0, 0.0
        )

        caps = []
        if usable_after_reserve > 0:
            caps.append(usable_after_reserve)

        if risk_pct > 0:
            caps.append(total_equity * risk_pct / 100.0)

        if cap_pct > 0:
            caps.append(total_equity * cap_pct / 100.0)

        if not caps:
            return 0.0, usable_after_reserve

        base_notional = min(caps)
        sizing = max(0.0, min(float(sizing_factor), 1.0))
        notional = round(base_notional * sizing, 2)
        return max(notional, 0.0), usable_after_reserve

    def _select_symbol(
        self, summary: Dict[str, object], settings: Settings
    ) -> tuple[Optional[str], Optional[Dict[str, object]]]:
        candidates: list[str] = []

        symbol = _safe_symbol(summary.get("symbol"))
        if symbol:
            candidates.append(symbol)

        primary = summary.get("primary_watch")
        if isinstance(primary, dict):
            primary_symbol = _safe_symbol(primary.get("symbol"))
            if primary_symbol and primary_symbol not in candidates:
                candidates.append(primary_symbol)

        extra = summary.get("candidate_symbols")
        if isinstance(extra, (list, tuple)):
            for candidate in extra:
                cleaned = _safe_symbol(candidate)
                if cleaned and cleaned not in candidates:
                    candidates.append(cleaned)

        for candidate in candidates:
            resolved, meta = self._map_symbol(candidate, settings=settings)
            if resolved:
                return resolved, meta

        return None, None

    def _map_symbol(
        self, symbol: str, *, settings: Settings
    ) -> tuple[Optional[str], Optional[Dict[str, object]]]:
        cleaned = _safe_symbol(symbol)
        if not cleaned:
            return None, None

        normalised, quote_source = ensure_usdt_symbol(cleaned)
        if not normalised:
            meta: Dict[str, object] = {"reason": "unsupported_quote", "requested": cleaned}
            if quote_source:
                meta["quote"] = quote_source
            return None, meta

        quote_meta: Optional[Dict[str, object]] = None
        if quote_source:
            quote_meta = {
                "requested": cleaned,
                "normalised": normalised,
                "from_quote": quote_source,
                "to_quote": "USDT",
            }

        cleaned = normalised

        if not getattr(settings, "testnet", True):
            meta_payload: Dict[str, object] = {}
            if quote_meta:
                meta_payload["quote_conversion"] = quote_meta
            return cleaned, meta_payload or None

        try:
            api = get_api_client()
        except Exception as exc:  # pragma: no cover - defensive
            return None, {"reason": "api_unavailable", "error": str(exc), "requested": cleaned}

        resolved, meta = resolve_trade_symbol(cleaned, api=api, allow_nearest=True)
        if resolved is None:
            if quote_meta:
                extra: Dict[str, object] = {}
                if isinstance(meta, Mapping):
                    extra.update(meta)
                extra["quote_conversion"] = quote_meta
                return None, extra
            return None, meta

        final_meta: Dict[str, object] = {}
        if isinstance(meta, Mapping):
            final_meta.update(meta)
        if quote_meta:
            final_meta.setdefault("quote_conversion", quote_meta)
        return resolved, final_meta or None

    def _signal_sizing_factor(
        self, summary: Dict[str, object], settings: Settings
    ) -> float:
        override = _safe_float(summary.get("auto_sizing_factor"))
        if override is not None and override > 0:
            return max(0.05, min(override, 1.0))

        contributions: List[float] = []

        mode = str(summary.get("mode") or "wait").lower()
        probability = _safe_float(summary.get("probability"))
        buy_threshold = _safe_float(getattr(settings, "ai_buy_threshold", None))
        sell_threshold = _safe_float(getattr(settings, "ai_sell_threshold", None))
        if buy_threshold is None or buy_threshold <= 0:
            buy_threshold = 0.6
        if sell_threshold is None or sell_threshold <= 0:
            sell_threshold = 0.45

        if probability is not None:
            span = 0.25
            if mode == "buy":
                alignment = probability - buy_threshold
            elif mode == "sell":
                alignment = sell_threshold - probability
            else:
                alignment = abs(probability - 0.5) - 0.02
            contributions.append(max(0.0, min(alignment / span, 1.0)))

        ev_bps = _safe_float(summary.get("ev_bps"))
        thresholds = summary.get("thresholds")
        min_ev_setting = _safe_float(getattr(settings, "ai_min_ev_bps", None))
        if isinstance(thresholds, dict):
            threshold_override = _safe_float(thresholds.get("min_ev_bps"))
            if threshold_override is not None:
                min_ev_setting = threshold_override
        min_ev = max(min_ev_setting or 0.0, 0.0)
        if ev_bps is not None:
            baseline = max(min_ev, 5.0)
            span = max(baseline * 1.5, 20.0)
            margin = ev_bps - baseline
            contributions.append(max(0.0, min(margin / span, 1.0)))

        primary = summary.get("primary_watch")
        if isinstance(primary, dict):
            edge_score = _safe_float(primary.get("edge_score"))
            if edge_score is not None and edge_score > 0:
                contributions.append(math.tanh(edge_score / 6.0))

        confidence_score = _safe_float(summary.get("confidence_score"))
        if confidence_score is not None:
            contributions.append(max(0.0, min(confidence_score, 1.0)))

        if contributions:
            average = sum(contributions) / len(contributions)
            floor = 0.3 if summary.get("actionable") else 0.2
            base_factor = floor + (1.0 - floor) * average
        else:
            base_factor = 1.0

        staleness = summary.get("staleness")
        if isinstance(staleness, dict):
            state = str(staleness.get("state") or "").lower()
            if state == "warning":
                base_factor = min(base_factor, 0.6)
            elif state == "stale":
                base_factor = min(base_factor, 0.3)

        return max(0.2, min(base_factor, 1.0))


class AutomationLoop:
    """Keep executing trading signals until stopped explicitly."""

    _SUCCESSFUL_STATUSES = {"filled", "dry_run", "skipped", "disabled", "rejected"}

    def __init__(
        self,
        executor: SignalExecutor,
        *,
        poll_interval: float = 15.0,
        success_cooldown: float = 120.0,
        error_backoff: float = 5.0,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None]
        | None = None,
    ) -> None:
        self.executor = executor
        self.poll_interval = max(float(poll_interval), 0.0)
        self.success_cooldown = max(float(success_cooldown), 0.0)
        self.error_backoff = max(float(error_backoff), 0.0)
        self._last_key: Optional[Tuple[Optional[str], Tuple[bool, bool, bool]]] = None
        self._last_status: Optional[str] = None
        self._last_result: Optional[ExecutionResult] = None
        self._on_cycle = on_cycle

    def _should_execute(
        self, signature: Optional[str], settings_marker: Tuple[bool, bool, bool]
    ) -> bool:
        key = (signature, settings_marker)
        if self._last_key != key:
            return True
        if self._last_status not in self._SUCCESSFUL_STATUSES:
            return True
        return False

    def _tick(self) -> float:
        signature = self.executor.current_signature()
        settings_marker = self.executor.settings_marker()
        key = (signature, settings_marker)

        if self._should_execute(signature, settings_marker):
            try:
                result = self.executor.execute_once()
            except Exception as exc:  # pragma: no cover - defensive
                log("guardian.auto.loop.error", err=str(exc))
                result = ExecutionResult(status="error", reason=str(exc))

            self._last_status = result.status
            self._last_key = key
            self._last_result = result

            if self._on_cycle is not None:
                try:
                    self._on_cycle(result, signature, settings_marker)
                except Exception:  # pragma: no cover - defensive callback guard
                    log("guardian.auto.loop.callback.error")

            if result.status in self._SUCCESSFUL_STATUSES:
                return self.success_cooldown or self.poll_interval
            if result.status == "error":
                return self.error_backoff or self.poll_interval or 1.0

        return self.poll_interval

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Process trading signals until ``stop_event`` is set."""

        event = stop_event or threading.Event()
        while not event.is_set():
            delay = self._tick()
            if delay <= 0:
                continue
            event.wait(delay)

