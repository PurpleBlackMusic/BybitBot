"""Automate execution of actionable Guardian signals."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
import threading
from typing import Callable, Dict, List, Optional, Tuple

from .envs import Settings, get_api_client, get_settings, creds_ok
from .live_checks import extract_wallet_totals
from .log import log
from .spot_market import place_spot_market_with_tolerance


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


class SignalExecutor:
    """Translate Guardian summaries into real trading actions."""

    def __init__(self, bot, settings: Optional[Settings] = None) -> None:
        self.bot = bot
        self._settings = settings

    # ------------------------------------------------------------------
    # public API
    def execute_once(self) -> ExecutionResult:
        summary = self._fetch_summary()
        if not summary.get("actionable"):
            return ExecutionResult(
                status="skipped",
                reason="Signal is not actionable according to current thresholds.",
            )

        settings = self._resolve_settings()
        if not getattr(settings, "ai_enabled", False):
            return ExecutionResult(
                status="disabled",
                reason="Автоматизация выключена — включите AI сигналы в настройках.",
            )

        mode = str(summary.get("mode") or "").lower()
        if mode not in {"buy", "sell"}:
            return ExecutionResult(
                status="skipped",
                reason=f"Режим {mode or 'wait'} не предполагает немедленного исполнения.",
            )

        symbol = self._select_symbol(summary)
        if symbol is None:
            return ExecutionResult(
                status="skipped",
                reason="Не удалось определить инструмент для сделки.",
            )

        side = "Buy" if mode == "buy" else "Sell"

        try:
            api, wallet_totals = self._resolve_wallet(
                require_success=not settings.dry_run
            )
        except Exception as exc:
            return ExecutionResult(
                status="error",
                reason=f"Не удалось получить баланс: {exc}",
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

        slippage_pct = max(
            float(getattr(settings, "ai_max_slippage_bps", 25) or 0.0) / 100.0, 0.01
        )

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
            return ExecutionResult(
                status="skipped",
                reason="Недостаточно свободного капитала для безопасной сделки.",
                context=order_context,
            )

        if getattr(settings, "dry_run", True):
            order = copy.deepcopy(order_context)
            order["slippage_percent"] = slippage_pct
            log("guardian.auto.preview", order=order)
            return ExecutionResult(status="dry_run", order=order, context=order_context)

        if api is None:
            return ExecutionResult(
                status="error",
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
        except Exception as exc:  # pragma: no cover - network/HTTP errors
            return ExecutionResult(
                status="error",
                reason=f"Не удалось отправить ордер: {exc}",
                context=order_context,
            )

        order = copy.deepcopy(order_context)
        order["slippage_percent"] = slippage_pct
        log("guardian.auto.execute", order=order, response=response)
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
        if isinstance(self._settings, Settings):
            return self._settings

        candidate = getattr(self.bot, "settings", None)
        if isinstance(candidate, Settings):
            self._settings = candidate
            return candidate

        resolved = get_settings()
        self._settings = resolved
        return resolved

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

    def _select_symbol(self, summary: Dict[str, object]) -> Optional[str]:
        symbol = _safe_symbol(summary.get("symbol"))
        if symbol:
            return symbol

        primary = summary.get("primary_watch")
        if isinstance(primary, dict):
            symbol = _safe_symbol(primary.get("symbol"))
            if symbol:
                return symbol

        candidates = summary.get("candidate_symbols")
        if isinstance(candidates, (list, tuple)):
            for candidate in candidates:
                symbol = _safe_symbol(candidate)
                if symbol:
                    return symbol

        return None

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

    _SUCCESSFUL_STATUSES = {"filled", "dry_run", "skipped", "disabled"}

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

