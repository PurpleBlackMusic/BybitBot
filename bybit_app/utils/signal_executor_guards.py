"""Runtime guard and portfolio helpers for the signal executor."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .envs import Settings
from .log import log
from .ai.deepseek_utils import resolve_deepseek_drawdown_limit
from .signal_executor_models import _safe_float


class SignalExecutorGuardsMixin:
    """Guards that enforce portfolio level risk and runtime health checks."""

    def _apply_runtime_guards(
        self,
        settings: Settings,
        summary: Mapping[str, object],
        *,
        total_equity: Optional[float],
        current_time: float,
        summary_meta: Optional[Tuple[Optional[float], Optional[float]]],
        price_meta: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> Optional[object]:
        guard = self._loss_streak_guard(settings)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)

        guard = self._portfolio_loss_guard(
            settings,
            summary,
            total_equity=total_equity,
            current_time=current_time,
            summary_meta=summary_meta,
            price_meta=price_meta,
        )
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)

        guard = self._daily_loss_guard(settings, total_equity=total_equity)
        if guard is not None:
            message, context = guard
            return self._decision(
                "disabled", reason=message, context=context
            )

        guard = self._drawdown_guard(settings, total_equity=total_equity)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)
        return None

    def _loss_streak_guard(self, settings: Settings) -> Optional[Tuple[str, Dict[str, object]]]:
        try:
            threshold = int(getattr(settings, "ai_kill_switch_loss_streak", 0) or 0)
        except (TypeError, ValueError):
            threshold = 0

        if threshold <= 0:
            return None

        snapshot = self._performance_state
        if snapshot is None or snapshot.loss_streak < threshold:
            return None

        context: Dict[str, object] = {
            "guard": "loss_streak_limit",
            "loss_streak": snapshot.loss_streak,
            "threshold": threshold,
            "recent_results": snapshot.recent_results(),
            "average_pnl": snapshot.average_pnl,
            "sample_count": snapshot.sample_count,
            "last_exit_ts": snapshot.last_exit_ts,
        }

        try:
            cooldown_minutes = float(
                getattr(settings, "ai_kill_switch_cooldown_min", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            cooldown_minutes = 0.0

        kill_until = None
        if cooldown_minutes > 0.0:
            reason = (
                "Серия убыточных сделок достигла лимита: {loss_streak} подряд."
            ).format(loss_streak=snapshot.loss_streak)
            kill_until = self._activate_kill_switch(cooldown_minutes, reason)
            context["kill_switch_until"] = kill_until

        log(
            "guardian.auto.guard.loss_streak",
            loss_streak=snapshot.loss_streak,
            threshold=threshold,
            sample_count=snapshot.sample_count,
            average_pnl=round(snapshot.average_pnl, 6),
        )

        message = (
            "Серия из {loss_streak} убыточных сделок подряд достигла лимита {threshold} —"
            " автоматика приостановлена"
        ).format(loss_streak=snapshot.loss_streak, threshold=threshold)

        if kill_until:
            try:
                resume_at = datetime.fromtimestamp(kill_until, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
            except (OSError, ValueError):
                resume_at = None
            else:
                message += f". Kill-switch активирован до {resume_at}."

        context["message"] = message
        return message, context

    def _kill_switch_guard(self) -> Optional[Tuple[str, Dict[str, object]]]:
        state = self._kill_switch_state()
        if not state.paused:
            return None

        pretty_until = None
        if isinstance(state.until, (int, float)) and state.until > 0:
            try:
                pretty_until = datetime.fromtimestamp(state.until, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
            except (OSError, ValueError):
                pretty_until = None

        message_parts = ["Автоторговля приостановлена защитой портфеля"]
        if state.reason:
            message_parts.append(f": {state.reason}")
        if pretty_until:
            message_parts.append(f" (до {pretty_until})")

        context = {"guard": "kill_switch", "until": state.until, "reason": state.reason}
        return "".join(message_parts), context

    def _portfolio_loss_guard(
        self,
        settings: Settings,
        summary: Mapping[str, object],
        *,
        total_equity: Optional[float],
        current_time: float,
        summary_meta: Optional[Tuple[Optional[float], Optional[float]]],
        price_meta: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        try:
            limit_pct = float(getattr(settings, "ai_portfolio_loss_limit_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            limit_pct = 0.0

        if limit_pct <= 0 or total_equity is None or total_equity <= 0:
            return None

        positions = self._collect_open_positions(
            settings,
            summary,
            current_time=current_time,
            summary_meta=summary_meta,
            price_meta=price_meta,
        )

        total_loss = 0.0
        per_symbol_losses: List[Dict[str, object]] = []
        for symbol, info in positions.items():
            pnl_value = _safe_float(info.get("pnl_value"))
            if pnl_value is None or pnl_value >= 0:
                continue
            loss_value = -pnl_value
            total_loss += loss_value
            per_symbol_losses.append({"symbol": symbol, "loss_value": loss_value})

        if total_loss <= 0:
            return None

        loss_pct = (total_loss / total_equity) * 100.0
        if loss_pct <= limit_pct:
            return None

        reason = (
            "Совокупный нереализованный убыток {loss:.2f} USDT ({pct:.2f}%) превышает лимит {limit:.2f}%."
        ).format(loss=total_loss, pct=loss_pct, limit=limit_pct)

        try:
            cooldown_minutes = float(
                getattr(settings, "ai_kill_switch_cooldown_min", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            cooldown_minutes = 0.0

        kill_until = None
        if cooldown_minutes > 0:
            kill_until = self._activate_kill_switch(cooldown_minutes, reason)

        log(
            "guardian.auto.guard.portfolio_loss",
            loss=round(total_loss, 2),
            percent=round(loss_pct, 4),
            limit=limit_pct,
            equity=round(total_equity, 2),
            cooldown_minutes=cooldown_minutes,
        )

        message = reason
        if kill_until:
            try:
                resume_at = datetime.fromtimestamp(kill_until, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
            except (OSError, ValueError):
                resume_at = None
            else:
                message += f" Kill-switch активирован до {resume_at}."

        context: Dict[str, object] = {
            "guard": "portfolio_loss_limit",
            "loss_value": total_loss,
            "loss_percent": loss_pct,
            "limit_percent": limit_pct,
            "total_equity": total_equity,
            "positions": per_symbol_losses,
        }
        if kill_until:
            context["kill_switch_until"] = kill_until
        context["message"] = message
        return message, context

    def _drawdown_guard(
        self,
        settings: Settings,
        *,
        total_equity: Optional[float],
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        try:
            limit_pct = float(getattr(settings, "ai_max_drawdown_limit_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            limit_pct = 0.0

        if limit_pct <= 0.0:
            fallback_limit = resolve_deepseek_drawdown_limit()
            if fallback_limit is not None:
                limit_pct = float(fallback_limit)

        if limit_pct <= 0.0:
            return None

        force_refresh = self._daily_pnl_force_refresh
        try:
            aggregated = self._daily_pnl(force_refresh=force_refresh)
        except TypeError:
            aggregated = self._daily_pnl()
            if force_refresh:
                self._daily_pnl_force_refresh = False
        except Exception:
            if force_refresh:
                self._daily_pnl_force_refresh = True
            return None
        else:
            if force_refresh:
                self._daily_pnl_force_refresh = False

        if not isinstance(aggregated, Mapping):
            return None

        history: list[Tuple[str, float]] = []
        cumulative = 0.0
        peak = 0.0
        last_drawdown = 0.0
        max_drawdown = 0.0

        for day in sorted(aggregated.keys()):
            net_result = self._extract_spot_daily_net(aggregated, day)
            if net_result is None:
                continue
            history.append((day, float(net_result)))
            cumulative += float(net_result)
            if cumulative > peak:
                peak = cumulative
            else:
                drop = peak - cumulative
                if drop > max_drawdown:
                    max_drawdown = drop
            last_drawdown = peak - cumulative

        if not history:
            return None
        if last_drawdown <= 0.0:
            return None

        equity_value: Optional[float] = None
        if isinstance(total_equity, (int, float)) and math.isfinite(total_equity) and total_equity > 0:
            equity_value = float(total_equity)
        if equity_value is None:
            try:
                _, wallet_totals, _, _ = self._resolve_wallet(require_success=False)
            except Exception:
                wallet_totals = None
            if isinstance(wallet_totals, Sequence) and wallet_totals:
                candidate = _safe_float(wallet_totals[0])
                if candidate is not None and candidate > 0.0:
                    equity_value = candidate
        if equity_value is None or equity_value <= 0.0:
            return None

        peak_equity = equity_value + last_drawdown
        if peak_equity <= 0.0:
            return None

        drawdown_pct = (last_drawdown / peak_equity) * 100.0
        if drawdown_pct <= limit_pct:
            return None

        try:
            cooldown_minutes = float(getattr(settings, "ai_kill_switch_cooldown_min", 0.0) or 0.0)
        except (TypeError, ValueError):
            cooldown_minutes = 0.0

        kill_until = None
        if cooldown_minutes > 0.0:
            reason = (
                "Просадка портфеля превысила лимит {limit:.2f}% ({loss:.2f}%)."
            ).format(limit=limit_pct, loss=drawdown_pct)
            kill_until = self._activate_kill_switch(cooldown_minutes, reason)

        log(
            "guardian.auto.guard.drawdown",
            drawdown=round(last_drawdown, 2),
            percent=round(drawdown_pct, 4),
            limit=limit_pct,
            equity=round(equity_value, 2),
            peak=round(peak_equity, 2),
            max_drawdown=round(max_drawdown, 2),
        )

        message = (
            "Просадка портфеля {loss:.2f} USDT ({percent:.2f}%) превысила лимит {limit:.2f}% —"
            " автоматика приостановлена"
        ).format(loss=last_drawdown, percent=drawdown_pct, limit=limit_pct)

        if kill_until:
            try:
                resume_at = datetime.fromtimestamp(kill_until, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
            except (OSError, ValueError):
                resume_at = None
            else:
                message += f". Kill-switch активирован до {resume_at}."

        context: Dict[str, object] = {
            "guard": "max_drawdown_limit",
            "drawdown_value": last_drawdown,
            "drawdown_percent": drawdown_pct,
            "limit_percent": limit_pct,
            "total_equity": equity_value,
            "peak_equity": peak_equity,
            "max_historical_drawdown": max_drawdown,
            "daily_history": [
                {"day": day, "net": net}
                for day, net in history[-10:]
            ],
        }
        if kill_until:
            context["kill_switch_until"] = kill_until

        context["message"] = message
        return message, context


    def _extract_spot_daily_net(
        self, aggregated: Mapping[str, object], day_key: str
    ) -> Optional[float]:
        day_bucket = aggregated.get(day_key)
        if not isinstance(day_bucket, Mapping):
            return None

        net_result = 0.0
        for payload in day_bucket.values():
            if not isinstance(payload, Mapping):
                continue

            spot_net = _safe_float(payload.get("spot_net"))
            if spot_net is not None:
                net_result += spot_net
                continue

            spot_pnl = _safe_float(payload.get("spot_pnl"))
            if spot_pnl is None:
                continue

            spot_fees = _safe_float(payload.get("spot_fees"))
            if spot_fees is not None:
                net_result += spot_pnl - spot_fees
                continue

            categories = payload.get("categories")
            if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
                if all(str(cat).lower() != "spot" for cat in categories):
                    continue

            category = payload.get("category")
            if isinstance(category, str) and category.strip().lower() != "spot":
                continue

            fees = _safe_float(payload.get("fees")) or 0.0
            net_result += (spot_pnl or 0.0) - fees

        return net_result

    def _daily_loss_guard(
        self, settings: Settings, *, total_equity: Optional[float] = None
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        try:
            limit_pct = float(getattr(settings, "ai_daily_loss_limit_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            limit_pct = 0.0

        if limit_pct <= 0.0:
            return None

        force_refresh = self._daily_pnl_force_refresh
        try:
            if force_refresh:
                aggregated = self._daily_pnl(force_refresh=True)
            else:
                aggregated = self._daily_pnl()
        except TypeError:
            aggregated = self._daily_pnl()
            if force_refresh:
                self._daily_pnl_force_refresh = False
        except Exception:
            if force_refresh:
                self._daily_pnl_force_refresh = True
            return None
        else:
            if force_refresh:
                self._daily_pnl_force_refresh = False

        if not isinstance(aggregated, Mapping):
            return None

        day_key = time.strftime("%Y-%m-%d", time.gmtime(self._current_time()))
        net_result = self._extract_spot_daily_net(aggregated, day_key)
        if net_result is None:
            return None

        if net_result >= 0.0:
            return None

        equity_value: Optional[float] = None
        if isinstance(total_equity, (int, float)) and math.isfinite(total_equity):
            if total_equity > 0:
                equity_value = float(total_equity)

        if equity_value is None:
            try:
                _, wallet_totals, _, _ = self._resolve_wallet(require_success=False)
            except Exception:
                return None

            if not isinstance(wallet_totals, Sequence) or len(wallet_totals) < 1:
                return None

            equity_candidate = _safe_float(wallet_totals[0])
            if equity_candidate is None or equity_candidate <= 0.0:
                return None
            equity_value = equity_candidate

        loss_value = -net_result
        loss_pct = (loss_value / equity_value) * 100.0 if equity_value > 0 else 0.0

        if loss_pct <= limit_pct:
            return None

        context: Dict[str, object] = {
            "guard": "daily_loss_limit",
            "day": day_key,
            "daily_pnl": net_result,
            "loss_value": loss_value,
            "loss_percent": loss_pct,
            "limit_percent": limit_pct,
            "total_equity": equity_value,
        }

        try:
            cooldown_minutes = float(
                getattr(settings, "ai_kill_switch_cooldown_min", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            cooldown_minutes = 0.0

        kill_until = None
        if cooldown_minutes > 0:
            reason = (
                "Дневной лимит убытка {limit:.2f}% превышен ({loss:.2f}%)."
            ).format(limit=limit_pct, loss=loss_pct)
            kill_until = self._activate_kill_switch(cooldown_minutes, reason)
            context["kill_switch_until"] = kill_until

        log(
            "guardian.auto.guard.daily_loss",
            loss=round(loss_value, 2),
            percent=round(loss_pct, 4),
            limit=limit_pct,
            equity=round(equity_value, 2),
        )

        message = (
            "Дневной убыток {loss:.2f} USDT ({percent:.2f}% капитала) превысил лимит {limit:.2f}% —"
            " автоматика приостановлена до конца суток"
        ).format(loss=loss_value, percent=loss_pct, limit=limit_pct)

        if kill_until:
            try:
                resume_at = datetime.fromtimestamp(kill_until, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
            except (OSError, ValueError):
                resume_at = None
            else:
                message += f" Kill-switch активирован до {resume_at}."

        context["message"] = message
        return message, context

    def _private_ws_guard(
        self, settings: Settings
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        if not getattr(settings, "ws_watchdog_enabled", False):
            return None

        threshold = _safe_float(getattr(settings, "ws_watchdog_max_age_sec", None))
        if threshold is None or threshold <= 0:
            return None

        manager = self._ws_manager()
        try:
            status = manager.status()
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
    ) -> Tuple[Optional[object], Tuple[float, float], Optional[float], Dict[str, object]]:
        metadata: Dict[str, object] = {}
        try:
            api = self._get_api_client()
        except Exception:
            if require_success:
                raise
            return None, (0.0, 0.0), None, metadata

        try:
            payload = self._wallet_balance_payload(api)
        except Exception:
            if require_success:
                raise
            return api, (0.0, 0.0), None, metadata

        totals = self._extract_wallet_totals(payload)
        quote_balance: Optional[float] = None
        try:
            balances = self._wallet_available_balances(api, required_asset="USDT")
        except Exception as exc:
            balances = None
            quote_balance = 0.0
            metadata["quote_wallet_cap_error"] = "wallet_balance_unavailable"
            log(
                "guardian.auto.wallet.available.error",
                err=str(exc),
            )
        if isinstance(balances, Mapping):
            quote_balance = 0.0
            raw_quote = None
            for candidate in ("USDT", "usdt"):
                raw_quote = balances.get(candidate)
                if raw_quote is not None:
                    break
            if raw_quote is not None:
                try:
                    quote_value = float(raw_quote)
                except (TypeError, ValueError):
                    quote_value = None
                else:
                    if not math.isfinite(quote_value) or quote_value < 0.0:
                        quote_value = 0.0
                if quote_value is not None:
                    quote_balance = quote_value

        return api, totals, quote_balance, metadata

    def _portfolio_quote_exposure(
        self,
        settings: Settings,
        summary: Mapping[str, object],
        *,
        current_time: float,
        summary_meta: Tuple[Optional[float], Optional[float]],
        price_meta: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[Dict[str, float], float]:
        try:
            positions = self._collect_open_positions(
                settings,
                summary,
                current_time=current_time,
                summary_meta=summary_meta,
                price_meta=price_meta,
            )
        except Exception:  # pragma: no cover - defensive guard
            return {}, 0.0

        exposures: Dict[str, float] = {}
        total = 0.0
        for raw_symbol, payload in positions.items():
            if not isinstance(raw_symbol, str) or not isinstance(payload, Mapping):
                continue
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            notional = _safe_float(payload.get("quote_notional"))
            if notional is None or notional <= 0:
                continue
            exposures[symbol] = float(notional)
            total += float(notional)
        return exposures, total
