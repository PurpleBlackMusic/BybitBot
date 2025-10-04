"""User-friendly helper that summarises the spot AI signals for beginners."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import json
import time

from .envs import Settings, get_settings
from .paths import DATA_DIR
from .trade_analytics import (
    ExecutionRecord,
    aggregate_execution_metrics,
    normalise_execution_payload,
)
from .spot_pnl import spot_inventory_and_pnl


@dataclass(frozen=True)
class GuardianBrief:
    """Human-readable snapshot of the current trading situation."""

    mode: str
    symbol: str
    headline: str
    action_text: str
    confidence_text: str
    ev_text: str
    caution: str
    updated_text: str
    analysis: str
    status_age: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        """Convert the brief to a JSON-serialisable payload."""

        return asdict(self)


@dataclass(frozen=True)
class GuardianLedgerView:
    """Cached aggregation of ledger-derived analytics."""

    portfolio: Dict[str, object]
    recent_trades: Tuple[Dict[str, object], ...]
    trade_stats: Dict[str, object]
    executions: Tuple[ExecutionRecord, ...]


@dataclass(frozen=True)
class GuardianSnapshot:
    """Aggregated view of bot state, built from disk once and reused."""

    status: Dict[str, object]
    brief: GuardianBrief
    status_summary: Dict[str, object]
    status_from_cache: bool
    portfolio: Dict[str, object]
    watchlist: List[Dict[str, object]]
    recent_trades: List[Dict[str, object]]
    trade_stats: Dict[str, object]
    executions: Tuple[ExecutionRecord, ...]
    generated_at: float
    status_signature: float
    ledger_signature: float


class GuardianBot:
    """Transforms raw AI outputs into safe, beginner-friendly explanations."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._ensure_dirs()
        self._settings = settings
        self._custom_settings = settings is not None
        self._last_status: Dict[str, object] = {}
        self._snapshot: Optional[GuardianSnapshot] = None
        self._ledger_signature: Optional[float] = None
        self._ledger_view: Optional[GuardianLedgerView] = None
        self._status_fallback_used: bool = False

    # ------------------------------------------------------------------
    # internal plumbing
    def _ensure_dirs(self) -> None:
        for sub in ("ai", "pnl"):
            (Path(self.data_dir) / sub).mkdir(parents=True, exist_ok=True)

    def _status_path(self) -> Path:
        return Path(self.data_dir) / "ai" / "status.json"

    def _ledger_path(self) -> Path:
        return Path(self.data_dir) / "pnl" / "executions.jsonl"

    @property
    def settings(self) -> Settings:
        if self._custom_settings:
            if self._settings is None:
                self._settings = Settings()
            return self._settings

        settings = get_settings()
        self._settings = settings
        return settings

    def reload_settings(self) -> None:
        if not self._custom_settings:
            self._settings = get_settings(force_reload=True)
        self._snapshot = None
        self._ledger_signature = None
        self._ledger_view = None

    def _load_status(self) -> Dict[str, object]:
        path = self._status_path()
        fallback_used = False

        raw: Dict[str, object] = {}
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                content = ""

            if content.strip():
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        raw = parsed
                except Exception:
                    raw = {}

        if not raw and self._last_status:
            raw = copy.deepcopy(self._last_status)
            fallback_used = True

        status = dict(raw)
        if status:
            try:
                ts = float(status.get("last_tick_ts") or 0.0)
            except Exception:
                ts = 0.0
            status["age_seconds"] = time.time() - ts if ts > 0 else None

        self._status_fallback_used = fallback_used
        return status

    @staticmethod
    def _extract_text(status: Dict[str, object], keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            value = status.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _narrative_from_status(self, status: Dict[str, object], mode: str, symbol: str) -> str:
        narrative = self._extract_text(
            status,
            (
                "explanation",
                "narrative",
                "commentary",
                "context",
                "reason",
                "summary",
            ),
        )
        if narrative:
            return narrative

        if mode == "buy":
            return (
                f"Модель заметила усиливающийся спрос на {symbol}. "
                "Мы готовимся покупать только ту долю, которая вписывается в лимиты по риску."
            )
        if mode == "sell":
            return (
                f"Цена по {symbol} дошла до зоны, где выгодно зафиксировать часть прибыли. "
                "Решение сопровождаем проверкой позиции и комиссии."
            )
        return (
            f"По {symbol} нет чёткой тенденции. Мы сохраняем капитал и ждём, пока данные дадут явное преимущество."
        )

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _age_to_text(age: Optional[float]) -> str:
        if age is None:
            return "Данные обновлены только что."
        if age < 60:
            return "Данные обновлены менее минуты назад."
        minutes = int(age // 60)
        if minutes == 0:
            return "Данные обновлены около минуты назад."
        if minutes < 5:
            return f"Данные обновлены {minutes} мин назад."
        return "Данные не поступали более пяти минут — убедитесь, что соединение с ботом активно."

    @staticmethod
    def _format_timestamp(value: object) -> str:
        """Convert diverse timestamp values to a short human string."""

        if value in (None, ""):
            return "—"

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        # heuristics: timestamps may come in seconds, milliseconds or nanoseconds
        if numeric > 1e18:
            numeric /= 1e9
        elif numeric > 1e12:
            numeric /= 1e3

        try:
            dt = datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return "—"
        return dt.strftime("%d.%m %H:%M")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return "менее минуты"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes} мин"
        hours = minutes // 60
        minutes = minutes % 60
        if hours < 24:
            if minutes == 0:
                return f"{hours} ч"
            return f"{hours} ч {minutes} мин"
        days = hours // 24
        hours = hours % 24
        if hours == 0:
            return f"{days} дн"
        return f"{days} дн {hours} ч"

    def _status_staleness(self, age: Optional[float]) -> Tuple[str, str]:
        """Categorise the freshness of the AI status file."""

        if age is None:
            return "fresh", "AI сигнал обновился только что."
        if age < 60:
            return "fresh", "AI сигнал обновился менее минуты назад."
        if age < 300:
            return "fresh", f"AI сигнал обновился {int(age // 60)} мин назад."
        if age < 900:
            return (
                "warning",
                f"AI сигнал обновлялся {self._format_duration(age)} назад — дождитесь скорого обновления.",
            )
        return (
            "stale",
            "AI сигнал не обновлялся более 15 минут — убедитесь, что сервис записи status.json активен.",
        )

    # snapshot helpers -------------------------------------------------
    def _snapshot_signature(self) -> Tuple[float, float]:
        status_path = self._status_path()
        ledger_path = self._ledger_path()
        try:
            status_mtime = status_path.stat().st_mtime
        except FileNotFoundError:
            status_mtime = 0.0
        try:
            ledger_mtime = ledger_path.stat().st_mtime
        except FileNotFoundError:
            ledger_mtime = 0.0
        return status_mtime, ledger_mtime

    def _load_ledger_events(self) -> List[Dict[str, object]]:
        path = self._ledger_path()
        if not path.exists():
            return []

        events: List[Dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        events.append(payload)
        except OSError:
            return []
        return events

    @staticmethod
    def _spot_events(events: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        spot: List[Dict[str, object]] = []
        for event in events:
            category = str(event.get("category") or "spot").lower()
            if category == "spot":
                spot.append(event)
        return spot

    def _ledger_view_for_signature(self, signature: float) -> GuardianLedgerView:
        cached_signature = self._ledger_signature
        cached_view = self._ledger_view
        if cached_signature == signature and cached_view is not None:
            return cached_view

        events = self._load_ledger_events()
        spot_events = self._spot_events(events)
        portfolio = self._build_portfolio(spot_events)
        recent_trades = tuple(self._build_recent_trades(spot_events))
        executions = self._build_execution_records(events)
        trade_stats = aggregate_execution_metrics(executions)

        view = GuardianLedgerView(
            portfolio=portfolio,
            recent_trades=recent_trades,
            trade_stats=trade_stats,
            executions=executions,
        )

        self._ledger_signature = signature
        self._ledger_view = view
        return view

    def _build_portfolio(self, spot_events: List[Dict[str, object]]) -> Dict[str, object]:
        inventory = spot_inventory_and_pnl(events=spot_events)
        positions: List[Dict[str, object]] = []
        total_realized = 0.0
        total_notional = 0.0
        open_positions = 0

        for symbol in sorted(inventory.keys()):
            rec = inventory[symbol]
            qty = float(rec.get("position_qty") or 0.0)
            avg_cost = float(rec.get("avg_cost") or 0.0)
            realized = float(rec.get("realized_pnl") or 0.0)
            notional = qty * avg_cost

            total_realized += realized
            total_notional += notional
            if qty > 0:
                open_positions += 1

            positions.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_cost": avg_cost,
                    "notional": notional,
                    "realized_pnl": realized,
                }
            )

        human_totals = {
            "realized": f"{total_realized:.2f} USDT",
            "open_notional": f"{total_notional:.2f} USDT",
            "open_positions": f"{open_positions}",
        }

        return {
            "positions": positions,
            "totals": {
                "realized_pnl": total_realized,
                "open_notional": total_notional,
                "open_positions": open_positions,
            },
            "human_totals": human_totals,
        }

    def _build_recent_trades(
        self, spot_events: List[Dict[str, object]], limit: int = 50
    ) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for event in reversed(spot_events):
            if len(records) >= limit:
                break
            symbol = str(event.get("symbol") or event.get("ticker") or "?")
            if not symbol:
                continue
            side = str(event.get("side") or event.get("direction") or "").capitalize()
            qty = float(event.get("execQty") or event.get("qty") or event.get("size") or 0.0)
            price = float(event.get("execPrice") or event.get("price") or 0.0)
            fee = float(event.get("execFee") or event.get("fee") or 0.0)
            ts = (
                event.get("execTime")
                or event.get("execTimeNs")
                or event.get("transactTime")
                or event.get("created_at")
                or event.get("tradeTime")
            )

            records.append(
                {
                    "symbol": symbol.upper(),
                    "side": side or "—",
                    "price": round(price, 6) if price else None,
                    "qty": round(qty, 6) if qty else None,
                    "fee": round(fee, 6) if fee else None,
                    "when": self._format_timestamp(ts),
                }
            )
        return records

    @staticmethod
    def _build_execution_records(
        events: Iterable[Dict[str, object]]
    ) -> Tuple[ExecutionRecord, ...]:
        records: List[ExecutionRecord] = []
        for event in events:
            record = normalise_execution_payload(event)
            if record is not None:
                records.append(record)
        return tuple(records)

    def _build_watchlist(self, status: Dict[str, object]) -> List[Dict[str, object]]:
        candidates = (
            status.get("watchlist")
            or status.get("heatmap")
            or status.get("opportunities")
            or status.get("signals")
        )
        entries: List[Dict[str, object]] = []

        if isinstance(candidates, dict):
            for symbol, payload in candidates.items():
                entries.append(self._normalise_watchlist_entry(str(symbol), payload))
        elif isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    symbol = item.get("symbol") or item.get("ticker") or "?"
                    entries.append(self._normalise_watchlist_entry(str(symbol), item))
                elif isinstance(item, (list, tuple)) and item:
                    symbol = str(item[0])
                    payload = item[1] if len(item) > 1 else None
                    entries.append(self._normalise_watchlist_entry(symbol, payload))
                elif isinstance(item, str):
                    entries.append(self._normalise_watchlist_entry(item, None))

        filtered = [entry for entry in entries if any(entry.values())]

        def sort_key(item: Dict[str, object]) -> tuple:
            score = item.get("score")
            if isinstance(score, (int, float)):
                return (0, -float(score), item["symbol"])
            return (1, item["symbol"])

        filtered.sort(key=sort_key)
        return filtered

    def _build_snapshot(self, signature: Tuple[float, float]) -> GuardianSnapshot:
        status = self._load_status()
        self._last_status = status
        settings = self.settings
        brief = self._brief_from_status(status, settings)
        status_summary = self._build_status_summary(
            status, brief, settings, self._status_fallback_used
        )

        ledger_view = self._ledger_view_for_signature(signature[1])
        portfolio = ledger_view.portfolio
        watchlist = self._build_watchlist(status)
        recent_trades = list(ledger_view.recent_trades)
        execution_records = ledger_view.executions
        trade_stats = ledger_view.trade_stats

        return GuardianSnapshot(
            status=status,
            brief=brief,
            status_summary=status_summary,
            status_from_cache=self._status_fallback_used,
            portfolio=portfolio,
            watchlist=watchlist,
            recent_trades=recent_trades,
            trade_stats=trade_stats,
            executions=execution_records,
            generated_at=time.time(),
            status_signature=signature[0],
            ledger_signature=signature[1],
        )

    @staticmethod
    def _copy_dict(payload: Dict[str, object]) -> Dict[str, object]:
        return copy.deepcopy(payload)

    @staticmethod
    def _copy_list(payload: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return copy.deepcopy(payload)

    def _get_snapshot(self, force: bool = False) -> GuardianSnapshot:
        signature = self._snapshot_signature()
        snapshot = self._snapshot
        if (
            force
            or snapshot is None
            or snapshot.status_signature != signature[0]
            or snapshot.ledger_signature != signature[1]
        ):
            snapshot = self._build_snapshot(signature)
            self._snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # public analytics helpers
    def _build_status_summary(
        self,
        status: Dict[str, object],
        brief: GuardianBrief,
        settings: Settings,
        fallback_used: bool,
    ) -> Dict[str, object]:
        probability = self._clamp_probability(float(status.get("probability") or 0.0))
        ev_bps = float(status.get("ev_bps") or 0.0)
        last_tick = status.get("last_tick_ts") or status.get("timestamp")

        buy_threshold = float(settings.ai_buy_threshold or 0.0)
        sell_threshold = float(settings.ai_sell_threshold or 0.0)
        min_ev = float(settings.ai_min_ev_bps or 0.0)

        staleness_state, staleness_message = self._status_staleness(brief.status_age)
        actionable, reasons = self._evaluate_actionability(
            brief.mode,
            probability,
            ev_bps,
            buy_threshold,
            sell_threshold,
            min_ev,
            staleness_state,
        )

        summary = {
            "symbol": brief.symbol,
            "mode": brief.mode,
            "headline": brief.headline,
            "probability": probability,
            "probability_pct": round(probability * 100.0, 2),
            "ev_bps": round(ev_bps, 2),
            "ev_text": brief.ev_text,
            "action_text": brief.action_text,
            "confidence_text": brief.confidence_text,
            "caution": brief.caution,
            "analysis": brief.analysis,
            "updated_text": brief.updated_text,
            "age_seconds": brief.status_age,
            "last_update": self._format_timestamp(last_tick),
            "actionable": actionable,
            "actionable_reasons": reasons,
            "thresholds": {
                "buy_probability_pct": round(buy_threshold * 100.0, 2),
                "sell_probability_pct": round(sell_threshold * 100.0, 2),
                "min_ev_bps": round(min_ev, 2),
            },
            "has_status": bool(status),
            "fallback_used": bool(fallback_used),
            "status_source": "cached" if fallback_used else "live",
            "staleness": {
                "state": staleness_state,
                "message": staleness_message,
            },
        }

        if status:
            summary["raw_keys"] = sorted(status.keys())

        return summary

    def _evaluate_actionability(
        self,
        mode: str,
        probability: float,
        ev_bps: float,
        buy_threshold: float,
        sell_threshold: float,
        min_ev: float,
        staleness_state: str,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if staleness_state == "stale":
            reasons.append(
                "Сигнал устарел — обновите status.json перед тем, как открывать сделки."
            )

        if mode == "buy":
            if probability < max(buy_threshold, 0.0):
                reasons.append(
                    "Уверенность модели ниже порога покупки — дождитесь более сильного сигнала."
                )
            if ev_bps < min_ev:
                reasons.append(
                    "Ожидаемая выгода ниже безопасного минимума — риск/прибыль не в нашу пользу."
                )
        elif mode == "sell":
            if probability < max(sell_threshold, 0.0):
                reasons.append(
                    "Уверенность продажи ниже заданного порога — можно не спешить с фиксацией."
                )
            if ev_bps < min_ev:
                reasons.append(
                    "Ожидаемая выгода по продаже ниже минимума — сделка может быть невыгодной."
                )
        else:
            reasons.append("Нет активного сигнала — бот предпочитает выжидать.")

        actionable = len(reasons) == 0
        return actionable, reasons

    def _brief_from_status(self, status: Dict[str, object], settings: Settings) -> GuardianBrief:
        symbols = (settings.ai_symbols or "BTCUSDT").split(",")
        symbol = symbols[0].strip().upper() or "BTCUSDT"

        probability = self._clamp_probability(float(status.get("probability") or 0.0))
        ev_bps = float(status.get("ev_bps") or 0.0)
        side_hint = (status.get("side") or "").lower()

        buy_threshold = settings.ai_buy_threshold or 0.55
        sell_threshold = settings.ai_sell_threshold or 0.45
        min_ev = max(float(settings.ai_min_ev_bps or 0.0), 0.0)

        if probability >= max(buy_threshold, 0.55) and ev_bps >= min_ev:
            mode = "buy"
        elif side_hint == "sell" and probability <= min(sell_threshold, buy_threshold) and ev_bps >= min_ev:
            mode = "sell"
        else:
            mode = "wait"

        confidence_pct = probability * 100.0
        if confidence_pct >= 70:
            level = "высокая"
        elif confidence_pct >= 55:
            level = "средняя"
        else:
            level = "низкая"

        confidence_text = (
            f"Уверенность модели: {confidence_pct:.1f}% — {level}. Чем выше процент, тем спокойнее можно действовать."
        )

        if ev_bps >= min_ev:
            ev_text = f"Потенциальная выгода: {ev_bps:.1f} б.п. Цель по безопасности — не ниже {min_ev:.1f} б.п."
        else:
            ev_text = (
                f"Потенциальная выгода: {ev_bps:.1f} б.п., это ниже безопасного порога {min_ev:.1f} б.п. — торгуем осторожно."
            )

        if mode == "buy":
            headline = f"{symbol}: появился шанс аккуратно купить."
            action_text = "Действие: готовим небольшую покупку. Бот рассчитает безопасный объём сам."
            caution = (
                "Покупаем только на ту сумму, которую готовы спокойно удерживать. Без плеча, без импульсивных решений."
            )
        elif mode == "sell":
            headline = f"{symbol}: бот предлагает зафиксировать прибыль."
            action_text = "Действие: выставляем продажу по рыночной цене, но без суеты и с учётом комиссии."
            caution = "Перед продажей убедитесь, что позиция действительно открыта и нет незакрытых заявок."
        else:
            headline = f"{symbol}: спокойный режим, явного преимущества нет."
            action_text = "Действие: ждём. Сохраняем депозит и не открываем новые сделки."
            caution = "Дисциплина важнее сделок. Пауза — это тоже стратегия."

        updated_text = self._age_to_text(status.get("age_seconds"))
        analysis = self._narrative_from_status(status, mode, symbol)

        return GuardianBrief(
            mode=mode,
            symbol=symbol,
            headline=headline,
            action_text=action_text,
            confidence_text=confidence_text,
            ev_text=ev_text,
            caution=caution,
            updated_text=updated_text,
            analysis=analysis,
            status_age=status.get("age_seconds"),
        )

    def generate_brief(self) -> GuardianBrief:
        snapshot = self._get_snapshot()
        return snapshot.brief

    def portfolio_overview(self) -> Dict[str, object]:
        snapshot = self._get_snapshot()
        return self._copy_dict(snapshot.portfolio)

    def risk_summary(self) -> str:
        s = self.settings
        mode = "учебный режим (демо)" if s.dry_run else "работаем с реальными деньгами"
        reserve = float(getattr(s, "spot_cash_reserve_pct", 0.0))
        per_trade = float(getattr(s, "ai_risk_per_trade_pct", 0.0))
        loss_limit = float(getattr(s, "ai_daily_loss_limit_pct", 0.0))
        concurrent = int(getattr(s, "ai_max_concurrent", 0))
        cash_only = bool(getattr(s, "spot_cash_only", True))

        lines = [
            f"• Режим: {mode}.",
            f"• Резерв безопасности: {reserve:.1f}% депозита хранится в кэше.",
            f"• Риск на сделку: до {per_trade:.2f}% капитала, дневной лимит убытка {loss_limit:.2f}%.",
            f"• Одновременно открывается не более {concurrent} сделок.",
        ]
        if cash_only:
            lines.append("• Используем только собственные средства, без кредитного плеча.")
        else:
            lines.append("• Разрешено использовать заёмные средства — контролируйте плечо самостоятельно.")

        return "\n".join(lines)

    def plan_steps(self, brief: Optional[GuardianBrief] = None) -> List[str]:
        brief = brief or self.generate_brief()
        steps = [
            "Проверяем баланс USDT и убеждаемся, что на бирже достаточно средств без плеча.",
        ]
        if brief.mode == "buy":
            steps.append("Бот рассчитает размер покупки и предложит цену. Мы подтверждаем только если готовы держать позицию.")
        elif brief.mode == "sell":
            steps.append("Проверяем открытую позицию и подтверждаем продажу, чтобы зафиксировать результат.")
        else:
            steps.append("Наблюдаем за рынком и не торопимся. Пауза защищает капитал.")
        steps.append("После сделки сверяем PnL и обновляем заметки: что получилось и почему.")
        return steps

    def safety_notes(self) -> List[str]:
        notes = [
            "Бот не открывает сделки без ваших API-ключей и всегда уважает лимиты риска.",
            "Перед реальной торговлей протестируйте логику на учебном аккаунте или с маленькой суммой.",
            "Следите за обновлениями данных: если сигнал устарел, не спешите входить в рынок.",
        ]
        if self.settings.dry_run:
            notes.insert(0, "Сейчас включен учебный режим: сделки не затрагивают реальные средства.")
        else:
            notes.insert(0, "Работаем с реальными средствами — подтверждайте только понятные сделки.")
        return notes

    def signal_scorecard(self, brief: Optional[GuardianBrief] = None) -> Dict[str, object]:
        snapshot = self._get_snapshot()
        brief = brief or snapshot.brief
        status = snapshot.status
        probability = float(status.get("probability") or 0.0)
        ev_bps = float(status.get("ev_bps") or 0.0)
        return {
            "symbol": brief.symbol,
            "mode": brief.mode,
            "probability_pct": round(self._clamp_probability(probability) * 100.0, 2),
            "ev_bps": round(ev_bps, 2),
            "buy_threshold": float(self.settings.ai_buy_threshold or 0.0) * 100.0,
            "sell_threshold": float(self.settings.ai_sell_threshold or 0.0) * 100.0,
            "min_ev_bps": float(self.settings.ai_min_ev_bps or 0.0),
            "last_update": brief.updated_text,
        }

    def _normalise_watchlist_entry(self, symbol: str, payload: object) -> Dict[str, object]:
        if isinstance(payload, dict):
            score = payload.get("score") or payload.get("probability") or payload.get("ev")
            trend = payload.get("trend") or payload.get("direction") or payload.get("side")
            note = payload.get("note") or payload.get("comment") or payload.get("reason")
        else:
            score = payload
            trend = None
            note = None

        numeric_score: Optional[float] = None
        if isinstance(score, (int, float)):
            numeric_score = round(float(score), 2)
        elif isinstance(score, str):
            stripped = score.strip()
            if stripped:
                try:
                    numeric_score = round(float(stripped), 2)
                except ValueError:
                    numeric_score = None

        entry = {
            "symbol": symbol.upper(),
            "score": numeric_score,
            "trend": str(trend) if trend not in (None, "") else None,
            "note": str(note) if note not in (None, "") else None,
        }
        return entry

    def market_watchlist(self) -> List[Dict[str, object]]:
        snapshot = self._get_snapshot()
        return self._copy_list(snapshot.watchlist)

    def recent_trades(self, limit: int = 10) -> List[Dict[str, object]]:
        snapshot = self._get_snapshot()
        trades = snapshot.recent_trades[:limit] if limit else snapshot.recent_trades
        return self._copy_list(list(trades))

    def trade_statistics(self, limit: Optional[int] = None) -> Dict[str, object]:
        """Return aggregated execution metrics for dashboards."""

        snapshot = self._get_snapshot()
        if limit is None or limit <= 0:
            return copy.deepcopy(snapshot.trade_stats)

        records: Tuple[ExecutionRecord, ...] = snapshot.executions[-limit:]
        return aggregate_execution_metrics(records)

    def data_health(self) -> Dict[str, Dict[str, object]]:
        """High level diagnostics for the UI to highlight stale inputs."""

        snapshot = self._get_snapshot()
        status = snapshot.status
        summary = snapshot.status_summary
        age = snapshot.brief.status_age

        staleness_state, staleness_message = self._status_staleness(age)

        if snapshot.status_from_cache:
            ai_ok = False
            ai_message = (
                "Используем сохранённый сигнал — не удалось прочитать актуальный status.json."
            )
            ai_details = (
                f"Последнее обновление: {summary.get('last_update', '—')}. "
                "Проверьте сервис, который пишет файл ai/status.json."
            )
        elif status:
            if staleness_state == "stale":
                ai_ok = False
            else:
                ai_ok = True
            ai_message = staleness_message
            symbol = str(status.get("symbol") or "?").upper()
            probability = float(status.get("probability") or 0.0) * 100.0
            ai_details = f"Текущий символ: {symbol}, уверенность {probability:.1f}%."
        else:
            ai_ok = False
            ai_message = "AI сигнал ещё не поступал — запустите Guardian Bot или загрузите демо-данные."
            ai_details = "Файл ai/status.json не найден."

        stats = snapshot.trade_stats
        trades = int(stats.get("trades", 0) or 0)
        last_trade_ts = stats.get("last_trade_ts")
        if trades == 0 or not last_trade_ts:
            exec_ok = False
            exec_message = "Журнал исполнений пуст — бот ещё не записывал сделки."
            exec_details = "Добавьте записи в pnl/executions.jsonl для проверки."
        else:
            last_trade_dt = datetime.fromtimestamp(float(last_trade_ts), tz=timezone.utc)
            exec_age = (datetime.now(timezone.utc) - last_trade_dt).total_seconds()
            if exec_age < 900:
                exec_ok = True
                exec_message = "Журнал сделок обновлялся менее 15 минут назад."
            elif exec_age < 3600:
                exec_ok = True
                exec_message = f"Журнал сделок обновлялся {self._format_duration(exec_age)} назад."
            else:
                exec_ok = False
                exec_message = (
                    "Журнал сделок не обновлялся более часа — проверьте исполнение ордеров и соединение."
                )
            exec_details = (
                f"Записей: {trades}, последняя сделка: {stats.get('last_trade_at', '—')}"
            )

        settings = self.settings
        has_keys = bool(settings.api_key and settings.api_secret)
        if has_keys:
            api_message = "API ключи подключены — можно переходить в боевой режим."
        else:
            api_message = "API ключи не заданы — бот работает в учебном режиме."
        api_details = (
            f"Сеть: {'Testnet' if settings.testnet else 'Mainnet'} · "
            f"Режим: {'DRY-RUN' if settings.dry_run else 'Live'}"
        )

        return {
            "ai_signal": {
                "title": "AI сигнал",
                "ok": ai_ok,
                "message": ai_message,
                "details": ai_details,
                "age_seconds": age,
            },
            "executions": {
                "title": "Журнал исполнений",
                "ok": exec_ok,
                "message": exec_message,
                "details": exec_details,
                "trades": trades,
                "last_trade_at": stats.get("last_trade_at"),
            },
            "api_keys": {
                "title": "Подключение API",
                "ok": has_keys,
                "message": api_message,
                "details": api_details,
            },
        }

    def refresh(self) -> GuardianSnapshot:
        """Drop cached aggregates and rebuild them on next access."""

        self._snapshot = None
        self._ledger_signature = None
        self._ledger_view = None
        return self._get_snapshot(force=True)

    def unified_report(self) -> Dict[str, object]:
        """Return a merged view of spot, risk and execution data."""

        snapshot = self._get_snapshot()
        return {
            "generated_at": snapshot.generated_at,
            "status": self._copy_dict(snapshot.status_summary),
            "brief": snapshot.brief.to_dict(),
            "portfolio": self._copy_dict(snapshot.portfolio),
            "watchlist": self._copy_list(snapshot.watchlist),
            "recent_trades": self._copy_list(snapshot.recent_trades),
            "statistics": copy.deepcopy(snapshot.trade_stats),
            "health": self.data_health(),
        }

    def brief_payload(self) -> Dict[str, object]:
        """Return the cached brief as a serialisable dict."""

        snapshot = self._get_snapshot()
        return snapshot.brief.to_dict()

    def status_summary(self) -> Dict[str, object]:
        """Return a user-friendly dictionary of the latest signal."""

        snapshot = self._get_snapshot()
        return self._copy_dict(snapshot.status_summary)

    # ------------------------------------------------------------------
    # conversation helpers
    @staticmethod
    def _contains_any(text: str, needles: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

    def _format_profit_answer(self, portfolio: Dict[str, object]) -> str:
        totals = portfolio.get("totals", {})
        realized = float(totals.get("realized_pnl", 0.0))
        open_notional = float(totals.get("open_notional", 0.0))
        open_positions = int(totals.get("open_positions", 0))
        return (
            f"Зафиксированная прибыль: {realized:.2f} USDT. "
            f"В позициях работает около {open_notional:.2f} USDT, активных сделок: {open_positions}. "
            "Плавный прирост капитала важнее быстрых прыжков, поэтому бот закрывает сделки только при понятном плюсе."
        )

    def _format_plan_answer(self, brief: GuardianBrief) -> str:
        steps = self.plan_steps(brief)
        formatted = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
        return f"План действий:\n{formatted}"

    def _default_response(self, brief: GuardianBrief, portfolio: Dict[str, object]) -> str:
        pieces = [
            brief.headline,
            brief.action_text,
            brief.analysis,
            brief.caution,
            brief.confidence_text,
            brief.ev_text,
            brief.updated_text,
        ]
        pieces.append(self._format_profit_answer(portfolio))
        pieces.append(self.risk_summary())
        return "\n\n".join(pieces)

    def initial_message(self) -> str:
        brief = self.generate_brief()
        portfolio = self.portfolio_overview()
        plan = self._format_plan_answer(brief)
        profit = self._format_profit_answer(portfolio)
        message_parts = [
            brief.headline,
            brief.action_text,
            brief.analysis,
            brief.confidence_text,
            brief.ev_text,
            brief.caution,
            brief.updated_text,
            plan,
            profit,
        ]
        staleness = self.staleness_alert(brief)
        if staleness:
            message_parts.append(staleness)
        return "\n\n".join(message_parts)

    def answer(self, question: str) -> str:
        prompt = (question or "").strip()
        if not prompt:
            return "Спросите меня о прибыли, риске или плане действий, и я объясню простыми словами."

        brief = self.generate_brief()
        portfolio = self.portfolio_overview()

        if self._contains_any(prompt, ["прибыл", "profit", "доход", "pnl"]):
            return self._format_profit_answer(portfolio)

        if self._contains_any(prompt, ["риск", "risk", "потер", "loss"]):
            return self.risk_summary()

        if self._contains_any(prompt, ["план", "что делать", "как начать", "инструкц"]):
            return self._format_plan_answer(brief)

        if self._contains_any(prompt, ["почему", "объясн", "анализ", "что видит", "поясн"]):
            return self.market_story(brief)

        if self._contains_any(prompt, ["куп", "buy", "long"]):
            if brief.mode == "buy":
                return "\n".join([brief.action_text, brief.caution, brief.confidence_text])
            return (
                "Сейчас входить в покупку рано. "
                "Ждём, пока вероятность и выгода станут выше безопасных порогов."
            )

        if self._contains_any(prompt, ["прод", "sell", "short"]):
            if brief.mode == "sell":
                return "\n".join([brief.action_text, brief.caution, brief.confidence_text])
            return "Пока бот не видит сигнала на выход. Держим позицию под контролем и наблюдаем."

        if self._contains_any(prompt, ["привет", "hello", "hi"]):
            return self.initial_message()

        return self._default_response(brief, portfolio)

    def market_story(self, brief: Optional[GuardianBrief] = None) -> str:
        snapshot = self._get_snapshot()
        brief = brief or snapshot.brief
        status = snapshot.status
        narrative = self._narrative_from_status(status, brief.mode, brief.symbol)
        staleness = self.staleness_alert(brief)
        if staleness:
            return "\n\n".join([narrative, staleness])
        return narrative

    def staleness_alert(self, brief: Optional[GuardianBrief] = None) -> Optional[str]:
        brief = brief or self.generate_brief()
        age = brief.status_age
        if age is None:
            return None
        if age >= 900:
            return (
                "Сигнал старше 15 минут. Без свежих данных лучше не открывать новые сделки и проверить подключение."
            )
        if age >= 300:
            return "Данные не обновлялись более 5 минут. Убедитесь, что бот подключён и обновляет сигнал."
        return None
