"""User-friendly helper that summarises the spot AI signals for beginners."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
import time

from .envs import Settings, get_settings
from .paths import DATA_DIR
from . import trade_control
from .trade_analytics import (
    ExecutionRecord,
    aggregate_execution_metrics,
    normalise_execution_payload,
)
from .spot_pnl import spot_inventory_and_pnl
from .live_checks import bybit_realtime_status


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
    manual_state: trade_control.TradeControlState
    manual_summary: Dict[str, object]
    generated_at: float
    status_signature: float
    ledger_signature: float
    manual_signature: float


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

    @staticmethod
    def _normalise_mode_hint(value: object) -> Optional[str]:
        """Convert heterogeneous directional hints into canonical modes."""

        if value is None:
            return None

        if isinstance(value, bool):
            return "buy" if value else "sell"

        if isinstance(value, (int, float)):
            if value > 0:
                return "buy"
            if value < 0:
                return "sell"
            return "wait"

        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower()
        if not cleaned:
            return None

        # Normalise delimiters and remove apostrophes for "don't" like phrases.
        cleaned = cleaned.replace("-", " ").replace("_", " ").replace("'", "")

        # Try to interpret numeric or boolean-looking strings early.
        if cleaned in {"true", "yes"}:
            return "buy"
        if cleaned in {"false", "no"}:
            return "sell"
        try:
            numeric = float(cleaned)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric > 0:
                return "buy"
            if numeric < 0:
                return "sell"
            return "wait"

        tokens = [token for token in cleaned.split() if token]
        joined = " ".join(tokens)

        phrase_map = {
            "no trade": "wait",
            "do not trade": "wait",
            "dont trade": "wait",
            "no signal": "wait",
            "stand aside": "wait",
            "stay aside": "wait",
            "не торгуй": "wait",
            "не торгуем": "wait",
            "не торговать": "wait",
            "ничего не делаем": "wait",
            "без сделки": "wait",
            "take profit": "sell",
            "take profits": "sell",
            "trim position": "sell",
            "фиксиру": "sell",
            "зафиксиру": "sell",
        }
        for phrase, mode in phrase_map.items():
            if phrase in joined:
                return mode

        buy_tokens = {
            "buy",
            "long",
            "bull",
            "bullish",
            "accumulate",
            "accumulation",
            "bid",
            "покупай",
            "покупаем",
            "покупка",
            "покупать",
            "лонг",
        }
        sell_tokens = {
            "sell",
            "short",
            "bear",
            "bearish",
            "distribute",
            "distribution",
            "exit",
            "reduce",
            "trim",
            "продавай",
            "продаем",
            "продаём",
            "продажа",
            "продавать",
            "фиксиру",
            "зафиксиру",
            "сократи",
            "сокращаем",
        }
        wait_tokens = {
            "wait",
            "hold",
            "holding",
            "flat",
            "neutral",
            "idle",
            "none",
            "stay",
            "pause",
            "observe",
            "sideline",
            "watch",
            "ждать",
            "ждём",
            "ждем",
            "ждите",
            "ждемс",
            "ожидаем",
            "ожидай",
            "держим",
            "держи",
            "держать",
            "пауза",
            "паузы",
            "ничего",
            "сидим",
            "выжидаем",
        }

        for token in tokens:
            if token in buy_tokens:
                return "buy"
            if token in sell_tokens:
                return "sell"
            if token in wait_tokens:
                return "wait"

        prefixes = {
            "buy": "buy",
            "long": "buy",
            "bull": "buy",
            "accum": "buy",
            "покуп": "buy",
            "лонг": "buy",
            "sell": "sell",
            "short": "sell",
            "bear": "sell",
            "distrib": "sell",
            "exit": "sell",
            "trim": "sell",
            "прода": "sell",
            "фикс": "sell",
            "сократ": "sell",
            "reduc": "sell",
            "wait": "wait",
            "hold": "wait",
            "flat": "wait",
            "neutral": "wait",
            "pause": "wait",
            "idle": "wait",
            "stay": "wait",
            "watch": "wait",
            "side": "wait",
            "жд": "wait",
            "ожид": "wait",
            "держ": "wait",
            "пауз": "wait",
            "ничег": "wait",
            "сид": "wait",
            "выжид": "wait",
        }
        for token in tokens:
            for prefix, mode in prefixes.items():
                if token.startswith(prefix):
                    return mode

        if tokens:
            first = tokens[0]
            if first in {"up", "increase", "add"}:
                return "buy"
            if first in {"down", "decrease", "cut"}:
                return "sell"

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
    def _snapshot_signature(self) -> Tuple[float, float, float]:
        status_path = self._status_path()
        ledger_path = self._ledger_path()
        manual_path = Path(self.data_dir) / "ai" / "trade_commands.jsonl"
        try:
            status_mtime = status_path.stat().st_mtime
        except FileNotFoundError:
            status_mtime = 0.0
        try:
            ledger_mtime = ledger_path.stat().st_mtime
        except FileNotFoundError:
            ledger_mtime = 0.0
        try:
            manual_mtime = manual_path.stat().st_mtime
        except FileNotFoundError:
            manual_mtime = 0.0
        return status_mtime, ledger_mtime, manual_mtime

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

    def _build_snapshot(self, signature: Tuple[float, float, float]) -> GuardianSnapshot:
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

        manual_state = trade_control.trade_control_state(data_dir=self.data_dir)
        manual_summary = self._manual_control_summary(manual_state)
        status_summary["manual_control"] = manual_summary

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
            manual_state=self._clone_trade_state(manual_state),
            manual_summary=manual_summary,
            generated_at=time.time(),
            status_signature=signature[0],
            ledger_signature=signature[1],
            manual_signature=signature[2],
        )

    @staticmethod
    def _copy_dict(payload: Dict[str, object]) -> Dict[str, object]:
        return copy.deepcopy(payload)

    @staticmethod
    def _copy_list(payload: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return copy.deepcopy(payload)

    @staticmethod
    def _clone_trade_entry(
        entry: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        return dict(entry)

    def _clone_trade_state(
        self, state: trade_control.TradeControlState
    ) -> trade_control.TradeControlState:
        commands = tuple(dict(entry) for entry in state.commands)
        return trade_control.TradeControlState(
            active=state.active,
            commands=commands,
            last_action=self._clone_trade_entry(state.last_action),
            last_start=self._clone_trade_entry(state.last_start),
            last_cancel=self._clone_trade_entry(state.last_cancel),
        )

    def _manual_control_summary(
        self, state: trade_control.TradeControlState
    ) -> Dict[str, object]:
        history = [dict(entry) for entry in state.commands]

        def _enrich(entry: Dict[str, Any]) -> Dict[str, Any]:
            enriched = dict(entry)
            enriched["ts_human"] = self._format_timestamp(entry.get("ts"))
            action = str(entry.get("action") or "").upper()
            if action:
                enriched["action_label"] = action
            prob = entry.get("probability_pct")
            try:
                if prob is not None:
                    enriched["probability_text"] = f"{float(prob):.1f}%"
            except (TypeError, ValueError):
                pass
            ev = entry.get("ev_bps")
            try:
                if ev is not None:
                    enriched["ev_text"] = f"{float(ev):.1f} б.п."
            except (TypeError, ValueError):
                pass
            note = entry.get("note") or entry.get("reason")
            if note:
                enriched["note_or_reason"] = str(note)
            return enriched

        enriched_history = [_enrich(entry) for entry in history]

        last_action = self._clone_trade_entry(state.last_action)
        last_start = self._clone_trade_entry(state.last_start)
        last_cancel = self._clone_trade_entry(state.last_cancel)

        symbol = None
        for candidate in (last_action, last_start, last_cancel):
            if candidate and candidate.get("symbol"):
                symbol = str(candidate["symbol"])
                break

        def _age_seconds(entry: Optional[Dict[str, Any]]) -> Optional[float]:
            if not entry:
                return None
            try:
                ts = float(entry.get("ts"))
            except (TypeError, ValueError):
                return None
            return max(0.0, time.time() - ts)

        last_action_age = _age_seconds(last_action)
        last_start_age = _age_seconds(last_start)
        last_cancel_age = _age_seconds(last_cancel)

        def _age_text(age: Optional[float]) -> Optional[str]:
            if age is None:
                return None
            return self._format_duration(age)

        if state.active:
            status_label = "active"
            if last_start_age is not None:
                duration = _age_text(last_start_age)
                status_text = (
                    f"Торговля по {symbol or 'выбранному символу'} активна — старт был {duration} назад."
                )
            else:
                status_text = "Торговля запущена вручную и отмечена как активная."
        elif last_cancel:
            status_label = "stopped"
            if last_cancel_age is not None:
                duration = _age_text(last_cancel_age)
                status_text = (
                    f"Торговля по {symbol or 'выбранному символу'} остановлена оператором {duration} назад."
                )
            else:
                status_text = "Торговля остановлена командой оператора."
        elif history:
            status_label = "pending"
            status_text = (
                "Последняя команда — запуск торговли, но подтверждения активной сделки пока нет."
            )
        else:
            status_label = "idle"
            status_text = "Ручные команды ещё не отправлялись."

        summary: Dict[str, object] = {
            "active": state.active,
            "symbol": symbol,
            "status_label": status_label,
            "status_text": status_text,
            "history": enriched_history,
            "history_count": len(enriched_history),
            "last_action": last_action,
            "last_start": last_start,
            "last_cancel": last_cancel,
            "last_action_at": self._format_timestamp((last_action or {}).get("ts")),
            "last_start_at": self._format_timestamp((last_start or {}).get("ts")),
            "last_cancel_at": self._format_timestamp((last_cancel or {}).get("ts")),
            "last_action_age_seconds": last_action_age,
            "last_start_age_seconds": last_start_age,
            "last_cancel_age_seconds": last_cancel_age,
        }

        return summary

    def _get_snapshot(self, force: bool = False) -> GuardianSnapshot:
        signature = self._snapshot_signature()
        snapshot = self._snapshot
        if (
            force
            or snapshot is None
            or snapshot.status_signature != signature[0]
            or snapshot.ledger_signature != signature[1]
            or snapshot.manual_signature != signature[2]
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

            for source_key in ("mode", "signal", "action", "bias", "side"):
                hint = self._normalise_mode_hint(status.get(source_key))
                if hint:
                    summary["mode_hint"] = hint
                    summary["mode_hint_source"] = source_key
                    break

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
        raw_symbol = status.get("symbol") or status.get("ticker") or status.get("pair")
        symbol: Optional[str] = None

        if isinstance(raw_symbol, str) and raw_symbol.strip():
            symbol = raw_symbol.strip().upper()
        else:
            for candidate in (settings.ai_symbols or "").split(","):
                candidate = candidate.strip()
                if candidate:
                    symbol = candidate.upper()
                    break

        if not symbol:
            symbol = "BTCUSDT"

        probability = self._clamp_probability(float(status.get("probability") or 0.0))
        ev_bps = float(status.get("ev_bps") or 0.0)

        mode_hint: Optional[str] = None
        for key in ("mode", "signal", "action", "bias"):
            mode_hint = self._normalise_mode_hint(status.get(key))
            if mode_hint:
                break
        side_hint = self._normalise_mode_hint(status.get("side"))

        buy_threshold = settings.ai_buy_threshold or 0.55
        sell_threshold = settings.ai_sell_threshold or 0.45
        min_ev = max(float(settings.ai_min_ev_bps or 0.0), 0.0)

        if mode_hint:
            mode = mode_hint
        elif side_hint == "wait":
            mode = "wait"
        elif probability >= max(buy_threshold, 0.55) and ev_bps >= min_ev:
            mode = "buy"
        elif (
            side_hint == "sell"
            and probability <= min(sell_threshold, buy_threshold)
            and ev_bps >= min_ev
        ):
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

    def manual_trade_state(self) -> trade_control.TradeControlState:
        """Return the manual trade control state for the current data directory."""

        snapshot = self._get_snapshot()
        return self._clone_trade_state(snapshot.manual_state)

    def manual_trade_history(self, limit: int = 50) -> Tuple[Dict[str, Any], ...]:
        """Return recent manual trade commands for the bot's data directory."""

        snapshot = self._get_snapshot()
        commands: List[Dict[str, Any]] = list(snapshot.manual_state.commands)
        if limit > 0:
            commands = commands[-limit:]
        return tuple(dict(command) for command in commands)

    def manual_trade_summary(self) -> Dict[str, object]:
        """Return cached metadata about manual control commands."""

        snapshot = self._get_snapshot()
        return self._copy_dict(snapshot.manual_summary)

    def manual_trade_start(
        self,
        *,
        symbol: Optional[str],
        mode: Optional[str],
        probability_pct: Optional[float] = None,
        ev_bps: Optional[float] = None,
        source: str = "manual",
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a manual request to start trading in the bot's workspace."""

        record = trade_control.request_trade_start(
            symbol=symbol,
            mode=mode,
            probability_pct=probability_pct,
            ev_bps=ev_bps,
            source=source,
            note=note,
            extra=extra,
            data_dir=self.data_dir,
        )
        self._snapshot = None
        return record

    def manual_trade_cancel(
        self,
        *,
        symbol: Optional[str],
        reason: Optional[str] = None,
        source: str = "manual",
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a manual request to cancel trading in the bot's workspace."""

        record = trade_control.request_trade_cancel(
            symbol=symbol,
            reason=reason,
            source=source,
            note=note,
            extra=extra,
            data_dir=self.data_dir,
        )
        self._snapshot = None
        return record

    def manual_trade_clear(self) -> None:
        """Remove manual trade commands stored for this bot's workspace."""

        trade_control.clear_trade_commands(data_dir=self.data_dir)
        self._snapshot = None

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

        realtime = bybit_realtime_status(settings)

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
            "realtime_trading": realtime,
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

    def _format_signal_quality_answer(self, summary: Dict[str, object]) -> str:
        if not summary.get("has_status"):
            return (
                "Живой сигнал пока не загружен — обновите ai/status.json, "
                "чтобы бот смог рассказать про вероятность и выгоду."
            )

        mode = str(summary.get("mode") or "wait")
        probability_pct = float(summary.get("probability_pct") or 0.0)
        ev_bps = float(summary.get("ev_bps") or 0.0)
        thresholds = summary.get("thresholds") or {}
        buy_threshold = float(thresholds.get("buy_probability_pct") or 0.0)
        sell_threshold = float(thresholds.get("sell_probability_pct") or 0.0)
        min_ev = float(thresholds.get("min_ev_bps") or 0.0)
        status_source = str(summary.get("status_source") or "live")
        actionable = bool(summary.get("actionable"))
        reasons = summary.get("actionable_reasons") or []

        lines = [
            (
                "Сигнал {mode} с уверенностью {prob:.2f}% и ожидаемой выгодой {ev:.2f} б.п.".format(
                    mode="на покупку" if mode == "buy" else "на продажу" if mode == "sell" else "в режиме ожидания",
                    prob=probability_pct,
                    ev=ev_bps,
                )
            )
        ]

        threshold_line = (
            "Пороги стратегии: покупка от {buy:.2f}%, продажа от {sell:.2f}%, минимальная выгода {ev:.2f} б.п.".format(
                buy=buy_threshold or 0.0,
                sell=sell_threshold or 0.0,
                ev=min_ev,
            )
        )
        lines.append(threshold_line)

        if actionable:
            lines.append("Показатели проходят контроль, сигнал можно исполнять при соблюдении риск-плана.")
        else:
            if reasons:
                for reason in reasons:
                    lines.append(f"⚠️ {reason}")
            else:
                lines.append("Сигнал пока наблюдаем — ждём совпадения с порогами стратегии.")

        staleness = summary.get("staleness") or {}
        staleness_message = staleness.get("message")
        if isinstance(staleness_message, str) and staleness_message.strip():
            lines.append(staleness_message.strip())

        if status_source == "cached" or summary.get("fallback_used"):
            lines.append(
                "Данные получены из последнего сохранённого файла — проверьте, что бот обновляет статус в реальном времени."
            )

        last_update = summary.get("last_update")
        if isinstance(last_update, str) and last_update.strip():
            lines.append(f"Последнее обновление: {last_update.strip()}.")

        return "\n".join(lines)

    def _format_update_answer(self, summary: Dict[str, object]) -> str:
        if not summary.get("has_status"):
            return (
                "Свежие данные ещё не поступали — файл ai/status.json отсутствует или пуст. "
                "Запустите сервис генерации сигнала или загрузите демо-статус, чтобы бот видел обновления."
            )

        lines: List[str] = []

        staleness = summary.get("staleness") or {}
        staleness_message = staleness.get("message")
        if isinstance(staleness_message, str) and staleness_message.strip():
            lines.append(staleness_message.strip())

        updated_text = summary.get("updated_text")
        if isinstance(updated_text, str) and updated_text.strip():
            lines.append(updated_text.strip())

        last_update = summary.get("last_update")
        if isinstance(last_update, str) and last_update.strip() and last_update.strip() != "—":
            lines.append(f"Последняя отметка обновления: {last_update.strip()} по UTC.")

        age_seconds = summary.get("age_seconds")
        if isinstance(age_seconds, (int, float)) and age_seconds is not None and age_seconds >= 0:
            lines.append(
                f"Текущий возраст сигнала: {self._format_duration(float(age_seconds))}."
            )

        if summary.get("fallback_used"):
            lines.append(
                "Ответ собран из кэшированной копии — убедитесь, что статус обновляется автоматически."
            )

        retrain_minutes = int(getattr(self.settings, "ai_retrain_minutes", 0) or 0)
        if retrain_minutes > 0:
            lines.append(
                f"Модель пересматривает веса примерно каждые {retrain_minutes} мин — держите статус свежим."
            )

        health = self.data_health()
        ai_health = health.get("ai_signal") if isinstance(health, dict) else None
        ai_details = ai_health.get("details") if isinstance(ai_health, dict) else None
        if isinstance(ai_details, str) and ai_details.strip():
            lines.append(f"Диагностика: {ai_details.strip()}")

        if not lines:
            lines.append(
                "Статус обновляется без предупреждений — можно продолжать следить за сигналом."
            )

        return "\n".join(lines)

    def _format_exposure_answer(self, portfolio: Dict[str, object]) -> str:
        settings = self.settings
        totals = portfolio.get("totals", {})
        positions = portfolio.get("positions", [])

        open_notional = float(totals.get("open_notional", 0.0))
        realized = float(totals.get("realized_pnl", 0.0))
        open_positions = int(totals.get("open_positions", 0))
        reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0))
        risk_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0))
        cash_only = bool(getattr(settings, "spot_cash_only", True))

        if open_notional <= 0:
            reserve_line = (
                "Капитал свободен — сделки не открыты. "
                f"По плану держим резерв не менее {reserve_pct:.1f}% в кэше"
            )
            if cash_only:
                reserve_line += ", работаем без плеча."
            else:
                reserve_line += ", при необходимости оператор может добавить плечо."
            return (
                reserve_line
                + " На новую идею откладываем не более "
                + f"{risk_pct:.2f}% капитала. Зафиксированная прибыль с начала сессии: "
                + f"{realized:.2f} USDT."
            )

        engaged_line = (
            f"В работе {open_notional:.2f} USDT через {open_positions} активных позиций. "
            f"Резерв безопасности по настройкам — {reserve_pct:.1f}% капитала."
        )

        leaders: List[str] = []
        for record in sorted(positions, key=lambda item: float(item.get("notional") or 0.0), reverse=True):
            notional = float(record.get("notional") or 0.0)
            if notional <= 0:
                continue
            symbol = str(record.get("symbol") or "?")
            qty = float(record.get("qty") or 0.0)
            share = (notional / open_notional * 100.0) if open_notional else 0.0
            leaders.append(
                f"{symbol}: {qty:.6g} шт ≈ {notional:.2f} USDT ({share:.1f}%)"
            )
            if len(leaders) == 3:
                break

        if leaders:
            leaders_line = "Крупнейшие позиции: " + "; ".join(leaders) + "."
        else:
            leaders_line = "Позиции с ненулевой стоимостью не найдены — проверьте журнал сделок."

        sizing_line = (
            "Новые сделки открываем небольшими частями:"
            f" до {risk_pct:.2f}% капитала на одну идею,"
        )
        if cash_only:
            sizing_line += " работаем только на собственные средства."
        else:
            sizing_line += " допускается аккуратное использование плеча оператором."

        return " ".join([engaged_line, leaders_line, sizing_line, f"Фиксированный результат: {realized:.2f} USDT."])

    def _format_plan_answer(self, brief: GuardianBrief) -> str:
        steps = self.plan_steps(brief)
        formatted = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
        return f"План действий:\n{formatted}"

    def _format_settings_answer(self) -> str:
        settings = self.settings

        lines: List[str] = []

        if settings.testnet:
            lines.append(
                "Работаем на тестнете Bybit — все сделки учебные, можно проверять логику без риска."
            )
        else:
            env_line = "Бот подключён к боевому счёту."
            if settings.dry_run:
                env_line += " Сделки подтверждаются в dry-run, исполнение нужно запускать вручную."
            else:
                env_line += " Реальные сделки включены — контролируйте лимиты тщательно."
            lines.append(env_line)

        if settings.testnet and not settings.dry_run:
            lines.append(
                "Несмотря на тестнет, dry-run выключен — ордера будут отправляться в симулятор биржи."
            )
        elif settings.dry_run and not settings.testnet:
            lines.append(
                "Dry-run включён: заявки не отправляются на биржу, но журнал и сигналы записываются."
            )

        ai_enabled = getattr(settings, "ai_enabled", False)
        ai_symbols = str(getattr(settings, "ai_symbols", "") or "").strip()
        if ai_enabled:
            if ai_symbols:
                raw_symbols = [symbol.strip().upper() for symbol in ai_symbols.split(",") if symbol.strip()]
                if raw_symbols:
                    symbol_text = ", ".join(sorted(raw_symbols))
                else:
                    symbol_text = "все доступные символы пресета"
            else:
                symbol_text = "все доступные символы пресета"
            lines.append(
                f"AI сигналы активны в категории {getattr(settings, 'ai_category', 'spot')} для: {symbol_text}."
            )
        else:
            lines.append("AI сигналы выключены — бот может действовать только по ручным командам.")

        buy_threshold = float(getattr(settings, "ai_buy_threshold", 0.0) or 0.0) * 100.0
        sell_threshold = float(getattr(settings, "ai_sell_threshold", 0.0) or 0.0) * 100.0
        min_ev = float(getattr(settings, "ai_min_ev_bps", 0.0) or 0.0)
        lines.append(
            "Порог входа: покупка от {buy:.2f}%, продажа от {sell:.2f}%, ожидаемая выгода не ниже {ev:.2f} б.п.".format(
                buy=buy_threshold,
                sell=sell_threshold,
                ev=min_ev,
            )
        )

        risk_per_trade = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0) or 0.0)
        daily_loss_limit = float(getattr(settings, "ai_daily_loss_limit_pct", 0.0) or 0.0)
        cash_only = bool(getattr(settings, "spot_cash_only", True))
        risk_line_parts = [f"Риск на сделку ограничен {risk_per_trade:.2f}% капитала"]
        risk_line_parts.append(f"резервируем в кэше не менее {reserve_pct:.1f}%")
        if daily_loss_limit > 0:
            risk_line_parts.append(f"дневной стоп по убытку {daily_loss_limit:.2f}%")
        if cash_only:
            risk_line_parts.append("работаем без заимствований")
        else:
            risk_line_parts.append("допускается использование плеча по усмотрению оператора")
        lines.append(
            ", ".join(risk_line_parts) + "."
        )

        max_concurrent = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        if max_concurrent > 0:
            lines.append(
                f"Одновременно AI ведёт до {max_concurrent} активных идей, чтобы не распылять капитал."
            )

        retrain_minutes = int(getattr(settings, "ai_retrain_minutes", 0) or 0)
        if retrain_minutes > 0:
            lines.append(
                f"Модель обновляет весы примерно каждые {retrain_minutes} минут для актуальности статистики."
            )

        watchdog_enabled = bool(getattr(settings, "ws_watchdog_enabled", False))
        execution_guard = int(getattr(settings, "execution_watchdog_max_age_sec", 0) or 0)
        if watchdog_enabled or execution_guard:
            guard_parts: List[str] = []
            if watchdog_enabled:
                guard_parts.append("веб-сокет сторож активен")
            if execution_guard:
                guard_parts.append(
                    f"за обновлением сделок следим, предел задержки {execution_guard} с"
                )
            lines.append("Сторожа соединений: " + ", ".join(guard_parts) + ".")

        return "\n".join(lines)

    def _format_positions_answer(self, portfolio: Dict[str, object]) -> str:
        positions = portfolio.get("positions") or []
        totals = portfolio.get("totals") or {}

        open_notional = float(totals.get("open_notional", 0.0))
        realized = float(totals.get("realized_pnl", 0.0))
        open_positions = int(totals.get("open_positions", 0))

        meaningful_positions: List[Dict[str, object]] = []
        for entry in positions:
            try:
                qty = float(entry.get("qty", 0.0))
                notional = float(entry.get("notional", 0.0))
            except (TypeError, ValueError):
                continue
            if qty <= 0 and notional <= 0:
                continue
            meaningful_positions.append(entry)

        if not meaningful_positions:
            return (
                "Открытых позиций нет — капитал ждёт нового сигнала. "
                f"Зафиксированная прибыль сейчас {realized:.2f} USDT."
            )

        sorted_positions = sorted(
            meaningful_positions,
            key=lambda item: float(item.get("notional") or 0.0),
            reverse=True,
        )

        lines: List[str] = []
        for entry in sorted_positions[:3]:
            symbol = str(entry.get("symbol") or "?").upper()
            qty = float(entry.get("qty") or 0.0)
            avg_cost = float(entry.get("avg_cost") or 0.0)
            notional = float(entry.get("notional") or 0.0)
            lines.append(
                (
                    f"{symbol}: {qty:.4f} шт по {avg_cost:.2f} USDT "
                    f"→ ~{notional:.2f} USDT"
                )
            )

        if len(sorted_positions) > 3:
            lines.append(
                f"Ещё {len(sorted_positions) - 3} позиция(и) с меньшим весом остаются под контролем."
            )

        header = (
            f"В портфеле {open_positions} актив(а) на сумму около {open_notional:.2f} USDT."
        )
        footer = (
            "Дисциплина важна: каждая позиция укладывается в риск-профиль,"
            " а свободные USDT страхуют волатильность."
        )

        return "\n".join([header, *lines, footer, f"Зафиксировано прибыли: {realized:.2f} USDT."])

    def _format_watchlist_answer(self) -> str:
        watchlist = self.market_watchlist()
        if not watchlist:
            return "Список наблюдения пуст — бот ждёт свежих сигналов, чтобы предложить идеи."

        lines: List[str] = []
        for entry in watchlist[:3]:
            symbol = str(entry.get("symbol") or "?").upper()
            bits: List[str] = []
            score = entry.get("score")
            trend = entry.get("trend")
            note = entry.get("note")
            if isinstance(trend, str) and trend:
                bits.append(trend.lower())
            if isinstance(score, (int, float)):
                bits.append(f"оценка {float(score):.2f}")
            if isinstance(note, str) and note:
                bits.append(note)
            detail = ", ".join(bits) if bits else "наблюдаем динамику"
            lines.append(f"{symbol}: {detail}")

        if len(watchlist) > 3:
            lines.append(
                f"В очереди наблюдения ещё {len(watchlist) - 3} инструмент(а) — бот отсортировал их по силе сигнала."
            )

        return "Список наблюдения:\n" + "\n".join(lines)

    def _format_health_answer(self) -> str:
        health = self.data_health()
        order = ("ai_signal", "executions", "api_keys", "realtime_trading")

        blocks: List[str] = []
        for key in order:
            block = health.get(key)
            if not isinstance(block, dict):
                continue

            title = str(block.get("title") or key.replace("_", " ").title())
            ok = bool(block.get("ok"))
            message = str(block.get("message") or "")
            details = block.get("details")
            icon = "✅" if ok else "⚠️"

            section = [f"{icon} {title}: {message}"]
            if isinstance(details, str) and details.strip():
                section.append(f"    {details.strip()}")
            blocks.append("\n".join(section))

        if not blocks:
            return (
                "Диагностика не дала результата — убедитесь, что данные бота обновляются хотя бы демо-файлами."
            )

        return "\n".join(blocks)

    def _format_manual_control_answer(self, summary: Dict[str, object]) -> str:
        if not summary:
            return "Ручных команд нет — бот работает автономно и следует сигналам ИИ."

        lines: List[str] = []

        status_text = str(summary.get("status_text") or "Ручные команды ещё не отправлялись.")
        lines.append(status_text)

        symbol = summary.get("symbol")
        if isinstance(symbol, str) and symbol:
            lines.append(f"Цель оператора: {symbol}.")

        last_action_age = summary.get("last_action_age_seconds")
        if isinstance(last_action_age, (int, float)) and last_action_age > 0:
            lines.append(
                f"Последняя команда была {self._format_duration(float(last_action_age))} назад."
            )

        history = summary.get("history") or []
        if history:
            lines.append("Последние команды оператора:")
            tail = list(history)[-3:]
            for entry in reversed(tail):
                action = str(entry.get("action_label") or entry.get("action") or "").lower()
                if action:
                    action_text = {
                        "start": "запуск торговли",
                        "cancel": "остановка",
                    }.get(action, action)
                else:
                    action_text = "команда"

                symbol_text = str(entry.get("symbol") or symbol or "—")
                probability = entry.get("probability_text")
                ev_text = entry.get("ev_text")
                note = entry.get("note_or_reason")
                when = entry.get("ts_human") or "—"

                extras: List[str] = []
                if isinstance(probability, str):
                    extras.append(f"уверенность {probability}")
                if isinstance(ev_text, str):
                    extras.append(ev_text)
                if isinstance(note, str) and note.strip():
                    extras.append(note.strip())
                details = ", ".join(extras)
                if details:
                    details = f" ({details})"

                lines.append(f"- {when}: {action_text} по {symbol_text}{details}")

            history_count = summary.get("history_count")
            if isinstance(history_count, int) and history_count > 3:
                lines.append(
                    (
                        f"Всего сохранено {history_count} команд — показаны последние три. "
                        "Остальной журнал доступен в приложении."
                    )
                )
        else:
            lines.append("Журнал команд пуст — управлять можно через панель оператора.")

        lines.append(
            "Даже при ручном запуске держите риск в рамках лимитов и проверяйте свежесть сигналов перед действиями."
        )

        return "\n".join(lines)

    def _format_trade_history_answer(
        self,
        trades: List[Dict[str, object]],
        stats: Dict[str, object],
    ) -> str:
        total_trades = int(stats.get("trades", len(trades)) or 0)
        gross_volume = float(stats.get("gross_volume") or 0.0)
        header = (
            f"В журнале {total_trades} сделок на сумму около {gross_volume:.2f} USDT."
        )

        if not trades:
            return (
                header
                + " Пока бот только готовится: новые записи появятся после исполнения первой сделки."
            )

        lines = [header, "Последние операции:"]

        def _format_side(value: object) -> str:
            mapping = {"buy": "покупка", "sell": "продажа"}
            if isinstance(value, str) and value:
                key = value.lower()
                return mapping.get(key, value.lower())
            return "операция"

        preview = trades[:3]
        for entry in preview:
            symbol = str(entry.get("symbol") or "?")
            side = _format_side(entry.get("side"))
            qty = entry.get("qty")
            price = entry.get("price")
            fee = entry.get("fee")
            when = entry.get("when") or "—"

            details: List[str] = []
            if isinstance(qty, (int, float)) and qty:
                details.append(f"{float(qty):.4f} шт")
            if isinstance(price, (int, float)) and price:
                details.append(f"по {float(price):.2f} USDT")
            detail_text = " ".join(details) if details else "без детализации"

            extra_bits: List[str] = []
            if isinstance(fee, (int, float)) and fee:
                extra_bits.append(f"комиссия {float(fee):.4f} USDT")
            if when and when != "—":
                extra_bits.append(f"в {when}")
            extra_text = ", ".join(extra_bits)
            if extra_text:
                extra_text = f" ({extra_text})"

            lines.append(f"- {symbol}: {side} {detail_text}{extra_text}")

        if total_trades > len(preview):
            lines.append(
                f"Показаны последние {len(preview)} записи из {total_trades}. Полный журнал — в разделе исполнений."
            )

        last_trade_at = stats.get("last_trade_at")
        if isinstance(last_trade_at, str) and last_trade_at.strip():
            lines.append(f"Последняя сделка отмечена как {last_trade_at} по UTC.")

        lines.append(
            "Следите, чтобы серия сделок оставалась в рамках дневного лимита потерь и резервов по USDT."
        )

        return "\n".join(lines)

    def _format_fee_activity_answer(self, stats: Dict[str, object]) -> str:
        trades = int(stats.get("trades", 0) or 0)
        if trades <= 0:
            return (
                "Комиссий ещё нет — журнал сделок пуст. "
                "Как только появятся исполнения, покажу расход по комиссиям и активность."
            )

        gross_volume = float(stats.get("gross_volume") or 0.0)
        fees_paid = float(stats.get("fees_paid") or 0.0)
        avg_trade = float(stats.get("avg_trade_value") or 0.0)
        maker_ratio = float(stats.get("maker_ratio") or 0.0) * 100.0
        last_trade_at = str(stats.get("last_trade_at") or "—")

        activity = stats.get("activity") or {}
        recent_15 = int(activity.get("15m") or 0)
        recent_hour = int(activity.get("1h") or 0)
        recent_day = int(activity.get("24h") or 0)

        lines = [
            (
                f"Совокупные комиссии: {fees_paid:.5f} USDT за {trades} сделк(и) "
                f"с объёмом около {gross_volume:.2f} USDT."
            ),
            (
                f"Средний размер сделки: {avg_trade:.2f} USDT. "
                f"Доля maker-исполнений: {maker_ratio:.1f}%."
            ),
            (
                "Активность: за 15 минут {recent_15}, за час {recent_hour}, за сутки {recent_day}. "
                f"Последняя запись: {last_trade_at} по UTC."
            ).format(
                recent_15=recent_15,
                recent_hour=recent_hour,
                recent_day=recent_day,
            ),
        ]

        per_symbol = stats.get("per_symbol") or []
        if per_symbol:
            top = per_symbol[0]
            symbol = str(top.get("symbol") or "—")
            volume = float(top.get("volume") or 0.0)
            buy_share = float(top.get("buy_share") or 0.0) * 100.0
            lines.append(
                (
                    f"Главный символ: {symbol} — {volume:.2f} USDT оборота, "
                    f"покупки занимают {buy_share:.1f}% трафика."
                )
            )

        lines.append(
            "Контролируйте комиссии: чем выше доля maker, тем дешевле обходятся сделки."
        )

        return "\n".join(lines)

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

        if self._contains_any(prompt, ["обнов", "свеж", "давно", "last update", "обновил", "age"]):
            summary = self.status_summary()
            return self._format_update_answer(summary)

        if self._contains_any(prompt, ["сигнал", "вероят", "threshold", "порог", "ev", "бп", "б.п."]):
            summary = self.status_summary()
            return self._format_signal_quality_answer(summary)

        if self._contains_any(prompt, ["план", "что делать", "как начать", "инструкц"]):
            return self._format_plan_answer(brief)

        if self._contains_any(prompt, ["экспоз", "загруж", "капитал", "резерв", "exposure", "занято"]):
            return self._format_exposure_answer(portfolio)

        if self._contains_any(prompt, ["настрой", "config", "параметр", "режим работы", "ограничен", "лимит"]):
            return self._format_settings_answer()

        if self._contains_any(prompt, ["портф", "позици", "баланс", "актив", "hold"]):
            return self._format_positions_answer(portfolio)

        if self._contains_any(prompt, ["воч", "watch", "наблюд", "лист", "монитор"]):
            return self._format_watchlist_answer()

        if self._contains_any(prompt, ["здоров", "health", "жив", "данн", "статус"]):
            return self._format_health_answer()

        if self._contains_any(prompt, ["комисс", "fee", "fees", "maker", "taker", "активн"]):
            stats = self.trade_statistics()
            return self._format_fee_activity_answer(stats)

        if self._contains_any(prompt, ["ручн", "manual", "оператор", "вручн", "start", "stop"]):
            summary = self.manual_trade_summary()
            return self._format_manual_control_answer(summary)

        if self._contains_any(prompt, ["сделк", "trade", "истор", "журнал", "execut"]):
            trades = self.recent_trades(limit=5)
            stats = self.trade_statistics()
            return self._format_trade_history_answer(trades, stats)

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
