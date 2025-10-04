"""User-friendly helper that summarises the spot AI signals for beginners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import json
import time

from .envs import Settings, get_settings
from .paths import DATA_DIR
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
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def reload_settings(self) -> None:
        if not self._custom_settings:
            self._settings = get_settings(force_reload=True)

    def _load_status(self) -> Dict[str, object]:
        path = self._status_path()
        if not path.exists():
            return {}

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        status = dict(raw)
        try:
            ts = float(status.get("last_tick_ts") or 0.0)
        except Exception:
            ts = 0.0
        status["age_seconds"] = time.time() - ts if ts > 0 else None
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

    # ------------------------------------------------------------------
    # public analytics helpers
    def generate_brief(self) -> GuardianBrief:
        status = self._load_status()
        self._last_status = status
        settings = self.settings

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

    def portfolio_overview(self) -> Dict[str, object]:
        inventory = spot_inventory_and_pnl(ledger_path=self._ledger_path())
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
        self.reload_settings()
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
        brief = brief or self.generate_brief()
        status = self._last_status if isinstance(self._last_status, dict) else {}
        if not status:
            status = {}
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
