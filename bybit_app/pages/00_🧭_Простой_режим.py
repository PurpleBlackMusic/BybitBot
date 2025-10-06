from __future__ import annotations

from datetime import datetime, timedelta, timezone

import copy

import pandas as pd
import streamlit as st

from utils.guardian_bot import GuardianBot
from utils.ui import rerun


def _format_age(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    delta = timedelta(seconds=int(seconds))
    if delta.days:
        return f"{delta.days} д. {delta.seconds // 3600:02d}:{(delta.seconds % 3600) // 60:02d}"
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    if minutes:
        return f"{minutes} мин {secs:02d} с"
    return f"{secs} с"


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (OSError, OverflowError, ValueError, TypeError):
        return "—"
    return dt.strftime("%d.%m.%Y %H:%M:%S UTC")


def _mode_label(mode: str | None) -> str:
    mapping = {
        "buy": "Покупаем",
        "sell": "Фиксируем",
        "wait": "Ждём",
    }
    return mapping.get((mode or "").lower(), "Ждём")


def _format_threshold_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "б.п.":
        return f"{value:.1f} б.п."
    return f"{value:.2f} {unit}"


def _progress_to_threshold(value: float, target: float) -> tuple[float | None, float]:
    if target <= 0:
        return None, target - value
    ratio = 0.0 if value <= 0 else value / target
    return ratio, target - value


def _recommendation(reason: str) -> str:
    text = reason.lower()
    if "устар" in text:
        return "Обновите status.json или дождитесь нового статуса от GuardianBot."
    if "уверенность" in text:
        return "Дождитесь, пока вероятность превысит заданный порог или скорректируйте настройки."
    if "выгода" in text:
        return "Пересмотрите параметры риска или ждите более выгодного соотношения риск/прибыль."
    if "нет активного сигнала" in text:
        return "Следите за обновлениями — AI пока не даёт явного направления."
    if "кэш" in text or "данн" in text:
        return "Проверьте соединение и убедитесь, что обновление прошло успешно."
    return "Уточните настройки GuardianBot или обратитесь к полному отчёту."


def _priority_for(reason: str) -> tuple[int, str, str]:
    text = reason.lower()
    if "устар" in text:
        return 0, "Освежить данные", "Перезагрузите status.json или запустите обновление GuardianBot."
    if "уверенность" in text:
        return 1, "Дождаться уверенности", "Проверьте силу сигнала или отрегулируйте пороги уверенности."
    if "выгода" in text:
        return 1, "Проверить выгоду", "Сравните ожидаемую выгоду с минимальным порогом и скорректируйте риск."
    if "нет активного сигнала" in text:
        return 2, "Ждать сигнала", "Следите за обновлениями — торговый сигнал ещё формируется."
    return 3, "Проверить настройки", "Убедитесь, что параметры GuardianBot соответствуют стратегии."


def _priority_label(rank: int) -> str:
    if rank <= 0:
        return "Срочно"
    if rank == 1:
        return "Важно"
    if rank == 2:
        return "Наблюдаем"
    return "Фоново"


def _readiness_guidance(title: str) -> str:
    title = title.lower()
    if "обновление данных" in title:
        return "Нажмите «Обновить данные» или проверьте, что status.json обновился у GuardianBot."
    if "источник данных" in title:
        return "Включите живой источник или дождитесь, пока бот перестанет использовать кэш."
    if "уверенность" in title:
        return "Ждите роста вероятности или скорректируйте порог в настройках GuardianBot."
    if "ожидаемая выгода" in title:
        return "Проверьте параметры риска и дождитесь, пока EV превысит минимум."
    if "активный сигнал" in title:
        return "Дождитесь, пока AI сформирует направление сделки."
    return "Проверьте расширенную диагностику в GuardianBot для уточнения."


def _append_history(
    summary: dict[str, object],
    readiness_score: float,
    passed_checks: int,
    total_checks: int,
) -> list[dict[str, object]]:
    history_list = list(st.session_state.get("readiness_history", []))
    next_index = len(history_list) + 1

    raw_label = summary.get("last_update") or summary.get("updated_text") or ""
    if isinstance(raw_label, str):
        raw_label = raw_label.strip()
    label = raw_label or f"Обновление #{next_index}"

    existing_labels = {entry.get("label") for entry in history_list}
    if label in existing_labels:
        label = f"{label} · {next_index}"

    probability_pct = float(summary.get("probability_pct") or 0.0)
    ev_bps = float(summary.get("ev_bps") or 0.0)
    age_seconds = float(summary.get("age_seconds") or 0.0)

    entry = {
        "label": label,
        "mode": _mode_label(summary.get("mode")),
        "probability_pct": round(probability_pct, 2),
        "ev_bps": round(ev_bps, 2),
        "readiness_score": round(readiness_score, 2),
        "checks": f"{passed_checks}/{total_checks}" if total_checks else "0/0",
        "actionable": "Да" if summary.get("actionable") else "Нет",
        "age": _format_age(age_seconds),
    }

    if history_list:
        last = history_list[-1]
        if (
            last.get("label") == entry["label"]
            and last.get("probability_pct") == entry["probability_pct"]
            and last.get("ev_bps") == entry["ev_bps"]
            and last.get("readiness_score") == entry["readiness_score"]
            and last.get("checks") == entry["checks"]
            and last.get("actionable") == entry["actionable"]
        ):
            return history_list

    history_list.append(entry)
    trimmed_history = history_list[-20:]
    st.session_state["readiness_history"] = trimmed_history
    return trimmed_history


def _recovery_steps(
    reasons: list[str],
    readiness_checks: list[tuple[str, bool, str]],
    summary: dict[str, object],
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for reason in reasons:
        rank, title, guidance = _priority_for(reason)
        if title in seen_titles:
            continue
        steps.append(
            {
                "Приоритет": _priority_label(rank),
                "Шаг": title,
                "Действие": guidance,
            }
        )
        seen_titles.add(title)

    if summary.get("fallback_used") and "Включить живые данные" not in seen_titles:
        steps.append(
            {
                "Приоритет": _priority_label(0),
                "Шаг": "Включить живые данные",
                "Действие": "Перезапустите бота или дождитесь живого статуса, чтобы уйти от кэша.",
            }
        )
        seen_titles.add("Включить живые данные")

    for title, is_ok, detail in readiness_checks:
        if is_ok or title in seen_titles:
            continue
        steps.append(
            {
                "Приоритет": _priority_label(1),
                "Шаг": title,
                "Действие": detail,
            }
        )
        seen_titles.add(title)

    return steps


st.set_page_config(page_title="Простой режим", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_guardian() -> GuardianBot:
    """Reuse a single GuardianBot instance across reruns."""

    return GuardianBot()


bot = _get_guardian()

st.title("🧭 Простой режим")
st.caption(
    "Минимальный интерфейс умного спотового бота: краткая сводка, пошаговый план и чат для вопросов."
)

refresh = st.button("🔄 Обновить данные", use_container_width=True)
if refresh:
    bot.refresh()
    rerun()

summary = bot.status_summary()
brief = bot.generate_brief()
plan_steps = bot.plan_steps(brief)
risk_text = bot.risk_summary()
portfolio = bot.portfolio_overview()
watchlist = bot.market_watchlist()
recent_trades = bot.recent_trades()
trade_stats = bot.trade_statistics()
health = bot.data_health()
automation_health = health.get("automation") or {}

previous_summary = st.session_state.get("previous_summary")
previous_readiness = st.session_state.get("previous_readiness")

thresholds = summary.get("thresholds") or {}
mode = summary.get("mode", "wait")

status_cols = st.columns(3)
mode_delta = None
if previous_summary:
    previous_mode = _mode_label(previous_summary.get("mode"))
    if previous_mode != _mode_label(mode):
        mode_delta = f"Было: {previous_mode}"
status_cols[0].metric("Режим", _mode_label(mode), mode_delta)
if summary.get("headline"):
    status_cols[0].caption(summary.get("headline"))

probability_pct = float(summary.get("probability_pct") or 0.0)
probability_delta = (
    probability_pct - float(previous_summary.get("probability_pct") or 0.0)
    if previous_summary
    else None
)
probability_delta_text = (
    f"{probability_delta:+.1f} п.п."
    if probability_delta is not None
    else None
)
status_cols[1].metric("Вероятность", f"{probability_pct:.1f}%", probability_delta_text)

if mode == "buy":
    prob_threshold = float(thresholds.get("buy_probability_pct") or 0.0)
elif mode == "sell":
    prob_threshold = float(thresholds.get("sell_probability_pct") or 0.0)
else:
    prob_threshold = 0.0
if prob_threshold:
    status_cols[1].caption(f"Порог: {prob_threshold:.0f}%")

ev_bps = float(summary.get("ev_bps") or 0.0)
ev_delta = (
    ev_bps - float(previous_summary.get("ev_bps") or 0.0)
    if previous_summary
    else None
)
ev_delta_text = f"{ev_delta:+.0f} б.п." if ev_delta is not None else None
status_cols[2].metric("Потенциал", f"{ev_bps:.0f} б.п.", ev_delta_text)
min_ev = float(thresholds.get("min_ev_bps") or 0.0)
if min_ev:
    status_cols[2].caption(f"Мин.: {min_ev:.0f} б.п.")

if summary.get("caution"):
    st.warning(summary["caution"])

st.write(summary.get("analysis", ""))
st.info(summary.get("action_text", ""))
st.caption(summary.get("updated_text", ""))
if summary.get("confidence_text"):
    st.caption(summary["confidence_text"])
if summary.get("ev_text"):
    st.caption(summary["ev_text"])

age_seconds = float(summary.get("age_seconds") or 0.0)
detail_cols = st.columns(3)
detail_cols[0].metric("Последнее обновление", summary.get("last_update", "—"))
detail_cols[1].metric("Возраст статуса", _format_age(age_seconds))
source_label = str(summary.get("status_source") or "").lower()
if source_label == "live":
    source_text = "Живой статус"
elif source_label == "file":
    source_text = "Файл status.json"
elif source_label == "cached":
    source_text = "Кэш"
else:
    source_text = "Нет данных"
detail_cols[2].metric("Источник данных", source_text)

reasons = summary.get("actionable_reasons") or []
staleness = summary.get("staleness") or {}

if summary.get("actionable"):
    st.success("Сигнал проходит фильтры риска — бот готов к действию.")
    if reasons:
        st.caption("Напоминание об ограничениях:")
        st.markdown("\n".join(f"• {reason}" for reason in reasons))
else:
    st.caption("Сигнал пока наблюдательный: бот ждёт лучших данных.")
    if reasons:
        st.warning(
            "\n".join(
                [
                    "Причины паузы:",
                    *(f"• {reason}" for reason in reasons),
                ]
            )
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Причина": reason,
                        "Что делать": _recommendation(reason),
                    }
                    for reason in reasons
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("#### Автоматизация")
with st.container(border=True):
    automation_ok = bool(automation_health.get("ok"))
    automation_message = (
        str(automation_health.get("message") or "AI готов к автоматическим сделкам.").strip()
    )
    automation_details = automation_health.get("details")
    reasons = summary.get("actionable_reasons") or []

    if automation_ok:
        st.success(automation_message)
    else:
        st.warning(automation_message)

    if isinstance(automation_details, str) and automation_details.strip():
        st.caption(automation_details.strip())

    if not automation_ok and reasons:
        st.caption("Почему бот ждёт:")
        st.markdown("\n".join(f"• {reason}" for reason in reasons))
    elif not reasons and not automation_ok:
        st.caption("AI ожидает подходящий сигнал для безопасного входа.")

readiness_checks = []
staleness_state = (staleness.get("state") or "").lower()
staleness_message = staleness.get("message") or ""
if staleness_state == "stale":
    readiness_checks.append(
        (
            "Обновление данных",
            False,
            staleness_message or "Сигнал устарел — обновите status.json перед сделкой.",
        )
    )
else:
    readiness_checks.append(
        (
            "Обновление данных",
            True,
            staleness_message or "Статус свежий — данные подходят для принятия решения.",
        )
    )

if summary.get("fallback_used"):
    readiness_checks.append(
        (
            "Источник данных",
            False,
            "Показаны кэшированные данные — убедитесь, что GuardianBot обновился недавно.",
        )
    )
else:
    readiness_checks.append(
        (
            "Источник данных",
            True,
            "Используются живые данные с последнего обновления.",
        )
    )

mode = summary.get("mode", "wait")
if mode == "buy":
    buy_threshold = float(thresholds.get("buy_probability_pct") or 0.0)
    has_threshold = buy_threshold > 0
    readiness_checks.append(
        (
            "Уверенность в покупке",
            probability_pct >= buy_threshold if has_threshold else True,
            (
                f"{probability_pct:.1f}% против порога {buy_threshold:.1f}%."
                if has_threshold
                else "Порог не задан — используется наблюдательный режим уверенности."
            ),
        )
    )
    min_ev = float(thresholds.get("min_ev_bps") or 0.0)
    readiness_checks.append(
        (
            "Ожидаемая выгода",
            ev_bps >= min_ev,
            f"{ev_bps:.0f} б.п. против минимума {min_ev:.0f} б.п.",
        )
    )
elif mode == "sell":
    sell_threshold = float(thresholds.get("sell_probability_pct") or 0.0)
    has_threshold = sell_threshold > 0
    readiness_checks.append(
        (
            "Уверенность в продаже",
            probability_pct >= sell_threshold if has_threshold else True,
            (
                f"{probability_pct:.1f}% против порога {sell_threshold:.1f}%."
                if has_threshold
                else "Порог не задан — используется наблюдательный режим уверенности."
            ),
        )
    )
    min_ev = float(thresholds.get("min_ev_bps") or 0.0)
    readiness_checks.append(
        (
            "Ожидаемая выгода",
            ev_bps >= min_ev,
            f"{ev_bps:.0f} б.п. против минимума {min_ev:.0f} б.п.",
        )
    )
else:
    readiness_checks.append(
        (
            "Активный сигнал",
            False,
            "AI не даёт чёткого указания на сделку — режим ожидания.",
        )
    )

st.markdown("#### Диагностика готовности")
for title, is_ok, detail in readiness_checks:
    icon = "✅" if is_ok else "⚠️"
    st.markdown(f"{icon} **{title}** — {detail}")

total_checks = len(readiness_checks)
passed_checks = sum(1 for _, is_ok, _ in readiness_checks if is_ok)
status_cols = st.columns(3)
status_cols[0].metric(
    "Готовность бота",
    f"{passed_checks}/{total_checks}",
    "Все фильтры пройдены" if passed_checks == total_checks else "Требуются действия",
)

readiness_score = (passed_checks / total_checks) * 100.0 if total_checks else 0.0
readiness_delta = (
    readiness_score - previous_readiness
    if previous_readiness is not None
    else None
)
readiness_delta_text = (
    f"{readiness_delta:+.0f} п.п."
    if readiness_delta is not None
    else None
)
status_cols[1].metric("Индекс готовности", f"{readiness_score:.0f}%", readiness_delta_text)

blocker = None
if total_checks:
    blocker = next(((title, detail) for title, ok, detail in readiness_checks if not ok), None)
status_cols[2].metric(
    "Главный блокер",
    blocker[0] if blocker else "—",
    blocker[1] if blocker else "Нет препятствий",
)

readiness_rows = []
for title, is_ok, detail in readiness_checks:
    readiness_rows.append(
        {
            "Проверка": title,
            "Статус": "✅ Пройдена" if is_ok else "⚠️ Требует внимания",
            "Комментарий": detail,
            "Рекомендация": "—" if is_ok else _readiness_guidance(title),
        }
    )

if readiness_rows:
    st.dataframe(
        pd.DataFrame(readiness_rows),
        use_container_width=True,
        hide_index=True,
    )

progress_items: list[tuple[str, float, float, str]] = []
buy_threshold_pct = float(thresholds.get("buy_probability_pct") or 0.0)
sell_threshold_pct = float(thresholds.get("sell_probability_pct") or 0.0)
min_ev_bps = float(thresholds.get("min_ev_bps") or 0.0)

if mode == "buy" and buy_threshold_pct:
    progress_items.append(("Уверенность к покупке", probability_pct, buy_threshold_pct, "%"))
elif mode == "sell" and sell_threshold_pct:
    progress_items.append(("Уверенность к продаже", probability_pct, sell_threshold_pct, "%"))

if min_ev_bps > 0:
    progress_items.append(("Ожидаемая выгода", ev_bps, min_ev_bps, "б.п."))

if progress_items:
    st.markdown("#### Прогресс к безопасной сделке")
    st.caption(
        "Показывает, насколько текущие показатели приблизились к порогам, которые снимают защитные фильтры."
    )
    cols = st.columns(len(progress_items))
    for col, (title, value, target, unit) in zip(cols, progress_items):
        col.markdown(f"**{title}**")
        ratio, gap = _progress_to_threshold(value, target)
        if ratio is None:
            col.info(
                f"Порог не задан — ориентируемся на текущие данные ({_format_threshold_value(value, unit)})."
            )
            continue
        progress = max(0.0, min(ratio, 1.0))
        col.progress(progress)
        formatted_value = _format_threshold_value(value, unit)
        formatted_target = _format_threshold_value(target, unit)
        formatted_gap = _format_threshold_value(abs(gap), unit)
        percent = progress * 100.0
        if gap > 0:
            col.caption(
                f"{formatted_value} из {formatted_target} ({percent:.0f}% от цели). Не хватает ещё {formatted_gap}."
            )
        else:
            col.caption(
                f"{formatted_value} из {formatted_target} (цель достигнута на {percent:.0f}%). Запас {formatted_gap}."
            )

recovery_plan = _recovery_steps(reasons, readiness_checks, summary)
if recovery_plan:
    st.markdown("#### Как разблокировать бота")
    st.dataframe(
        pd.DataFrame(recovery_plan),
        use_container_width=True,
        hide_index=True,
    )

history = _append_history(summary, readiness_score, passed_checks, total_checks)
if history:
    st.markdown("#### Хронология сигнала")
    st.caption(
        "Фиксирует обновления статуса, чтобы отслеживать, как меняются вероятность, EV и готовность.",
    )
    history_df = pd.DataFrame(history)
    chart_df = history_df.set_index("label")[
        ["probability_pct", "ev_bps", "readiness_score"]
    ]
    chart_df = chart_df.rename(
        columns={
            "probability_pct": "Вероятность, %",
            "ev_bps": "EV, б.п.",
            "readiness_score": "Готовность, %",
        }
    )
    st.line_chart(chart_df, height=240)

    display_df = history_df.copy()
    display_df["probability_pct"] = display_df["probability_pct"].map(lambda v: f"{v:.1f}%")
    display_df["ev_bps"] = display_df["ev_bps"].map(lambda v: f"{v:.0f} б.п.")
    display_df["readiness_score"] = display_df["readiness_score"].map(lambda v: f"{v:.0f}%")
    display_df = display_df.rename(
        columns={
            "label": "Обновление",
            "mode": "Режим",
            "checks": "Чек-лист",
            "actionable": "Готов к сделке",
            "age": "Возраст",
            "probability_pct": "Вероятность",
            "ev_bps": "EV",
            "readiness_score": "Готовность",
        }
    )
    display_df = display_df[
        [
            "Обновление",
            "Режим",
            "Готов к сделке",
            "Чек-лист",
            "Возраст",
            "Вероятность",
            "EV",
            "Готовность",
        ]
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.session_state["previous_summary"] = copy.deepcopy(summary)
st.session_state["previous_readiness"] = readiness_score

with st.expander("Технические детали сигнала", expanded=False):
    mode_hint = summary.get("mode_hint")
    if mode_hint:
        st.markdown(
            f"**Подсказка режима:** {mode_hint} (источник: {summary.get('mode_hint_source', '—')})."
        )
    st.caption(
        "Доступные поля статуса: "
        + ", ".join(summary.get("raw_keys", []))
        if summary.get("raw_keys")
        else "Сырые поля статуса пока не загружены."
    )
    st.caption(
        (
            {
                "live": "Источник статуса: живой",
                "file": "Источник статуса: файл",
                "cached": "Источник статуса: кэш",
            }.get(
                str(summary.get("status_source") or "").lower(),
                "Источник статуса: нет данных",
            )
        )
    )

st.divider()
st.subheader("Пошаговый план")
if plan_steps:
    st.markdown("\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps)))
else:
    st.info("План сформируется после следующего обновления сигнала.")

st.subheader("Риски и защита")
st.markdown(risk_text)

st.divider()
st.subheader("Портфель")
portfolio_totals = portfolio.get("human_totals", {})
cols = st.columns(3)
cols[0].metric("Реализовано", portfolio_totals.get("realized", "0.00 USDT"))
cols[1].metric("В позиции", portfolio_totals.get("open_notional", "0.00 USDT"))
cols[2].metric("Активных сделок", portfolio_totals.get("open_positions", "0"))

positions = portfolio.get("positions", [])
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
else:
    st.caption("Позиции отсутствуют — капитал в резерве.")

st.divider()
st.subheader("Дополнительная аналитика")

health_cards = [
    health.get("ai_signal"),
    health.get("executions"),
    health.get("api_keys"),
    health.get("realtime_trading"),
]
health_cards = [card for card in health_cards if card]
if health_cards:
    cols = st.columns(len(health_cards))
    for col, card in zip(cols, health_cards):
        icon = "✅" if card.get("ok") else "⚠️"
        col.metric(card.get("title", "Статус"), icon, card.get("message", ""))
        details = card.get("details")
        if details:
            col.caption(details)

if watchlist:
    st.markdown("#### Наблюдаемые пары")
    st.dataframe(pd.DataFrame(watchlist), use_container_width=True, hide_index=True)

if recent_trades:
    st.markdown("#### Последние сделки")
    st.dataframe(pd.DataFrame(recent_trades), use_container_width=True, hide_index=True)

if trade_stats.get("trades"):
    st.markdown("#### Статистика исполнения")
    stats_cols = st.columns(3)
    stats_cols[0].metric("Сделок", int(trade_stats.get("trades", 0)))
    stats_cols[1].metric("Оборот", trade_stats.get("gross_volume_human", "0.00 USDT"))
    maker_ratio = float(trade_stats.get("maker_ratio", 0.0) or 0.0) * 100.0
    stats_cols[2].metric("Мейкер", f"{maker_ratio:.0f}%")
    st.caption(
        " · ".join(
            [
                f"{trade_stats.get('activity', {}).get('15m', 0)} за 15 минут",
                f"{trade_stats.get('activity', {}).get('1h', 0)} за час",
                f"{trade_stats.get('activity', {}).get('24h', 0)} за сутки",
                f"последняя: {trade_stats.get('last_trade_at', '—')}",
            ]
        )
    )

st.divider()
st.subheader("Пообщайтесь с ботом")
if "guardian_chat" not in st.session_state:
    st.session_state["guardian_chat"] = [
        {"role": "assistant", "content": bot.initial_message()},
    ]

for message in st.session_state["guardian_chat"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Спросите о рисках, прибыли или плане…"):
    st.session_state["guardian_chat"].append({"role": "user", "content": prompt})
    st.session_state["guardian_chat"].append({"role": "assistant", "content": bot.answer(prompt)})
    rerun()
