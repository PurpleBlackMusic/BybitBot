from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.guardian_bot import GuardianBot


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
    st.experimental_rerun()

summary = bot.status_summary()
brief = bot.generate_brief()
plan_steps = bot.plan_steps(brief)
risk_text = bot.risk_summary()
portfolio = bot.portfolio_overview()
watchlist = bot.market_watchlist()
recent_trades = bot.recent_trades()
trade_stats = bot.trade_statistics()
health = bot.data_health()

status_cols = st.columns(3)
status_cols[0].metric(
    "Режим",
    {
        "buy": "Покупаем",
        "sell": "Фиксируем",
        "wait": "Ждём",
    }.get(summary.get("mode", "wait"), "Ждём"),
    summary.get("headline", "—"),
)
status_cols[1].metric(
    "Вероятность",
    f"{summary.get('probability_pct', 0.0):.1f}%",
    f"Порог {summary.get('thresholds', {}).get('buy_probability_pct', 0.0):.0f}%",
)
status_cols[2].metric(
    "Потенциал",
    f"{summary.get('ev_bps', 0.0):.0f} б.п.",
    f"Мин. {summary.get('thresholds', {}).get('min_ev_bps', 0.0):.0f} б.п.",
)

if summary.get("caution"):
    st.warning(summary["caution"])

st.write(summary.get("analysis", ""))
st.info(summary.get("action_text", ""))
st.caption(summary.get("updated_text", ""))

if summary.get("actionable"):
    st.success("Сигнал проходит фильтры риска — бот готов к действию.")
else:
    st.caption("Сигнал пока наблюдательный: бот ждёт лучших данных.")

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

health_cols = st.columns(3)
health_cols[0].metric(
    health["ai_signal"]["title"],
    "✅" if health["ai_signal"]["ok"] else "⚠️",
    health["ai_signal"]["message"],
)
health_cols[1].metric(
    health["executions"]["title"],
    "✅" if health["executions"]["ok"] else "⚠️",
    health["executions"]["message"],
)
health_cols[2].metric(
    health["api_keys"]["title"],
    "✅" if health["api_keys"]["ok"] else "⚠️",
    health["api_keys"]["message"],
)

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
    st.experimental_rerun()
