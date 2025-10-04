from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.guardian_bot import GuardianBot

st.set_page_config(page_title="Дружелюбный спот-бот", page_icon="🛡", layout="wide")

st.title("🛡 Дружелюбный спот-бот")
st.caption(
    "Интерфейс, где можно увидеть, что делает умный спотовый бот, зачем он это делает и какие риски учтены."
)

bot = GuardianBot()
brief = bot.generate_brief()
portfolio = bot.portfolio_overview()
plan_steps = bot.plan_steps(brief)
risk_text = bot.risk_summary()
safety_notes = bot.safety_notes()
story = bot.market_story(brief)
staleness = bot.staleness_alert(brief)
scorecard = bot.signal_scorecard(brief)
watchlist = bot.market_watchlist()
recent_trades = bot.recent_trades()
trade_stats = bot.trade_statistics()

with st.container(border=True):
    st.subheader("Текущий сигнал")
    st.markdown(f"**{brief.headline}**")
    st.write(brief.action_text)
    st.write(brief.analysis)
    st.write(brief.confidence_text)
    st.write(brief.ev_text)
    st.info(brief.caution)
    st.caption(brief.updated_text)
    if staleness:
        st.warning(staleness)

with st.container(border=True):
    st.subheader("Как бот видит рынок")
    st.write(story)
    st.caption("Короткое объяснение, почему выбрано именно такое действие.")

with st.container(border=True):
    st.subheader("Деньги под контролем")
    totals = portfolio.get("human_totals", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Реализованный результат", totals.get("realized", "0.00 USDT"))
    col2.metric("В работе", totals.get("open_notional", "0.00 USDT"))
    col3.metric("Открытых сделок", totals.get("open_positions", "0"))

    positions = portfolio.get("positions", [])
    if positions:
        df = pd.DataFrame(positions)
        df = df.rename(
            columns={
                "symbol": "Символ",
                "qty": "Количество",
                "avg_cost": "Средняя цена",
                "notional": "Объём в работе",
                "realized_pnl": "Реализованный PnL",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Сделок нет — депозит в безопасности, ждём хороший сигнал.")

with st.container(border=True):
    st.subheader("Пошаговый план")
    st.markdown("\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps)))

with st.container(border=True):
    st.subheader("Контроль риска")
    st.markdown(risk_text)
    st.caption("Если что-то непонятно — лучше задайте вопрос боту ниже.")

with st.container(border=True):
    st.subheader("Безопасность")
    for note in safety_notes:
        st.markdown(f"- {note}")

with st.expander("🫥 Скрытая аналитика для бота"):
    st.caption(
        "Продвинутые сигналы и история сделок доступны здесь, чтобы не перегружать основной экран."
    )

    if scorecard:
        st.markdown("#### Карточка сигнала")
        df_score = pd.DataFrame([scorecard])
        st.dataframe(df_score, use_container_width=True, hide_index=True)

    if watchlist:
        st.markdown("#### Наблюдаемые пары")
        df_watch = pd.DataFrame(watchlist)
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    else:
        st.info("Дополнительных сигналов пока нет — бот сосредоточен на главной паре.")

    if recent_trades:
        st.markdown("#### Последние сделки")
        df_trades = pd.DataFrame(recent_trades)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.caption("Журнал сделок пуст — ждём подтверждённых исполнений от биржи.")

    if trade_stats.get("trades"):
        st.markdown("#### Как бот исполняет сделки")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Совершено сделок", int(trade_stats.get("trades", 0)))
        col_b.metric("Оборот", trade_stats.get("gross_volume_human", "0.00 USDT"))
        maker_ratio = float(trade_stats.get("maker_ratio", 0.0)) * 100.0
        col_c.metric("Доля мейкер", f"{maker_ratio:.0f}%")

        activity = trade_stats.get("activity", {})
        st.caption(
            " · ".join(
                [
                    f"{activity.get('15m', 0)} сделок за 15 минут",
                    f"{activity.get('1h', 0)} за час",
                    f"{activity.get('24h', 0)} за сутки",
                    f"последняя: {trade_stats.get('last_trade_at', '—')}",
                ]
            )
        )
        st.caption(
            "Торгуемые пары: "
            + (", ".join(trade_stats.get("symbols", [])) or "бот ждёт идеальный сигнал")
        )

st.divider()
st.subheader("Пообщайтесь с ботом")
if "guardian_chat" not in st.session_state:
    st.session_state["guardian_chat"] = [
        {"role": "assistant", "content": bot.initial_message()},
    ]

for msg in st.session_state["guardian_chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос простыми словами…"):
    st.session_state["guardian_chat"].append({"role": "user", "content": prompt})
    answer = bot.answer(prompt)
    st.session_state["guardian_chat"].append({"role": "assistant", "content": answer})
    st.experimental_rerun()
