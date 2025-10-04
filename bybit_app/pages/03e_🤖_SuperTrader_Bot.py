from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.guardian_bot import GuardianBot


st.title("🛡 Дружелюбный спот-бот")
st.caption(
    "Простое рабочее место для тех, кто только знакомится с рынком: бот объясняет сигналы, риск и прибыль без сложных терминов."
)

bot = GuardianBot()
brief = bot.generate_brief()
portfolio = bot.portfolio_overview()
plan_steps = bot.plan_steps(brief)
risk_text = bot.risk_summary()
safety_notes = bot.safety_notes()
story = bot.market_story(brief)
staleness = bot.staleness_alert(brief)


with st.container(border=True):
    st.subheader("Что происходит сейчас")
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
    st.caption("Здесь коротко описано, почему модель выбрала текущее действие.")

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
                "notional": "В работе",
                "realized_pnl": "Результат",
            }
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Сделок нет — это нормально. Мы не рискуем депозитом без преимущества.")

with st.container(border=True):
    st.subheader("Пошаговый план")
    st.markdown("\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan_steps)))

with st.container(border=True):
    st.subheader("Контроль риска")
    st.markdown(risk_text)
    st.caption("Если что-то в риск-параметрах непонятно — лучше остановиться и уточнить.")

with st.container(border=True):
    st.subheader("Важно новичку")
    for note in safety_notes:
        st.markdown(f"- {note}")

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
