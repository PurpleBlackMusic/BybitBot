from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.guardian_bot import GuardianBot
from utils.ui import safe_set_page_config

safe_set_page_config(page_title="DeepSeek сигналы", page_icon="🧠", layout="wide")

st.title("🧠 DeepSeek сигналы и влияние на решения")
st.caption(
    "На этой странице собраны торговые подсказки от DeepSeek, их уверенность и то, как они усиливают или ослабляют решения бота."
)

bot = GuardianBot()
watchlist = bot.market_watchlist()

rows: list[dict[str, object]] = []
for entry in watchlist:
    if not isinstance(entry, dict):
        continue
    deepseek_info = entry.get("deepseek")
    if not isinstance(deepseek_info, dict) or not deepseek_info:
        continue
    guidance = deepseek_info.get("guidance")
    if not isinstance(guidance, dict):
        guidance = {}
    rows.append(
        {
            "symbol": entry.get("symbol"),
            "trend": entry.get("trend"),
            "probability": entry.get("probability"),
            "ev_bps": entry.get("ev_bps"),
            "score": deepseek_info.get("score"),
            "direction": deepseek_info.get("direction"),
            "summary": deepseek_info.get("summary"),
            "stop_loss": deepseek_info.get("stop_loss"),
            "take_profit": deepseek_info.get("take_profit"),
            "guidance": guidance,
        }
    )

if not rows:
    st.info(
        "В актуальном вотчлисте пока нет сигналов с подробной информацией от DeepSeek."
    )
else:
    table_data: list[dict[str, object]] = []
    for row in rows:
        guidance = row.get("guidance") if isinstance(row.get("guidance"), dict) else {}
        influence = guidance.get("influence") if isinstance(guidance, dict) else None
        multiplier = guidance.get("multiplier") if isinstance(guidance, dict) else None
        table_data.append(
            {
                "Символ": row.get("symbol"),
                "Тренд": row.get("trend"),
                "Вероятность": row.get("probability"),
                "EV (б.п.)": row.get("ev_bps"),
                "DeepSeek score": row.get("score"),
                "DeepSeek направление": row.get("direction"),
                "Влияние": influence,
                "Множитель": multiplier,
                "Стоп-лосс": row.get("stop_loss"),
                "Тейк-профит": row.get("take_profit"),
                "Кратко": row.get("summary"),
            }
        )

    df = pd.DataFrame(table_data)
    df["Вероятность"] = pd.to_numeric(df["Вероятность"], errors="coerce")
    df["EV (б.п.)"] = pd.to_numeric(df["EV (б.п.)"], errors="coerce")
    df["DeepSeek score"] = pd.to_numeric(df["DeepSeek score"], errors="coerce")
    df["Множитель"] = pd.to_numeric(df["Множитель"], errors="coerce")
    df["Стоп-лосс"] = pd.to_numeric(df["Стоп-лосс"], errors="coerce")
    df["Тейк-профит"] = pd.to_numeric(df["Тейк-профит"], errors="coerce")

    st.subheader("Фильтр доверия")
    min_score = st.slider(
        "Минимальная уверенность DeepSeek",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )
    filtered = df[df["DeepSeek score"].fillna(0.0) >= min_score].copy()
    if filtered.empty:
        st.warning("По заданному порогу пока нет сигналов.")
    else:
        col1, col2, col3 = st.columns(3)
        avg_score = filtered["DeepSeek score"].mean()
        boost_count = (filtered["Влияние"].str.lower() == "boost").sum()
        reduce_count = (filtered["Влияние"].str.lower() == "reduce").sum()
        col1.metric("Средний score", f"{avg_score:.2f}")
        col2.metric("Усиленных сигналов", int(boost_count))
        col3.metric("Ослабленных сигналов", int(reduce_count))

        display_df = filtered.rename(
            columns={
                "Вероятность": "Вероятность, %",
                "Множитель": "Множитель риска",
            }
        )
        display_df["Вероятность, %"] = display_df["Вероятность, %"].apply(
            lambda v: None if pd.isna(v) else round(float(v) * 100, 2)
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    status = bot.status_summary()
    current_symbol = str(status.get("symbol") or "").upper()
    current_entry = None
    for row in rows:
        symbol_value = str(row.get("symbol") or "").upper()
        if symbol_value == current_symbol:
            current_entry = row
            break

    with st.container(border=True):
        st.subheader("Текущее решение бота")
        if not current_entry:
            st.caption(
                "Бот ещё не получил сигнал DeepSeek по текущему активному инструменту."
            )
        else:
            guidance = (
                current_entry.get("guidance")
                if isinstance(current_entry.get("guidance"), dict)
                else {}
            )
            cols = st.columns(3)
            cols[0].metric("Актив", current_symbol or "—")
            cols[1].metric(
                "Влияние", guidance.get("influence", "—") or "—"
            )
            multiplier = guidance.get("multiplier")
            cols[2].metric(
                "Множитель риска",
                f"{multiplier:.2f}" if isinstance(multiplier, (int, float)) else "—",
            )
            allow_trade = guidance.get("allow")
            if allow_trade is False:
                st.warning(
                    "DeepSeek рекомендует воздержаться от сделки — бот пропустит сигнал до обновления оценки."
                )
            elif guidance.get("influence") == "reduce":
                st.info(
                    "DeepSeek указывает на повышенные риски, объём позиции адаптирован."
                )
            elif guidance.get("influence") == "boost":
                st.success(
                    "Сигнал DeepSeek усиливает уверенность — размер сделки скорректирован вверх в допустимых пределах."
                )
            summary_text = current_entry.get("summary")
            if summary_text:
                st.markdown(f"**Краткое резюме DeepSeek:** {summary_text}")
            stop_loss = current_entry.get("stop_loss")
            take_profit = current_entry.get("take_profit")
            if stop_loss or take_profit:
                st.caption(
                    "Автопилот учитывает предложенные уровни и сопоставляет их со встроенными лимитами риск-менеджмента."
                )
                bullet_lines: list[str] = []
                if isinstance(stop_loss, (int, float)):
                    bullet_lines.append(f"- Стоп-лосс: {float(stop_loss):.4f}")
                else:
                    bullet_lines.append("- Стоп-лосс: нет предложения")
                if isinstance(take_profit, (int, float)):
                    bullet_lines.append(f"- Тейк-профит: {float(take_profit):.4f}")
                else:
                    bullet_lines.append("- Тейк-профит: нет предложения")
                st.markdown("\n".join(bullet_lines))

    with st.expander("Сырые данные DeepSeek", expanded=False):
        st.json(rows)
