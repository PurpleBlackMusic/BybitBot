from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.guardian_bot import GuardianBot
from utils.paths import DATA_DIR

st.set_page_config(page_title="Мониторинг сделок", page_icon="📈", layout="wide")

st.title("📈 Мониторинг сделок")
st.caption("Следим, какие сделки открыл спотовый бот, каков результат и нет ли задержек в данных.")

bot = GuardianBot()
brief = bot.generate_brief()
portfolio = bot.portfolio_overview()
trade_stats = bot.trade_statistics()
health = bot.data_health()

with st.container(border=True):
    st.subheader("Диагностика источников данных")
    for key in ("ai_signal", "executions", "api_keys"):
        info = health.get(key, {})
        if not info:
            continue
        icon = "✅" if info.get("ok") else "⚠️"
        title = info.get("title", key)
        message = info.get("message", "")
        st.markdown(f"{icon} **{title}** — {message}")
        details = info.get("details")
        if details:
            st.caption(details)

st.divider()

with st.container(border=True):
    st.subheader("Итоги по портфелю")
    totals = portfolio.get("human_totals", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Реализованный PnL", totals.get("realized", "0.00 USDT"))
    col2.metric("Объём в работе", totals.get("open_notional", "0.00 USDT"))
    col3.metric("Открытых сделок", totals.get("open_positions", "0"))

    col4, col5, col6 = st.columns(3)
    col4.metric("Совершено сделок", int(trade_stats.get("trades", 0)))
    col5.metric("Оборот", trade_stats.get("gross_volume_human", "0.00 USDT"))
    col6.metric("Комиссии", trade_stats.get("fees_paid_human", "0.0000 USDT"))

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
        st.info("Позиции не обнаружены — бот ждёт подходящего момента.")

    if trade_stats.get("per_symbol"):
        st.markdown("#### Где бот торгует активнее всего")
        df_activity = pd.DataFrame(trade_stats.get("per_symbol", []))
        df_activity = df_activity.rename(
            columns={
                "symbol": "Символ",
                "trades": "Сделок",
                "volume_human": "Оборот",
                "buy_share": "Доля покупок",
            }
        )
        df_activity["Доля покупок"] = (df_activity["Доля покупок"] * 100).round(1).astype(str) + "%"
        st.dataframe(df_activity, use_container_width=True, hide_index=True)
    else:
        st.caption("Как только появятся сделки, здесь появится разбивка по парам.")

st.divider()

st.subheader("Последние исполнения")
ledger_path = Path(DATA_DIR) / "pnl" / "executions.jsonl"
if not ledger_path.exists():
    st.warning("Журнал исполнений пока пуст.")
else:
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines()[-200:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "ts": payload.get("execTime"),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "price": float(payload.get("execPrice") or 0.0),
                "qty": float(payload.get("execQty") or 0.0),
                "fee": float(payload.get("execFee") or 0.0),
                "is_maker": payload.get("isMaker"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("ts", ascending=False)
        df = df.rename(
            columns={
                "ts": "Время",
                "symbol": "Символ",
                "side": "Сторона",
                "price": "Цена",
                "qty": "Количество",
                "fee": "Комиссия",
                "is_maker": "Мейкер?",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Ещё не было сделок, которые бот записал в журнал.")

st.divider()

st.subheader("Наличие данных")
st.caption("Если данные устарели, бот предупредит на странице «Дружелюбный спот-бот». Здесь можно проверить вручную.")
age = brief.status_age
if age is None:
    st.success("Статус обновлён только что.")
elif age < 300:
    st.success(f"Последнее обновление {int(age // 60)} минут назад.")
else:
    st.warning("Данных от бота давно не было — проверьте соединение и журнал логов.")
