
from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.dataframe import arrow_safe
from utils.envs import get_api_client, get_settings

st.title("📈 Скринер — объём и волатильность (spot)")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Укажите API-ключи на странице «Подключение и состояние».")
    st.stop()

api = get_api_client()
q = st.text_input("Фильтр по подстроке тикера", value="USDT")
top_n = int(st.slider("Сколько пар показать (по 24h объёму)", 5, 100, 20))
try:
    tk = api.tickers(category="spot")
    rows = (tk.get("result") or {}).get("list") or []
    # простая фильтрация и сортировка по turnover24h
    def as_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    rows = [r for r in rows if q.upper() in str(r.get("symbol","")).upper()]
    rows.sort(key=lambda r: as_float(r.get("turnover24h",0)), reverse=True)
    if rows:
        st.dataframe(
            arrow_safe(pd.DataFrame(rows[:top_n])),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Нет данных под текущий фильтр.")
except Exception as e:
    st.error(f"Ошибка загрузки тикеров: {e}")
