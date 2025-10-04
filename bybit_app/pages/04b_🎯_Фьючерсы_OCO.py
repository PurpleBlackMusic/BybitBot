
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.oco_futures import place_linear_oco
from utils.quant import clamp_qty, gte_min_notional
from utils.log import log

st.title("🎯 Фьючерсы — OCO (Linear) — BETA")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Сначала укажите API ключи на странице «Подключение и состояние».")
    st.stop()

api = get_api_client()

with st.form("foco"):
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Тикер", value="BTCUSDT").strip().upper()
        side = st.selectbox("Сторона", ["Buy", "Sell"])
    with c2:
        qty = st.text_input("Количество (контракты)", value="0.001")
        entry_price = st.text_input("Цена входа (Limit)", value="")
    with c3:
        tp_price = st.text_input("TP (Trigger)", value="")
        sl_price = st.text_input("SL (Trigger)", value="")
    reduce = st.checkbox("reduceOnly", value=True)
    run = st.form_submit_button("🚀 Отправить Futures OCO")

if run:
    try:
        price = entry_price or "0"
        r = place_linear_oco(api, symbol, side, qty, price, take_profit=tp_price, stop_loss=sl_price, reduce_only=reduce)
        st.success("Futures OCO отправлен.")
        st.json(r)
    except Exception as e:
        st.error(f"Ошибка: {e}")
