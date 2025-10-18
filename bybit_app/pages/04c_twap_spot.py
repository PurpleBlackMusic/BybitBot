
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.twap_spot import twap_spot

st.title("🧩 TWAP Исполнитель (Spot)")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Сначала укажите API ключи на странице «Подключение и состояние».")
    st.stop()

with st.form("twap"):
    symbol = st.text_input("Символ", value="BTCUSDT").upper().strip()
    side = st.selectbox("Сторона", ["Buy","Sell"])
    qty = st.text_input("Общий объём (qty)", value="0.01")
    slices = st.number_input("Число срезов", 2, 50, int(s.twap_slices or 5))
    child_secs = st.number_input("Интервал между срезами (сек)", 1, 120, int(s.twap_child_secs or 10))
    agg_bps = st.number_input("Агрессивность (bps)", 0.0, 50.0, float(s.twap_aggressiveness_bps or 2.0))
    batch = st.checkbox("Batch mode (до 10 ордеров за раз)", value=False)
    run = st.form_submit_button("▶️ Запустить TWAP")

if run:
    api = get_api_client()
    try:
        if batch:
            from utils.twap_spot_batch import twap_spot_batch

            result = twap_spot_batch(api, symbol, side, float(qty), int(slices), float(agg_bps))
            st.success("TWAP (batch) запущен.")
            st.json(result)
        else:
            result = twap_spot(api, symbol, side, float(qty), int(slices), int(child_secs), float(agg_bps))
            st.success(f"Запущено. Дочерних ордеров: {len(result)}")
            for idx, payload in enumerate(result):
                st.json({"child": idx, "resp": payload})
    except Exception as e:
        st.error(f"Ошибка: {e}")
