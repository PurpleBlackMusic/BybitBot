
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.ai.tune import tune_buy_threshold
from utils.envs import update_settings

st.title("🎯 Buy Threshold — авто‑подстройка (Spot)")

symbol = st.text_input("Символ", value="BTCUSDT").upper().strip()
rr = st.number_input("RR (для расчёта скоринга)", 0.1, 5.0, 1.8, 0.1)
fee = st.number_input("Комиссия bps (на всякий случай)", 0.0, 50.0, 7.0)
if st.button("▶️ Рассчитать"):
    best, table = tune_buy_threshold(symbol, rr=rr, fee_bps=fee)
    if best:
        st.success(f"Рекомедованный threshold: {best:.3f}")
        st.dataframe(pd.DataFrame(table), use_container_width=True)
        if st.button("💾 Применить"):
            update_settings(ai_buy_threshold=float(best))
            st.success("Порог сохранён в настройки.")
    else:
        st.info("Недостаточно данных.")
