
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.impact import estimate_vwap_from_orderbook

st.title("🧪 Impact-Cost Анализатор (Spot)")

s = get_settings()
with st.form("impact"):
    symbol = st.text_input("Символ", value="BTCUSDT").upper().strip()
    side = st.selectbox("Сторона", ["Buy","Sell"])
    qty = st.text_input("Объём (qty, base)", value="0.01")
    limit = st.number_input("Глубина стакана (уровней)", 50, 1000, 200)
    run = st.form_submit_button("▶️ Оценить")
if run:
    api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
    ob = api.orderbook(category="spot", symbol=symbol, limit=int(limit))
    est = estimate_vwap_from_orderbook(ob, side=side, qty_base=float(qty))
    st.json(est)
    if est.get("vwap"):
        st.metric("VWAP", f"{est['vwap']:.8f}")
        st.metric("Impact (bps)", f"{est['impact_bps']:.2f}")
