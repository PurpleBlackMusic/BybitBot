
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.ws_orderbook import LiveOrderbook

st.title("⚡ Live Orderbook Impact (Spot) — VWAP оценка")

s = get_settings()
api = get_api_client()
lob = LiveOrderbook(api, category="spot")

sym = st.text_input("Символ", value="BTCUSDT")
side = st.selectbox("Сторона", ["buy","sell"])
qty = st.number_input("Количество (base)", 0.0, 1e9, 0.01, 0.001)
limit = st.selectbox("Глубина расчёта", [50,200,1000], index=1)

if st.button("🔎 Оценить VWAP"):
    ob = lob.get_book(sym, limit=int(limit))
    est = lob.vwap_for(sym, side, float(qty), limit=int(limit))
    st.json({"best": est.get("best"), "vwap": est.get("vwap"), "levels": est.get("levels")})
    if est.get("vwap") and est.get("best"):
        slip_bps = (float(est["vwap"])/float(est["best"]) - 1.0)*10000.0 if side=='buy' else (1.0 - float(est["vwap"])/float(est["best"]))*10000.0
        st.metric("Оценка слиппеджа (bps)", f"{slip_bps:.2f}")
st.caption("По докам: REST orderbook даёт слепок до 1000 уровней; Public WS пушит 1/50/200/1000 уровни с частотой 10–300мс. Мы используем REST как фоллбэк.")
