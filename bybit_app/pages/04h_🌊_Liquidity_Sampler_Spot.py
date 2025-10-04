
from __future__ import annotations
import streamlit as st, pandas as pd, numpy as np, time
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.impact import estimate_vwap_from_orderbook

st.title("🌊 Ликвидность (Sampler) — Spot")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
symbol = st.text_input("Символ", value="BTCUSDT").upper().strip()
side = st.selectbox("Сторона", ["Buy","Sell"])
qtys = st.text_input("Объёмы (через запятую, base)", value="0.001,0.003,0.005,0.01")
samples = st.number_input("Сэмплов стакана", 1, 50, 5)
pause = st.number_input("Пауза между сэмплами, сек", 0, 60, 1)

if st.button("▶️ Замерить"):
    qs = [float(x.strip()) for x in qtys.split(",") if x.strip()]
    rows = []
    for i in range(samples):
        ob = api.orderbook(category="spot", symbol=symbol, limit=200)
        for q in qs:
            est = estimate_vwap_from_orderbook(ob, side=side, qty_base=q)
            rows.append({"sample": i, "qty": q, "impact_bps": est.get("impact_bps"), "vwap": est.get("vwap"), "mid": est.get("mid")})
        time.sleep(pause)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.caption("Совет: используйте p75 impact для порога импакта на паре.")
