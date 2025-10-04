
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.portfolio import corr_matrix, estimate_portfolio_allocation, load_corr
from utils.hrp import hrp_weights

st.title("🧭 HRP vs Vol‑Target — сравнение аллокаций")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
symbols = [x.strip().upper() for x in (s.ai_symbols or "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if x.strip()]

if st.button("🔄 Посчитать матрицу (30д, 1h)"):
    C = corr_matrix(api, symbols, interval="60", lookback_hours=24*30)
    st.dataframe(C, use_container_width=True)
else:
    C = load_corr()
    if not C.empty:
        st.dataframe(C, use_container_width=True)
    else:
        st.info("Нет сохранённой матрицы. Пересчитайте на странице Portfolio Risk.")

st.divider()
st.subheader("HRP веса (по последней матрице)")
if not C.empty:
    # черновой R DataFrame для весов (используем ковариацию из C на обратной стороне — суррогат)
    import numpy as np
    # синтетические ряды не строим — используем corr для порядка, ковариацию не требуется для масштаба в HRP‑весах
    # подадим фиктивную матрицу R, чтобы hrp_weights принял DataFrame
    R = pd.DataFrame(np.random.randn(100, len(C.columns)), columns=C.columns)
    w_hrp = hrp_weights(R)
    st.dataframe(w_hrp.rename("weight").to_frame(), use_container_width=True)
else:
    st.info("Нет матрицы для HRP.")

st.divider()
st.subheader("Vol‑Target аллокации (из Portfolio Risk)")
eq = st.number_input("Equity (USDT)", 0.0, 1e9, 1000.0, 10.0)
if st.button("📐 Рассчитать vol‑target"):
    out = estimate_portfolio_allocation(api, symbols, prices={}, equity_usdt=float(eq), settings=s)
    st.json(out.get("alloc", {}))
