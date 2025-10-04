
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.portfolio import corr_matrix, save_corr, load_corr, estimate_portfolio_allocation

st.title("🧮 Portfolio Risk (Spot) — корреляции и лимиты")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))

symbols = [x.strip().upper() for x in (s.ai_symbols or "BTCUSDT,ETHUSDT,SOLUSDT").split(',') if x.strip()]
if st.button("🔄 Пересчитать корреляции (30д, 1h)"):
    C = corr_matrix(api, symbols, interval="60", lookback_hours=24*30)
    if not C.empty:
        save_corr(C)
        st.success("Обновлено.")
        st.dataframe(C, use_container_width=True)
else:
    C = load_corr()
    if not C.empty:
        st.dataframe(C, use_container_width=True)
    else:
        st.info("Пока нет сохранённой матрицы. Нажмите «Пересчитать».")

st.divider()
st.subheader("Оценка аллокаций (vol-target + портфельные лимиты)")
eq = st.number_input("Equity (USDT)", 0.0, 1e9, 1000.0, 10.0)
if st.button("📐 Рассчитать"):
    out = estimate_portfolio_allocation(api, symbols, prices={}, equity_usdt=float(eq), settings=s)
    st.json({k: v for k, v in out.items() if k in ["alloc","group_caps"]})


st.divider()
st.subheader("ENB — Effective Number of Bets")
from utils.portfolio import load_corr, effective_number_of_bets
C = load_corr()
if not C.empty:
    enb = effective_number_of_bets(C)
    st.metric("ENB (по корреляциям)", f"{enb:.2f}")
    st.caption("ENB≈количество независимых рисков в портфеле. Ближе к числу символов — лучше диверсификация.")
else:
    st.info("Матрица корреляций ещё не посчитана.")
