
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.dataframe import arrow_safe
from utils.symbol_overrides import load_overrides, set_override
st.title("🧰 Пер‑символьные настройки (Spot)")
ovr = load_overrides()
if ovr:
    st.dataframe(arrow_safe(pd.DataFrame(ovr).T), use_container_width=True)
sym = st.text_input("Символ", value="BTCUSDT").upper().strip()
imp = st.number_input("Лимит импакта (bps) — пусто чтобы не менять", 0.0, 500.0, 0.0)
th  = st.number_input("Buy threshold — пусто чтобы не менять", 0.0, 1.0, 0.0, step=0.01)
rr  = st.number_input("RR override — пусто чтобы не менять", 0.0, 10.0, 0.0, step=0.1)
if st.button("💾 Сохранить override"):
    kwargs = {}
    if imp>0: kwargs["spot_max_impact_bps"] = float(imp)
    if th>0: kwargs["ai_buy_threshold"] = float(th)
    if rr>0: kwargs["ai_rr_base"] = float(rr)
    set_override(sym, **kwargs)
    st.success("Сохранено.")
