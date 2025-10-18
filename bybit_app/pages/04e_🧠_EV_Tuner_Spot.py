from __future__ import annotations
import streamlit as st, pandas as pd
from utils.dataframe import arrow_safe
from utils.eval_exec import realized_impact_report
from utils.envs import get_settings, update_settings

st.title("🧠 EV Tuner — авто-порог impact для спота")

rep = realized_impact_report(window_sec=1800)
if not rep:
    st.info("Недостаточно данных (нужны недавние исполнения и решения).")
else:
    df = pd.DataFrame(rep).T
    st.dataframe(arrow_safe(df), use_container_width=True)
    st.caption("Порог = 1.1 × p75 реализованного импакта, но не ниже 5 bps.")
    s = get_settings()
    sym = (
        st.text_input(
            "Символ для применения (точно как в отчёте)",
            value=(list(rep.keys())[0] if rep else "BTCUSDT"),
        )
        .upper()
        .strip()
    )
    lim = st.number_input(
        "Новый лимит импакта (bps)",
        1.0,
        200.0,
        float(df.loc[sym, "suggest_limit_bps"])
        if rep and sym in df.index and df.loc[sym, "suggest_limit_bps"]
        else 25.0,
    )
    if st.button("💾 Применить как общий лимит (spot)"):
        update_settings(spot_max_impact_bps=float(lim))
        st.success(f"Установлен общий лимит импакта: {lim} bps")
