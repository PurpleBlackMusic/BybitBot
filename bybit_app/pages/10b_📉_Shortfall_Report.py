
from __future__ import annotations
import streamlit as st, pandas as pd, numpy as np
from utils.eval_exec import realized_impact_report
from utils.decision_log import read_decisions
from utils.pnl import read_ledger

st.title("📉 Implementation Shortfall — отчёт (Spot)")

decs = read_decisions(10000)
fills = read_ledger(10000)
st.caption(f"Решений: {len(decs)}, исполнений: {len(fills)}")

rep = realized_impact_report(window_sec=1800)
if rep:
    st.subheader("Сводка реализованного импакта (bps)")
    st.dataframe(pd.DataFrame(rep).T, use_container_width=True)
else:
    st.info("Недостаточно свежих данных для сводки импакта.")

