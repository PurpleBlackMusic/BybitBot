
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.dataframe import arrow_safe
from utils.trade_pairs import pair_trades
st.title("🔗 Trade Pairs (Spot) — ex‑post EV & R")
if st.button("🔄 Пересчитать пары"):
    trs = pair_trades()
else:
    trs = pair_trades()
if trs:
    df = pd.DataFrame(trs)
    st.dataframe(arrow_safe(df.sort_values("exit_ts")), use_container_width=True)
    win = (df["bps_realized"]>0).mean() if not df.empty else None
    st.metric("Win‑rate", f"{win*100:.1f}%" if win is not None else "—")
else:
    st.info("Пока нет данных для паринга сделок.")
