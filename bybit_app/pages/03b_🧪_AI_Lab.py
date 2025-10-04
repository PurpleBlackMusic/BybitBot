
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.ai.engine import AIPipeline
from utils.ai.backtest import walk_forward
from utils.paths import DATA_DIR

st.title("🧪 AI Lab — Walk-Forward Backtest")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))

col1, col2, col3 = st.columns(3)
with col1:
    category = st.selectbox("Категория", ["spot","linear"], index=(0 if s.ai_category=='spot' else 1))
with col2:
    symbol = st.text_input("Символ", value=(s.ai_symbols.split(",")[0].strip() if s.ai_symbols else "BTCUSDT"))
with col3:
    interval = st.selectbox("Интервал", ["1","3","5","15","30","60","240","D"], index=["1","3","5","15","30","60","240","D"].index(s.ai_interval if s.ai_interval in ["1","3","5","15","30","60","240","D"] else "60"))

horizon = st.slider("Горизонт метки (баров)", 3, 96, int(s.ai_horizon_bars or 12))
n_splits = st.slider("Сколько фолдов", 3, 10, 5)
rr = st.slider("RR (TP/SL)", 1.2, 3.0, 1.8, 0.1)
fee_bps = st.number_input("Комиссия, б.п. (в сумме за вход/выход будет 2*fee)", 0.0, 50.0, float(s.ai_fee_bps or 7.0))
slip_bps = st.number_input("Слиппедж, б.п. (в сумме за вход/выход будет 2*slip)", 0.0, 100.0, float(s.ai_slippage_bps or 10.0))

if st.button("▶️ Запустить Walk-Forward"):
    pipe = AIPipeline(DATA_DIR / "ai")
    try:
        df = pipe.fetch_klines(api, category, symbol.strip().upper(), interval, limit=1000)
        res = walk_forward(df, horizon=horizon, n_splits=n_splits, rr=rr, fee_bps=fee_bps, slip_bps=slip_bps)
        st.success("Готово.")
        st.json(res)
        st.caption("Подсказка: установите buy_threshold близко к best_threshold и задайте ai_min_ev_bps >= ev_bps для фильтра сделок.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
