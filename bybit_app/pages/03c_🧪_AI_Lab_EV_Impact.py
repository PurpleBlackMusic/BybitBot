
from __future__ import annotations
import streamlit as st, numpy as np
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.impact import estimate_vwap_from_orderbook
from utils.ai.costs import expected_value_dynamic
from utils.cache_helpers import cached_fee_rate

st.title("🧪 AI Lab — EV по VWAP/Impact (Spot)")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))

symbol = st.text_input("Символ", value="BTCUSDT").upper().strip()
side = st.selectbox("Сторона", ["Buy","Sell"])
p_up = st.slider("Вероятность роста p_up", 0.5, 0.8, 0.58, 0.01)
rr = st.slider("RR (TP/SL)", 1.2, 3.0, 1.8, 0.1)
qtys = st.text_input("Кандидаты объёма (через запятую, base)", value="0.002,0.005,0.01,0.02")

fr = cached_fee_rate(category="spot", symbol=symbol)
fee_bps = 7.0
try:
    rate = ((fr.get("result") or {}).get("list") or [])
    if rate:
        taker = abs(float(rate[0].get("takerFeeRate", 0))*10000.0)
        fee_bps = taker
except Exception:
    pass

if st.button("▶️ Рассчитать EV для объёмов"):
    ob = api.orderbook(category="spot", symbol=symbol, limit=200)
    nums = [float(x.strip()) for x in qtys.split(",") if x.strip()]
    rows = []
    for q in nums:
        est = estimate_vwap_from_orderbook(ob, side=side, qty_base=q)
        imp = est.get("impact_bps") or 0.0
        ev = expected_value_dynamic(p_up, rr, fee_bps, fee_bps, imp, imp)
        rows.append({"qty": q, "impact_bps": imp, "EV_bps": ev, "VWAP": est.get("vwap")})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
