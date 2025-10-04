
from __future__ import annotations
import streamlit as st
from utils.bybit_api import BybitAPI, BybitCreds
from utils.envs import get_settings
from utils.nanny_limits import spot_order_counters

st.title("📏 Spot Order Limits — мониторинг капов")
s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
c = spot_order_counters(api)
st.json(c)
st.caption("По докам: общий лимит открытых спотовых ордеров — до 500 на аккаунт; до 30 открытых TP/SL и до 30 открытых условных ордеров на символ/аккаунт.")
