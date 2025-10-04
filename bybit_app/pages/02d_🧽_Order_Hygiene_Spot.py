
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.hygiene import cancel_stale_orders

st.title("🧽 Order Hygiene — отмена «старых» заявок")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))

older = st.number_input("Старше, сек", 60, 24*3600, 900)
sym = st.text_input("Символ (пусто = все)", value="")
if st.button("🧹 Отменить старые"):
    r = cancel_stale_orders(api, category="spot", symbol=(sym.upper().strip() or None), older_than_sec=int(older))
    st.json(r)
