
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.hygiene import cancel_stale_orders

st.title("🧽 Order Hygiene — отмена «старых» заявок")

s = get_settings()
api = get_api_client()

older = st.number_input("Старше, сек", 60, 24*3600, 900)
sym = st.text_input("Символ (пусто = все)", value="")
if st.button("🧹 Отменить старые"):
    r = cancel_stale_orders(api, category="spot", symbol=(sym.upper().strip() or None), older_than_sec=int(older))
    st.json(r)
