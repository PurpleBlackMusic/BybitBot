
from __future__ import annotations
import streamlit as st, time, json

from utils.envs import get_api_client, get_settings

st.title("🛑 KillSwitch and API Nanny")

# Build API client from settings
s = get_settings()
api = get_api_client()

st.subheader("Быстрые отмены (Spot)")

with st.expander("⚡ Cancel tools", expanded=True):
    colA, colB = st.columns(2)
    with colA:
        sym_cancel = st.text_input("Символ (опционально для Cancel All)", value="")
        if st.button("❌ Cancel All (spot)"):
            try:
                r = api.cancel_all(category="spot", symbol=(sym_cancel or None))
                st.json(r)
            except Exception as e:
                st.error(f"Ошибка: {e}")

    with colB:
        st.caption("Batch Cancel по orderId/orderLinkId (JSON-массив объектов).")
        payload = st.text_area("Requests JSON", value='[{"symbol":"BTCUSDT","orderLinkId":"AI-BTCUSDT-123"}]')
        if st.button("🧺 Batch Cancel"):
            try:
                reqs = json.loads(payload)
                r = api.batch_cancel(category="spot", requests=reqs)
                st.json(r)
            except Exception as e:
                st.error(f"Ошибка: {e}")
