
from __future__ import annotations
import streamlit as st, time, json

from utils.envs import get_api_client, get_settings
from ..ui.actions import run_api_action

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
            run_api_action(
                lambda: api.cancel_all(category="spot", symbol=(sym_cancel or None)),
                success=st.json,
                error_message="Не удалось отменить ордера.",
                description="cancel_all_spot",
            )

    with colB:
        st.caption("Batch Cancel по orderId/orderLinkId (JSON-массив объектов).")
        payload = st.text_area("Requests JSON", value='[{"symbol":"BTCUSDT","orderLinkId":"AI-BTCUSDT-123"}]')
        if st.button("🧺 Batch Cancel"):
            def _load_requests():
                data = json.loads(payload)
                if not isinstance(data, list):
                    raise ValueError("Ожидался JSON-массив заявок")
                return data

            requests_payload = run_api_action(
                _load_requests,
                error_message="Некорректный JSON с заявками.",
                description="parse_batch_cancel_payload",
            )
            if requests_payload is not None:
                run_api_action(
                    lambda: api.batch_cancel(category="spot", requests=requests_payload),
                    success=st.json,
                    error_message="Не удалось выполнить пакетную отмену.",
                    description="batch_cancel_spot",
                )
