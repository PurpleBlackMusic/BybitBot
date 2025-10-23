
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings, update_settings
from utils.time_sync import check_time_drift_seconds

st.title("🩺 Health — Time Sync & API Window")

settings = get_settings()
drift = check_time_drift_seconds()
st.metric("Drift (sec, local - server)", f"{drift:.3f}")

current_window = int(getattr(settings, "recv_window_ms", 15000) or 15000)

with st.form("recv_window_form"):
    new_window = st.number_input(
        "recvWindow (ms)",
        min_value=1000,
        max_value=120000,
        step=1000,
        value=current_window,
        help="Гарантированное окно допустимого рассинхрона между клиентом и сервером Bybit.",
    )
    submitted = st.form_submit_button("Сохранить окно")
    if submitted:
        update_settings(recv_window_ms=int(new_window))
        st.success("recvWindow обновлён")
        settings = get_settings(force_reload=True)  # type: ignore[call-arg]
        current_window = int(getattr(settings, "recv_window_ms", new_window) or new_window)

st.caption(f"Текущее окно: {current_window} мс")

if abs(drift * 1000) > current_window / 2:
    st.warning(
        "Дрифт времени велик относительно recvWindow — синхронизируйте системные часы/NTP.",
    )
