
from __future__ import annotations
import streamlit as st
from utils.time_sync import check_time_drift_seconds
from utils.envs import get_settings
st.title("🩺 Health — Time Sync & API Window")
drift = check_time_drift_seconds()
s = get_settings()
st.metric("Drift (sec, local - server)", f"{drift:.3f}")
st.write(f"recvWindow (ms): {int(getattr(s,'recv_window_ms',5000) or 5000)}")
if abs(drift*1000) > int(getattr(s,'recv_window_ms',5000) or 5000)/2:
    st.warning("Дрифт времени велик относительно recvWindow — синхронизируйте системные часы/NTP.")
