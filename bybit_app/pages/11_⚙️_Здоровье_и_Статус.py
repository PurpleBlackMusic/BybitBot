from __future__ import annotations
import streamlit as st, time, datetime as dt
from utils.ws_manager import manager as ws_manager
from utils.envs import get_settings
from utils.telegram_notify import send_telegram

st.set_page_config(page_title="Здоровье и статус", layout="wide")

s = get_settings()
st.title("⚙️ Здоровье и статус")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("WebSocket")
    st.json(ws_manager.status())
    if st.button("Перезапустить публичный WS"):
        ws_manager.stop_public()
        ok = ws_manager.start_public(tuple(getattr(s, "ws_public_topics", ["tickers.BTCUSDT"])))
        st.success(f"WS public restarted: {ok}")
with col2:
    st.subheader("Telegram")
    if st.button("Отправить тест-сообщение"):
        r = send_telegram("Тест: бот на связи ✅")
        st.write(r)
with col3:
    st.subheader("Info")
    st.write({"testnet": s.testnet, "dry_run": getattr(s, "dry_run", True)})
