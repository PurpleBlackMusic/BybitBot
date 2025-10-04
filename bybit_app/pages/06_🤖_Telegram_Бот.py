
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings, update_settings
from utils.telegram_notify import send_telegram

st.title("🤖 Telegram Бот")

s = get_settings()
col1, col2 = st.columns(2)
with col1:
    token = st.text_input("Bot Token", value=s.telegram_token, type="password")
with col2:
    chat = st.text_input("Chat ID", value=s.telegram_chat_id)

hb = st.toggle("Включить heartbeat", value=s.heartbeat_enabled, help="Периодические сообщения о состоянии WS/бота.")
mins = st.number_input("Частота heartbeat (мин)", value=int(s.heartbeat_minutes or 5), min_value=1, max_value=120)

if st.button("💾 Сохранить"):
    update_settings(telegram_token=token.strip(), telegram_chat_id=chat.strip(), heartbeat_enabled=bool(hb), heartbeat_minutes=int(mins))
    st.success("Сохранено.")

st.divider()
if st.button("📨 Отправить тестовое сообщение"):
    r = send_telegram("Тестовое сообщение: бот на связи ✅")
    st.json(r)
