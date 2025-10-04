
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings, update_settings
from utils.ui import section

st.title("⚙️ Настройки")

s = get_settings()
section("Telegram")
col1, col2 = st.columns(2)
with col1:
    token = st.text_input("Bot Token", value=s.telegram_token, type="password")
with col2:
    chat = st.text_input("Chat ID", value=s.telegram_chat_id)

if st.button("💾 Сохранить Telegram"):
    update_settings(telegram_token=token.strip(), telegram_chat_id=chat.strip())
    st.success("Telegram настройки сохранены.")
