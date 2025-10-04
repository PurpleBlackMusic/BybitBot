from __future__ import annotations

import streamlit as st

from utils.envs import get_settings, update_settings
from utils.telegram_notify import send_telegram

st.set_page_config(page_title="Telegram бот", page_icon="🤖", layout="centered")

st.title("🤖 Telegram-бот для уведомлений")
st.caption(
    "Настройте токен и чат, чтобы получать оперативные сообщения от умного спотового бота."
)

settings = get_settings()

with st.form("telegram-creds"):
    st.subheader("Доступ к Telegram")
    col1, col2 = st.columns(2)
    with col1:
        token = st.text_input("Bot Token", value=settings.telegram_token, type="password")
    with col2:
        chat_id = st.text_input("Chat ID", value=settings.telegram_chat_id)
    heartbeat = st.toggle(
        "Пульс бота",
        value=settings.heartbeat_enabled,
        help="Периодические сообщения о том, что подключение и торговый бот активны.",
    )
    interval = st.number_input(
        "Частота heartbeat (мин)",
        min_value=1,
        max_value=180,
        value=int(settings.heartbeat_minutes or 5),
        help="Как часто бот отправляет служебное сообщение о состоянии.",
    )
    submitted = st.form_submit_button("💾 Сохранить настройки")
    if submitted:
        update_settings(
            telegram_token=token.strip(),
            telegram_chat_id=chat_id.strip(),
            heartbeat_enabled=bool(heartbeat),
            heartbeat_minutes=int(interval),
        )
        st.success("Настройки сохранены. Бот готов к отправке уведомлений.")

st.divider()
st.subheader("Проверка связи")

if st.button("📨 Отправить тестовое сообщение"):
    response = send_telegram("Тестовое сообщение: бот на связи ✅")
    st.json(response)
else:
    st.info("Тестовое сообщение поможет убедиться, что токен и Chat ID заданы верно.")
