
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings, update_settings, creds_ok
from utils.log import log

st.title("✅ Подключение и состояние")

s = get_settings()
with st.form("creds"):
    st.subheader("Доступ к Bybit")
    api_key = st.text_input("API Key", value=s.api_key, type="password")
    api_secret = st.text_input("API Secret", value=s.api_secret, type="password")
    testnet = st.toggle("Testnet", value=s.testnet)
    dry = st.toggle("DRY-RUN (симуляция)", value=s.dry_run, help="Если включено — заявки не отправляются, только логируются.")
    submitted = st.form_submit_button("💾 Сохранить")
    if submitted:
        update_settings(api_key=api_key.strip(), api_secret=api_secret.strip(), testnet=testnet, dry_run=dry)
        st.success("Сохранено.")

st.divider()
st.subheader("Проверка соединения")
if creds_ok():
    api = get_api_client()
    try:
        t = api.server_time()
        st.success(f"Связь с API есть. Серверное время: {t.get('result',{}).get('timeSecond', '—')}")
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")

    st.write("---")
    st.caption("Локальная проверка пары")
    sym = st.text_input("Тикер для проверки", value="BTCUSDT")
    if st.button("Проверить инструменты и тикер"):
        try:
            info = api.instruments_info(category="spot", symbol=sym.strip().upper())
            st.json(info)
            tk = api.tickers(category="spot", symbol=sym.strip().upper())
            st.json(tk)
        except Exception as e:
            st.error(f"Ошибка: {e}")
else:
    st.warning("Укажите API ключи выше и сохраните.")
