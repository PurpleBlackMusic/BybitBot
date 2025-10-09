
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings, update_settings, creds_ok
from utils.log import log

st.title("✅ Подключение и состояние")

s = get_settings()


def _mask(value: str) -> str:
    if not value:
        return ""
    return "•" * min(len(value), 6)


with st.form("creds"):
    st.subheader("Доступ к Bybit")
    api_key_placeholder = _mask(s.api_key)
    api_secret_placeholder = _mask(s.api_secret)
    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder=api_key_placeholder or "",
        help="Поле оставьте пустым, чтобы сохранить ранее введённый ключ.",
    )
    api_secret = st.text_input(
        "API Secret",
        value="",
        type="password",
        placeholder=api_secret_placeholder or "",
        help="Поле оставьте пустым, чтобы сохранить ранее введённый секрет.",
    )
    testnet = st.toggle("Testnet", value=s.testnet)
    dry = st.toggle(
        "DRY-RUN (симуляция)",
        value=s.dry_run,
        help="Если включено — заявки не отправляются, только логируются.",
    )
    clear_keys = st.checkbox(
        "Очистить сохранённые ключи", value=False, help="Установите флажок, если хотите удалить ключи из хранилища."
    )
    submitted = st.form_submit_button("💾 Сохранить")
    if submitted:
        payload: dict[str, object] = {}

        if testnet != s.testnet:
            payload["testnet"] = testnet
        if dry != s.dry_run:
            payload["dry_run"] = dry

        if clear_keys:
            payload.update({"api_key": "", "api_secret": ""})
        else:
            cleaned_key = api_key.strip()
            cleaned_secret = api_secret.strip()
            if cleaned_key:
                payload["api_key"] = cleaned_key
            if cleaned_secret:
                payload["api_secret"] = cleaned_secret
        update_settings(**payload)
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
