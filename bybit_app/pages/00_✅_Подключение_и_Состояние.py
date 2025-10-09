
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
    st.caption("Ключи и режим DRY-RUN можно задавать отдельно для каждой сети.")

    testnet_active = st.toggle(
        "Активная сеть: Testnet",
        value=s.testnet,
        help="Переключает рабочую сеть по умолчанию для приложения.",
    )

    col_testnet, col_mainnet = st.columns(2)

    with col_testnet:
        st.markdown("#### Testnet")
        api_key_testnet_placeholder = _mask(s.get_api_key(testnet=True))
        api_secret_testnet_placeholder = _mask(s.get_api_secret(testnet=True))
        api_key_testnet = st.text_input(
            "API Key (Testnet)",
            value="",
            type="password",
            placeholder=api_key_testnet_placeholder or "",
            help="Оставьте пустым, чтобы не менять сохранённый ключ.",
        )
        api_secret_testnet = st.text_input(
            "API Secret (Testnet)",
            value="",
            type="password",
            placeholder=api_secret_testnet_placeholder or "",
            help="Оставьте пустым, чтобы не менять сохранённый секрет.",
        )
        dry_testnet = st.toggle(
            "DRY-RUN Testnet",
            value=s.get_dry_run(testnet=True),
            key="dry_run_testnet_toggle",
            help="Если включено — ордера в тестовой сети не отправляются на биржу.",
        )
        clear_testnet = st.checkbox(
            "Очистить тестовые ключи",
            value=False,
            key="clear_keys_testnet",
        )

    with col_mainnet:
        st.markdown("#### Mainnet")
        api_key_mainnet_placeholder = _mask(s.get_api_key(testnet=False))
        api_secret_mainnet_placeholder = _mask(s.get_api_secret(testnet=False))
        api_key_mainnet = st.text_input(
            "API Key (Mainnet)",
            value="",
            type="password",
            placeholder=api_key_mainnet_placeholder or "",
            help="Оставьте пустым, чтобы не менять сохранённый ключ.",
        )
        api_secret_mainnet = st.text_input(
            "API Secret (Mainnet)",
            value="",
            type="password",
            placeholder=api_secret_mainnet_placeholder or "",
            help="Оставьте пустым, чтобы не менять сохранённый секрет.",
        )
        dry_mainnet = st.toggle(
            "DRY-RUN Mainnet",
            value=s.get_dry_run(testnet=False),
            key="dry_run_mainnet_toggle",
            help="Если включено — ордера в боевой сети не отправляются на биржу.",
        )
        clear_mainnet = st.checkbox(
            "Очистить боевые ключи",
            value=False,
            key="clear_keys_mainnet",
        )

    submitted = st.form_submit_button("💾 Сохранить")
    if submitted:
        payload: dict[str, object] = {}

        if testnet_active != s.testnet:
            payload["testnet"] = testnet_active

        network_entries = [
            (
                True,
                clear_testnet,
                api_key_testnet.strip(),
                api_secret_testnet.strip(),
                dry_testnet,
                s.get_dry_run(testnet=True),
            ),
            (
                False,
                clear_mainnet,
                api_key_mainnet.strip(),
                api_secret_mainnet.strip(),
                dry_mainnet,
                s.get_dry_run(testnet=False),
            ),
        ]

        for is_testnet, clear, key_value, secret_value, dry_value, current_dry in network_entries:
            key_field = "api_key_testnet" if is_testnet else "api_key_mainnet"
            secret_field = "api_secret_testnet" if is_testnet else "api_secret_mainnet"
            dry_field = "dry_run_testnet" if is_testnet else "dry_run_mainnet"

            if clear:
                payload[key_field] = ""
                payload[secret_field] = ""
            else:
                if key_value:
                    payload[key_field] = key_value
                if secret_value:
                    payload[secret_field] = secret_value

            if dry_value != current_dry:
                payload[dry_field] = dry_value

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
