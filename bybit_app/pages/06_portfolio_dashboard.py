from __future__ import annotations

import pandas as pd
import streamlit as st
from utils.dataframe import arrow_safe

from utils.envs import get_api_client, get_settings
from utils.spot_market import wallet_balance_payload

st.title("📊 Портфель — Дашборд")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Укажите API ключи на странице «Подключение и состояние».")
    st.stop()

api = get_api_client()
try:
    wb = wallet_balance_payload(api)
    raw_list = ((wb.get("result") or {}).get("list") or [{}])
    first_entry = raw_list[0] if raw_list else {}
    coins_raw = first_entry.get("coin") or []

    if isinstance(coins_raw, dict):
        coins_iterable = list(coins_raw.values())
    elif isinstance(coins_raw, list):
        coins_iterable = coins_raw
    else:
        coins_iterable = []

    table = pd.DataFrame.from_records(coins_iterable)

    st.caption("Unified Wallet — доступные активы")
    if table.empty:
        st.info("Баланс не содержит монет для отображения.")
    else:
        st.dataframe(arrow_safe(table), use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Ошибка чтения баланса: {e}")
