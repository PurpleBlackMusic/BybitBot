
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds

st.title("📊 Портфель — Дашборд")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Укажите API ключи на странице «Подключение и состояние».")
    st.stop()

api = BybitAPI(creds=BybitCreds(s.api_key, s.api_secret, s.testnet))
try:
    wb = api.wallet_balance()
    coins = (((wb.get("result") or {}).get("list") or [{}])[0].get("coin") or [])
    st.caption("Unified Wallet — доступные активы")
    st.dataframe(coins, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка чтения баланса: {e}")
