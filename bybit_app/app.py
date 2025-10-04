
from __future__ import annotations
import streamlit as st
from utils.ui import safe_set_page_config
from utils.paths import APP_ROOT
from utils.envs import get_settings

safe_set_page_config(page_title="Bybit Smart OCO — PRO", page_icon="🧠", layout="wide")

st.title("Bybit Smart OCO — PRO")
st.caption("Улучшенная 3Commas: умный OCO, понятный интерфейс, живые статусы.")

s = get_settings()
ok = bool(s.api_key and s.api_secret)
st.info(f"API key: {'✅' if s.api_key else '❌'} | Secret: {'✅' if s.api_secret else '❌'} | Сеть: {'Testnet' if s.testnet else 'Mainnet'} | DRY-RUN: {'ON' if s.dry_run else 'OFF'}")

st.write("Файлы приложения:", APP_ROOT)
st.write("Используйте меню слева для разделов. Начните со страницы **Подключение и состояние**.")
