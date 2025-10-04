
from __future__ import annotations
import streamlit as st
from utils.ui import safe_set_page_config
from utils.paths import APP_ROOT
from utils.envs import get_settings

safe_set_page_config(page_title="Bybit Smart OCO — PRO", page_icon="🧠", layout="wide")

st.title("Bybit Smart OCO — PRO")
st.caption("Улучшенная 3Commas: умный OCO, понятный интерфейс, живые статусы.")

st.subheader("🎯 Миссия приложения")
st.markdown(
    """
    - предоставляет живую аналитику крипторынка, чтобы вы понимали **что происходит прямо сейчас**;
    - прогнозирует движение цены на базе AI‑модели и помогает решить, что сегодня **покупать, продавать или пропустить**;
    - автоматизирует сделки OCO и TWAP, контролируя комиссии и спрэды, чтобы забирать **максимум прибыли** с биржи;
    - делится всей служебной информацией: отчёты, статус WebSocket, исполнение ордеров, уведомления в Telegram;
    - включает строгие risk‑guards и kill‑switch'и, чтобы **не дать счёту уйти в минус** и остановить торговлю при угрозе убытка.
    """
)

s = get_settings()
ok = bool(s.api_key and s.api_secret)
st.info(f"API key: {'✅' if s.api_key else '❌'} | Secret: {'✅' if s.api_secret else '❌'} | Сеть: {'Testnet' if s.testnet else 'Mainnet'} | DRY-RUN: {'ON' if s.dry_run else 'OFF'}")

st.subheader("🛡 Контроль капитала")
col1, col2, col3 = st.columns(3)
col1.metric("Риск на сделку", f"{getattr(s, 'ai_risk_per_trade_pct', 0.25):.2f}%", help="Максимальная доля капитала, которую бот рискует в одной сделке.")
col2.metric("Дневной лимит убытка", f"{getattr(s, 'ai_daily_loss_limit_pct', 3.0):.2f}%", help="При достижении порога торговля ставится на паузу.")
cap_guard = 100 - float(getattr(s, 'spot_cash_reserve_pct', 10.0) or 0.0)
col3.metric("Задействованный капитал", f"≤ {cap_guard:.0f}%", help="Часть средств зарезервирована, чтобы портфель не уходил в минус.")

st.caption("Настройки защиты можно изменить в разделах 🧠 AI-Трейдер и 🧭 Простой режим. Включённая опция DRY-RUN гарантирует демонстрационный режим без реальных ордеров.")

st.write("Файлы приложения:", APP_ROOT)
st.write("Используйте меню слева для разделов. Начните со страницы **Подключение и состояние**.")
