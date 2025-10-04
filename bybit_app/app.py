
from __future__ import annotations
import streamlit as st
from bybit_app.utils.ui import safe_set_page_config, inject_css
from bybit_app.utils.paths import APP_ROOT
from bybit_app.utils.envs import get_settings

safe_set_page_config(page_title="Bybit Smart OCO — PRO", page_icon="🧠", layout="wide")
inject_css(
    """
    .bybit-hero {
        background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,118,110,0.88));
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 20px;
        box-shadow: 0 18px 48px rgba(15, 118, 110, 0.28);
        margin-bottom: 1.25rem;
    }
    .bybit-hero h1, .bybit-hero h2, .bybit-hero p { color: inherit; }
    .bybit-hero .hero-sub { opacity: 0.9; font-size: 1.05rem; }
    .stMetric { background: rgba(15,118,110,0.12); border-radius: 16px; padding: 0.75rem 1rem; }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255,255,255,0.14);
    }
    .status-pill.negative { background: rgba(220,38,38,0.16); }
    .status-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .status-card.ok { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.3); }
    .status-card.warn { background: rgba(250, 204, 21, 0.12); border-color: rgba(250, 204, 21, 0.35); }
    .status-card__title { font-size: 1rem; font-weight: 600; margin-bottom: 0.35rem; display: flex; gap: 0.4rem; align-items: center; }
    .status-card p { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    """
)

with st.container():
    st.markdown(
        """
        <div class="bybit-hero">
            <h1>Bybit Smart OCO — PRO</h1>
            <p class="hero-sub">Умные торговые сценарии, прозрачные статусы и контроль рисков для трейдера, который ценит скорость принятия решений.</p>
            <div style="margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
                <span class="status-pill">⚡ Реакция &lt; 1s</span>
                <span class="status-pill">🧠 AI OCO &amp; TWAP</span>
                <span class="status-pill">🛡 Risk Guards</span>
                <span class="status-pill">🔔 Telegram Ping</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
with st.container(border=True):
    st.markdown("#### ⚙️ Технический статус профиля")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        status = "Все ключи подключены" if ok else "Требуется подключение API"
        status_hint = "Готовы к торговле и автоматическим стратегиям." if ok else "Добавьте API ключ и секрет, чтобы активировать торговлю."
        status_class = "ok" if ok else "warn"
        status_icon = "🔐" if ok else "⚠️"
        st.markdown(
            f"""
            <div class=\"status-card {status_class}\">
                <div class=\"status-card__title\">{status_icon} {status}</div>
                <p>{status_hint}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.metric("Режим", "Testnet" if s.testnet else "Mainnet", help="Переключите сеть в настройках окружения.")
        st.metric("DRY-RUN", "ON" if s.dry_run else "OFF", help="Включите DRY-RUN, чтобы проверять стратегию без реальных ордеров.")

    st.caption(
        f"API key: {'✅' if s.api_key else '❌'} · Secret: {'✅' if s.api_secret else '❌'} · Последнее обновление настроек: {s.updated_at.strftime('%d.%m.%Y %H:%M:%S') if getattr(s, 'updated_at', None) else '—'}"
    )

st.subheader("🛡 Контроль капитала")
cap_guard = 100 - float(getattr(s, 'spot_cash_reserve_pct', 10.0) or 0.0)
risk_cols = st.columns(3)
risk_cols[0].metric(
    "Риск на сделку",
    f"{getattr(s, 'ai_risk_per_trade_pct', 0.25):.2f}%",
    help="Максимальная доля капитала, которую бот рискует в одной сделке.",
)
risk_cols[1].metric(
    "Дневной лимит убытка",
    f"{getattr(s, 'ai_daily_loss_limit_pct', 3.0):.2f}%",
    help="При достижении порога торговля ставится на паузу.",
)
risk_cols[2].metric(
    "Задействованный капитал",
    f"≤ {cap_guard:.0f}%",
    help="Часть средств зарезервирована, чтобы портфель не уходил в минус.",
)

st.caption(
    "Настройки защиты можно изменить в разделах 🧠 AI-Трейдер и 🧭 Простой режим. Включённая опция DRY-RUN гарантирует демонстрационный режим без реальных ордеров."
)

st.divider()

with st.container(border=True):
    st.markdown("#### 🚀 Быстрые действия")
    quick_cols = st.columns(2)
    quick_actions = [
        ("🔌 Подключение и состояние", "pages/00_✅_Подключение_и_Состояние.py"),
        ("📈 AI-скринер рынка", "pages/01_📈_Скринер.py"),
        ("🎯 Смарт сделки OCO", "pages/04_🎯_Смарт_Сделки_OCO.py"),
        ("🧮 Управление риском портфеля", "pages/05_🧮_Portfolio_Risk_Spot.py"),
        ("📊 Портфельный дашборд", "pages/06_📊_Портфель_Дашборд.py"),
        ("🪵 Логи и уведомления", "pages/07_🪵_Логи.py"),
    ]
    for col, actions in zip(quick_cols, (quick_actions[:3], quick_actions[3:])):
        for label, page in actions:
            if col.button(label, use_container_width=True, key=f"quick_{page}"):
                st.switch_page(page)
    st.caption("Самые частые шаги вынесены сюда, чтобы вы быстрее переходили к анализу и действиям.")

st.divider()

st.markdown("#### 🧭 Режимы работы")
mode_tabs = st.tabs(["AI-трейдер", "Простой режим", "Инфраструктура"])
with mode_tabs[0]:
    st.success(
        "AI-модуль управляет OCO и TWAP, пересчитывает лимиты, следит за комиссиями и подстраивает стратегию под текущий рынок.")
    st.write(
        "- Планируйте сделки по сигналам модели и следите за качеством исполнения в разделе OCO.\n"
        "- Используйте лаборатории AI, чтобы откалибровать допущения и скоринг инструментов."
    )
with mode_tabs[1]:
    st.info(
        "Простой режим помогает быстро выставить защитные ордера и управлять капиталом вручную без избыточных настроек.")
    st.write(
        "- Настройте лимиты и kill-switch в разделе Order Hygiene.\n"
        "- Оцените момент входа через скринер и портфельные дашборды."
    )
with mode_tabs[2]:
    st.warning(
        "Следите за состоянием API, синхронизацией времени и WebSocket: стабильность канала важна для мгновенного исполнения.")
    st.write(
        "- Проверьте разделы WebSocket Status и Health TimeSync.\n"
        "- Настройте Telegram-бота, чтобы не пропустить критические события.")

st.write("Файлы приложения:", APP_ROOT)
st.write("Используйте меню слева для разделов. Начните со страницы **Подключение и состояние**.")
