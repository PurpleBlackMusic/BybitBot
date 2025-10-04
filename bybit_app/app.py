
from __future__ import annotations

from textwrap import dedent

import streamlit as st

from bybit_app.utils.ui import (
    safe_set_page_config,
    inject_css,
    build_pill,
    build_status_card,
)
from bybit_app.utils.paths import APP_ROOT
from bybit_app.utils.envs import get_settings

safe_set_page_config(page_title="Bybit Smart OCO — PRO", page_icon="🧠", layout="wide")

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.block-container { padding: 1.6rem 2.4rem 3rem; max-width: 1240px; }
[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,118,110,0.75)); }
[data-testid="stSidebar"] nav { padding-top: 0.5rem; }
.stButton>button {
    border-radius: 14px;
    padding: 0.75rem 1.1rem;
    font-weight: 600;
    background: linear-gradient(120deg, rgba(16,185,129,0.9), rgba(45,212,191,0.85));
    border: none;
    color: white;
    box-shadow: 0 12px 30px rgba(45,212,191,0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 18px 40px rgba(45,212,191,0.32); }
.stTabs [role="tablist"] { gap: 0.6rem; }
.stTabs [role="tab"] {
    padding: 0.6rem 1.35rem;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.12);
    color: rgba(226, 232, 240, 0.9);
    border: 1px solid transparent;
    transition: background 0.2s ease, color 0.2s ease, border 0.2s ease;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(16,185,129,0.95), rgba(45,212,191,0.85));
    color: white;
    border-color: rgba(16,185,129,0.55);
    box-shadow: 0 14px 32px rgba(16,185,129,0.25);
}
.stMetric {
    background: rgba(15,118,110,0.12);
    border-radius: 18px;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(45,212,191,0.35);
}
.metric-subtitle { font-size: 0.8rem; opacity: 0.65; margin-top: 0.2rem; }
.bybit-hero {
    background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,118,110,0.88));
    color: white;
    padding: 1.6rem 1.9rem;
    border-radius: 22px;
    box-shadow: 0 20px 52px rgba(15, 118, 110, 0.28);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.bybit-hero::after {
    content: "";
    position: absolute;
    inset: -40% 40% auto auto;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(56,189,248,0.45) 0%, rgba(15,118,110,0) 70%);
    opacity: 0.7;
}
.bybit-hero h1 { margin-bottom: 0.4rem; font-size: 2.4rem; }
.bybit-hero__sub { opacity: 0.9; font-size: 1.05rem; max-width: 760px; }
.hero-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.85rem; position: relative; z-index: 1; }
.hero-grid .bybit-pill { background: rgba(255,255,255,0.18); color: white; }
.hero-grid .bybit-pill.bybit-pill--success { background: rgba(45,212,191,0.28); color: #0f172a; }
.hero-grid .bybit-pill.bybit-pill--warning { background: rgba(250,204,21,0.3); color: #0f172a; }
.hero-grid .bybit-pill.bybit-pill--danger { background: rgba(248,113,113,0.3); color: #0f172a; }
.quick-actions__desc { font-size: 0.85rem; opacity: 0.75; margin-top: 0.45rem; }
@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1.2rem 2.4rem; }
    .bybit-hero { padding: 1.35rem 1.4rem; }
    .bybit-hero h1 { font-size: 2rem; }
}
"""

inject_css(GLOBAL_CSS)

def render_hero() -> None:
    hero_badges = [
        build_pill("Реакция < 1s", icon="⚡"),
        build_pill("AI OCO & TWAP", icon="🧠"),
        build_pill("Risk Guards", icon="🛡", tone="success"),
        build_pill("Telegram Ping", icon="🔔"),
    ]

    st.markdown(
        dedent(
            f"""
            <div class="bybit-hero">
                <h1>Bybit Smart OCO — PRO</h1>
                <p class="bybit-hero__sub">Умные торговые сценарии, прозрачные статусы и контроль рисков для трейдера, который ценит скорость принятия решений.</p>
                <div class="hero-grid">{''.join(hero_badges)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.caption("Улучшенная 3Commas: умный OCO, понятный интерфейс, живые статусы.")


def render_mission() -> None:
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


def render_status_section(settings) -> None:
    ok = bool(settings.api_key and settings.api_secret)
    with st.container(border=True):
        st.markdown("#### ⚙️ Технический статус профиля")
        col_a, col_b = st.columns([1.2, 1])
        status_title = "Все ключи подключены" if ok else "Требуется подключение API"
        status_hint = (
            "Готовы к торговле и автоматическим стратегиям."
            if ok
            else "Добавьте API ключ и секрет, чтобы активировать торговлю."
        )
        status_html = build_status_card(
            status_title,
            status_hint,
            icon="🔐" if ok else "⚠️",
            tone="success" if ok else "warning",
        )
        with col_a:
            st.markdown(status_html, unsafe_allow_html=True)
        with col_b:
            st.metric("Режим", "Testnet" if settings.testnet else "Mainnet", help="Переключите сеть в настройках окружения.")
            st.metric(
                "DRY-RUN",
                "ON" if settings.dry_run else "OFF",
                help="Включите DRY-RUN, чтобы проверять стратегию без реальных ордеров.",
            )

        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M:%S") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if settings.api_key else '❌'} · Secret: {'✅' if settings.api_secret else '❌'} · Последнее обновление настроек: {last_update}"
        )


def _render_metric(column, *, label: str, value: str, help_text: str, hint: str) -> None:
    column.metric(label, value, help=help_text)
    column.markdown(
        f"<div class='metric-subtitle'>{hint}</div>",
        unsafe_allow_html=True,
    )


def _risk_metrics(settings) -> list[dict[str, str]]:
    cap_guard = 100 - float(getattr(settings, "spot_cash_reserve_pct", 10.0) or 0.0)
    return [
        {
            "label": "Риск на сделку",
            "value": f"{getattr(settings, 'ai_risk_per_trade_pct', 0.25):.2f}%",
            "help_text": "Максимальная доля капитала, которую бот рискует в одной сделке.",
            "hint": "Рекомендуется ≤ 0.50% для спокойной торговли.",
        },
        {
            "label": "Дневной лимит убытка",
            "value": f"{getattr(settings, 'ai_daily_loss_limit_pct', 3.0):.2f}%",
            "help_text": "При достижении порога торговля ставится на паузу.",
            "hint": "После паузы проверьте рынок и скорректируйте риск.",
        },
        {
            "label": "Задействованный капитал",
            "value": f"≤ {cap_guard:.0f}%",
            "help_text": "Часть средств зарезервирована, чтобы портфель не уходил в минус.",
            "hint": "Резерв помогает пережить повышенную волатильность.",
        },
    ]


def render_risk_controls(settings) -> None:
    st.subheader("🛡 Контроль капитала")
    risk_cols = st.columns(3)
    for column, metric in zip(risk_cols, _risk_metrics(settings)):
        _render_metric(column, **metric)
    st.caption(
        "Настройки защиты можно изменить в разделах 🧠 AI-Трейдер и 🧭 Простой режим. Включённая опция DRY-RUN гарантирует демонстрационный режим без реальных ордеров."
    )


_QUICK_ACTIONS = [
    {
        "label": "🔌 Подключение и состояние",
        "page": "pages/00_✅_Подключение_и_Состояние.py",
        "description": "Проверьте API-ключи, синхронизацию времени и доступ к бирже.",
    },
    {
        "label": "📈 AI-скринер рынка",
        "page": "pages/01_📈_Скринер.py",
        "description": "Получите свежие сигналы и тепловые карты волатильности.",
    },
    {
        "label": "🎯 Смарт сделки OCO",
        "page": "pages/04_🎯_Смарт_Сделки_OCO.py",
        "description": "Запускайте умные OCO-сценарии и проверяйте качество исполнения.",
    },
    {
        "label": "🧮 Управление риском портфеля",
        "page": "pages/05_🧮_Portfolio_Risk_Spot.py",
        "description": "Выравнивайте позиции и лимиты риска по выбранной методике.",
    },
    {
        "label": "📊 Портфельный дашборд",
        "page": "pages/06_📊_Портфель_Дашборд.py",
        "description": "Следите за PnL, распределением активов и динамикой портфеля.",
    },
    {
        "label": "🪵 Логи и уведомления",
        "page": "pages/07_🪵_Логи.py",
        "description": "Проверьте историю действий, алертов и уведомлений Telegram.",
    },
]


def render_quick_actions() -> None:
    with st.container(border=True):
        st.markdown("#### 🚀 Быстрые действия")
        quick_cols = st.columns(3, gap="large")
        for idx, action in enumerate(_QUICK_ACTIONS):
            column = quick_cols[idx % len(quick_cols)]
            with column.container(border=True):
                if st.button(action["label"], use_container_width=True, key=f"quick_{action['page']}"):
                    st.switch_page(action["page"])
                st.markdown(
                    f"<div class='quick-actions__desc'>{action['description']}</div>",
                    unsafe_allow_html=True,
                )
        st.caption("Самые частые шаги вынесены сюда, чтобы вы быстрее переходили к анализу и действиям.")


def render_modes_section() -> None:
    st.markdown("#### 🧭 Режимы работы")
    mode_tabs = st.tabs(["AI-трейдер", "Простой режим", "Инфраструктура"])
    with mode_tabs[0]:
        st.success(
            "AI-модуль управляет OCO и TWAP, пересчитывает лимиты, следит за комиссиями и подстраивает стратегию под текущий рынок."
        )
        st.write(
            "- Планируйте сделки по сигналам модели и следите за качеством исполнения в разделе OCO.\n"
            "- Используйте лаборатории AI, чтобы откалибровать допущения и скоринг инструментов."
        )
    with mode_tabs[1]:
        st.info(
            "Простой режим помогает быстро выставить защитные ордера и управлять капиталом вручную без избыточных настроек."
        )
        st.write(
            "- Настройте лимиты и kill-switch в разделе Order Hygiene.\n"
            "- Оцените момент входа через скринер и портфельные дашборды."
        )
    with mode_tabs[2]:
        st.warning(
            "Следите за состоянием API, синхронизацией времени и WebSocket: стабильность канала важна для мгновенного исполнения."
        )
        st.write(
            "- Проверьте разделы WebSocket Status и Health TimeSync.\n"
            "- Настройте Telegram-бота, чтобы не пропустить критические события."
        )


def render_footer() -> None:
    st.write("Файлы приложения:", APP_ROOT)
    st.write("Используйте меню слева для разделов. Начните со страницы **Подключение и состояние**.")


def main() -> None:
    settings = get_settings()

    render_hero()
    render_mission()
    render_status_section(settings)
    st.divider()

    render_risk_controls(settings)
    st.divider()

    render_quick_actions()
    st.divider()

    render_modes_section()
    render_footer()


if __name__ == "__main__":
    main()
