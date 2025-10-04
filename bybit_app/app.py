
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from bybit_app.utils.ui import (
    build_status_card,
    inject_css,
    navigation_link,
    safe_set_page_config,
)
from bybit_app.utils.envs import get_settings
from bybit_app.utils.guardian_bot import GuardianBot

safe_set_page_config(page_title="Bybit Spot Guardian", page_icon="🧠", layout="centered")

MINIMAL_CSS = """
:root { color-scheme: dark; }
.block-container { max-width: 900px; padding-top: 1.5rem; }
.bybit-card { border-radius: 18px; border: 1px solid rgba(148, 163, 184, 0.2); padding: 1.2rem 1.4rem; background: rgba(15, 23, 42, 0.35); }
.bybit-card h3 { margin-bottom: 0.6rem; }
.shortcut-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin-top: 0.8rem; }
.shortcut { display: block; border-radius: 14px; padding: 0.85rem 1rem; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.28); font-weight: 600; text-align: left; }
.shortcut small { display: block; font-weight: 400; opacity: 0.75; margin-top: 0.2rem; }
.stButton>button { width: 100%; border-radius: 14px; padding: 0.7rem 1rem; font-weight: 600; }
.stMetric { border-radius: 12px; padding: 0.4rem 0.6rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.pill-row span { background: rgba(148, 163, 184, 0.22); border-radius: 999px; padding: 0.3rem 0.75rem; font-size: 0.85rem; font-weight: 600; }
[data-testid="stPageLinkContainer"] { margin-top: 0.35rem; }
[data-testid="stPageLinkContainer"] a, .bybit-shortcut {
    display: block;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    font-weight: 600;
    text-decoration: none;
    color: inherit;
}
[data-testid="stPageLinkContainer"] a:hover, .bybit-shortcut:hover {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.18);
}
[data-testid="stPageLinkContainer"] a:focus, .bybit-shortcut:focus {
    outline: 2px solid rgba(16, 185, 129, 0.6);
}
"""

inject_css(MINIMAL_CSS)


def render_header() -> None:
    st.title("Bybit Spot Guardian")
    st.caption(
        "Центр управления умным спотовым ботом: статус подключения, аналитика и тихие помощники."
    )
    st.markdown(
        """
        <div class="pill-row">
            <span>🛡 Контроль рисков</span>
            <span>⚡ Быстрый запуск</span>
            <span>📊 Чёткий статус</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(settings) -> None:
    ok = bool(settings.api_key and settings.api_secret)
    status = build_status_card(
        "Ключи подключены" if ok else "Добавьте API ключи",
        "Готовы к размещению ордеров." if ok else "Введите ключ и секрет в разделе подключения.",
        icon="🔐" if ok else "⚠️",
        tone="success" if ok else "warning",
    )
    with st.container(border=True):
        st.markdown(status, unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Сеть", "Testnet" if settings.testnet else "Mainnet")
        col_b.metric("Режим", "DRY-RUN" if settings.dry_run else "Live")
        cap_guard = getattr(settings, "spot_cash_reserve_pct", 10.0)
        col_c.metric("Резерв кэша", f"{cap_guard:.0f}%")
        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if settings.api_key else '❌'} · Secret: {'✅' if settings.api_secret else '❌'} · Настройки обновлены: {last_update}"
        )


def render_onboarding() -> None:
    st.subheader("Первые шаги")
    st.markdown(
        """
        1. Откройте раздел **«Подключение и состояние»** и сохраните API ключи.
        2. Загляните на страницу **«Дружелюбный спот-бот»**, чтобы увидеть текущий сигнал и план действий.
        3. Используйте **«Мониторинг сделок»** для контроля исполнений и результата.
        4. Дополнительные помощники (Telegram, журналы) спрятаны в блоке **«Скрытые инструменты»** ниже.
        """
    )


def render_shortcuts() -> None:
    st.subheader("Основные разделы")
    shortcuts = [
        (
            "🔌 Подключение",
            "pages/00_✅_Подключение_и_Состояние.py",
            "API ключи, проверка связи и режим DRY-RUN.",
        ),
        (
            "🛡 Спот-бот",
            "pages/03_🛡_Спот_Бот.py",
            "Актуальный сигнал, риск и чат с ботом.",
        ),
        (
            "📈 Мониторинг",
            "pages/05_📈_Мониторинг_Сделок.py",
            "Последние сделки и состояние портфеля.",
        ),
    ]

    columns = st.columns(len(shortcuts))
    for column, shortcut in zip(columns, shortcuts):
        label, page, description = shortcut
        with column:
            navigation_link(page, label=label)
            st.caption(description)


def render_data_health() -> None:
    bot = GuardianBot()
    health = bot.data_health()

    with st.container(border=True):
        st.subheader("Диагностика бота")
        st.caption(
            "Следим за свежестью сигнала, журналом исполнений и подключением API, чтобы не пропускать проблемы."
        )
        for key in ("ai_signal", "executions", "api_keys"):
            info = health.get(key, {})
            if not info:
                continue
            icon = "✅" if info.get("ok") else "⚠️"
            title = info.get("title", key)
            message = info.get("message", "")
            st.markdown(f"{icon} **{title}** — {message}")
            details = info.get("details")
            if details:
                st.caption(details)


def render_hidden_tools() -> None:
    with st.expander("🫥 Скрытые инструменты для бота"):
        st.caption(
            "Продвинутые аналитические и инженерные панели доступны здесь, чтобы не перегружать основной сценарий."
        )

        groups = [
            (
                "Рынок и сигналы",
                [
                    ("📈 Скринер", "pages/01_📈_Скринер.py", "Топ ликвидных пар и волатильность."),
                    (
                        "🌐 Universe Builder",
                        "pages/01d_🌐_Universe_Builder_Spot.py",
                        "Подбор и фильтрация тикеров для спот-бота.",
                    ),
                    ("🧠 AI трейдер", "pages/03_🧠_AI_Трейдер.py", "Подробный срез сигнала AI."),
                    ("🧪 AI Lab", "pages/03b_🧪_AI_Lab.py", "Эксперименты и симуляции стратегий."),
                    (
                        "🧪 EV Impact",
                        "pages/03c_🧪_AI_Lab_EV_Impact.py",
                        "Как изменяется ожидаемая доходность от параметров.",
                    ),
                    ("🎯 Порог покупки", "pages/03d_🎯_Buy_Threshold_Tuner.py", "Калибровка триггеров входа."),
                    ("🧭 Простой режим", "pages/00_🧭_Простой_режим.py", "Обучающий обзор для новичков."),
                ],
            ),
            (
                "Риск и безопасность",
                [
                    ("⚙️ Настройки", "pages/02_⚙️_Настройки.py", "Глобальные параметры бота."),
                    ("🛑 KillSwitch", "pages/02c_🛑_KillSwitch_and_API_Nanny.py", "Быстрые предохранители."),
                    ("🧽 Гигиена ордеров", "pages/02d_🧽_Order_Hygiene_Spot.py", "Чистка зависших заявок."),
                    ("📏 Лимиты ордеров", "pages/02e_📏_Spot_Order_Limits.py", "Контроль размеров и частоты."),
                    ("🧮 Риск портфеля", "pages/05_🧮_Portfolio_Risk_Spot.py", "Декомпозиция риска по позициям."),
                    ("🧭 HRP vs VolTarget", "pages/05b_🧭_HRP_vs_VolTarget_Spot.py", "Сравнение ребалансировок."),
                    ("⚡ WS контроль", "pages/05_⚡_WS_Контроль.py", "Статус real-time соединений."),
                    ("🕸️ WS монитор", "pages/05b_🕸️_WS_Монитор.py", "Трафик и задержки WebSocket."),
                    ("🧰 Reconcile", "pages/09_🧰_Reconcile.py", "Сверка позиций и журналов."),
                    ("⚙️ Здоровье", "pages/11_⚙️_Здоровье_и_Статус.py", "Диагностика инфраструктуры."),
                    ("🩺 Time Sync", "pages/00c_🩺_Health_TimeSync.py", "Синхронизация времени и задержки."),
                ],
            ),
            (
                "Торговые инструменты",
                [
                    ("🧩 TWAP", "pages/04c_🧩_TWAP_Spot.py", "Пакетное исполнение крупного объёма."),
                    ("⚡ Live OB Impact", "pages/04d_⚡_Live_OB_Impact_Spot.py", "Воздействие на стакан в режиме live."),
                    ("🧪 Impact Analyzer", "pages/04d_🧪_Impact_Analyzer_Spot.py", "Аналитика влияния сделок."),
                    ("🧠 EV Tuner", "pages/04e_🧠_EV_Tuner_Spot.py", "Оптимизация ожидаемой доходности."),
                    ("🔁 Правила", "pages/04f_🔁_Rules_Refresher_Spot.py", "Напоминания по дисциплине."),
                    ("🧰 Overrides", "pages/04g_🧰_Overrides_Spot.py", "Ручные корректировки сигналов."),
                    ("🌊 Liquidity", "pages/04h_🌊_Liquidity_Sampler_Spot.py", "Замер ликвидности по бирже."),
                    ("🔗 Trade Pairs", "pages/06_🔗_Trade_Pairs_Spot.py", "Связанные пары для хеджей."),
                ],
            ),
            (
                "PnL и отчётность",
                [
                    ("💰 PnL дашборд", "pages/06_💰_PnL_Дашборд.py", "История доходности и метрики."),
                    ("📊 Портфель", "pages/06_📊_Портфель_Дашборд.py", "Структура активов и динамика."),
                    ("💰 PnL мониторинг", "pages/10_💰_PnL_Мониторинг.py", "Детальный журнал сделок."),
                    ("📉 Shortfall", "pages/10b_📉_Shortfall_Report.py", "Контроль просадок и недополученной прибыли."),
                ],
            ),
            (
                "Коммуникации",
                [
                    ("🤖 Telegram", "pages/06_🤖_Telegram_Бот.py", "Настройка уведомлений и heartbeat."),
                    ("🪵 Логи", "pages/07_🪵_Логи.py", "Журнал действий и системных сообщений."),
                ],
            ),
        ]

        for title, items in groups:
            st.markdown(f"#### {title}")
            for idx in range(0, len(items), 3):
                row = items[idx : idx + 3]
                cols = st.columns(len(row))
                for column, shortcut in zip(cols, row):
                    label, page, description = shortcut
                    with column:
                        navigation_link(page, label=label)
                        st.caption(description)


def render_tips(settings) -> None:
    with st.container(border=True):
        st.markdown("### Быстрые подсказки")
        st.markdown(
            """
            - DRY-RUN оставляет заявки в журналах, не отправляя их на биржу.
            - Убедитесь, что резерв кэша не опускается ниже 10%, чтобы торговля оставалась устойчивой.
            - Если требуется автоматизация, используйте Guardian Bot — он уже настроен на защиту депозита.
            - За уведомления отвечает Telegram-бот: включите его в блоке «Скрытые инструменты».
            """
        )
        if settings.dry_run:
            st.info("DRY-RUN активен: безопасно тестируйте стратегии перед реальной торговлей.")
        else:
            st.warning("DRY-RUN выключен. Проверьте лимиты риска перед запуском торговых сценариев.")


def main() -> None:
    settings = get_settings()

    render_header()
    st.divider()
    render_status(settings)
    st.divider()
    render_onboarding()
    st.divider()
    render_shortcuts()
    st.divider()
    render_data_health()
    st.divider()
    render_hidden_tools()
    st.divider()
    render_tips(settings)


if __name__ == "__main__":
    main()
