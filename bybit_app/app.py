
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from bybit_app.utils.ui import (
    build_pill,
    build_status_card,
    inject_css,
    navigation_link,
    safe_set_page_config,
)
from bybit_app.utils.envs import get_settings
from bybit_app.utils.guardian_bot import GuardianBot, GuardianBrief

safe_set_page_config(page_title="Bybit Spot Guardian", page_icon="🧠", layout="centered")

MINIMAL_CSS = """
:root { color-scheme: dark; }
.block-container { max-width: 900px; padding-top: 1.5rem; }
.bybit-card { border-radius: 18px; border: 1px solid rgba(148, 163, 184, 0.2); padding: 1.2rem 1.4rem; background: rgba(15, 23, 42, 0.35); }
.bybit-card h3 { margin-bottom: 0.6rem; }
.stButton>button { width: 100%; border-radius: 14px; padding: 0.7rem 1rem; font-weight: 600; }
.stMetric { border-radius: 12px; padding: 0.4rem 0.6rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.pill-row span { background: rgba(148, 163, 184, 0.22); border-radius: 999px; padding: 0.3rem 0.75rem; font-size: 0.85rem; font-weight: 600; }
[data-testid="stTabs"] { margin-top: 0.6rem; }
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
.signal-card { display: flex; flex-direction: column; gap: 0.55rem; }
.signal-card__badge { display: flex; gap: 0.45rem; align-items: center; }
.signal-card__symbol { font-weight: 600; opacity: 0.8; }
.signal-card__headline { font-size: 1.05rem; font-weight: 700; }
.signal-card__body { font-size: 0.95rem; line-height: 1.45; opacity: 0.92; }
.signal-card__footer { display: flex; flex-wrap: wrap; gap: 0.6rem; font-size: 0.85rem; opacity: 0.75; }
.checklist { list-style: decimal; padding-left: 1.15rem; line-height: 1.5; }
.checklist li { margin-bottom: 0.35rem; }
.safety-list { list-style: disc; padding-left: 1.1rem; line-height: 1.5; }
.safety-list li { margin-bottom: 0.3rem; }
"""

inject_css(MINIMAL_CSS)


@st.cache_resource(show_spinner=False)
def _load_guardian_bot() -> GuardianBot:
    return GuardianBot()


def get_bot() -> GuardianBot:
    """Return a cached GuardianBot instance."""

    return _load_guardian_bot()


def render_navigation_grid(shortcuts: list[tuple[str, str, str]], *, columns: int = 2) -> None:
    """Render navigation links in a compact grid layout."""

    if not shortcuts:
        return

    for idx in range(0, len(shortcuts), columns):
        row = shortcuts[idx : idx + columns]
        cols = st.columns(len(row))
        for column, shortcut in zip(cols, row):
            label, page, description = shortcut
            with column:
                navigation_link(page, label=label)
                st.caption(description)


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
        status_col, metrics_col = st.columns([2, 1])
        with status_col:
            st.markdown(status, unsafe_allow_html=True)
        with metrics_col:
            st.metric("Сеть", "Testnet" if settings.testnet else "Mainnet")
            st.metric("Режим", "DRY-RUN" if settings.dry_run else "Live")
            reserve = getattr(settings, "spot_cash_reserve_pct", 10.0)
            st.metric("Резерв кэша", f"{reserve:.0f}%")

        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if settings.api_key else '❌'} · Secret: {'✅' if settings.api_secret else '❌'} · Настройки обновлены: {last_update}"
        )

        if not ok:
            st.warning(
                "Без API ключей бот не сможет размещать ордера. Перейдите в раздел «Подключение» и добавьте их."
            )


def _mode_meta(mode: str) -> tuple[str, str, str]:
    mapping: dict[str, tuple[str, str, str]] = {
        "buy": ("Покупка", "🟢", "success"),
        "sell": ("Продажа", "🔴", "warning"),
        "wait": ("Наблюдаем", "⏸", "neutral"),
    }
    return mapping.get(mode, ("Наблюдаем", "⏸", "neutral"))


def render_signal_brief(bot: GuardianBot) -> GuardianBrief:
    brief = bot.generate_brief()
    score = bot.signal_scorecard(brief)
    settings = bot.settings
    mode_label, mode_icon, tone = _mode_meta(brief.mode)

    st.subheader("Сводка сигнала")
    with st.container(border=True):
        st.markdown(
            """
            <div class="signal-card__badge">
                {pill}<span class="signal-card__symbol">· {symbol}</span>
            </div>
            """.format(
                pill=build_pill(mode_label, icon=mode_icon, tone=tone),
                symbol=brief.symbol,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__headline'>{brief.headline}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.analysis}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.action_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.confidence_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.ev_text}</div>",
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Вероятность",
            f"{score['probability_pct']:.1f}%",
            f"Порог {score['buy_threshold']:.0f}%",
        )
        metric_cols[1].metric(
            "Потенциал",
            f"{score['ev_bps']:.1f} б.п.",
            f"Мин. {score['min_ev_bps']:.1f} б.п.",
        )
        trade_mode = "DRY-RUN" if settings.dry_run else "Live"
        metric_cols[2].metric("Тактика", mode_label, trade_mode)
        st.caption(f"Обновление: {score['last_update']}")

    if brief.caution:
        st.warning(brief.caution)
    if brief.status_age and brief.status_age > 300:
        st.error(
            "Сигнал не обновлялся более пяти минут — проверьте соединение с данными или перезапустите источник."
        )

    return brief


def render_onboarding() -> None:
    st.subheader("Первые шаги")
    st.markdown(
        """
        1. Откройте раздел **«Подключение и состояние»** и сохраните API ключи.
        2. Загляните в **«Простой режим»**, чтобы увидеть текущий сигнал, план действий и чат с ботом.
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
            "🧭 Простой режим",
            "pages/00_🧭_Простой_режим.py",
            "Актуальный сигнал, план и чат с ботом.",
        ),
        (
            "📈 Мониторинг",
            "pages/05_📈_Мониторинг_Сделок.py",
            "Последние сделки и состояние портфеля.",
        ),
    ]

    render_navigation_grid(shortcuts, columns=3)


def render_data_health(bot: GuardianBot) -> None:
    health = bot.data_health()

    with st.container(border=True):
        st.subheader("Диагностика бота")
        st.caption(
            "Следим за свежестью сигнала, журналом исполнений и подключением API, чтобы не пропускать проблемы."
        )
        cards: list[tuple[str, str, str, str]] = []
        for key in ("ai_signal", "executions", "api_keys"):
            info = health.get(key, {})
            if not info:
                continue
            tone = "success" if info.get("ok") else "warning"
            icon = "✅" if info.get("ok") else "⚠️"
            title = info.get("title", key)
            message = info.get("message", "")
            cards.append((title, message, icon, tone))

        if not cards:
            st.caption("Пока нет диагностических данных.")
            return

        cols = st.columns(min(3, len(cards)))
        for column, (title, message, icon, tone) in zip(cols, cards):
            with column:
                st.markdown(
                    build_status_card(title, message, icon=icon, tone=tone),
                    unsafe_allow_html=True,
                )


def render_market_watchlist(bot: GuardianBot) -> None:
    st.subheader("Наблюдаемые активы")
    items = bot.market_watchlist()

    if not items:
        st.caption("Пока нет тикеров в списке наблюдения — бот ждёт новый сигнал.")
        return

    st.dataframe(items, hide_index=True, use_container_width=True)


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

        tab_titles = [title for title, _ in groups]
        tabs = st.tabs(tab_titles)

        for tab, (_, items) in zip(tabs, groups):
            with tab:
                render_navigation_grid(items)


def render_action_plan(bot: GuardianBot, brief: GuardianBrief) -> None:
    steps = bot.plan_steps(brief)
    notes = bot.safety_notes()

    plan_html = "".join(f"<li>{step}</li>" for step in steps)
    safety_html = "".join(f"<li>{note}</li>" for note in notes)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Что делаем дальше")
        st.markdown(f"<ol class='checklist'>{plan_html}</ol>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("#### Памятка безопасности")
        st.markdown(f"<ul class='safety-list'>{safety_html}</ul>", unsafe_allow_html=True)
        st.caption(bot.risk_summary().replace("\n", "  \n"))


def render_guides(settings, bot: GuardianBot, brief: GuardianBrief) -> None:
    st.subheader("Поддержка и советы")
    plan_tab, onboarding_tab, tips_tab = st.tabs(["План действий", "Первые шаги", "Подсказки"])

    with plan_tab:
        render_action_plan(bot, brief)

    with onboarding_tab:
        render_onboarding()

    with tips_tab:
        render_tips(settings, brief)


def render_tips(settings, brief: GuardianBrief) -> None:
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
        if brief.status_age and brief.status_age > 300:
            st.error("Данные сигнала устарели. Проверьте, что пайплайн сигналов работает корректно.")
        if not (settings.api_key and settings.api_secret):
            st.warning("API ключи не добавлены: без них торговля невозможна.")


def main() -> None:
    settings = get_settings()
    bot = get_bot()

    render_header()
    st.divider()
    render_status(settings)
    st.divider()
    brief = render_signal_brief(bot)
    st.divider()
    render_shortcuts()
    st.divider()
    render_data_health(bot)
    st.divider()
    render_market_watchlist(bot)
    st.divider()
    render_guides(settings, bot, brief)
    st.divider()
    render_hidden_tools()


if __name__ == "__main__":
    main()
