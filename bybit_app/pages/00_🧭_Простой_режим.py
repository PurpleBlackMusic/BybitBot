from __future__ import annotations

import json
import time
from typing import Any, Callable

import streamlit as st

from utils.ai.live import AIRunner
from utils.coach import build_autopilot_settings, market_health
from utils.envs import get_api_client, get_settings, update_settings
from utils.paths import DATA_DIR
from utils.reporter import send_daily_report, send_test_message, summarize_today
from utils.scheduler import (
    _load_state_file,
    _save_state_file,
    start_background_loop,
)


def _settings_attr(settings: Any, name: str, default: Any) -> Any:
    """Return a settings attribute with a convenient default."""

    return getattr(settings, name, default)


def _with_spinner(label: str, callback: Callable[[], Any]) -> Any:
    """Run ``callback`` while showing a spinner and surface exceptions."""

    with st.spinner(label):
        return callback()


@st.cache_resource(show_spinner=False)
def _get_cached_api_client():
    """Reuse a single API client instance between reruns."""

    return get_api_client()


@st.cache_data(ttl=120, show_spinner=False)
def _load_market_health_cached() -> dict[str, Any]:
    """Fetch and cache market health hints for the briefing block."""

    api = _get_cached_api_client()
    return market_health(api, category="spot")


@st.cache_data(ttl=60, show_spinner=False)
def _load_today_summary_cached() -> dict[str, Any]:
    """Cache the daily summary to avoid touching the log files on each rerun."""

    return summarize_today()


st.set_page_config(page_title="Простой режим", page_icon="🧭", layout="wide")
st.title("🧭 Простой режим")

st.caption(
    "Эта страница для тех, кто НЕ хочет разбираться в крипте и настройках. "
    "Здесь — краткая подсказка на сегодня и **одна кнопка**, чтобы запустить умного бота."
)

s = get_settings()


def _persist_settings(**kwargs: Any) -> None:
    """Update settings while keeping Telegram defaults in sync."""

    current = get_settings()
    payload = dict(kwargs)
    payload.setdefault(
        "tg_trade_notifs",
        bool(_settings_attr(current, "tg_trade_notifs", False)),
    )
    payload.setdefault(
        "tg_trade_notifs_min_notional",
        float(_settings_attr(current, "tg_trade_notifs_min_notional", 50.0)),
    )
    update_settings(**payload)


def _persist_with_feedback(message: str, **kwargs: Any) -> None:
    """Persist settings while showing success/error feedback."""

    try:
        _persist_settings(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.error(f"Не удалось сохранить настройки: {exc}")
    else:
        st.success(message)


api = _get_cached_api_client()
# фоновая автоматика
start_background_loop()


def _render_briefing() -> None:
    st.subheader("Сегодняшний брифинг")
    refresh = st.button("🔄 Обновить брифинг", key="refresh_briefing")
    try:
        if refresh:
            _load_market_health_cached.clear()
            info = _with_spinner("Обновляем брифинг...", _load_market_health_cached)
        else:
            info = _load_market_health_cached()
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.warning(f"Не удалось получить рыночные подсказки: {exc}")
        return

    light = info.get("light")
    reason = info.get("reason", "")
    cols = st.columns([1, 6])
    with cols[0]:
        st.metric(
            "Статус рынка",
            {"green": "✅ ОК", "yellow": "⚠️ Риск", "red": "⛔ Стоп"}.get(light, "—"),
        )
    with cols[1]:
        st.write(reason)

    st.caption("✅ ОК — можно торговать • ⚠️ Риск — аккуратно • ⛔ Стоп — лучше не торговать")

    top = info.get("top") or []
    with st.expander("Рекомендованные монеты на сегодня", expanded=False):
        if top:
            table = {
                "symbol": [row.get("symbol", "—") for row in top],
                "turnover24h": [row.get("turnover24h") for row in top],
                "spread (bps)": [row.get("spread_bps") for row in top],
            }
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("Подходящих монет пока нет — попробуйте обновить позже.")


def _render_autopilot(settings: Any, api_client: Any) -> None:
    st.subheader("Авто-бот (одной кнопкой)")
    left, right = st.columns([2, 1])

    with left:
        st.write(
            "Бот **сам подберёт монеты и параметры**, включит защиту капитала и TWAP, "
            "и запустится в фоне. Вы увидите статус и отчёты в разделе «Логи».",
        )
        runner: AIRunner = st.session_state.setdefault("ai_runner", AIRunner())

        if st.button("🤖 Автоподбор и запуск", use_container_width=True):
            try:
                pack = _with_spinner(
                    "Подбираем монеты и параметры...",
                    lambda: build_autopilot_settings(get_settings(), api_client),
                )
                _persist_settings(**pack["settings"])
                runner.start()
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Не удалось запустить: {exc}")
            else:
                st.success(
                    "Бот запущен. Прогнозная подготовка: ~"
                    f"{pack.get('eta_minutes', '—')} мин.",
                )

        col_stop, col_panic = st.columns(2)
        if col_stop.button("⏹ Остановить бота"):
            try:
                runner.stop()
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Ошибка остановки: {exc}")
            else:
                st.info("Бот остановлен.")

        if col_panic.button("🛑 Паник-стоп (до завтра)"):
            stop_error: Exception | None = None
            try:
                runner.stop()
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                stop_error = exc
                st.error(f"Ошибка остановки: {exc}")

            def _activate_panic() -> None:
                state = _load_state_file() or {}
                state["stop_day_locked"] = True
                state["stop_day_reason"] = "panic"
                state["stop_day_date"] = time.strftime("%Y-%m-%d")
                if not _save_state_file(state):
                    raise RuntimeError("state not saved")

            try:
                _activate_panic()
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Не удалось активировать паник-стоп: {exc}")
            else:
                if stop_error is None:
                    st.warning("Паник-стоп активирован: автозапуск заблокирован до завтра.")

        dry_default = bool(_settings_attr(settings, "dry_run", True))
        dry = st.toggle(
            "Демо-режим (без реальных ордеров)",
            value=dry_default,
            help="В демо-запуске бот **не отправляет** реальные заявки.",
        )
        if dry != dry_default:
            _persist_settings(dry_run=bool(dry))
            st.rerun()

    with right:
        st.write("**Статус бота**")
        status_path = DATA_DIR / "ai" / "status.json"
        try:
            if status_path.exists():
                st.json(json.loads(status_path.read_text(encoding="utf-8")))
            else:
                st.info("Пока статуса нет. Нажмите «Автоподбор и запуск».")
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            st.warning(f"Статус недоступен: {exc}")

    st.subheader("Что сегодня делает бот? (человеческим языком)")
    st.markdown(
        """
        - Выбирает **самые ликвидные** пары с узким спредом.
        - На каждой итерации смотрит **ленту цен/спред** и принимает **простое решение** купить/продать/подождать.
        - Риск на сделку ограничен, включены **ограничения на заём средств**, лимиты на символ и сделку.
        - **DRY RUN** выключается только когда вы снимете тумблер «Демо».
        - Все действия видно в **🪵 Логи** (ищите записи `ai.*`).
        """
    )


def _render_automation(settings: Any) -> None:
    st.subheader("📅 Автоматизация")
    with st.expander("Расписание: автозапуск/автостоп бота и дневной отчёт"):
        with st.form("auto_schedule_form"):
            col1, col2 = st.columns(2)
            with col1:
                auto_enabled = st.toggle(
                    "Включить авто-торговлю по расписанию",
                    value=bool(_settings_attr(settings, "auto_trade_enabled", False)),
                )
                start_time = st.text_input(
                    "Время авто-старта (чч:мм)",
                    value=str(_settings_attr(settings, "auto_start_time", "09:00")),
                )
                stop_time = st.text_input(
                    "Время авто-стопа (чч:мм)",
                    value=str(_settings_attr(settings, "auto_stop_time", "21:00")),
                )
                auto_dry = st.toggle(
                    "Торговать в демо-режиме при автозапуске",
                    value=bool(_settings_attr(settings, "auto_dry_run", True)),
                )
            with col2:
                report_enabled = st.toggle(
                    "Ежедневный отчёт в Telegram",
                    value=bool(_settings_attr(settings, "daily_report_enabled", False)),
                )
                report_time = st.text_input(
                    "Время отправки отчёта (чч:мм)",
                    value=str(_settings_attr(settings, "daily_report_time", "20:00")),
                )
                loss_limit = st.number_input(
                    "Дневной лимит убытка (%)",
                    value=float(_settings_attr(settings, "ai_daily_loss_limit_pct", 1.0)),
                    step=0.1,
                )
                profit_target = st.number_input(
                    "Дневная цель прибыли (%)",
                    value=float(_settings_attr(settings, "ai_daily_profit_target_pct", 0.0)),
                    step=0.1,
                )

            if st.form_submit_button("💾 Сохранить расписание"):
                _persist_with_feedback(
                    "Расписание сохранено. Фоновая автоматика уже работает.",
                    auto_trade_enabled=bool(auto_enabled),
                    auto_start_time=start_time,
                    auto_stop_time=stop_time,
                    auto_dry_run=bool(auto_dry),
                    daily_report_enabled=bool(report_enabled),
                    daily_report_time=report_time,
                    ai_daily_loss_limit_pct=float(loss_limit),
                    ai_daily_profit_target_pct=float(profit_target),
                )


def _render_universe_filters(settings: Any) -> None:
    st.subheader("⚙️ Доп. настройки универсума")
    with st.expander("Фильтрация монет (для автоподбора)"):
        with st.form("universe_filters_form"):
            whitelist = st.text_input(
                "Белый список монет (через запятую, опционально)",
                value=str(_settings_attr(settings, "ai_symbols_whitelist", "")),
                help="Если заполнено — автоподбор берёт монеты только из этого списка.",
            )
            blacklist = st.text_input(
                "Чёрный список монет (через запятую, опционально)",
                value=str(_settings_attr(settings, "ai_symbols_blacklist", "")),
                help="Эти монеты исключаются из автоподбора.",
            )
            manual = st.text_input(
                "Я сам задам монеты (перечисли через запятую)",
                value=str(_settings_attr(settings, "ai_symbols_manual", "")),
                help="Если указано — автоподбор возьмёт именно эти монеты.",
            )

            if st.form_submit_button("💾 Сохранить списки монет"):
                _persist_with_feedback(
                    "Списки сохранены.",
                    ai_symbols_whitelist=whitelist,
                    ai_symbols_blacklist=blacklist,
                    ai_symbols_manual=manual,
                )


def _render_export_import(_: Any) -> None:
    st.subheader("🗂 Экспорт/импорт настроек")
    col_export, col_import = st.columns(2)

    with col_export:
        st.caption("Скачайте текущую конфигурацию (подходит для резервной копии).")

        def _prepare_dump() -> bytes:
            data_obj = get_settings()
            try:
                data = data_obj.dict() if hasattr(data_obj, "dict") else data_obj.__dict__
            except Exception:  # pragma: no cover - defensive
                data = {}
            return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

        if st.button("⬇️ Подготовить JSON"):
            blob = _with_spinner("Готовим файл...", _prepare_dump)
            st.download_button(
                "Скачать settings.json",
                data=blob,
                file_name="settings.json",
                mime="application/json",
            )

    with col_import:
        st.caption("Загрузите ранее сохранённый файл.")
        uploaded = st.file_uploader("Загрузить settings.json", type=["json"])
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                _persist_settings(**payload)
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Ошибка импорта: {exc}")
            else:
                st.success("Настройки импортированы.")


def _render_telegram(settings: Any) -> None:
    st.subheader("🔔 Telegram-отчёты")
    with st.form("telegram_trade_notifs"):
        trade_notifs = st.checkbox(
            "Уведомления о сделках в Telegram",
            value=bool(_settings_attr(settings, "tg_trade_notifs", False)),
        )
        min_notional = st.number_input(
            "Минимальный объём сделки для уведомлений (USDT)",
            value=float(_settings_attr(settings, "tg_trade_notifs_min_notional", 50.0)),
            step=10.0,
        )
        if st.form_submit_button("💾 Сохранить уведомления"):
            _persist_with_feedback(
                "Настройки уведомлений сохранены.",
                tg_trade_notifs=bool(trade_notifs),
                tg_trade_notifs_min_notional=float(min_notional),
            )

    with st.expander("Настройки Telegram (для уведомлений и отчётов)"):
        with st.form("telegram_credentials_form"):
            token = st.text_input(
                "Bot Token",
                type="password",
                value=str(_settings_attr(settings, "telegram_token", "")),
            )
            chat_id = st.text_input(
                "Chat ID",
                value=str(_settings_attr(settings, "telegram_chat_id", "")),
            )
            notify = st.toggle(
                "Включить уведомления",
                value=bool(_settings_attr(settings, "telegram_notify", False)),
                help="Когда включено — приложение будет присылать короткие уведомления о старте/остановке и заявках.",
            )
            submitted = st.form_submit_button("✅ Сохранить Telegram-настройки")
            if submitted:
                _persist_with_feedback(
                    "Сохранено.",
                    telegram_token=token,
                    telegram_chat_id=chat_id,
                    telegram_notify=bool(notify),
                )

        if st.button("🧪 Отправить тестовое сообщение"):
            try:
                response = _with_spinner(
                    "Отправляем сообщение...",
                    lambda: send_test_message("Привет! Telegram настроен ✅"),
                )
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Не удалось отправить сообщение: {exc}")
            else:
                st.write(f"Ответ: {response}")


def _render_daily_report() -> None:
    st.subheader("Отчёт за сегодня")
    if st.button("🔄 Обновить сводку", key="refresh_summary"):
        _load_today_summary_cached.clear()

    try:
        summary = _load_today_summary_cached()
    except Exception as exc:  # pragma: no cover - defensive UI feedback
        st.error(f"Не удалось получить сводку: {exc}")
        summary = None

    if summary:
        st.write(
            "Событий: **{events}**, сигналов: **{signals}**, заявок: **{orders}**, ошибок: **{errors}**.".format(
                events=summary.get("events", 0),
                signals=summary.get("signals", 0),
                orders=summary.get("orders", 0),
                errors=summary.get("errors", 0),
            )
        )
    else:
        st.info("Пока нет данных за сегодня.")

    col_report, col_unlock = st.columns(2)
    if col_report.button("📤 Отправить отчёт в Telegram"):
        try:
            result = _with_spinner("Отправляем отчёт...", send_daily_report)
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            st.error(f"Не удалось отправить отчёт: {exc}")
        else:
            _load_today_summary_cached.clear()
            st.success(f"Отчёт отправлен: {result}")

    if col_unlock.button("🔓 Снять ‘стоп-день’ до завтра"):
        def _unlock() -> None:
            state = _load_state_file() or {}
            state["stop_day_locked"] = False
            state["stop_day_reason"] = ""
            if not _save_state_file(state):
                raise RuntimeError("state not saved")

        try:
            _unlock()
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            st.error(f"Не удалось обновить состояние стоп-дня: {exc}")
        else:
            st.success("Ограничение снято до следующего срабатывания.")


def _render_order_preview(settings: Any, api_client: Any) -> None:
    st.subheader("🔎 Предпросмотр заявки")
    with st.expander("Проверить, как биржа скорректирует параметры"):
        from utils.safety import guard_order

        with st.form("preview_order_form"):
            col_symbol, col_side, col_category = st.columns(3)
            with col_symbol:
                default_symbol = (
                    str(_settings_attr(settings, "ai_symbols_manual", "")) or "BTCUSDT"
                ).split(",")[0].strip()
                symbol = st.text_input("Символ", value=default_symbol)
            with col_side:
                side = st.selectbox("Сторона", ["BUY", "SELL"], index=0)
            with col_category:
                category = st.selectbox("Категория", ["spot"], index=0)

            col_price, col_qty = st.columns(2)
            with col_price:
                price_str = st.text_input("Цена (опционально, для лимитной)", value="")
            with col_qty:
                qty_str = st.text_input("Кол-во", value="10")

            if st.form_submit_button("Проверить"):
                try:
                    price = float(price_str) if price_str else None
                    qty = float(qty_str)
                    response = guard_order(
                        api_client,
                        category=category,
                        symbol=symbol.upper(),
                        side=side,
                        orderType="Limit" if price else "Market",
                        qty=qty,
                        price=price,
                    )
                except ValueError:
                    st.error("Введите корректные числовые значения цены и количества.")
                except Exception as exc:  # pragma: no cover - defensive UI feedback
                    st.error(f"Ошибка проверки: {exc}")
                else:
                    st.json(response)
                    decision = response.get("decision")
                    if decision == "ok":
                        st.success("OK — параметры соответствуют требованиям биржи.")
                    elif decision == "adjusted":
                        st.warning("Биржа потребует коррекцию — ниже показаны корректные значения.")
                    else:
                        st.error(f"Заявка будет отклонена: {response.get('reason')}")


def _render_universe_presets(settings: Any) -> None:
    st.subheader("🗺️ Пресеты универсума")
    with st.expander("Фильтры автоподбора монет"):
        options = ["Консервативный", "Стандарт", "Агрессивный"]
        default = str(_settings_attr(settings, "ai_universe_preset", "Стандарт"))
        index = options.index(default) if default in options else 1
        preset = st.selectbox("Пресет", options, index=index)

        if preset == "Консервативный":
            max_spread_default = float(_settings_attr(settings, "ai_max_spread_bps", 10.0))
            min_turnover_default = float(_settings_attr(settings, "ai_min_turnover_usd", 5_000_000.0))
        elif preset == "Агрессивный":
            max_spread_default = float(_settings_attr(settings, "ai_max_spread_bps", 50.0))
            min_turnover_default = float(_settings_attr(settings, "ai_min_turnover_usd", 500_000.0))
        else:
            max_spread_default = float(_settings_attr(settings, "ai_max_spread_bps", 25.0))
            min_turnover_default = float(_settings_attr(settings, "ai_min_turnover_usd", 2_000_000.0))

        max_spread = st.number_input("Макс. спред (бпс)", value=max_spread_default, step=1.0)
        min_turnover = st.number_input("Мин. оборот (USD)", value=min_turnover_default, step=100000.0)

        if st.button("💾 Сохранить пресет"):
            _persist_with_feedback(
                "Сохранено. Автоподбор будет учитывать фильтры.",
                ai_universe_preset=preset,
                ai_max_spread_bps=float(max_spread),
                ai_min_turnover_usd=float(min_turnover),
            )


def _render_watchdog(settings: Any) -> None:
    st.subheader("⚡ WS Watchdog")
    with st.form("watchdog_form"):
        wd_enabled = st.checkbox(
            "Включить авто-перезапуск WS",
            value=bool(_settings_attr(settings, "ws_watchdog_enabled", True)),
        )
        wd_max_age = st.number_input(
            "Макс. задержка heartbeat (сек)",
            value=int(_settings_attr(settings, "ws_watchdog_max_age_sec", 90)),
            step=10,
        )
        if st.form_submit_button("💾 Сохранить Watchdog"):
            _persist_with_feedback(
                "Сохранено.",
                ws_watchdog_enabled=bool(wd_enabled),
                ws_watchdog_max_age_sec=int(wd_max_age),
            )


def _render_tick_preview(api_client: Any) -> None:
    st.subheader("🧪 Смоделировать следующий тик (превью)")
    with st.expander("Показать, что бот потенциально отправит дальше (оценка)"):
        try:
            from utils.preview import next_tick_preview
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            st.error(f"Модуль превью недоступен: {exc}")
            return

        if st.button("🔍 Обновить превью"):
            try:
                preview = _with_spinner(
                    "Получаем оценку...", lambda: next_tick_preview(api_client)
                )
            except Exception as exc:  # pragma: no cover - defensive UI feedback
                st.error(f"Не удалось получить превью: {exc}")
                return

            st.json(preview)
            decision = preview.get("decision")
            if decision == "skip":
                st.warning(
                    "Сейчас заявка была бы отклонена фильтрами биржи. Отрегулируйте параметры/пресет.",
                )
            elif decision == "adjusted":
                st.info("Заявка потребует коррекции: см. итоговые qty/price в блоке preview.")
            else:
                st.success("Оценка в норме. Реальный AI может принять иное решение — это только превью.")


st.divider()
_render_briefing()

st.divider()
_render_autopilot(s, api)

st.divider()
_render_automation(s)

st.divider()
_render_universe_filters(s)

st.divider()
_render_export_import(s)

st.divider()
_render_telegram(s)

st.divider()
_render_daily_report()

st.divider()
_render_order_preview(s, api)

st.divider()
_render_universe_presets(s)

st.divider()
_render_watchdog(s)

st.divider()
_render_tick_preview(api)
