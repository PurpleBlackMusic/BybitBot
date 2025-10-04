
from __future__ import annotations
import streamlit as st, json, time
from utils.envs import get_api_client, get_settings, update_settings
from utils.coach import market_health, build_autopilot_settings
from utils.scheduler import start_background_loop, _load_state_file, _save_state_file
from utils.reporter import send_daily_report, summarize_today, send_test_message
from utils.ai.live import AIRunner
from utils.paths import DATA_DIR

st.set_page_config(page_title="Простой режим", page_icon="🧭", layout="wide")
st.title("🧭 Простой режим")

st.caption("Эта страница для тех, кто НЕ хочет разбираться в крипте и настройках. "
           "Здесь — краткая подсказка на сегодня и **одна кнопка**, чтобы запустить умного бота.")

s = get_settings()

# ##__TG_DEFAULTS__ ensure variables exist before buttons use them
tg_trd = bool(getattr(s, 'tg_trade_notifs', False))
tg_min = float(getattr(s, 'tg_trade_notifs_min_notional', 50.0))
api = get_api_client()
# фоновая автоматика
start_background_loop()

# --- Daily briefing ---
st.subheader("Сегодняшний брифинг")
try:
    info = market_health(api, category="spot")
    light = info.get("light")
    reason = info.get("reason","")
    cols = st.columns([1,6])
    with cols[0]:
        st.metric("Статус рынка", {"green":"✅ ОК","yellow":"⚠️ Риск","red":"⛔ Стоп"}.get(light,"—"))
    with cols[1]:
        st.write(reason)
    st.caption("✅ ОК — можно торговать • ⚠️ Риск — аккуратно • ⛔ Стоп — лучше не торговать")
    with st.expander("Рекомендованные монеты на сегодня"):
        st.table({"symbol":[x["symbol"] for x in info.get("top", [])],
                  "turnover24h":[x["turnover24h"] for x in info.get("top", [])],
                  "spread (bps)":[x["spread_bps"] for x in info.get("top", [])]})
except Exception as e:
    st.warning(f"Не удалось получить рыночные подсказки: {e}")

st.divider()

# --- One-click autopilot ---
st.subheader("Авто-бот (одной кнопкой)")
left, right = st.columns([2,1])
with left:
    st.write("Бот **сам подберёт монеты и параметры**, включит защиту капитала и TWAP, "
             "и запустится в фоне. Вы увидите статус и отчёты в разделе «Логи».")
    if "ai_runner" not in st.session_state:
        st.session_state["ai_runner"] = AIRunner()
    runner: AIRunner = st.session_state["ai_runner"]

    if st.button("🤖 Автоподбор и запуск"):
        try:
            pack = build_autopilot_settings(s, api)
            update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), **pack["settings"])
            runner.start()
            st.success(f"Бот запущен. Прогнозная подготовка: ~{pack['eta_minutes']} мин.")
        except Exception as e:
            st.error(f"Не удалось запустить: {e}")

    colA, colB = st.columns(2)
    if colA.button("⏹ Остановить бота"):
        try:
            runner.stop()
            st.info("Бот остановлен.")
        except Exception as e:
            st.error(f"Ошибка остановки: {e}")
    if colB.button("🛑 Паник-стоп (до завтра)"):
        stop_err: Exception | None = None
        try:
            runner.stop()
        except Exception as e:
            stop_err = e
            st.error(f"Ошибка остановки: {e}")
        try:
            stf = _load_state_file() or {}
            stf['stop_day_locked'] = True
            stf['stop_day_reason'] = 'panic'
            stf['stop_day_date'] = time.strftime('%Y-%m-%d')
            if not _save_state_file(stf):
                raise RuntimeError('state not saved')
        except Exception as e:
            st.error(f"Не удалось активировать паник-стоп: {e}")
        else:
            if stop_err is None:
                st.warning("Паник-стоп активирован: автозапуск заблокирован до завтра.")
    dry = st.toggle("Демо-режим (без реальных ордеров)", value=getattr(s, 'dry_run', True),
                    help="В демо-запуске бот **не отправляет** реальные заявки.")
    if dry != getattr(s, 'dry_run', True):
        update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), dry_run=bool(dry))
        st.rerun()

with right:
    st.write("**Статус бота**")
    try:
        p = DATA_DIR / "ai" / "status.json"
        if p.exists():
            st.json(json.loads(p.read_text(encoding="utf-8")))
        else:
            st.info("Пока статуса нет. Нажмите «Автоподбор и запуск».")
    except Exception as e:
        st.warning(f"Статус недоступен: {e}")

st.divider()
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



st.divider()

st.divider()
st.subheader("📅 Автоматизация")
with st.expander("Расписание: автозапуск/автостоп бота и дневной отчёт"):
    col1, col2 = st.columns(2)
    with col1:
        en = st.toggle("Включить авто-торговлю по расписанию", value=bool(getattr(s, "auto_trade_enabled", False)))
        start_t = st.text_input("Время авто-старта (чч:мм)", value=str(getattr(s, "auto_start_time", "09:00")))
        stop_t  = st.text_input("Время авто-стопа (чч:мм)", value=str(getattr(s, "auto_stop_time", "21:00")))
        auto_dry = st.toggle("Торговать в демо-режиме при автозапуске", value=bool(getattr(s, "auto_dry_run", True)))
    with col2:
        rep = st.toggle("Ежедневный отчёт в Telegram", value=bool(getattr(s, "daily_report_enabled", False)))
        rep_t = st.text_input("Время отправки отчёта (чч:мм)", value=str(getattr(s, "daily_report_time", "20:00")))
        loss = st.number_input("Дневной лимит убытка (%)", value=float(getattr(s, "ai_daily_loss_limit_pct", 1.0)), step=0.1)
        prof = st.number_input("Дневная цель прибыли (%)", value=float(getattr(s, "ai_daily_profit_target_pct", 0.0)), step=0.1)

    if st.button("💾 Сохранить расписание"):
        update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), auto_trade_enabled=bool(en), auto_start_time=start_t, auto_stop_time=stop_t, auto_dry_run=bool(auto_dry),
                        daily_report_enabled=bool(rep), daily_report_time=rep_t,
                        ai_daily_loss_limit_pct=float(loss), ai_daily_profit_target_pct=float(prof))
        st.success("Расписание сохранено. Фоновая автоматика уже работает.")


st.subheader("⚙️ Доп. настройки универсума")
with st.expander("Фильтрация монет (для автоподбора)"):
    wl = st.text_input("Белый список монет (через запятую, опционально)", value=str(getattr(s, 'ai_symbols_whitelist', '')),
                       help="Если заполнено — автоподбор берёт монеты только из этого списка.")
    bl = st.text_input("Чёрный список монет (через запятую, опционально)", value=str(getattr(s, 'ai_symbols_blacklist', '')),
                       help="Эти монеты исключаются из автоподбора.")
    man = st.text_input("Я сам задам монеты (перечисли через запятую)", value=str(getattr(s, 'ai_symbols_manual', '')),
                        help="Если указано — автоподбор возьмёт именно эти монеты.")
    if st.button("💾 Сохранить списки монет"):
        update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), ai_symbols_whitelist=wl, ai_symbols_blacklist=bl, ai_symbols_manual=man)
        st.success("Списки сохранены.")

st.subheader("🗂 Экспорт/импорт настроек")
colx, coly = st.columns(2)
with colx:
    if st.button("⬇️ Экспорт настроек в JSON"):
        from utils.envs import get_settings
        s = get_settings()
        try:
            data = s.dict() if hasattr(s, "dict") else (s.__dict__ if hasattr(s, "__dict__") else {})
        except Exception:
            data = {}
        import io, json
        buf = io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button("Скачать settings.json", data=buf.getvalue(), file_name="settings.json", mime="application/json")
with coly:
    up = st.file_uploader("Загрузить settings.json", type=["json"])
    if up is not None:
        try:
            import json
            payload = json.loads(up.read().decode("utf-8"))
            from utils.envs import update_settings
            update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), **payload)
            st.success("Настройки импортированы.")
        except Exception as e:
            st.error(f"Ошибка импорта: {e}")

st.subheader("🔔 Telegram-отчёты")
tg_trd = st.checkbox("Уведомления о сделках в Telegram", value=bool(getattr(s, "tg_trade_notifs", False)))
tg_min = st.number_input("Минимальный объём сделки для уведомлений (USDT)", value=float(getattr(s, "tg_trade_notifs_min_notional", 50.0)), step=10.0)

with st.expander("Настройки Telegram (для уведомлений и отчётов)"):
    tok = st.text_input("Bot Token", type="password", value=str(getattr(s, "telegram_token", "")))
    chat = st.text_input("Chat ID", value=str(getattr(s, "telegram_chat_id", "")))
    en = st.toggle("Включить уведомления", value=bool(getattr(s, "telegram_notify", False)),
                   help="Когда включено — приложение будет присылать короткие уведомления о старте/остановке и заявках.")
    if st.button("✅ Сохранить Telegram-настройки"):
        update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), telegram_token=tok, telegram_chat_id=chat, telegram_notify=bool(en))
        st.success("Сохранено.")
    if st.button("🧪 Отправить тестовое сообщение"):
        r = send_test_message("Привет! Telegram настроен ✅")
        st.write(f"Ответ: {r}")


st.divider()
st.subheader("Отчёт за сегодня")
try:
    summary = summarize_today()
    st.write(f"Событий: **{summary.get('events',0)}**, сигналов: **{summary.get('signals',0)}**, заявок: **{summary.get('orders',0)}**, ошибок: **{summary.get('errors',0)}**.")
except Exception as e:
    st.write("Пока нет данных за сегодня.")
if st.button("📤 Отправить отчёт в Telegram"):
    r = send_daily_report()
    st.success(f"Отчёт отправлен: {r}")

if st.button("🔓 Снять ‘стоп-день’ до завтра"):
    stf = _load_state_file() or {}
    stf['stop_day_locked'] = False
    stf['stop_day_reason'] = ''
    if _save_state_file(stf):
        st.success('Ограничение снято до следующего срабатывания.')
    else:
        st.error('Не удалось обновить состояние стоп-дня.')

st.divider()
st.subheader("🔎 Предпросмотр заявки")
with st.expander("Проверить, как биржа скорректирует параметры"):
    from utils.safety import guard_order
    ps1, ps2, ps3 = st.columns(3)
    with ps1:
        sym_prev = st.text_input("Символ", value=(getattr(s, "ai_symbols_manual", "") or "BTCUSDT").split(",")[0].strip())
    with ps2:
        side_prev = st.selectbox("Сторона", ["BUY","SELL"], index=0)
    with ps3:
        cat_prev = st.selectbox("Категория", ["spot"], index=0)
    colp, colq = st.columns(2)
    with colp:
        price_prev = st.text_input("Цена (опционально, для лимитной)", value="")
    with colq:
        qty_prev = st.text_input("Кол-во", value="10")
    if st.button("Проверить"):
        try:
            pr = float(price_prev) if price_prev else None
            qv = float(qty_prev)
            res = guard_order(api, category=cat_prev, symbol=sym_prev.upper(), side=side_prev, orderType="Limit" if pr else "Market", qty=qv, price=pr)
            st.json(res)
            if res.get("decision") == "ok":
                st.success("OK — параметры соответствуют требованиям биржи.")
            elif res.get("decision") == "adjusted":
                st.warning("Биржа потребует коррекцию — ниже показаны корректные значения.")
            else:
                st.error(f"Заявка будет отклонена: {res.get('reason')}")
        except Exception as e:
            st.error(f"Ошибка проверки: {e}")

st.divider()
st.subheader("🗺️ Пресеты универсума")
with st.expander("Фильтры автоподбора монет"):
    preset = st.selectbox("Пресет", ["Консервативный","Стандарт","Агрессивный"], index={"Консервативный":0,"Стандарт":1,"Агрессивный":2}[str(getattr(s, "ai_universe_preset", "Стандарт")) if hasattr(s, "ai_universe_preset") else "Стандарт"])
    # spread threshold in bps and min daily turnover USD (heuristic, based on tickers endpoint)
    if preset == "Консервативный":
        max_spread_bps = st.number_input("Макс. спред (бпс)", value=float(getattr(s, "ai_max_spread_bps", 10.0)), step=1.0)
        min_turnover_usd = st.number_input("Мин. оборот (USD)", value=float(getattr(s, "ai_min_turnover_usd", 5_000_000.0)), step=100000.0)
    elif preset == "Агрессивный":
        max_spread_bps = st.number_input("Макс. спред (бпс)", value=float(getattr(s, "ai_max_spread_bps", 50.0)), step=1.0)
        min_turnover_usd = st.number_input("Мин. оборот (USD)", value=float(getattr(s, "ai_min_turnover_usd", 500_000.0)), step=50000.0)
    else:
        max_spread_bps = st.number_input("Макс. спред (бпс)", value=float(getattr(s, "ai_max_spread_bps", 25.0)), step=1.0)
        min_turnover_usd = st.number_input("Мин. оборот (USD)", value=float(getattr(s, "ai_min_turnover_usd", 2_000_000.0)), step=100000.0)
    if st.button("💾 Сохранить пресет"):
        update_settings(tg_trade_notifs=bool(tg_trd), tg_trade_notifs_min_notional=float(tg_min), ai_universe_preset=preset, ai_max_spread_bps=float(max_spread_bps), ai_min_turnover_usd=float(min_turnover_usd))
        st.success("Сохранено. Автоподбор будет учитывать фильтры.")


st.divider()
st.subheader("⚡ WS Watchdog")
colw1, colw2 = st.columns(2)
with colw1:
    wd_on = st.checkbox("Включить авто-перезапуск WS", value=bool(getattr(s, "ws_watchdog_enabled", True)))
with colw2:
    wd_max = st.number_input("Макс. задержка heartbeat (сек)", value=int(getattr(s, "ws_watchdog_max_age_sec", 90)), step=10)
if st.button("💾 Сохранить Watchdog"):
    update_settings(ws_watchdog_enabled=bool(wd_on), ws_watchdog_max_age_sec=int(wd_max))
    st.success("Сохранено.")


st.divider()
st.subheader("🧪 Смоделировать следующий тик (превью)")
with st.expander("Показать, что бот потенциально отправит дальше (оценка)"):
    try:
        from utils.preview import next_tick_preview
        pr = next_tick_preview(api)
        st.json(pr)
        if pr.get("decision") == "skip":
            st.warning("Сейчас заявка была бы отклонена фильтрами биржи. Отрегулируйте параметры/пресет.")
        elif pr.get("decision") == "adjusted":
            st.info("Заявка потребует коррекции: см. итоговые qty/price в блоке preview.")
        else:
            st.success("Оценка в норме. Реальный AI может принять иное решение — это только превью.")
    except Exception as e:
        st.error(f"Не удалось получить превью: {e}")
