
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings, update_settings
from utils.guardian_bot import GuardianBot
from utils.ui import section

st.title("⚙️ Настройки")

s = get_settings()

with st.form("automation_toggle"):
    ai_enabled_initial = bool(getattr(s, "ai_enabled", False))
    ai_enabled_choice = st.checkbox(
        "Включить автоматическое исполнение сигналов",
        value=ai_enabled_initial,
        help="Когда флажок активен, GuardianBot исполняет сделки автоматически при выполнении всех ограничений.",
    )
    toggle_submitted = st.form_submit_button("💾 Обновить режим")

if toggle_submitted:
    if ai_enabled_choice != ai_enabled_initial:
        update_settings(ai_enabled=ai_enabled_choice)
        st.success("Настройка автоматизации обновлена.")
        s = get_settings(force_reload=True)
    else:
        st.info("Настройки автоматизации без изменений.")

bot = GuardianBot(settings=s)
summary = bot.status_summary()
automation_health = bot.data_health().get("automation") or {}
brief = bot.generate_brief()

ai_enabled = bool(getattr(s, "ai_enabled", False))

if ai_enabled:
    automation_ok = bool(automation_health.get("ok"))
    automation_message = (
        str(automation_health.get("message") or "AI готов к автоматическим сделкам.").strip()
    )
    automation_details = automation_health.get("details")
    status_renderer = st.success if automation_ok else st.warning
    status_renderer(automation_message)
    if isinstance(automation_details, str) and automation_details.strip():
        st.caption(automation_details.strip())
    blockers = [reason for reason in summary.get("actionable_reasons", []) if reason]
    if blockers:
        st.markdown("**Почему бот пока ждёт:**")
        for reason in blockers:
            st.markdown(f"- {reason}")
else:
    st.info(
        "AI сигналы выключены — включите автоматизацию выше, чтобы GuardianBot исполнял сделки самостоятельно."
    )

plan_steps = bot.plan_steps(brief=brief)
if plan_steps:
    with st.expander("Чек-лист действий от AI", expanded=True):
        for idx, step in enumerate(plan_steps, start=1):
            st.markdown(f"{idx}. {step}")

risk_outline = bot.risk_summary()
if risk_outline:
    with st.expander("Контроль рисков", expanded=False):
        st.markdown(risk_outline.replace("\n", "  \n"))

safety_notes = bot.safety_notes()
if safety_notes:
    with st.expander("Напоминания по безопасности", expanded=False):
        for note in safety_notes:
            st.markdown(f"- {note}")

thresholds = summary.get("thresholds") or {}
if thresholds:
    st.subheader("Ориентиры GuardianBot")
    st.caption(
        "Пороговые значения из текущего сигнала помогают понять, когда AI готов действовать без лишнего риска."
    )
    cols = st.columns(3)
    cols[0].metric(
        "Порог покупки",
        f"{float(thresholds.get('effective_buy_probability_pct') or 0.0):.1f}%",
    )
    cols[1].metric(
        "Порог продажи",
        f"{float(thresholds.get('effective_sell_probability_pct') or 0.0):.1f}%",
    )
    cols[2].metric(
        "Мин. ожидаемая выгода",
        f"{float(thresholds.get('min_ev_bps') or 0.0):.1f} б.п.",
    )

st.caption(
    "GuardianBot автоматически подстраивает параметры риска и аккуратно ведёт сделки. Если нужны ручные правки, обновите settings.json напрямую."
)

section("Telegram")
col1, col2 = st.columns(2)
with col1:
    token = st.text_input("Bot Token", value=s.telegram_token, type="password")
with col2:
    chat = st.text_input("Chat ID", value=s.telegram_chat_id)

if st.button("💾 Сохранить Telegram"):
    update_settings(telegram_token=token.strip(), telegram_chat_id=chat.strip())
    st.success("Telegram настройки сохранены.")
