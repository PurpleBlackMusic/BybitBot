
from __future__ import annotations
import streamlit as st
from utils.envs import get_settings, update_settings
from utils.guardian_bot import GuardianBot
from utils.ui import section

st.title("⚙️ Настройки")

s = get_settings()

section("Режим работы GuardianBot")

mode_options = {
    "auto": {
        "label": "🤖 Автоматически",
        "description": "Бот сам обновляет параметры риска и исполняет сигналы.",
    },
    "manual": {
        "label": "🛠 Ручной контроль",
        "description": "Вы запускаете сделки сами, а AI подсказывает лучшие действия.",
    },
}

current_mode = str(getattr(s, "operation_mode", "manual") or "manual").lower()
if current_mode not in mode_options:
    current_mode = "manual"

mode_keys = list(mode_options.keys())
default_index = mode_keys.index(current_mode)

with st.form("operation_mode_form"):
    selected_mode = st.radio(
        "Как бот должен работать?",
        options=mode_keys,
        index=default_index,
        format_func=lambda key: mode_options[key]["label"],
    )
    submitted = st.form_submit_button("💾 Применить режим")

if submitted:
    updates = {"operation_mode": selected_mode}
    if selected_mode == "auto" and not getattr(s, "ai_enabled", False):
        updates["ai_enabled"] = True
    update_settings(**updates)
    st.success("Режим работы обновлён.")
    s = get_settings(force_reload=True)
    current_mode = selected_mode
else:
    current_mode = selected_mode

st.caption(mode_options[current_mode]["description"])

bot = GuardianBot(settings=s)
summary = bot.status_summary()

if current_mode == "auto":
    st.success(
        "GuardianBot управляет параметрами автоматически: при совпадении условий сигнал исполняется без ручного вмешательства."
    )
    blockers = [reason for reason in summary.get("actionable_reasons", []) if reason]
    if blockers:
        st.caption("Чтобы авто-режим торговал, убедитесь, что устранены ограничения ниже:")
        for reason in blockers:
            st.markdown(f"- {reason}")
else:
    guidance = summary.get("manual_guidance") or {}
    st.info(
        "В ручном режиме сделки не отправляются автоматически — AI готовит вам подсказки и чек-лист действий."
    )
    headline = guidance.get("headline")
    if headline:
        st.subheader(headline)
    control_status = guidance.get("control_status") or {}
    if control_status:
        awaiting_operator = bool(control_status.get("awaiting_operator"))
        is_active = bool(control_status.get("active"))
        message = control_status.get("message") or ""
        label = control_status.get("label") or "Статус ручного управления"
        status_block = st.warning if awaiting_operator else st.success if is_active else st.info
        status_lines = [f"**{label}**"]
        if message:
            status_lines.append("")
            status_lines.append(message)
        status_block("\n\n".join(status_lines))

        meta_bits = []
        symbol = control_status.get("symbol")
        if symbol:
            meta_bits.append(f"Символ: {symbol}")
        last_action_at = control_status.get("last_action_at")
        if last_action_at:
            meta_bits.append(f"Последняя команда: {last_action_at}")
        last_action_age = control_status.get("last_action_age")
        if last_action_age:
            meta_bits.append(f"Прошло времени: {last_action_age}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        history_preview = control_status.get("history_preview") or []
        if history_preview:
            with st.expander(
                "История ручных команд", expanded=False
            ):
                for entry in history_preview:
                    st.markdown(f"- {entry}")
    notes = guidance.get("notes") or []
    if notes:
        st.markdown("**Что советует AI:**")
        for note in notes:
            st.markdown(f"- {note}")
    thresholds = guidance.get("thresholds") or {}
    if thresholds:
        col_buy, col_sell, col_ev = st.columns(3)
        col_buy.metric("Порог покупки", f"{thresholds.get('effective_buy_probability_pct', 0):.2f}%")
        col_sell.metric("Порог продажи", f"{thresholds.get('effective_sell_probability_pct', 0):.2f}%")
        col_ev.metric("Мин. выгода", f"{thresholds.get('min_ev_bps', 0):.2f} б.п.")

    blockers = guidance.get("reasons") or summary.get("actionable_reasons", [])
    blockers = [reason for reason in blockers if reason]
    if blockers:
        st.markdown("**Почему автоматическое исполнение сейчас выключено:**")
        for reason in blockers:
            st.markdown(f"- {reason}")

    plan_steps = guidance.get("plan_steps") or []
    if plan_steps:
        with st.expander("Чек-лист действий", expanded=True):
            for idx, step in enumerate(plan_steps, start=1):
                st.markdown(f"{idx}. {step}")

    risk_outline = guidance.get("risk_summary") or ""
    if risk_outline:
        with st.expander("Контроль рисков", expanded=False):
            st.markdown(risk_outline.replace("\n", "  \n"))

    safety_notes = guidance.get("safety_notes") or []
    if safety_notes:
        with st.expander("Напоминания по безопасности", expanded=False):
            for note in safety_notes:
                st.markdown(f"- {note}")

    thresholds = summary.get("thresholds") or {}
    recommended_buy = thresholds.get("effective_buy_probability_pct", 0.0)
    recommended_sell = thresholds.get("effective_sell_probability_pct", 0.0)
    recommended_ev = thresholds.get("min_ev_bps", 0.0)

    st.subheader("Настройка параметров под себя")
    st.caption("AI делится актуальными ориентирами по вероятности сигнала и ожидаемой выгоде. Вы можете скорректировать лимиты вручную ниже.")

    current_buy = float(getattr(s, "ai_buy_threshold", 0.55) or 0.0) * 100.0
    current_sell = float(getattr(s, "ai_sell_threshold", 0.45) or 0.0) * 100.0
    current_ev = float(getattr(s, "ai_min_ev_bps", 0.0) or 0.0)
    current_risk = float(getattr(s, "ai_risk_per_trade_pct", 0.25) or 0.0)
    current_reserve = float(getattr(s, "spot_cash_reserve_pct", 10.0) or 0.0)
    current_daily_stop = float(getattr(s, "ai_daily_loss_limit_pct", 3.0) or 0.0)
    ai_analysis_active = bool(getattr(s, "ai_enabled", False))

    with st.form("manual_ai_tuning"):
        col_buy_sell, col_ev, col_risk = st.columns([1.2, 1.0, 1.0])
        with col_buy_sell:
            buy_pct = st.slider(
                "Порог покупки, %",
                min_value=0.0,
                max_value=100.0,
                value=current_buy,
                step=0.5,
                help=f"AI советует ≥ {recommended_buy:.2f}%",
            )
            sell_pct = st.slider(
                "Порог продажи, %",
                min_value=0.0,
                max_value=100.0,
                value=current_sell,
                step=0.5,
                help=f"AI советует ≥ {recommended_sell:.2f}%",
            )
        with col_ev:
            min_ev_bps = st.number_input(
                "Мин. ожидаемая выгода, б.п.",
                min_value=0.0,
                max_value=200.0,
                value=current_ev,
                step=0.5,
                help=f"AI не рекомендует опускаться ниже {recommended_ev:.2f} б.п.",
            )
            daily_stop = st.number_input(
                "Дневной стоп по убытку, %",
                min_value=0.0,
                max_value=100.0,
                value=current_daily_stop,
                step=0.5,
            )
        with col_risk:
            risk_pct = st.number_input(
                "Риск на сделку, % капитала",
                min_value=0.0,
                max_value=100.0,
                value=current_risk,
                step=0.1,
            )
            reserve_pct = st.number_input(
                "Резерв в кэше, %",
                min_value=0.0,
                max_value=100.0,
                value=current_reserve,
                step=1.0,
            )

        ai_analysis = st.checkbox(
            "AI анализ сигналов включён",
            value=ai_analysis_active,
            help="Оставьте анализ включённым, чтобы получать подсказки без автоторговли.",
        )

        tuning_submitted = st.form_submit_button("💾 Сохранить параметры")

    if tuning_submitted:
        update_settings(
            ai_buy_threshold=buy_pct / 100.0,
            ai_sell_threshold=sell_pct / 100.0,
            ai_min_ev_bps=min_ev_bps,
            ai_risk_per_trade_pct=risk_pct,
            spot_cash_reserve_pct=reserve_pct,
            ai_daily_loss_limit_pct=daily_stop,
            ai_enabled=ai_analysis,
        )
        st.success("Параметры сохранены. GuardianBot учтёт новые значения при следующем анализе.")
        s = get_settings(force_reload=True)

section("Telegram")
col1, col2 = st.columns(2)
with col1:
    token = st.text_input("Bot Token", value=s.telegram_token, type="password")
with col2:
    chat = st.text_input("Chat ID", value=s.telegram_chat_id)

if st.button("💾 Сохранить Telegram"):
    update_settings(telegram_token=token.strip(), telegram_chat_id=chat.strip())
    st.success("Telegram настройки сохранены.")
