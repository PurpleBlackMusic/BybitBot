
from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from bybit_app.utils.dataframe import arrow_safe
from bybit_app.utils.ui import (
    build_pill,
    build_status_card,
    inject_css,
    navigation_link,
    page_slug_from_path,
    safe_set_page_config,
    auto_refresh,
)
from bybit_app.utils.formatting import tabular_numeric_css
from bybit_app.utils.ai.kill_switch import get_state as get_kill_switch_state
from bybit_app.utils.background import (
    ensure_background_services,
    restart_automation,
    restart_guardian,
    restart_websockets,
)
from bybit_app.utils.envs import (
    CredentialValidationError,
    active_api_key,
    active_api_secret,
    active_dry_run,
    get_settings,
    validate_runtime_credentials,
    update_settings,
)
from bybit_app.ui.state import (
    BASE_SESSION_STATE,
    cached_api_client,
    cached_guardian_snapshot,
    cached_preflight_snapshot,
    cached_ws_snapshot,
    clear_data_caches,
    get_last_interaction_timestamp,
    get_auto_refresh_holds,
    note_user_interaction,
    track_value_change,
    ensure_keys,
)
from bybit_app.ui.components import (
    _StatusBarContext,
    command_palette,
    log_viewer,
    metrics_strip,
    orders_table,
    show_error_banner,
    render_connection_gate,
    signals_table,
    status_bar,
    trade_ticket,
    wallet_overview,
)
from bybit_app.ui.backend_client import (
    pause_kill_switch as backend_pause_kill_switch,
    resume_kill_switch as backend_resume_kill_switch,
)





def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_tone(value: object) -> str:
    if not isinstance(value, str):
        return "warning"
    tone = value.strip().lower()
    mapping = {
        "critical": "danger",
        "danger": "danger",
        "error": "danger",
        "severe": "danger",
        "warn": "warning",
        "warning": "warning",
        "caution": "warning",
        "info": "info",
        "information": "info",
        "notice": "info",
        "success": "success",
        "ok": "success",
    }
    return mapping.get(tone, "warning")


def _tone_priority(tone: str) -> int:
    return {"danger": 0, "warning": 1, "info": 2, "success": 3}.get(tone, 1)


def _freshness_fill(value: float | None, *, warn_after: float, danger_after: float) -> float:
    """Return a visual fill percentage for freshness bars.

    Fresh data stays near 100%, warning decays to ~40%, danger clamps lower for quick scanning.
    """

    if value is None:
        return 100.0

    if value <= warn_after:
        return 100.0

    if value >= danger_after:
        return 30.0

    span = max(danger_after - warn_after, 1.0)
    decay = (value - warn_after) / span
    return max(30.0, 100.0 - decay * 60.0)


def _normalise_brief(raw: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raw = {}

    def _text(key: str, fallback: str = "") -> str:
        value = raw.get(key)
        if value is None:
            return fallback
        return str(value)

    mode = _text("mode", "wait").lower() or "wait"
    status_age = _safe_float(raw.get("status_age"), None)

    return {
        "mode": mode,
        "symbol": _text("symbol", "—"),
        "headline": _text("headline"),
        "action_text": _text("action_text"),
        "confidence_text": _text("confidence_text"),
        "ev_text": _text("ev_text"),
        "caution": _text("caution"),
        "updated_text": _text("updated_text"),
        "analysis": _text("analysis"),
        "status_age": status_age,
    }


def _normalise_key_fragment(value: str) -> str:
    """Return a Streamlit-safe fragment for widget keys."""

    fragment = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_")
    return fragment or "page"


def render_navigation_grid(
    shortcuts: list[tuple[str, str, str]], *, columns: int = 2, key_prefix: str = "nav"
) -> None:
    """Render navigation links in a compact grid layout."""

    if not shortcuts:
        return

    prefix_fragment = _normalise_key_fragment(str(key_prefix))

    for idx in range(0, len(shortcuts), columns):
        row = shortcuts[idx : idx + columns]
        cols = st.columns(len(row))
        for column_offset, (column, shortcut) in enumerate(zip(cols, row)):
            label, page, description = shortcut
            slug_fragment = _normalise_key_fragment(page_slug_from_path(page))
            unique_key = f"{prefix_fragment}_{slug_fragment}_{idx + column_offset}"
            with column:
                navigation_link(page, label=label, key=unique_key)
                st.caption(description)


def render_header(
    settings: Any,
    *,
    report: Mapping[str, Any] | None = None,
    guardian_snapshot: Mapping[str, Any] | None = None,
    ws_snapshot: Mapping[str, Any] | None = None,
    kill_switch: Any | None = None,
) -> None:
    def _as_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    def _tone_from_age(value: float | None, *, warn_after: float, danger_after: float) -> str:
        if value is None:
            return "muted"
        if value >= danger_after:
            return "danger"
        if value >= warn_after:
            return "warn"
        return "ok"

    context = _StatusBarContext.from_inputs(
        settings,
        _as_mapping(guardian_snapshot),
        _as_mapping(ws_snapshot),
        _as_mapping(report),
        kill_switch,
    )

    stats = _as_mapping(_as_mapping(report).get("statistics"))
    plan = _as_mapping(_as_mapping(report).get("symbol_plan"))
    portfolio = _as_mapping(_as_mapping(report).get("portfolio"))
    totals = _as_mapping(portfolio.get("totals"))
    brief = _normalise_brief(_as_mapping(_as_mapping(report).get("brief")))

    actionable = int(_safe_float(stats.get("actionable_count"), 0.0) or 0)
    ready = int(_safe_float(stats.get("ready_count"), 0.0) or 0)
    positions = int(_safe_float(stats.get("position_count"), 0.0) or 0)
    tracked_pairs = len(plan)
    equity_value = _safe_float(totals.get("total_equity") or totals.get("equity"))
    available_value = _safe_float(totals.get("available_balance") or totals.get("available"))
    equity_text = f"{equity_value:,.2f}" if equity_value is not None else "—"
    available_text = f"{available_value:,.2f}" if available_value is not None else "—"
    readiness_pct = 0.0 if actionable <= 0 else (ready / max(actionable, 1)) * 100
    readiness_tone = "danger" if readiness_pct < 40 else ("warn" if readiness_pct < 75 else "ok")
    signal_fill = _freshness_fill(context.signal_age, warn_after=120.0, danger_after=300.0)
    ws_fill = _freshness_fill(context.ws_worst_age, warn_after=60.0, danger_after=90.0)
    status_age_text = _format_seconds_ago(brief.get("status_age"))

    mode_tag = "Тестнет" if context.testnet else "Основной режим"
    run_tag = "DRY-RUN" if context.dry_run else "Боевой режим"
    kill_tag = "На паузе" if context.kill_switch.paused else "Готово"

    signal_tone = _tone_from_age(context.signal_age, warn_after=120.0, danger_after=300.0)
    ws_tone = _tone_from_age(context.ws_worst_age, warn_after=60.0, danger_after=90.0)
    auto_tone = "ok" if context.automation_ok else "warn"
    kill_tone = "danger" if context.kill_switch.paused else "ok"
    realtime_tone = "ok" if context.realtime_ok else "danger"

    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-hero__title">
                <div class="app-hero__eyebrow-row">
                    <p class="app-hero__eyebrow">Bybit Spot Guardian</p>
                    <span class="app-hero__tag">Обзор · {mode_tag}</span>
                    <span class="app-hero__tag app-hero__tag--accent">{run_tag}</span>
                    <span class="app-hero__tag app-hero__tag--muted">{kill_tag}</span>
                </div>
                <h1>Центр решений по споту</h1>
                <p class="app-hero__lede">
                    Единая панель для режима, свежести сигналов и автоматизации. Сразу видно, готов ли бот к запуску и где сосредоточить внимание.
                </p>
                <div class="app-hero__toolbar">
                    <div class="app-hero__chip app-hero__chip--accent">{mode_tag}</div>
                    <div class="app-hero__chip app-hero__chip--ghost">{run_tag}</div>
                    <div class="app-hero__chip app-hero__chip--{kill_tone}">Kill-Switch: {kill_tag}</div>
                    <div class="app-hero__chip app-hero__chip--muted">Последний сигнал: {context.signal_caption or '—'}</div>
                </div>
                <ul class="app-hero__bullets">
                    <li>Подключение, защита и авто-режим сведены в верхние карточки — видно, что готово.</li>
                    <li>Свежесть сигналов и потоков маркируется прогрессом, подсвечивая устаревание.</li>
                    <li>Сигналы, позиции и трекинг пар вынесены в стат-блоки для быстрых решений.</li>
                </ul>
                <div class="app-hero__meta-row">
                    <div class="app-hero__meta-card app-hero__meta-card--soft">
                        <div class="app-hero__meta-label">Свежесть потоков</div>
                        <div class="app-hero__meta-value">{context.signal_caption or '—'}</div>
                        <p class="app-hero__meta-note">Сигналы и WebSocket: {context.ws_caption or '—'}</p>
                    </div>
                    <div class="app-hero__meta-card">
                        <div class="app-hero__meta-label">Защита</div>
                        <div class="app-hero__meta-value">Kill-Switch: {kill_tag}</div>
                        <p class="app-hero__meta-note">Авто-режим: {context.automation_caption}</p>
                    </div>
                    <div class="app-hero__meta-card">
                        <div class="app-hero__meta-label">Баланс</div>
                        <div class="app-hero__meta-value">{equity_text} / {available_text}</div>
                        <p class="app-hero__meta-note">Equity · Доступно (USD)</p>
                    </div>
                    <div class="app-hero__meta-card app-hero__meta-card--ghost">
                        <div class="app-hero__meta-label">Guardian</div>
                        <div class="app-hero__meta-value">{context.guardian_caption or '—'}</div>
                        <p class="app-hero__meta-note">Мониторинг и рестарты</p>
                    </div>
                </div>
                <div class="app-hero__meters">
                    <div class="app-hero__meter">
                        <div class="app-hero__meter-head">
                            <span>Сигналы</span>
                            <span class="app-hero__meter-chip app-hero__meter-chip--{signal_tone}">{context.signal_caption or '—'}</span>
                        </div>
                        <div class="app-hero__meter-bar">
                            <span class="app-hero__meter-fill app-hero__meter-fill--{signal_tone}" style="width:{signal_fill:.0f}%"></span>
                        </div>
                        <p class="app-hero__meter-caption">Свежесть отчёта guardian с учётом последнего сигнала.</p>
                    </div>
                    <div class="app-hero__meter">
                        <div class="app-hero__meter-head">
                            <span>WebSocket</span>
                            <span class="app-hero__meter-chip app-hero__meter-chip--{ws_tone}">{context.ws_caption or 'Нет данных'}</span>
                        </div>
                        <div class="app-hero__meter-bar">
                            <span class="app-hero__meter-fill app-hero__meter-fill--{ws_tone}" style="width:{ws_fill:.0f}%"></span>
                        </div>
                        <p class="app-hero__meter-caption">Пульс pub/priv каналов — падение покажет устаревание.</p>
                    </div>
                </div>
                <div class="app-hero__hints">
                    <span class="app-hero__hint app-hero__hint--{realtime_tone}">Биржа: {context.realtime_caption}</span>
                    <span class="app-hero__hint app-hero__hint--{auto_tone}">Авто: {context.automation_caption}</span>
                    <span class="app-hero__hint app-hero__hint--{kill_tone}">Kill-Switch: {context.kill_caption}</span>
                </div>
                <div class="app-hero__health-grid">
                    <div class="app-hero__health app-hero__health--{signal_tone}">
                        <div class="app-hero__health-label">Сигналы</div>
                        <div class="app-hero__health-value">{context.signal_caption or '—'}</div>
                        <div class="app-hero__health-caption">Возраст обновления</div>
                    </div>
                    <div class="app-hero__health app-hero__health--{ws_tone}">
                        <div class="app-hero__health-label">WebSocket</div>
                        <div class="app-hero__health-value">{context.ws_caption or 'Нет данных'}</div>
                        <div class="app-hero__health-caption">pub/priv канал</div>
                    </div>
                    <div class="app-hero__health app-hero__health--{realtime_tone}">
                        <div class="app-hero__health-label">Биржа</div>
                        <div class="app-hero__health-value">{context.realtime_caption}</div>
                        <div class="app-hero__health-caption">Статус API</div>
                    </div>
                    <div class="app-hero__health app-hero__health--{auto_tone}">
                        <div class="app-hero__health-label">Авто-режим</div>
                        <div class="app-hero__health-value">{context.automation_caption}</div>
                        <div class="app-hero__health-caption">{'Готов к действию' if context.automation_ready else 'Статус автоматики'}</div>
                    </div>
                </div>
                <div class="app-hero__progress">
                    <div class="app-hero__progress-header">
                        <span>Готовность сигналов</span>
                        <span>{ready}/{actionable} · {readiness_pct:.0f}%</span>
                    </div>
                    <div class="app-hero__progress-bar">
                        <span class="app-hero__progress-fill app-hero__progress-fill--{readiness_tone}" style="width:{min(readiness_pct, 100):.0f}%"></span>
                    </div>
                    <p class="app-hero__progress-caption">Готовые сигналы выходят сразу на авто-процессы, остальным нужны правки.</p>
                </div>
                <div class="app-hero__digest">
                    <div class="app-hero__digest-label">Следующее действие</div>
                    <div class="app-hero__digest-headline">{brief.get('headline') or 'Нет активных подсказок'}</div>
                    <div class="app-hero__digest-flags">
                        <span class="app-hero__flag app-hero__flag--accent">{brief.get('ev_text') or 'EV обновится после оценки'}</span>
                        <span class="app-hero__flag app-hero__flag--muted">{brief.get('updated_text') or 'Ожидаем новое обновление'}</span>
                    </div>
                    <div class="app-hero__digest-meta">
                        <span>Пара: {brief.get('symbol')}</span>
                        <span>{brief.get('action_text') or '—'}</span>
                        <span>{brief.get('confidence_text') or ''}</span>
                        <span>Обновлено: {status_age_text}</span>
                    </div>
                    <p class="app-hero__digest-body">{brief.get('analysis') or 'План обновится, когда появится новый сигнал или рекомендации от стража.'}</p>
                    {f"<div class='app-hero__digest-note'>⚠️ {brief.get('caution')}</div>" if brief.get('caution') else ''}
                </div>
            </div>
            <div class="app-hero__panel">
                <div class="app-hero__panel-heading">Быстрый обзор</div>
                <div class="app-hero__panel-grid">
                    <div class="app-hero__stat">
                        <div class="app-hero__stat-label">Готово к действию</div>
                        <div class="app-hero__stat-value">{ready}/{actionable}</div>
                        <small>сигналов готовы без правок</small>
                    </div>
                    <div class="app-hero__stat">
                        <div class="app-hero__stat-label">Активные позиции</div>
                        <div class="app-hero__stat-value">{positions}</div>
                        <small>слежение за открытыми сделками</small>
                    </div>
                    <div class="app-hero__stat">
                        <div class="app-hero__stat-label">Отслеживаемые пары</div>
                        <div class="app-hero__stat-value">{tracked_pairs}</div>
                        <small>в планах сигнала</small>
                    </div>
                </div>
                <div class="app-hero__panel-footer">
                    <div class="app-hero__pill">💰 Equity: {equity_text} USD</div>
                    <div class="app-hero__pill">📥 Доступно: {available_text} USD</div>
                    <div class="app-hero__pill app-hero__pill--muted">⏱ Kill-Switch: {context.kill_caption}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(settings) -> None:
    api_key_value = active_api_key(settings)
    api_secret_value = active_api_secret(settings)
    ok = bool(api_key_value and api_secret_value)
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
            st.metric("Режим", "DRY-RUN" if active_dry_run(settings) else "Live")
            reserve = getattr(settings, "spot_cash_reserve_pct", 10.0)
            st.metric("Резерв кэша", f"{reserve:.0f}%")

        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if api_key_value else '❌'} · Secret: {'✅' if api_secret_value else '❌'} · Настройки обновлены: {last_update}"
        )

    if not ok:
        st.warning(
            "Без API ключей бот не сможет размещать ордера. Перейдите в раздел «Подключение» и добавьте их."
        )
        navigation_link(
            "pages/00_connection_status.py",
            label="Настроить подключение",
            icon="🔌",
            key="dashboard_setup_link",
        )


def _format_seconds_ago(value: object | None) -> str:
    try:
        seconds = float(value) if value is not None else None
    except (TypeError, ValueError):
        seconds = None

    if seconds is None or seconds < 0:
        return "—"
    if seconds < 1:
        return "< 1 с назад"
    if seconds < 60:
        return f"{seconds:.0f} с назад"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} мин назад"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} ч назад"
    days = hours / 24
    return f"{days:.1f} дн назад"


def _pick_freshest(records: Mapping[str, Mapping[str, object]]) -> tuple[str, Mapping[str, object]] | None:
    freshest: tuple[float, str, Mapping[str, object]] | None = None
    for topic, payload in records.items():
        age_raw = payload.get("age_seconds") if isinstance(payload, Mapping) else None
        try:
            age = float(age_raw) if age_raw is not None else float("inf")
        except (TypeError, ValueError):
            age = float("inf")
        if freshest is None or age < freshest[0]:
            freshest = (age, topic, payload)
    if freshest is None:
        return None
    return freshest[1], freshest[2]


def _summarise_order(order: Mapping[str, object] | None) -> str:
    if not isinstance(order, Mapping):
        return "Нет свежих ордеров"
    symbol = str(order.get("symbol") or "—")
    side = str(order.get("side") or "—").upper()
    status = str(order.get("status") or order.get("orderStatus") or "—")
    return f"{symbol} · {side} · {status}"


def _summarise_execution(execution: Mapping[str, object] | None) -> str:
    if not isinstance(execution, Mapping):
        return "Нет свежих исполнений"
    symbol = str(execution.get("symbol") or "—")
    side = str(execution.get("side") or "—").upper()
    qty = execution.get("execQty") or execution.get("qty")
    price = execution.get("execPrice") or execution.get("price")
    qty_text = f"{qty}" if qty not in (None, "") else "?"
    price_text = f"{price}" if price not in (None, "") else "?"
    return f"{symbol} · {side} · {qty_text}@{price_text}"


def render_ws_telemetry(snapshot: Mapping[str, object] | None) -> None:
    if not snapshot:
        return

    realtime = snapshot.get("realtime") if isinstance(snapshot, Mapping) else None
    realtime = realtime if isinstance(realtime, Mapping) else {}
    generated_at = realtime.get("generated_at") if isinstance(realtime, Mapping) else None
    try:
        snapshot_age = time.time() - float(generated_at) if generated_at is not None else None
    except (TypeError, ValueError):
        snapshot_age = None
    public_records = realtime.get("public") if isinstance(realtime, Mapping) else {}
    if not isinstance(public_records, Mapping):
        public_records = {}
    private_records = realtime.get("private") if isinstance(realtime, Mapping) else {}
    if not isinstance(private_records, Mapping):
        private_records = {}

    last_order = snapshot.get("last_order") if isinstance(snapshot, Mapping) else None
    last_execution = snapshot.get("last_execution") if isinstance(snapshot, Mapping) else None
    public_stale = bool(snapshot.get("public_stale")) if isinstance(snapshot, Mapping) else False
    private_stale = bool(snapshot.get("private_stale")) if isinstance(snapshot, Mapping) else False

    with st.container(border=True):
        st.markdown("#### Живой поток данных")
        cols = st.columns(2)

        latest_public = _pick_freshest(public_records) if public_records else None
        if latest_public is None:
            delta = "ожидаем обновление" if not public_stale else "данные устарели"
            cols[0].metric("Публичный поток", "нет данных", delta)
        else:
            topic, payload = latest_public
            age_text = _format_seconds_ago(payload.get("age_seconds") if isinstance(payload, Mapping) else None)
            delta = "устарели" if public_stale else age_text
            cols[0].metric("Публичный поток", topic, delta)
            cols[0].caption(f"Тем {len(public_records)} · последнее обновление {age_text}")

        latest_private = _pick_freshest(private_records) if private_records else None
        if latest_private is None:
            delta = "ожидаем обновление" if not private_stale else "данные устарели"
            cols[1].metric("Приватный поток", "нет данных", delta)
        else:
            topic, payload = latest_private
            age_text = _format_seconds_ago(payload.get("age_seconds") if isinstance(payload, Mapping) else None)
            delta = "устарели" if private_stale else age_text
            cols[1].metric("Приватный поток", topic, delta)
            cols[1].caption(f"Тем {len(private_records)} · последнее обновление {age_text}")

        info_bits: list[str] = []
        if last_order:
            info_bits.append(f"🧾 { _summarise_order(last_order) }")
        if last_execution:
            info_bits.append(f"⚡ { _summarise_execution(last_execution) }")
        if snapshot_age is not None:
            info_bits.append(f"⏱ Снимок обновлён { _format_seconds_ago(snapshot_age) }")
        if info_bits:
            st.markdown("<br />".join(info_bits), unsafe_allow_html=True)

def _mode_meta(mode: str) -> tuple[str, str, str]:
    mapping: dict[str, tuple[str, str, str]] = {
        "buy": ("Покупка", "🟢", "success"),
        "sell": ("Продажа", "🔴", "warning"),
        "wait": ("Наблюдаем", "⏸", "neutral"),
    }
    return mapping.get(mode, ("Наблюдаем", "⏸", "neutral"))


def render_signal_brief(
    brief_raw: Mapping[str, object] | None,
    score: Mapping[str, object] | None,
    *,
    settings,
) -> dict[str, object]:
    brief = _normalise_brief(brief_raw)
    probability_pct = _safe_float(
        score.get("probability_pct") if isinstance(score, Mapping) else None, 0.0
    )
    buy_threshold = _safe_float(
        score.get("buy_threshold") if isinstance(score, Mapping) else None, 0.0
    )
    ev_bps = _safe_float(
        score.get("ev_bps") if isinstance(score, Mapping) else None, 0.0
    )
    min_ev_bps = _safe_float(
        score.get("min_ev_bps") if isinstance(score, Mapping) else None, 0.0
    )
    last_update = (
        score.get("last_update") if isinstance(score, Mapping) else None
    ) or "—"

    mode_label, mode_icon, tone = _mode_meta(brief.get("mode", "wait"))

    st.subheader("Сводка сигнала")
    with st.container(border=True):
        st.markdown(
            """
            <div class="signal-card__badge">
                {pill}<span class="signal-card__symbol">· {symbol}</span>
            </div>
            """.format(
                pill=build_pill(mode_label, icon=mode_icon, tone=tone),
                symbol=brief.get("symbol", "—"),
            ),
            unsafe_allow_html=True,
        )
        for key in ("headline", "analysis", "action_text", "confidence_text", "ev_text"):
            text = str(brief.get(key) or "").strip()
            if not text:
                continue
            st.markdown(
                f"<div class='signal-card__body'>{text}</div>",
                unsafe_allow_html=True,
            )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Вероятность",
            f"{probability_pct or 0.0:.1f}%",
            f"Порог {buy_threshold or 0.0:.0f}%",
        )
        metric_cols[1].metric(
            "Потенциал",
            f"{ev_bps or 0.0:.1f} б.п.",
            f"Мин. {min_ev_bps or 0.0:.1f} б.п.",
        )
        trade_mode = "DRY-RUN" if active_dry_run(settings) else "Live"
        metric_cols[2].metric("Тактика", mode_label, trade_mode)
        st.caption(f"Обновление: {last_update}")

    caution = str(brief.get("caution") or "").strip()
    if caution:
        st.warning(caution)
    status_age = _safe_float(brief.get("status_age"), None)
    if status_age is not None and status_age > 300:
        st.error(
            "Сигнал не обновлялся более пяти минут — проверьте соединение с данными или перезапустите источник."
        )

    return brief


def _normalise_health(health: Mapping[str, object] | Sequence[tuple[str, object]] | None) -> dict[str, object]:
    """Return a dictionary representation of the health payload."""

    if health is None:
        return {}
    if isinstance(health, Mapping):
        return dict(health)
    try:
        return dict(health)
    except Exception:
        return {}


def _normalise_watchlist(watchlist: object) -> list[Mapping[str, object] | object]:
    """Convert watchlist payloads to a list consumable by the UI."""

    if watchlist is None:
        return []

    if hasattr(watchlist, "to_dict"):
        try:
            records = watchlist.to_dict("records")  # type: ignore[call-arg]
        except Exception:
            records = None
        else:
            if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
                return list(records)

    if isinstance(watchlist, Mapping):
        return [watchlist]

    if isinstance(watchlist, Sequence) and not isinstance(watchlist, (str, bytes)):
        return list(watchlist)

    if isinstance(watchlist, Iterable) and not isinstance(watchlist, (str, bytes)):
        return list(watchlist)

    return [watchlist]


_TONE_ICON_MAP: dict[str, str] = {
    "danger": "⛔",
    "warning": "⚠️",
    "info": "ℹ️",
    "success": "✅",
}

def _combine_descriptions(primary: str, extra: str) -> str:
    primary = (primary or "").strip()
    extra = (extra or "").strip()
    if not extra:
        return primary
    if not primary:
        return extra
    if extra.lower() == primary.lower():
        return primary
    if extra in primary:
        return primary
    if primary in extra:
        return extra
    joiner = " " if primary.endswith((".", "!", "?", ":", "—", "-", "–")) else " · "
    return f"{primary}{joiner}{extra}".strip()


def _format_details(details: object) -> str:
    if not details:
        return ""
    if isinstance(details, str):
        return details
    if isinstance(details, Mapping):
        return "; ".join(f"{key}: {value}".strip() for key, value in details.items() if str(value).strip())
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
        return "; ".join(str(item) for item in details)
    return str(details)


def _normalise_step_item(item: object) -> str | None:
    if isinstance(item, Mapping):
        for key in ("title", "text", "description", "label", "message"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        values = [str(value).strip() for value in item.values() if str(value).strip()]
        if values:
            return " ".join(values)
        return None
    if isinstance(item, (str, bytes)):
        text = item.decode() if isinstance(item, bytes) else item
    else:
        text = str(item)
    text = text.strip()
    return text or None


def _normalise_steps(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [
            part.strip(" •-–—")
            for part in re.split(r"[\n;,•·]+", raw)
            if part.strip(" •-–—")
        ]
        return parts
    if isinstance(raw, Mapping):
        return [
            f"{key}: {value}".strip()
            for key, value in raw.items()
            if str(value).strip()
        ]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        steps: list[str] = []
        for item in raw:
            normalised = _normalise_step_item(item)
            if normalised:
                steps.append(normalised)
        return steps
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return [
            step
            for item in raw
            if (step := _normalise_step_item(item))
        ]
    normalised = _normalise_step_item(raw)
    return [normalised] if normalised else []


def _collect_steps(info: Mapping[str, Any]) -> list[str]:
    fields = ("checklist", "steps", "actions", "remediation", "recommendations")
    steps: list[str] = []
    for field in fields:
        steps.extend(_normalise_steps(info.get(field)))
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for step in steps:
        lowered = step.lower()
        if lowered in seen_keys:
            continue
        seen_keys.add(lowered)
        deduped.append(step)
    return deduped


@dataclass
class _ActionCandidate:
    title: str
    description: str
    icon: str
    tone: str
    page: str | None
    page_label: str | None
    priority: int
    order: int

    def merge_with(self, other: "_ActionCandidate") -> None:
        if other.priority < self.priority:
            combined = _combine_descriptions(other.description, self.description)
            self.title = other.title
            self.description = combined
            self.icon = other.icon
            self.tone = other.tone
            self.page = other.page
            self.page_label = other.page_label
            self.priority = other.priority
            self.order = min(self.order, other.order)
            return

        self.description = _combine_descriptions(self.description, other.description)
        if not self.page and other.page:
            self.page = other.page
        if not self.page_label and other.page_label:
            self.page_label = other.page_label
        self.order = min(self.order, other.order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "tone": self.tone,
            "page": self.page,
            "page_label": self.page_label,
            "priority": self.priority,
        }


class _ActionBuilder:
    def __init__(self) -> None:
        self._actions: list[_ActionCandidate] = []
        self._seen: dict[tuple[str, str], _ActionCandidate] = {}
        self._order = 0

    def _next_order(self) -> int:
        self._order += 1
        return self._order

    def add(
        self,
        title: str,
        description: str,
        *,
        icon: str | None = None,
        tone: str | None = None,
        page: str | None = None,
        page_label: str | None = None,
        priority: int | None = None,
        identity_hint: tuple[str, str] | None = None,
    ) -> None:
        resolved_tone = _normalise_tone(tone)
        resolved_icon = icon or _TONE_ICON_MAP.get(resolved_tone, "⚠️")
        resolved_priority = priority if priority is not None else _tone_priority(resolved_tone)
        identity = identity_hint or (title.strip(), description.strip())
        candidate = _ActionCandidate(
            title=title,
            description=description,
            icon=resolved_icon,
            tone=resolved_tone,
            page=page,
            page_label=page_label,
            priority=resolved_priority,
            order=self._next_order(),
        )

        existing = self._seen.get(identity)
        if existing is not None:
            existing.merge_with(candidate)
            return

        self._seen[identity] = candidate
        self._actions.append(candidate)

    def as_list(self) -> list[dict[str, Any]]:
        ordered = sorted(self._actions, key=lambda item: (item.priority, item.order))
        return [item.to_dict() for item in ordered]


def collect_user_actions(
    settings,
    brief: Mapping[str, object] | None,
    health: dict[str, dict[str, object]] | None,
    watchlist: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Compile context-aware next steps for the home dashboard."""

    builder = _ActionBuilder()

    brief_map = dict(brief) if isinstance(brief, Mapping) else {}
    brief_caution = str(brief_map.get("caution") or "").strip()
    brief_status_age = _safe_float(brief_map.get("status_age"), None)

    has_keys = bool(active_api_key(settings) and active_api_secret(settings))
    dry_run_enabled = bool(active_dry_run(settings))
    reserve_pct = getattr(settings, "spot_cash_reserve_pct", None)

    if not has_keys:
        builder.add(
            "Добавьте API ключи",
            "Сохраните ключ и секрет Bybit в разделе подключения, чтобы бот смог размещать ордера.",
            icon="🔑",
            tone="warning",
            page="pages/00_connection_status.py",
            page_label="Открыть «Подключение»",
        )
    else:
        if dry_run_enabled:
            builder.add(
                "DRY-RUN активен",
                "Живые заявки не отправляются. Отключите учебный режим, когда будете готовы к реальной торговле.",
                icon="🧪",
                tone="warning",
                page="pages/02_settings.py",
                page_label="Перейти к настройкам",
            )

    if isinstance(reserve_pct, (int, float)) and reserve_pct < 10:
        builder.add(
            "Резерв кэша ниже рекомендации",
            f"Сейчас отложено {reserve_pct:.0f}% — держите не меньше 10%, чтобы бот не истощил депозит.",
            icon="💧",
            tone="warning",
            page="pages/02_settings.py",
            page_label="Настроить резерв",
        )

    if brief_caution:
        builder.add(
            "Проверка сигнала",
            brief_caution,
            icon="🛟",
            tone="warning",
            page="pages/00_simple_mode.py",
            page_label="Изучить сигнал",
        )

    if brief_status_age is not None and brief_status_age > 300:
        builder.add(
            "Сигнал устарел",
            "Данные не обновлялись более пяти минут — перезапустите источник или обновите пайплайн сигналов.",
            icon="⏱",
            tone="danger",
            page="pages/00_simple_mode.py",
            page_label="Проверить сигнал",
        )

    health_map = health or {}
    page_lookup: dict[str, tuple[str | None, str | None]] = {
        "ai_signal": ("pages/00_simple_mode.py", "Открыть «Простой режим»"),
        "executions": ("pages/05_trade_monitoring.py", "Открыть «Мониторинг сделок»"),
        "realtime_trading": ("pages/05_ws_control.py", "Проверить real-time"),
        "api_keys": ("pages/00_connection_status.py", "Проверить подключение"),
    }
    priority_lookup: dict[str, int] = {
        "ai_signal": -1,
    }

    for key, info in health_map.items():
        if not isinstance(info, Mapping):
            continue
        if info.get("ok") is not False:
            continue
        if key == "realtime_trading" and (dry_run_enabled or not has_keys):
            continue

        title = str(info.get("title") or key)
        message = str(info.get("message") or "").strip()
        details_text = _format_details(info.get("details"))
        description = " ".join(part for part in (message, details_text) if part).strip() or "Подробности недоступны."
        default_page, default_page_label = page_lookup.get(key, (None, None))
        page = info.get("page") or info.get("link") or default_page
        if not isinstance(page, str):
            page = default_page
        page_label = info.get("page_label") or info.get("link_label") or info.get("action") or default_page_label
        if not isinstance(page_label, str):
            page_label = default_page_label
        tone = info.get("tone") or info.get("status") or info.get("severity")
        normalised_tone = _normalise_tone(tone)
        computed_priority = _tone_priority(normalised_tone)
        raw_priority = info.get("priority") if isinstance(info.get("priority"), int) else None
        effective_priority = raw_priority if raw_priority is not None else computed_priority
        if raw_priority is not None:
            effective_priority = min(raw_priority, computed_priority)
        default_priority = priority_lookup.get(key)
        if default_priority is not None:
            effective_priority = min(effective_priority, default_priority)
        raw_icon = info.get("icon")
        icon = raw_icon if isinstance(raw_icon, str) else None
        steps = _collect_steps(info)
        if steps:
            limit = 4
            trimmed = steps[:limit]
            steps_text = "Шаги: " + " · ".join(trimmed)
            if len(steps) > limit:
                steps_text += f" (+{len(steps) - limit})"
            description = _combine_descriptions(description, steps_text)

        builder.add(
            title,
            description,
            icon=icon,
            tone=normalised_tone,
            page=page,
            page_label=page_label,
            priority=effective_priority,
            identity_hint=(
                title.strip(),
                str(
                    info.get("slug")
                    or info.get("id")
                    or message
                    or description
                    or key
                    or title.strip()
                ),
            ),
        )

    if not watchlist:
        builder.add(
            "Добавьте пары в наблюдение",
            "Список пуст — соберите рабочий универсум через Universe Builder или добавьте тикеры вручную.",
            icon="👀",
            tone="warning",
            page="pages/01d_universe_builder_spot.py",
            page_label="Открыть Universe Builder",
        )

    return builder.as_list()


def render_user_actions(
    settings,
    brief: Mapping[str, object] | None,
    health: dict[str, dict[str, object]] | None,
    watchlist: Sequence[object] | None,
) -> None:
    st.subheader("Быстрые действия")
    actions = collect_user_actions(settings, brief, health, watchlist)

    if not actions:
        st.success("Все проверки зелёные — можно сосредоточиться на торговле.")
        return

    for index, action in enumerate(actions):
        with st.container(border=True):
            st.markdown(
                build_status_card(
                    str(action["title"]),
                    str(action["description"]),
                    icon=str(action.get("icon") or ""),
                    tone=str(action.get("tone") or "warning"),
                ),
                unsafe_allow_html=True,
            )
            page = action.get("page")
            if isinstance(page, str) and page:
                navigation_link(
                    page,
                    label=action.get("page_label") or "Перейти",
                    key=f"action_nav_{index}_{page}",
                )

        st.markdown("")


def render_onboarding() -> None:
    st.markdown("<div id='onboarding'></div>", unsafe_allow_html=True)
    st.subheader("Первые шаги")
    st.markdown(
        """
        1. Откройте раздел **«Подключение и состояние»** и сохраните API ключи.
        2. Загляните в **«Простой режим»**, чтобы увидеть текущий сигнал, план действий и чат с ботом.
        3. Используйте **«Мониторинг сделок»** для контроля исполнений и результата.
        4. Дополнительные помощники (Telegram, журналы) спрятаны в блоке **«Скрытые инструменты»** ниже.
        """
    )


def primary_shortcuts() -> list[tuple[str, str, str]]:
    return [
        (
            "🔌 Подключение",
            "pages/00_connection_status.py",
            "API ключи, проверка связи и режим DRY-RUN.",
        ),
        (
            "⚙️ Настройки",
            "pages/02_settings.py",
            "Порог сигналов, торговые режимы и лимиты риска.",
        ),
        (
            "🛡 Риск-менеджмент",
            "pages/05_portfolio_risk_spot.py",
            "Контроль экспозиции и баланс портфеля.",
        ),
        (
            "🧭 Простой режим",
            "pages/00_simple_mode.py",
            "Актуальный сигнал, план и чат с ботом.",
        ),
        (
            "📈 Мониторинг",
            "pages/05_trade_monitoring.py",
            "Последние сделки и состояние портфеля.",
        ),
    ]


def render_shortcuts(shortcuts: Sequence[tuple[str, str, str]] | None = None) -> None:
    st.subheader("Основные разделы")
    st.caption(
        "Не знаете, где искать нужный инструмент? Эти кнопки откроют ключевые рабочие страницы."
    )
    items = list(shortcuts) if shortcuts is not None else primary_shortcuts()
    render_navigation_grid(items, columns=3, key_prefix="shortcuts")


def render_data_health(health: dict[str, dict[str, object]] | None) -> None:
    health = health or {}
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
            ok = bool(info.get("ok"))
            tone_candidates = [
                _normalise_tone(info.get(field))
                for field in ("tone", "severity", "status", "level")
                if info.get(field) is not None
            ]
            if ok:
                tone = "success"
            else:
                tone = (
                    min(tone_candidates, key=_tone_priority)
                    if tone_candidates
                    else _normalise_tone(None)
                )
                if tone == "success":
                    tone = "warning"
            if tone not in {"success", "warning", "danger"}:
                tone = "warning" if not ok else "success"
            icon = {"success": "✅", "warning": "⚠️", "danger": "⛔"}.get(tone, "⚠️")
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


def render_market_watchlist(
    watchlist: Sequence[dict[str, object]] | Sequence[Mapping[str, object]]
) -> None:
    st.subheader("Наблюдаемые активы")
    if not watchlist:
        st.caption("Пока нет тикеров в списке наблюдения — бот ждёт новый сигнал.")
        return

    st.dataframe(
        arrow_safe(pd.DataFrame(watchlist)),
        hide_index=True,
        use_container_width=True,
    )


def render_hidden_tools() -> None:
    with st.expander("🫥 Скрытые инструменты для бота"):
        st.caption(
            "Продвинутые аналитические и инженерные панели доступны здесь, чтобы не перегружать основной сценарий."
        )

        groups = [
            (
                "Рынок и сигналы",
                [
                    ("📈 Скринер", "pages/01_screener.py", "Топ ликвидных пар и волатильность."),
                    (
                        "🌐 Universe Builder",
                        "pages/01d_universe_builder_spot.py",
                        "Подбор и фильтрация тикеров для спот-бота.",
                    ),
                ],
            ),
            (
                "Риск и безопасность",
                [
                    ("⚙️ Настройки", "pages/02_settings.py", "Глобальные параметры бота."),
                    ("🛑 KillSwitch", "pages/02c_killswitch_and_api_nanny.py", "Быстрые предохранители."),
                    ("🧽 Гигиена ордеров", "pages/02d_order_hygiene_spot.py", "Чистка зависших заявок."),
                    ("📏 Лимиты ордеров", "pages/02e_spot_order_limits.py", "Контроль размеров и частоты."),
                    ("🧮 Риск портфеля", "pages/05_portfolio_risk_spot.py", "Декомпозиция риска по позициям."),
                    ("🧭 HRP vs VolTarget", "pages/05b_hrp_vs_voltarget_spot.py", "Сравнение ребалансировок."),
                    ("⚡ WS контроль", "pages/05_ws_control.py", "Статус real-time соединений."),
                    ("🕸️ WS монитор", "pages/05b_ws_monitor.py", "Трафик и задержки WebSocket."),
                    ("🧰 Reconcile", "pages/09_reconcile.py", "Сверка позиций и журналов."),
                    ("⚙️ Здоровье", "pages/11_health_and_status.py", "Диагностика инфраструктуры."),
                    ("🩺 Time Sync", "pages/00c_health_timesync.py", "Синхронизация времени и задержки."),
                ],
            ),
            (
                "Торговые инструменты",
                [
                    ("🧩 TWAP", "pages/04c_twap_spot.py", "Пакетное исполнение крупного объёма."),
                    ("⚡ Live OB Impact", "pages/04d_live_ob_impact_spot.py", "Воздействие на стакан в режиме live."),
                    ("🧪 Impact Analyzer", "pages/04d_impact_analyzer_spot.py", "Аналитика влияния сделок."),
                    ("🧠 EV Tuner", "pages/04e_ev_tuner_spot.py", "Оптимизация ожидаемой доходности."),
                    ("🔁 Правила", "pages/04f_rules_refresher_spot.py", "Напоминания по дисциплине."),
                    ("🧰 Overrides", "pages/04g_overrides_spot.py", "Ручные корректировки сигналов."),
                    ("🌊 Liquidity", "pages/04h_liquidity_sampler_spot.py", "Замер ликвидности по бирже."),
                    ("🔗 Trade Pairs", "pages/06_trade_pairs_spot.py", "Связанные пары для хеджей."),
                ],
            ),
            (
                "PnL и отчётность",
                [
                    ("💰 PnL дашборд", "pages/06_pnl_dashboard.py", "История доходности и метрики."),
                    ("📊 Портфель", "pages/06_portfolio_dashboard.py", "Структура активов и динамика."),
                    ("💰 PnL мониторинг", "pages/10_pnl_monitoring.py", "Детальный журнал сделок."),
                    ("📉 Shortfall", "pages/10b_shortfall_report.py", "Контроль просадок и недополученной прибыли."),
                ],
            ),
            (
                "Коммуникации",
                [
                    ("🤖 Telegram", "pages/06_telegram_bot.py", "Настройка уведомлений и heartbeat."),
                    ("🪵 Логи", "pages/07_logs.py", "Журнал действий и системных сообщений."),
                ],
            ),
        ]

        tab_titles = [title for title, _ in groups]
        tabs = st.tabs(tab_titles)

        for group_index, (tab, (_, items)) in enumerate(zip(tabs, groups)):
            with tab:
                render_navigation_grid(items, key_prefix=f"hidden_{group_index}")


def render_action_plan(
    plan_steps: Sequence[object] | None,
    safety_notes: Sequence[object] | None,
    risk_summary: str | None,
) -> None:
    steps = [str(step) for step in plan_steps or [] if str(step).strip()]
    notes = [str(note) for note in safety_notes or [] if str(note).strip()]

    plan_html = "".join(f"<li>{step}</li>" for step in steps)
    safety_html = "".join(f"<li>{note}</li>" for note in notes)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Что делаем дальше")
        st.markdown(f"<ol class='checklist'>{plan_html}</ol>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("#### Памятка безопасности")
        st.markdown(f"<ul class='safety-list'>{safety_html}</ul>", unsafe_allow_html=True)
        summary_text = str(risk_summary or "").replace("\n", "  \n")
        if summary_text.strip():
            st.caption(summary_text)


def render_guides(
    settings,
    plan_steps: Sequence[object] | None,
    safety_notes: Sequence[object] | None,
    risk_summary: str | None,
    brief: Mapping[str, object] | None,
) -> None:
    st.subheader("Поддержка и советы")
    plan_tab, onboarding_tab, tips_tab = st.tabs(["План действий", "Первые шаги", "Подсказки"])

    with plan_tab:
        render_action_plan(plan_steps, safety_notes, risk_summary)

    with onboarding_tab:
        render_onboarding()

    with tips_tab:
        render_tips(settings, brief)


def render_tips(settings, brief: Mapping[str, object] | None) -> None:
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
        if active_dry_run(settings):
            st.info("DRY-RUN активен: безопасно тестируйте стратегии перед реальной торговлей.")
        else:
            st.warning("DRY-RUN выключен. Проверьте лимиты риска перед запуском торговых сценариев.")
        status_age = _safe_float(brief.get("status_age") if isinstance(brief, Mapping) else None, None)
        if status_age is not None and status_age > 300:
            st.error(
                "Данные сигналов не обновлялись более 5 минут. Нажмите «Обновить сейчас» или убедитесь, что процесс обновления сигналов запущен.",
            )
        if not (active_api_key(settings) and active_api_secret(settings)):
            st.error("API ключи не указаны — торговля недоступна, пока вы не добавите их.")
            navigation_link(
                "pages/00_connection_status.py",
                label="Добавить API ключи",
                icon="🔑",
                key="tips_api_keys_link",
            )


def main() -> None:
    safe_set_page_config(page_title="Bybit Spot Guardian", page_icon="🧠", layout="wide")

    ensure_keys()
    state = st.session_state

    theme_dir = Path(__file__).resolve().parent / "ui"
    theme_files = {"dark": "theme.css", "light": "theme_light.css"}
    theme_name = str(state.get("ui_theme", "dark")).lower()
    theme_path = theme_dir / theme_files.get(theme_name, "theme.css")
    if not theme_path.exists():
        theme_path = theme_dir / "theme.css"
    if theme_path.exists():
        try:
            inject_css(theme_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - IO errors
            pass

    # Ensure numeric values line up across tables and metrics.
    st.markdown(tabular_numeric_css(), unsafe_allow_html=True)

    settings = get_settings()

    key_present = bool(active_api_key(settings))
    secret_present = bool(active_api_secret(settings))
    if not (key_present and secret_present):
        missing_fields = []
        if not key_present:
            missing_fields.append("API Key")
        if not secret_present:
            missing_fields.append("API Secret")
        show_error_banner(
            "API ключ и секрет не указаны. Добавьте их на странице подключения, чтобы запустить бота.",
            title="Требуется подключение",
        )
        render_connection_gate(settings, missing_fields=missing_fields)
        st.stop()

    try:
        validate_runtime_credentials()
    except CredentialValidationError as cred_err:
        show_error_banner(str(cred_err), title="Проверка ключей")
        render_connection_gate(
            settings,
            missing_fields=[],
            validation_error=str(cred_err),
        )
        st.stop()

    ensure_background_services()

    kill_state = get_kill_switch_state()

    auto_enabled = bool(state.get("auto_refresh_enabled", BASE_SESSION_STATE["auto_refresh_enabled"]))
    refresh_interval = int(state.get("refresh_interval", BASE_SESSION_STATE["refresh_interval"]))
    auto_holds = get_auto_refresh_holds(state)

    def _trigger_refresh(*, delay: float = 0.0) -> None:
        clear_data_caches()
        if delay > 0:
            time.sleep(delay)
        st.experimental_rerun()

    shortcuts = primary_shortcuts()
    in_page_shortcuts = [
        ("🟢 Обзор: статус", "#status-bar", "Прокрутить к статус-бару здоровья бота."),
        ("⚡ Обзор: быстрые действия", "#quick-actions", "Перейти к списку рекомендаций и CTA."),
        ("🚀 Обзор: онбординг", "#onboarding", "Шаги быстрого запуска и подсказки."),
    ]
    command_palette(shortcuts + in_page_shortcuts)

    with st.sidebar:
        st.header("🚀 Быстрый ордер")
        trade_ticket(
            settings=settings,
            client_factory=cached_api_client,
            state=state,
            on_success=[_trigger_refresh],
            key_prefix="quick_trade",
            compact=True,
            submit_label="Отправить ордер",
            instance="primary",
        )

        st.divider()
        st.header("🛡️ Пауза и Kill-Switch")
        st.caption(
            "Выберите режим: кратковременная пауза с автоматическим возобновлением или полная "
            "остановка до ручного запуска (Kill-Switch)."
        )
        kill_reason = st.text_input(
            "Комментарий",
            value=state.get("kill_reason", BASE_SESSION_STATE.get("kill_reason", "Manual kill-switch")),
            key="kill_reason",
            help="Будет прикреплён к выбранному режиму остановки.",
        )

        selected_mode = state.get("kill_mode", BASE_SESSION_STATE.get("kill_mode", "pause"))
        mode_label_map = {
            "pause": "⏸ Пауза на время",
            "kill": "🛑 Kill-Switch (ручной перезапуск)",
        }
        mode = st.radio(
            "Режим остановки",
            options=list(mode_label_map.keys()),
            index=0 if selected_mode == "pause" else 1,
            format_func=lambda key: mode_label_map.get(key, key),
            key="kill_mode",
        )
        if mode != "kill":
            state.pop("kill_switch_confirm_pending", None)

        pause_minutes_widget = st.number_input(
            "Пауза (мин)",
            min_value=5,
            max_value=1440,
            step=5,
            value=int(state.get("pause_minutes", BASE_SESSION_STATE.get("pause_minutes", 60))),
            disabled=kill_state.paused or mode == "kill",
            key="pause_minutes",
            help="Включить паузу автоматики на заданное количество минут.",
        )
        pause_minutes = float(state.get("pause_minutes", pause_minutes_widget))

        if kill_state.paused:
            if getattr(kill_state, "manual", False):
                st.warning("Kill-Switch активен — автоматизация остановлена до ручного возобновления.")
            else:
                st.success("Автоматизация приостановлена.")
                if kill_state.until:
                    remaining_minutes = max((kill_state.until - time.time()) / 60.0, 0.0)
                    st.caption(f"До возобновления ≈ {remaining_minutes:.1f} мин.")
            if kill_state.reason:
                st.caption(f"Причина: {kill_state.reason}")
            if st.button("▶️ Возобновить работу", use_container_width=True):
                resume_ok = backend_resume_kill_switch()
                if not resume_ok:
                    st.info("Backend недоступен, Kill-Switch возобновлен локально.")
                _trigger_refresh()
        else:
            if mode == "pause":
                if st.button("⏸ Поставить на паузу", use_container_width=True):
                    pause_ok = backend_pause_kill_switch(pause_minutes, kill_reason or "Paused via dashboard")
                    if not pause_ok:
                        st.info("Backend недоступен, пауза применена локально.")
                    _trigger_refresh()
            else:
                confirm_pending = bool(state.get("kill_switch_confirm_pending", False))
                if not confirm_pending:
                    if st.button("🛑 Активировать Kill-Switch", use_container_width=True):
                        state["kill_switch_confirm_pending"] = True
                else:
                    st.warning("Вы уверены, что хотите вручную остановить бота?")
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button(
                        "Да, остановить",
                        use_container_width=True,
                        key="kill_switch_confirm_yes",
                    ):
                        pause_ok = backend_pause_kill_switch(None, kill_reason or "Manual kill-switch")
                        if not pause_ok:
                            st.info("Backend недоступен, Kill-Switch активирован локально.")
                        state["kill_switch_confirm_pending"] = False
                        _trigger_refresh()
                    if cancel_col.button(
                        "Отмена",
                        use_container_width=True,
                        key="kill_switch_confirm_no",
                    ):
                        state["kill_switch_confirm_pending"] = False

        if kill_state.paused and getattr(kill_state, "manual", False):
            st.caption("Kill-Switch активен до ручного возобновления.")

        st.divider()
        trade_ticket(
            settings=settings,
            client_factory=cached_api_client,
            state=state,
            on_success=[lambda: _trigger_refresh(delay=1.0)],
            key_prefix="quick_trade",
            compact=True,
            submit_label="Отправить ордер",
            instance="secondary",
        )

        st.divider()
        st.header("🌐 Фильтры сигналов")
        actionable_only = st.checkbox(
            "Только actionable",
            value=bool(state.get("signals_actionable_only", False)),
            key="signals_actionable_only",
            help="Показывать только сигналы, по которым можно действовать прямо сейчас.",
        )
        track_value_change(
            state,
            "signals_actionable_only",
            actionable_only,
            reason="Фильтры сигналов обновлены",
            cooldown=3.0,
        )
        ready_only = st.checkbox(
            "Только готовые",
            value=bool(state.get("signals_ready_only", False)),
            key="signals_ready_only",
            help="Оставлять только сигналы, прошедшие подготовку Guardian Bot.",
        )
        track_value_change(
            state,
            "signals_ready_only",
            ready_only,
            reason="Фильтры сигналов обновлены",
            cooldown=3.0,
        )
        hide_skipped = st.checkbox(
            "Скрыть пропуски",
            value=bool(state.get("signals_hide_skipped", False)),
            key="signals_hide_skipped",
            help="Скрывать сигналы, пропущенные из-за лимитов риска.",
        )
        track_value_change(
            state,
            "signals_hide_skipped",
            hide_skipped,
            reason="Фильтры сигналов обновлены",
            cooldown=3.0,
        )
        min_ev = st.number_input(
            "Мин. EV (bps)",
            min_value=0.0,
            step=1.0,
            value=float(state.get("signals_min_ev", 0.0)),
            key="signals_min_ev",
            help="Минимальная ожидаемая выгода в базисных пунктах (1 б.п. = 0.01%).",
        )
        track_value_change(
            state,
            "signals_min_ev",
            float(min_ev),
            reason="Фильтры сигналов обновлены",
            cooldown=3.0,
        )
        min_prob = st.slider(
            "Мин. вероятность (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=float(state.get("signals_min_probability", 0.0)),
            key="signals_min_probability",
            help="Минимальная вероятность, при которой сигнал попадёт в список.",
        )
        track_value_change(
            state,
            "signals_min_probability",
            float(min_prob),
            reason="Фильтры сигналов обновлены",
            cooldown=3.0,
        )

        st.divider()
        st.header("⏱ Обновление данных")
        auto_enabled = st.toggle(
            "Автообновление",
            value=auto_enabled,
            help="Автоматически обновлять данные без ручного вмешательства.",
        )
        track_value_change(
            state,
            "auto_refresh_enabled",
            auto_enabled,
            reason="Настройки автообновления изменены",
            cooldown=4.0,
        )
        refresh_interval = st.slider(
            "Интервал, сек",
            min_value=5,
            max_value=120,
            value=refresh_interval,
            help="Как часто обновлять данные при активном автообновлении.",
        )
        track_value_change(
            state,
            "refresh_interval",
            refresh_interval,
            reason="Настройки автообновления изменены",
            cooldown=4.0,
        )
        idle_interval_default = int(
            state.get("refresh_idle_interval", BASE_SESSION_STATE.get("refresh_idle_interval", 8))
        )
        idle_interval = st.slider(
            "Когда просто смотрю (сек)",
            min_value=3,
            max_value=60,
            value=idle_interval_default,
            help="Интервал обновления, когда вы наблюдаете за дашбордом без активных действий.",
        )
        track_value_change(
            state,
            "refresh_idle_interval",
            idle_interval,
            reason="Настройки автообновления изменены",
            cooldown=4.0,
        )
        idle_after_default = int(
            state.get("refresh_idle_after", BASE_SESSION_STATE.get("refresh_idle_after", 45.0))
        )
        idle_after = st.slider(
            "Переход в наблюдение через (сек)",
            min_value=10,
            max_value=300,
            step=5,
            value=idle_after_default,
            help="Через сколько секунд без взаимодействий ускорять обновления.",
        )
        track_value_change(
            state,
            "refresh_idle_after",
            float(idle_after),
            reason="Настройки автообновления изменены",
            cooldown=4.0,
        )
        refresh_now = st.button("Обновить сейчас", use_container_width=True)
        state["auto_refresh_enabled"] = auto_enabled
        state["refresh_interval"] = refresh_interval
        state["refresh_idle_interval"] = int(idle_interval)
        state["refresh_idle_after"] = float(idle_after)
        if refresh_now:
            note_user_interaction("Ручное обновление", cooldown=1.0)
            _trigger_refresh()

        last_interaction_ts = get_last_interaction_timestamp(state)
        elapsed_since_interaction = None
        if last_interaction_ts is not None:
            elapsed_since_interaction = max(time.time() - last_interaction_ts, 0.0)

        if not auto_enabled:
            st.caption("Автообновление отключено — используйте ручное обновление при необходимости.")
        elif auto_holds:
            st.caption(
                "Автообновление временно приостановлено: "
                + "; ".join(auto_holds)
            )
        else:
            use_idle_mode = (
                elapsed_since_interaction is None
                or elapsed_since_interaction >= float(idle_after)
            )
            current_interval = idle_interval if use_idle_mode else refresh_interval
            mode_label = "наблюдение" if use_idle_mode else "активный ввод"
            st.caption(
                f"Сейчас: каждые {int(current_interval)} с ({mode_label})."
            )

    effective_auto_refresh = auto_enabled and not auto_holds

    adaptive_interval = max(1, int(refresh_interval))
    idle_interval_seconds = max(1, int(state.get("refresh_idle_interval", 8)))
    idle_after_seconds = float(state.get("refresh_idle_after", 45.0))
    last_interaction_ts = get_last_interaction_timestamp(state)
    if last_interaction_ts is None or (time.time() - last_interaction_ts) >= idle_after_seconds:
        adaptive_interval = idle_interval_seconds

    if effective_auto_refresh:
        auto_refresh(adaptive_interval, key="home_auto_refresh_v2")

    guardian_snapshot = cached_guardian_snapshot()
    ws_snapshot = cached_ws_snapshot()
    preflight_snapshot = cached_preflight_snapshot()

    guardian_state = guardian_snapshot.get("state") if isinstance(guardian_snapshot, Mapping) else {}
    guardian_state = guardian_state if isinstance(guardian_state, Mapping) else {}
    report = guardian_state.get("report") if isinstance(guardian_state.get("report"), Mapping) else {}

    brief_payload = guardian_state.get("brief") if isinstance(guardian_state.get("brief"), Mapping) else {}
    if not brief_payload and isinstance(report.get("brief"), Mapping):
        brief_payload = report.get("brief", {})  # type: ignore[assignment]

    health = _normalise_health(report.get("health"))
    watchlist = _normalise_watchlist(report.get("watchlist"))
    actions = collect_user_actions(settings, brief_payload, health, watchlist)

    guardian_error = guardian_snapshot.get("error")
    if guardian_error:
        show_error_banner(
            "Фоновый процесс Guardian сообщил об ошибке. Проверьте логи и перезапустите автоматику при необходимости.",
            title="Фоновый сервис Guardian",
            details=str(guardian_error),
        )

    preflight_error = preflight_snapshot.get("error")
    if preflight_error:
        show_error_banner(
            "Проверка перед запуском завершилась с ошибкой. Проверьте настройки и повторите попытку.",
            title="Проверка перед запуском",
            details=str(preflight_error),
        )

    def _state_float(key: str, default: float = 0.0) -> float:
        value = state.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    signal_filters = {
        "actionable_only": bool(state.get("signals_actionable_only", False)),
        "ready_only": bool(state.get("signals_ready_only", False)),
        "hide_skipped": bool(state.get("signals_hide_skipped", False)),
        "min_ev_bps": _state_float("signals_min_ev", 0.0),
        "min_probability": _state_float("signals_min_probability", 0.0),
    }

    render_header(
        settings,
        report=report,
        guardian_snapshot=guardian_snapshot,
        ws_snapshot=ws_snapshot,
        kill_switch=kill_state,
    )

    st.markdown("### Обзор")
    with st.container(border=True):
        st.markdown("<div id='status-bar'></div>", unsafe_allow_html=True)
        status_bar(
            settings,
            guardian_snapshot=guardian_snapshot,
            ws_snapshot=ws_snapshot,
            report=report,
            kill_switch=kill_state,
        )
        metrics_strip(report)
        if not guardian_state:
            st.info(
                "Фоновые службы подготавливают данные бота — свежая сводка появится через несколько секунд."
            )

    summary_cols = st.columns([1.5, 1.2, 1.1])
    with summary_cols[0]:
        render_signal_brief(
            brief_payload,
            report.get("score") if isinstance(report, Mapping) else {},
            settings=settings,
        )
    with summary_cols[1]:
        render_user_actions(settings, brief_payload, health, watchlist)
        render_data_health(health)
    with summary_cols[2]:
        render_status(settings)
        render_ws_telemetry(ws_snapshot)
        render_shortcuts(shortcuts)

    if watchlist:
        render_market_watchlist(watchlist)

    render_hidden_tools()

    plan_steps = report.get("plan_steps") if isinstance(report, Mapping) else None
    safety_notes = report.get("safety_notes") if isinstance(report, Mapping) else None
    risk_summary = report.get("risk_summary") if isinstance(report, Mapping) else None
    render_guides(settings, plan_steps, safety_notes, risk_summary, brief_payload)

    detail_tabs = st.tabs(["Торговля", "Кошелёк", "Настройки", "Логи"])

    with detail_tabs[0]:
        st.markdown("#### Сигналы и сделки")
        signals_table(
            report.get("symbol_plan") if isinstance(report, Mapping) else {},
            filters=signal_filters,
            table_key="signals_table_main",
        )
        caution = ""
        if isinstance(brief_payload, Mapping):
            caution = str(brief_payload.get("caution") or "").strip()
        if caution:
            st.warning(caution)

        st.divider()
        trade_cols = st.columns([1.4, 1])
        with trade_cols[0]:
            orders_table(report, state=state)
        with trade_cols[1]:
            trade_ticket(
                settings,
                client_factory=cached_api_client,
                state=state,
                on_success=[lambda: _trigger_refresh(delay=1.0)],
            )

    with detail_tabs[1]:
        wallet_overview(report)

    with detail_tabs[2]:
        st.subheader("Стратегия и среда")
        buy_threshold = float(getattr(settings, "ai_buy_threshold", 0.52) * 100.0)
        sell_threshold = float(getattr(settings, "ai_sell_threshold", 0.42) * 100.0)
        min_ev = float(getattr(settings, "ai_min_ev_bps", 12.0))
        kill_streak = int(getattr(settings, "ai_kill_switch_loss_streak", 0) or 0)
        kill_cooldown = float(getattr(settings, "ai_kill_switch_cooldown_min", 60.0) or 0.0)
        refresh_interval = int(state.get("refresh_interval", BASE_SESSION_STATE.get("refresh_interval", 12)))
        theme_name = str(state.get("ui_theme", "dark")).lower()

        with st.form("strategy_settings"):
            st.markdown("#### Пороговые значения")
            buy_value = st.number_input(
                "Порог покупки (%)",
                min_value=0.0,
                max_value=100.0,
                value=buy_threshold,
                step=0.5,
                help="Минимальная вероятность для входа в сделку.",
            )
            sell_value = st.number_input(
                "Порог продажи (%)",
                min_value=0.0,
                max_value=100.0,
                value=sell_threshold,
                step=0.5,
                help="Максимальная вероятность, ниже которой бот закрывает позицию.",
            )
            ev_value = st.number_input(
                "Минимальная выгода (bps)",
                min_value=0.0,
                value=min_ev,
                step=1.0,
                help="Минимальная ожидаемая выгода в базисных пунктах (1 б.п. = 0.01%).",
            )
            kill_streak_value = st.number_input(
                "Kill-switch: серия убыточных сделок",
                min_value=0,
                value=kill_streak,
                step=1,
                help="После скольких убыточных сделок подряд включать аварийную паузу.",
            )
            kill_cooldown_value = st.number_input(
                "Kill-switch: пауза (мин)",
                min_value=0.0,
                value=kill_cooldown,
                step=5.0,
                help="Сколько минут ждать перед возобновлением после срабатывания kill-switch.",
            )

            st.subheader("Режим работы")
            dry_run_value = st.toggle(
                "Учебный режим (DRY-RUN)",
                value=active_dry_run(settings),
                help="В тестовом режиме сделки не отправляются на биржу.",
            )
            st.caption("DRY-RUN ведёт только локальный журнал и безопасен для проверки сигналов без риска для депозита.")
            network_value = st.selectbox(
                "Сеть",
                ["Testnet", "Mainnet"],
                index=0 if settings.testnet else 1,
                help="Выберите торговую среду: тестовую или основную.",
            )
            st.caption(
                "Testnet — биржевой полигон без реальных средств, Mainnet — рабочие ордера на живом счёте."
            )

            st.subheader("Интерфейс")
            refresh_slider = st.slider("Интервал автообновления (сек)", min_value=5, max_value=120, value=refresh_interval, key="settings_refresh_interval")
            if refresh_slider != state.get("refresh_interval"):
                state["refresh_interval"] = refresh_slider
            theme_options = [("dark", "Тёмная тема"), ("light", "Светлая тема")]
            current_theme_index = next((index for index, (value, _) in enumerate(theme_options) if value == theme_name), 0)
            selected_theme = st.selectbox(
                "Тема интерфейса",
                theme_options,
                index=current_theme_index,
                format_func=lambda item: item[1],
            )
            if isinstance(selected_theme, tuple):
                chosen_theme = selected_theme[0]
            else:
                chosen_theme = theme_name
            if chosen_theme != theme_name:
                state["ui_theme"] = chosen_theme
                st.experimental_rerun()

            if st.button("Сохранить настройки"):
                update_settings(
                    ai_buy_threshold=buy_value / 100.0,
                    ai_sell_threshold=sell_value / 100.0,
                    ai_min_ev_bps=ev_value,
                    ai_kill_switch_loss_streak=kill_streak_value,
                    ai_kill_switch_cooldown_min=kill_cooldown_value,
                    dry_run=dry_run_value,
                    testnet=(network_value == "Testnet"),
                )
                settings = get_settings(force_reload=True)
                clear_data_caches()
                st.success("Настройки сохранены.")

    with detail_tabs[3]:
        log_path = Path(__file__).resolve().parent / "_data" / "logs" / "app.log"
        log_viewer(log_path, state=state)


if __name__ == "__main__":
    main()
