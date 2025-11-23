"""Composable Streamlit components used across the dashboard tabs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, get_args
import json
from string import Template

import pandas as pd
import streamlit as st

from bybit_app.utils.ai.kill_switch import KillSwitchState
from bybit_app.utils.envs import (
    active_api_key,
    active_api_secret,
    active_dry_run,
    last_api_client_error,
)
from bybit_app.utils.ui import build_pill, navigation_link, page_slug_from_path
from bybit_app.utils.spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    prepare_spot_market_order,
    prepare_spot_trade_snapshot,
)
from bybit_app.utils.formatting import (
    format_money,
    format_percent,
    format_quantity,
    format_datetime,
)
from bybit_app.ui.state import (
    clear_auto_refresh_hold,
    note_user_interaction,
    set_auto_refresh_hold,
    track_value_change,
)

_STATUS_BADGE_CSS = """
<style>
.status-badge{padding:0.5rem;border-radius:0.75rem;background-color:rgba(15,23,42,0.55);border:1px solid rgba(148,163,184,0.2);margin-bottom:0.5rem;}
.status-badge__label{display:block;margin-bottom:0.2rem;}
.status-badge__value{font-weight:600;font-size:1.05rem;margin-top:0.25rem;}
.status-badge small{display:block;color:rgba(148,163,184,0.85);}
</style>
"""
BadgeTone = Literal["neutral", "success", "warning", "danger", "info"]
_VALID_BADGE_TONES = set(get_args(BadgeTone))


@dataclass(frozen=True)
class StatusBadge:
    """Immutable description of a status badge rendered in the dashboard header."""

    label: str
    value: str
    tone: BadgeTone = "neutral"
    caption: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", str(self.value))
        object.__setattr__(self, "caption", str(self.caption or ""))
        if self.tone not in _VALID_BADGE_TONES:
            object.__setattr__(self, "tone", "neutral")

    def render(self) -> str:
        pill = build_pill(self.label, tone=self.tone)
        caption_html = f"<small>{self.caption}</small>" if self.caption else ""
        return (
            "<div class='status-badge'>"
            f"<div class='status-badge__label'>{pill}</div>"
            f"<div class='status-badge__value'>{self.value}</div>"
            f"{caption_html}"
            "</div>"
        )


def _ensure_status_badge_css() -> None:
    """Inject badge styling once per session to avoid duplicate style blocks."""

    flag = "_status_badge_css_injected"
    if st.session_state.get(flag):
        return
    st.session_state[flag] = True
    st.markdown(_STATUS_BADGE_CSS, unsafe_allow_html=True)


def _render_badge_grid(badges: Sequence[StatusBadge], *, columns: int = 4) -> None:
    if not badges:
        return

    for start in range(0, len(badges), columns):
        row = badges[start : start + columns]
        cols = st.columns(len(row))
        for column, badge in zip(cols, row):
            with column:
                st.markdown(badge.render(), unsafe_allow_html=True)


def render_connection_gate(
    settings: Any,
    *,
    missing_fields: Sequence[str],
    validation_error: str | None = None,
    key: str = "connection_gate",
) -> None:
    """Gate the dashboard until both API credentials are present and validated."""

    key_present = bool(active_api_key(settings))
    secret_present = bool(active_api_secret(settings))
    missing = [field for field in missing_fields]
    validation_issue = bool(validation_error)
    network = "Testnet" if getattr(settings, "testnet", True) else "Mainnet"
    dry_run = active_dry_run(settings)
    ready_count = int(key_present) + int(secret_present)
    readiness_pct = int((ready_count / 2) * 100)
    ready = readiness_pct == 100 and not validation_issue
    api_error = last_api_client_error(settings)

    tone = "ok" if ready else "danger"
    hint = (
        "Проверка не пройдена, исправьте ключи"
        if validation_issue
        else "Осталось добавить секрет"
        if key_present and not secret_present
        else "Заполните ключ и секрет"
    )
    missing_text = (
        ", ".join(missing)
        if missing
        else "ничего — требуется перепроверка" if validation_issue else "ничего — можно запускать"
    )
    error_reason = validation_error or api_error
    error_text = str(error_reason) if error_reason else "Ошибок подключения не обнаружено"
    error_tone = "danger" if error_reason else "muted"
    status_label = (
        "Готов к запуску"
        if ready
        else "Требуется проверка" if validation_issue else "Заполните ключи"
    )

    st.markdown("<div class='gate gate--panel'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='gate__head'>
            <span class='gate__pill'>🔒 Доступ закрыт до ввода ключей</span>
            <h2 class='gate__title'>Один маршрут подключения</h2>
            <p class='gate__lede'>Ключ и секрет вводятся на странице подключения. После валидации мы автоматически поднимаем Guardian, WebSocket и автообновление.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1.2, 1])

    with cols[0]:
        st.markdown(
            """
            <div class='gate__progress gate__progress--wide'>
                <div class='gate__progress-head'>
                    <div class='gate__progress-meta'>
                        <span class='gate__badge gate__badge--muted'>Готовность</span>
                        <span class='gate__progress-score'>{ready}/{total}</span>
                    </div>
                    <span class='gate__badge gate__badge--status gate__badge--{tone}'>{status}</span>
                </div>
                <div class='gate__progress-bar'>
                    <span class='gate__progress-fill gate__progress-fill--{tone}' style='width:{pct}%;'></span>
                </div>
                <div class='gate__progress-caption'>{pct}% · {hint}</div>
                <div class='gate__steps'>
                    <div class='gate__step gate__step--{key_tone}'>
                        <div class='gate__step-title'>API Key</div>
                        <div class='gate__step-status'>{key_emoji} {key_status}</div>
                    </div>
                    <div class='gate__step gate__step--{secret_tone}'>
                        <div class='gate__step-title'>API Secret</div>
                        <div class='gate__step-status'>{secret_emoji} {secret_status}</div>
                    </div>
                </div>
            </div>
            """.format(
                tone=tone,
                pct=readiness_pct,
                hint=hint,
                status=status_label,
                ready=ready_count,
                total=2,
                key_tone="ok" if key_present else "danger",
                secret_tone="ok" if secret_present else "danger",
                key_emoji="✅" if key_present else "⚠️",
                secret_emoji="✅" if secret_present else "⚠️",
                key_status="Есть" if key_present else "Отсутствует",
                secret_status="Есть" if secret_present else "Отсутствует",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='gate__card gate__card--list'>
                <div class='gate__card-title'>Что нужно заполнить</div>
                <ul class='gate__list'>
                    <li class='gate__item gate__item--{key_tone}'><span>API Key</span><span class='gate__badge gate__badge--{key_tone}'>{key_status}</span></li>
                    <li class='gate__item gate__item--{secret_tone}'><span>API Secret</span><span class='gate__badge gate__badge--{secret_tone}'>{secret_status}</span></li>
                </ul>
                <p class='gate__note'>Заполняется один раз: placeholders показывают, что данные сохранены.</p>
                <p class='gate__note'>Отсутствует: {missing}</p>
            </div>
            """.format(
                key_tone="ok" if key_present else "danger",
                secret_tone="ok" if secret_present else "danger",
                key_status="Есть" if key_present else "Отсутствует",
                secret_status="Есть" if secret_present else "Отсутствует",
                missing=missing_text,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='gate__cta'>
                <div>
                    <div class='gate__cta-title'>Один клик к подключению</div>
                    <p class='gate__note'>После сохранения ключей фоновые службы стартуют автоматически.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        navigation_link(
            "pages/00_connection_status.py",
            label="🔑 Перейти к настройке API",
            icon=None,
            key=f"{key}_cta",
        )

    with cols[1]:
        st.markdown(
            """
            <div class='gate__card gate__card--meta'>
                <div class='gate__card-title'>Профиль окружения</div>
                <div class='gate__meta-row'>
                    <span>Сеть</span>
                    <span class='gate__badge gate__badge--muted'>{network}</span>
                </div>
                <div class='gate__meta-row'>
                    <span>Режим</span>
                    <span class='gate__badge gate__badge--muted'>{mode}</span>
                </div>
                <div class='gate__meta-row gate__meta-row--strong'>
                    <span>Доступ к боту</span>
                    <span class='gate__badge gate__badge--{tone}'>{status}</span>
                </div>
                <div class='gate__meta-row gate__meta-row--ghost'>
                    <span>API проверка</span>
                    <span class='gate__badge gate__badge--{error_tone}'>{error_label}</span>
                </div>
                <p class='gate__hint'>Креды проверяются сразу; запуск фоновых служб происходит автоматически после подтверждения.</p>
                <div class='gate__hint gate__hint--inline'>{error_text}</div>
            </div>
            """.format(
                network=network,
                mode="DRY-RUN" if dry_run else "Live",
                tone=tone,
                status="Разблокирован" if ready else "Заблокирован",
                error_tone=error_tone,
                error_label="Ошибка" if error_reason else "Ошибок нет",
                error_text=error_text,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='gate__card gate__card--services'>
                <div class='gate__card-title'>Что разблокируется</div>
                <div class='gate__services'>
                    <div class='gate__service'>
                        <div class='gate__service-icon'>🛡️</div>
                        <div>
                            <div class='gate__service-title'>Guardian</div>
                            <div class='gate__service-note'>Мониторинг сигналов и авто-перезапуск.</div>
                        </div>
                    </div>
                    <div class='gate__service'>
                        <div class='gate__service-icon'>🌐</div>
                        <div>
                            <div class='gate__service-title'>WebSocket</div>
                            <div class='gate__service-note'>Потоки цен и ордеров для дашборда.</div>
                        </div>
                    </div>
                    <div class='gate__service'>
                        <div class='gate__service-icon'>⚡</div>
                        <div>
                            <div class='gate__service-title'>Автообновление</div>
                            <div class='gate__service-note'>Фоновые рефреши без ручных перезагрузок.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def command_palette(
    shortcuts: Sequence[tuple[str, str, str]], *, key: str = "command_palette"
) -> None:
    """Inject a lightweight command palette bound to ``Ctrl/Cmd + K``."""

    if not shortcuts:
        return

    flag = f"_bybit_{key}_injected"
    if st.session_state.get(flag):
        return

    items: list[dict[str, str]] = []
    for label, page, description in shortcuts:
        path = str(page)
        target = "page"
        if isinstance(page, str) and page.startswith("#"):
            slug = page.lstrip("#")
            target = "anchor"
        else:
            slug = page_slug_from_path(page)
        items.append(
            {
                "label": str(label),
                "slug": slug,
                "description": str(description),
                "path": path,
                "target": target,
            }
        )

    if not items:
        return

    palette_id = f"bybit-cmd-{uuid4().hex}"
    data_json = json.dumps(items)
    st.session_state[flag] = True

    template = Template(
        """
    <style>
    #$palette_id {
        position: fixed;
        inset: 0;
        display: none;
        align-items: flex-start;
        justify-content: center;
        padding-top: 10vh;
        z-index: 1000;
        background: rgba(15, 23, 42, 0.35);
        backdrop-filter: blur(2px);
    }
    #$palette_id.is-visible {
        display: flex;
    }
    #$palette_id .palette-panel {
        width: min(520px, 90vw);
        background: rgba(15, 23, 42, 0.97);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.45);
        color: #f8fafc;
        font-family: inherit;
    }
    #$palette_id .palette-search {
        width: 100%;
        padding: 0.6rem 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(15, 23, 42, 0.7);
        color: inherit;
        font-size: 1rem;
        outline: none;
        margin-bottom: 0.75rem;
    }
    #$palette_id .palette-search::placeholder {
        color: rgba(148, 163, 184, 0.9);
    }
    #$palette_id .palette-list {
        list-style: none;
        margin: 0;
        padding: 0;
        max-height: 60vh;
        overflow-y: auto;
    }
    #$palette_id .palette-item {
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        transition: background 80ms ease;
    }
    #$palette_id .palette-item[aria-selected="true"] {
        background: rgba(14, 165, 233, 0.25);
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    #$palette_id .palette-item:not([aria-selected="true"]):hover {
        background: rgba(148, 163, 184, 0.16);
    }
    #$palette_id .palette-label {
        font-weight: 600;
        font-size: 1rem;
    }
    #$palette_id .palette-description {
        font-size: 0.85rem;
        color: rgba(226, 232, 240, 0.85);
    }
    </style>
    <div id="$palette_id" class="bybit-command-palette" aria-hidden="true">
        <div class="palette-panel" role="dialog" aria-modal="true">
            <input class="palette-search" type="text" placeholder="Куда перейти? (Ctrl/Cmd + K)" />
            <ul class="palette-list" role="listbox"></ul>
        </div>
    </div>
    <script>
    (function() {
        const items = $data_json;
        const palette = document.getElementById('$palette_id');
        if (!palette || !Array.isArray(items) || !items.length) {
            return;
        }
        const input = palette.querySelector('.palette-search');
        const list = palette.querySelector('.palette-list');
        let filtered = items.slice();
        let activeIndex = 0;
        let open = false;

        function closePalette() {
            open = false;
            palette.classList.remove('is-visible');
            palette.setAttribute('aria-hidden', 'true');
        }

        function selectIndex(index) {
            const options = list.querySelectorAll('.palette-item');
            options.forEach((option, optionIndex) => {
                option.setAttribute('aria-selected', optionIndex === index ? 'true' : 'false');
            });
            activeIndex = index;
        }

        function navigateTo(item) {
            if (!item) {
                return;
            }
            if (item.target === 'anchor' && item.path) {
                closePalette();
                const hash = item.path.startsWith('#') ? item.path : '#' + item.path;
                setTimeout(() => {
                    window.location.hash = hash;
                }, 0);
                return;
            }
            const url = new URL(window.location.href);
            if (item.slug) {
                url.searchParams.set('page', item.slug);
            } else if (item.path) {
                url.searchParams.set('page', item.path);
            }
            window.location.href = url.toString();
        }

        function renderList() {
            list.innerHTML = '';
            filtered.forEach((item, index) => {
                const element = document.createElement('li');
                element.className = 'palette-item';
                element.setAttribute('role', 'option');
                element.dataset.index = String(index);
                element.innerHTML =
                    '<span class="palette-label">' + item.label + '</span>' +
                    '<span class="palette-description">' + item.description + '</span>';
                list.appendChild(element);
            });
            if (!filtered.length) {
                const empty = document.createElement('li');
                empty.className = 'palette-item';
                empty.textContent = 'Не найдено подходящих разделов';
                empty.setAttribute('aria-disabled', 'true');
                list.appendChild(empty);
                activeIndex = -1;
                return;
            }
            const boundedIndex = Math.max(0, Math.min(activeIndex, filtered.length - 1));
            selectIndex(boundedIndex);
        }

        function openPalette() {
            open = true;
            palette.classList.add('is-visible');
            palette.setAttribute('aria-hidden', 'false');
            filtered = items.slice();
            activeIndex = 0;
            renderList();
            setTimeout(() => input.focus(), 0);
        }

        input.addEventListener('input', (event) => {
            const value = String(event.target.value || '').trim().toLowerCase();
            filtered = items.filter((item) => {
                const slugMatch = item.slug && item.slug.toLowerCase().includes(value);
                const pathMatch = item.path && item.path.toLowerCase().includes(value);
                return (
                    item.label.toLowerCase().includes(value) ||
                    item.description.toLowerCase().includes(value) ||
                    slugMatch ||
                    pathMatch
                );
            });
            activeIndex = 0;
            renderList();
        });

        input.addEventListener('keydown', (event) => {
            if (event.key === 'ArrowDown') {
                event.preventDefault();
                if (!filtered.length) {
                    return;
                }
                const next = (activeIndex + 1) % filtered.length;
                selectIndex(next);
            } else if (event.key === 'ArrowUp') {
                event.preventDefault();
                if (!filtered.length) {
                    return;
                }
                const prev = (activeIndex - 1 + filtered.length) % filtered.length;
                selectIndex(prev);
            } else if (event.key === 'Enter') {
                event.preventDefault();
                if (activeIndex >= 0 && filtered[activeIndex]) {
                    navigateTo(filtered[activeIndex]);
                }
            } else if (event.key === 'Escape') {
                event.preventDefault();
                closePalette();
            }
        });

        list.addEventListener('click', (event) => {
            const target = event.target.closest('.palette-item');
            if (!target) {
                return;
            }
            const index = Number(target.dataset.index);
            if (!Number.isNaN(index) && filtered[index]) {
                navigateTo(filtered[index]);
            }
        });

        document.addEventListener('keydown', (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
                event.preventDefault();
                if (open) {
                    closePalette();
                } else {
                    openPalette();
                }
            } else if (event.key === 'Escape' && open) {
                closePalette();
            }
        });

        palette.addEventListener('click', (event) => {
            if (event.target === palette) {
                closePalette();
            }
        });
    })();
    </script>
    """
    )
    html = template.substitute(palette_id=palette_id, data_json=data_json)

    st.markdown(html, unsafe_allow_html=True)


def _format_age(seconds: object) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if value < 1:
        return "<1s"
    if value < 60:
        return f"{value:.0f}s"
    minutes = value / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_tone(
    value: float | None, *, warn_after: float, danger_after: float
) -> BadgeTone:
    if value is None:
        return "warning"
    if value >= danger_after:
        return "danger"
    if value >= warn_after:
        return "warning"
    return "success"


def _numeric_tone(
    value: Any, *, warn_at: float | None = None, danger_at: float | None = None
) -> BadgeTone:
    number = _coerce_float(value)
    if number is None:
        return "warning"
    if danger_at is not None and number <= danger_at:
        return "danger"
    if warn_at is not None and number <= warn_at:
        return "warning"
    return "success"


@dataclass(frozen=True)
class _NumericBadgeSpec:
    label: str
    value: Any
    warn_at: float | None = None
    danger_at: float | None = None
    precision: int = 2
    caption: str = ""

    def build(self) -> StatusBadge:
        return StatusBadge(
            self.label,
            _format_number(self.value, precision=self.precision),
            tone=_numeric_tone(
                self.value, warn_at=self.warn_at, danger_at=self.danger_at
            ),
            caption=self.caption,
        )


@dataclass(frozen=True)
class _AgeBadgeSpec:
    label: str
    value: float | None
    warn_after: float
    danger_after: float
    caption: str = ""
    empty_placeholder: str = "—"

    def build(self) -> StatusBadge:
        display = _format_age(self.value) or self.empty_placeholder
        return StatusBadge(
            self.label,
            display,
            tone=_age_tone(
                self.value, warn_after=self.warn_after, danger_after=self.danger_after
            ),
            caption=self.caption,
        )


def _format_number(value: Any, *, precision: int = 2) -> str:
    number = _coerce_float(value)
    if number is None:
        return "—"
    return f"{number:,.{precision}f}"


def _ensure_mapping(payload: object) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


@dataclass(frozen=True)
class _StatusBarContext:
    testnet: bool
    dry_run: bool
    equity: float | None
    available: float | None
    kill_switch: KillSwitchState
    kill_caption: str
    automation_ok: bool
    automation_ready: bool
    automation_caption: str
    realtime_ok: bool
    realtime_caption: str
    signal_age: float | None
    signal_caption: str
    guardian_age: float | None
    guardian_caption: str
    public_age: float | None
    private_age: float | None
    ws_caption: str
    ws_worst_age: float | None

    @classmethod
    def from_inputs(
        cls,
        settings: Any,
        guardian_snapshot: Mapping[str, Any],
        ws_snapshot: Mapping[str, Any],
        report: Mapping[str, Any] | None,
        kill_switch: KillSwitchState | None,
    ) -> "_StatusBarContext":
        now = time.time()

        def _summarise(payload: Mapping[str, Any]) -> str:
            if not isinstance(payload, Mapping):
                return ""
            message = str(payload.get("message") or "").strip()
            if message:
                return message
            details = payload.get("details")
            if isinstance(details, str):
                detail_text = details.strip()
                if detail_text:
                    return detail_text
            elif isinstance(details, Sequence):
                for entry in details:
                    if isinstance(entry, Mapping):
                        for key in ("title", "description", "message"):
                            text = str(entry.get(key) or "").strip()
                            if text:
                                return text
                    else:
                        text = str(entry or "").strip()
                        if text:
                            return text
            return ""

        def _compact_text(value: str, *, limit: int = 120) -> str:
            value = value.strip()
            if len(value) <= limit:
                return value
            return value[: limit - 1].rstrip() + "…"

        guardian = _ensure_mapping(guardian_snapshot)
        ws = _ensure_mapping(ws_snapshot)
        report_mapping = _ensure_mapping(report)
        portfolio = _ensure_mapping(report_mapping.get("portfolio"))
        totals = _ensure_mapping(portfolio.get("totals"))
        health_payload = _ensure_mapping(report_mapping.get("health"))
        automation_payload = _ensure_mapping(health_payload.get("automation"))
        realtime_payload = _ensure_mapping(health_payload.get("realtime_trading"))

        automation_ok = bool(automation_payload.get("ok"))
        automation_ready = bool(automation_payload.get("actionable"))
        automation_caption = _compact_text(
            _summarise(automation_payload) or str(automation_payload.get("title") or "")
        )
        if not automation_caption:
            automation_caption = "Нет данных"

        realtime_ok = bool(realtime_payload.get("ok"))
        realtime_caption = _compact_text(
            _summarise(realtime_payload) or str(realtime_payload.get("title") or "")
        )
        if not realtime_caption:
            realtime_caption = "Нет данных"

        signal_state = _ensure_mapping(guardian.get("state"))
        signal_age_value = _coerce_float(signal_state.get("age_seconds"))
        signal_report = _ensure_mapping(signal_state.get("report"))
        signal_generated = signal_report.get("generated_at") or signal_report.get(
            "timestamp"
        )
        if signal_age_value is None and signal_generated is not None:
            generated_ts = _coerce_float(signal_generated)
            if generated_ts is not None:
                signal_age_value = max(now - generated_ts, 0.0)

        if signal_age_value is None and signal_generated:
            signal_caption = "Время генерации неизвестно"
        elif signal_age_value is not None:
            signal_caption = f"обновл. {_format_age(signal_age_value)} назад"
        else:
            signal_caption = ""

        guardian_age_value = _coerce_float(guardian.get("age_seconds"))
        restart_count = _coerce_float(guardian.get("restart_count"))
        guardian_caption = f"рестартов: {int(restart_count)}" if restart_count else ""

        ws_status = _ensure_mapping(ws.get("status"))
        private_age_value = _coerce_float(
            _ensure_mapping(ws_status.get("private")).get("age_seconds")
        )
        public_age_value = _coerce_float(
            _ensure_mapping(ws_status.get("public")).get("age_seconds")
        )
        ws_age_candidates = [
            value
            for value in (private_age_value, public_age_value)
            if value is not None
        ]
        ws_worst_age = max(ws_age_candidates) if ws_age_candidates else None

        ws_age_value = _coerce_float(ws.get("age_seconds"))
        ws_caption = (
            f"обновл. {_format_age(ws_age_value)} назад"
            if ws_age_value is not None
            else ""
        )

        kill_state = kill_switch or KillSwitchState(
            paused=False, until=None, reason=None, manual=False
        )
        kill_caption = "Готов"
        if kill_state.paused:
            if getattr(kill_state, "manual", False):
                kill_caption = "до ручного запуска"
            elif kill_state.until:
                remaining = max(kill_state.until - now, 0.0)
                kill_caption = f"до {_format_age(remaining)}"
            else:
                kill_caption = "Активен"
            if kill_state.reason:
                kill_caption += f" · {kill_state.reason}"

        return cls(
            testnet=bool(getattr(settings, "testnet", True)),
            dry_run=active_dry_run(settings),
            equity=_coerce_float(totals.get("total_equity") or totals.get("equity")),
            available=_coerce_float(
                totals.get("available_balance") or totals.get("available")
            ),
            kill_switch=kill_state,
            kill_caption=kill_caption,
            automation_ok=automation_ok,
            automation_ready=automation_ready,
            automation_caption=automation_caption,
            realtime_ok=realtime_ok,
            realtime_caption=realtime_caption,
            signal_age=signal_age_value,
            signal_caption=signal_caption,
            guardian_age=guardian_age_value,
            guardian_caption=guardian_caption,
            public_age=public_age_value,
            private_age=private_age_value,
            ws_caption=ws_caption,
            ws_worst_age=ws_worst_age,
        )

    def badges(self) -> list[StatusBadge]:
        ws_value = (
            f"pub {_format_age(self.public_age)} · priv {_format_age(self.private_age)}"
        )
        badge_factories: Sequence[Callable[[], StatusBadge]] = (
            lambda: StatusBadge(
                "Network",
                "Testnet" if self.testnet else "Mainnet",
                tone="warning" if self.testnet else "success",
            ),
            lambda: StatusBadge(
                "Mode",
                "DRY-RUN" if self.dry_run else "Live",
                tone="warning" if self.dry_run else "success",
            ),
            lambda: _NumericBadgeSpec("Equity", self.equity, warn_at=0.0).build(),
            lambda: _NumericBadgeSpec(
                "Balance",
                self.available,
                danger_at=0.0,
            ).build(),
            lambda: StatusBadge(
                "Kill-Switch",
                "Paused" if self.kill_switch.paused else "Ready",
                tone="danger" if self.kill_switch.paused else "success",
                caption=self.kill_caption,
            ),
            lambda: StatusBadge(
                "Auto",
                "Готов"
                if self.automation_ready
                else ("Вкл." if self.automation_ok else "Выкл."),
                tone="success" if self.automation_ok else "warning",
                caption=self.automation_caption,
            ),
            lambda: StatusBadge(
                "Bybit",
                "Онлайн" if self.realtime_ok else "Offline",
                tone="success" if self.realtime_ok else "danger",
                caption=self.realtime_caption,
            ),
            lambda: _AgeBadgeSpec(
                "Signal",
                self.signal_age,
                warn_after=120.0,
                danger_after=300.0,
                caption=self.signal_caption,
            ).build(),
            lambda: _AgeBadgeSpec(
                "Guardian",
                self.guardian_age,
                warn_after=120.0,
                danger_after=300.0,
                caption=self.guardian_caption,
            ).build(),
            lambda: StatusBadge(
                "WebSocket",
                ws_value,
                tone=_age_tone(
                    self.ws_worst_age,
                    warn_after=60.0,
                    danger_after=90.0,
                ),
                caption=self.ws_caption,
            ),
        )

        return [factory() for factory in badge_factories]


def show_error_banner(
    message: str,
    *,
    title: str | None = None,
    details: Mapping[str, Any] | str | None = None,
) -> None:
    """Render a consistent error banner with optional structured details."""

    header = message.strip()
    if title:
        header = f"**{title.strip()}:** {header}" if header else f"**{title.strip()}**"
    if isinstance(details, Mapping) and details:
        with st.container(border=True):
            st.error(header or "Произошла ошибка")
            st.json(details, expanded=False)
    elif isinstance(details, str) and details.strip():
        st.error(f"{header}\n\n{details.strip()}")
    else:
        st.error(header or "Произошла ошибка")


def status_bar(
    settings: Any,
    *,
    guardian_snapshot: Mapping[str, Any],
    ws_snapshot: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
    kill_switch: KillSwitchState | None = None,
) -> None:
    """Display the high level status strip with connection, balance and latency hints."""

    st.subheader("⚙️ Системный пульс")

    api_error = last_api_client_error()
    context = _StatusBarContext.from_inputs(
        settings,
        guardian_snapshot,
        ws_snapshot,
        report,
        kill_switch,
    )

    _ensure_status_badge_css()
    with st.container(border=True):
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.caption("Режим")
            mode = "Тестнет" if context.testnet else "Основной"
            st.markdown(
                build_pill(mode, tone="info" if context.testnet else "success"),
                unsafe_allow_html=True,
            )
        with meta_cols[1]:
            st.caption("DRY-RUN")
            st.markdown(
                build_pill(
                    "Включен" if context.dry_run else "Выключен",
                    tone="warning" if context.dry_run else "neutral",
                ),
                unsafe_allow_html=True,
            )
        with meta_cols[2]:
            st.caption("Баланс")
            balance_text = (
                format_money(context.equity, currency="USD")
                if context.equity is not None
                else "—"
            )
            st.markdown(build_pill(balance_text, tone="neutral"), unsafe_allow_html=True)
        with meta_cols[3]:
            st.caption("Доступно")
            available_text = (
                format_money(context.available, currency="USD")
                if context.available is not None
                else "—"
            )
            st.markdown(
                build_pill(available_text, tone="neutral"),
                unsafe_allow_html=True,
            )

        st.caption("Показатели стабильности")
        _render_badge_grid(context.badges(), columns=4)

    has_keys = bool(active_api_key(settings) and active_api_secret(settings))
    if not has_keys:
        show_error_banner(
            "Добавьте API ключ и секрет, чтобы бот мог размещать ордера. Нажмите кнопку ниже, чтобы открыть настройки.",
            title="API ключи отсутствуют",
        )
        navigation_link(
            "pages/00_✅_Подключение_и_Состояние.py",
            label="Добавить API ключи",
            icon="🔑",
            key="status_api_keys_link",
        )
    elif api_error:
        show_error_banner(
            "Нет связи с биржей Bybit. Проверьте интернет или повторите попытку позже.",
            title="API клиент недоступен",
            details=api_error,
        )


def metrics_strip(report: Mapping[str, Any]) -> None:
    """Render quick metrics summarising actionable opportunities and exposure."""

    st.subheader("📊 Snapshot")
    stats = report.get("statistics") if isinstance(report, Mapping) else {}
    plan = report.get("symbol_plan") if isinstance(report, Mapping) else {}
    portfolio = report.get("portfolio") if isinstance(report, Mapping) else {}
    totals = portfolio.get("totals") if isinstance(portfolio, Mapping) else {}

    def _num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    actionable = stats.get("actionable_count") if isinstance(stats, Mapping) else 0
    ready = stats.get("ready_count") if isinstance(stats, Mapping) else 0
    positions = stats.get("position_count") if isinstance(stats, Mapping) else 0
    limit = (
        stats.get("limit")
        if isinstance(stats, Mapping)
        else plan.get("limit")
        if isinstance(plan, Mapping)
        else 0
    )
    equity = (
        totals.get("total_equity")
        if isinstance(totals, Mapping)
        else totals.get("equity")
        if isinstance(totals, Mapping)
        else None
    )
    pnl = totals.get("realized_pnl") if isinstance(totals, Mapping) else None

    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Actionable", actionable)
        cols[1].metric("Ready", ready)
        cols[2].metric("Positions", positions, f"Limit {limit}" if limit else None)
        if equity is not None:
            cols[3].metric("Equity", f"{_num(equity):,.2f}")
        else:
            cols[3].metric("Realized PnL", f"{_num(pnl):,.2f}")

        if limit:
            utilisation = max(0.0, min(1.0, positions / limit)) if limit else 0.0
            st.progress(
                utilisation,
                text=f"Использование лимита: {positions}/{limit}",
            )


def _normalise_priority_table(plan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan, Mapping):
        return []
    table = plan.get("priority_table")
    if not isinstance(table, Sequence):
        return []

    rows: list[dict[str, Any]] = []
    for entry in table:
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "symbol": entry.get("symbol"),
                "priority": entry.get("priority"),
                "trend": entry.get("trend"),
                "probability_pct": entry.get("probability_pct"),
                "ev_bps": entry.get("ev_bps"),
                "actionable": bool(entry.get("actionable")),
                "ready": bool(entry.get("ready")),
                "skip_reason": entry.get("skip_reason") or "",
            }
        )
    return rows


def signals_table(
    plan: Mapping[str, Any] | None,
    *,
    filters: Mapping[str, Any] | None = None,
    table_key: str = "signals_table",
) -> None:
    """Display the prioritised signal table with actionable context."""

    st.subheader("🚦 Signals")
    rows = _normalise_priority_table(plan)
    if not rows:
        st.caption("Пока нет активных кандидатов. Обновите данные позже.")
        return

    frame = pd.DataFrame(rows)
    if filters:
        if filters.get("actionable_only"):
            frame = frame[frame["actionable"]]
        if filters.get("ready_only"):
            frame = frame[frame["ready"]]
        min_ev = filters.get("min_ev_bps")
        if isinstance(min_ev, (int, float)):
            ev_numeric = pd.to_numeric(frame["ev_bps"], errors="coerce")
            frame = frame[ev_numeric >= float(min_ev)]
        min_prob = filters.get("min_probability")
        if isinstance(min_prob, (int, float)):
            prob_numeric = pd.to_numeric(frame["probability_pct"], errors="coerce")
            frame = frame[prob_numeric >= float(min_prob)]
        if filters.get("hide_skipped"):
            frame = frame[frame["skip_reason"].fillna("").str.strip() == ""]

    if frame.empty:
        st.info("Фильтры скрыли все сигналы.")
        return

    frame = frame.sort_values(["priority", "ev_bps"], ascending=[True, False])

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        key=table_key,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "priority": st.column_config.NumberColumn("Priority", format="%d"),
            "trend": st.column_config.TextColumn("Trend"),
            "probability_pct": st.column_config.NumberColumn("Prob %", format="%.2f"),
            "ev_bps": st.column_config.NumberColumn("EV (bps)", format="%.2f"),
            "actionable": st.column_config.CheckboxColumn("Actionable"),
            "ready": st.column_config.CheckboxColumn("Ready"),
            "skip_reason": st.column_config.TextColumn("Skip reason"),
        },
    )


_OPTIMISTIC_ORDERS_KEY = "_optimistic_orders"
_OPTIMISTIC_TTL = 120.0  # seconds


def _state_container(state: Any | None = None):
    return state if state is not None else st.session_state


def _optimistic_orders(state: Any | None = None) -> list[dict[str, Any]]:
    holder = _state_container(state)
    orders = holder.get(_OPTIMISTIC_ORDERS_KEY, [])
    if isinstance(orders, list):
        return [order for order in orders if isinstance(order, dict)]
    return []


def _record_optimistic_order(
    *,
    state: Any | None,
    symbol: str,
    side: str,
    qty: Any,
    price: Any,
    notional: Any,
) -> str:
    """Store optimistic order metadata in the session state."""

    token = uuid4().hex
    entry = {
        "token": token,
        "symbol": symbol,
        "side": side.capitalize(),
        "qty": qty,
        "price": price,
        "notional": notional,
        "status": "pending",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    holder = _state_container(state)
    existing = _optimistic_orders(holder)
    holder[_OPTIMISTIC_ORDERS_KEY] = existing + [entry]
    return token


def _update_optimistic_order(
    token: str,
    *,
    state: Any | None,
    status: str,
    message: str | None = None,
) -> None:
    holder = _state_container(state)
    orders = _optimistic_orders(holder)
    updated: list[dict[str, Any]] = []
    now = time.time()
    for entry in orders:
        if entry.get("token") != token:
            if now - float(entry.get("created_at", 0.0)) < _OPTIMISTIC_TTL:
                updated.append(entry)
            continue
        entry = dict(entry)
        entry["status"] = status
        if message:
            entry["message"] = message
        entry["updated_at"] = now
        updated.append(entry)
    holder[_OPTIMISTIC_ORDERS_KEY] = updated


def _cleanup_optimistic_orders(state: Any | None = None) -> None:
    holder = _state_container(state)
    orders = _optimistic_orders(holder)
    if not orders:
        return
    now = time.time()
    holder[_OPTIMISTIC_ORDERS_KEY] = [
        order
        for order in orders
        if now - float(order.get("updated_at", order.get("created_at", 0.0)))
        < _OPTIMISTIC_TTL
    ]


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError, AttributeError):
        return None


def orders_table(report: Mapping[str, Any], *, state: Any | None = None) -> None:
    """Show recent trades and current positions."""

    st.subheader("🧾 Orders & Trades")
    portfolio = report.get("portfolio") if isinstance(report, Mapping) else {}
    positions = portfolio.get("positions") if isinstance(portfolio, Mapping) else []
    recent_trades = report.get("recent_trades") if isinstance(report, Mapping) else []

    _cleanup_optimistic_orders(state)
    optimistic = _optimistic_orders(state)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Open positions**")
        if positions:
            positions_frame = pd.DataFrame(positions)
            st.dataframe(
                positions_frame,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "avg_cost": st.column_config.NumberColumn(
                        "Avg cost", format="%.4f"
                    ),
                    "notional": st.column_config.NumberColumn(
                        "Notional", format="%.2f"
                    ),
                    "realized_pnl": st.column_config.NumberColumn(
                        "Realized PnL", format="%.2f"
                    ),
                },
            )
        else:
            st.caption("Нет открытых позиций.")
    with cols[1]:
        st.markdown("**Recent trades**")
        if recent_trades or optimistic:
            trades_frame = pd.DataFrame(recent_trades)
            if optimistic:
                optimistic_frame = pd.DataFrame(optimistic)
                if "when" not in optimistic_frame.columns:
                    optimistic_frame["when"] = format_datetime(
                        datetime.now(timezone.utc)
                    )
                optimistic_frame["status"] = optimistic_frame.get("status", "pending")
                trades_frame = pd.concat(
                    [optimistic_frame, trades_frame], ignore_index=True, sort=False
                )
            if "status" not in trades_frame.columns:
                trades_frame["status"] = "done"

            if {"qty", "price"}.issubset(
                trades_frame.columns
            ) and "notional" not in trades_frame.columns:
                try:
                    qty_numeric = pd.to_numeric(trades_frame["qty"], errors="coerce")
                    price_numeric = pd.to_numeric(
                        trades_frame["price"], errors="coerce"
                    )
                    trades_frame["notional"] = qty_numeric * price_numeric
                except Exception:
                    pass

            def _status_style(value: str) -> str:
                tone = str(value or "").lower()
                if tone in {"pending", "sending"}:
                    color = "rgba(56,189,248,0.2)"
                    border = "rgba(56,189,248,0.4)"
                    text = "#0369a1"
                    label = "В обработке"
                elif tone in {"failed", "error"}:
                    color = "rgba(248,113,113,0.2)"
                    border = "rgba(248,113,113,0.4)"
                    text = "#b91c1c"
                    label = "Ошибка"
                else:
                    color = "rgba(74,222,128,0.25)"
                    border = "rgba(74,222,128,0.5)"
                    text = "#166534"
                    label = "Исполнено"
                return (
                    f"background-color:{color};color:{text};"
                    f"border:1px solid {border};border-radius:1rem;padding:0.15rem 0.6rem;"
                    f"font-weight:600;text-align:center;"
                ), label

            status_labels: list[str] = []
            status_styles: list[str] = []
            for tone in trades_frame["status"].tolist():
                style, label = _status_style(str(tone))
                status_styles.append(style)
                status_labels.append(label)
            trades_frame["status"] = status_labels

            formatter_map: dict[str, Callable[[Any], str]] = {}
            if "qty" in trades_frame.columns:
                formatter_map["qty"] = (
                    lambda value: format_quantity(value) if pd.notna(value) else "—"
                )
            if "price" in trades_frame.columns:
                formatter_map["price"] = lambda value: format_money(
                    value, currency="", precision=4
                )
            if "fee" in trades_frame.columns:
                formatter_map["fee"] = lambda value: format_money(
                    value, currency="", precision=4
                )
            if "notional" in trades_frame.columns:
                formatter_map["notional"] = lambda value: format_money(
                    value, precision=2
                )

            styled = trades_frame.style.format(formatter_map)
            styled = styled.set_properties(subset=["status"], **{"font-weight": "600"})

            def _apply_status(_: pd.Series) -> list[str]:
                return status_styles

            styled = styled.apply(_apply_status, subset=["status"], axis=0)
            st.dataframe(styled, hide_index=True, use_container_width=True)
        else:
            st.caption("Сделок пока не было.")


def wallet_overview(report: Mapping[str, Any]) -> None:
    """Render a simple wallet overview with totals and exposure."""

    st.subheader("💼 Wallet")
    portfolio = report.get("portfolio") if isinstance(report, Mapping) else {}
    totals = portfolio.get("totals") if isinstance(portfolio, Mapping) else {}
    positions = portfolio.get("positions") if isinstance(portfolio, Mapping) else []

    cols = st.columns(3)
    equity = totals.get("total_equity") or totals.get("equity")
    available = totals.get("available_balance") or totals.get("available")
    reserve = totals.get("cash_reserve_pct") or totals.get("reserve_pct")
    cols[0].metric("Equity", format_money(equity))
    cols[1].metric("Available", format_money(available))
    if reserve is None:
        reserve_display = "—"
    else:
        reserve_value = float(reserve) if isinstance(reserve, (int, float)) else reserve
        if isinstance(reserve_value, (int, float)) and reserve_value <= 1:
            reserve_value = reserve_value * 100
        reserve_display = format_percent(reserve_value)
    cols[2].metric("Reserve %", reserve_display)

    if positions:
        st.markdown("**Positions**")
        frame = pd.DataFrame(positions)
        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol"),
                "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
                "avg_cost": st.column_config.NumberColumn("Avg cost", format="%.4f"),
                "notional": st.column_config.NumberColumn("Notional", format="%.2f"),
                "realized_pnl": st.column_config.NumberColumn(
                    "Realized PnL", format="%.2f"
                ),
            },
        )
    else:
        st.caption("Активных позиций нет.")

    extras = {
        key: value
        for key, value in totals.items()
        if key
        not in {
            "total_equity",
            "equity",
            "available_balance",
            "available",
            "cash_reserve_pct",
            "reserve_pct",
        }
    }
    if extras:
        st.markdown("**Totals context**")
        st.json(extras, expanded=False)


def _validation_details(exc: OrderValidationError) -> Mapping[str, Any] | None:
    details = getattr(exc, "details", None)
    if isinstance(details, Mapping) and details:
        return details
    return None


def trade_ticket(
    settings: Any,
    *,
    client_factory,
    state,
    on_success: Iterable[Callable[[], None]] | None = None,
    key_prefix: str = "trade",
    compact: bool = False,
    submit_label: str | None = None,
    instance: str | None = None,
) -> None:
    """Render an interactive trade ticket tied to ``place_spot_market_with_tolerance``.

    The ``instance`` parameter allows hosting multiple tickets with the same ``key_prefix``
    on a single page without causing Streamlit widget key collisions.
    """

    heading = "⚡ Быстрый ордер" if compact else "🛒 Ордер"
    st.subheader(heading)
    if on_success is None:
        on_success = []

    instance_suffix = instance.strip() if isinstance(instance, str) else ""
    if instance_suffix:
        base_prefix = (
            f"{key_prefix}_{instance_suffix}" if key_prefix else instance_suffix
        )
    else:
        base_prefix = key_prefix

    def _state_key(suffix: str) -> str:
        suffix = suffix.lstrip("_")
        return f"{base_prefix}_{suffix}" if base_prefix else suffix

    defaults = {
        "symbol": state.get(_state_key("symbol"), "BTCUSDT"),
        "side": state.get(_state_key("side"), "Buy"),
        "notional": float(state.get(_state_key("notional"), 100.0) or 0.0),
        "tolerance": int(state.get(_state_key("tolerance_bps"), 50) or 0),
    }

    form_key = f"{base_prefix}-ticket-form" if base_prefix else "trade-ticket-form"
    submit_text = submit_label or (
        "Отправить" if compact else "Разместить маркет-ордер"
    )

    hold_key = _state_key("auto_refresh_hold")
    pause_label = "При заполнении ордера приостановить автообновление"
    pause_help = (
        "Предотвращает автоматическую перезагрузку страницы, пока вы вводите параметры ордера. "
        "Снимите флажок, чтобы возобновить автообновление."
    )
    pause_checkbox_key = _state_key("auto_pause")
    pause_active = st.checkbox(pause_label, key=pause_checkbox_key, help=pause_help)
    track_value_change(
        state,
        pause_checkbox_key,
        pause_active,
        reason="Настройки формы ордера обновлены",
        cooldown=4.0,
    )
    pause_reason = (
        "Форма быстрого ордера открыта" if compact else "Форма ордера открыта"
    )
    if pause_active:
        set_auto_refresh_hold(hold_key, pause_reason)
        st.caption(
            "Автообновление приостановлено до отправки формы или отключения флажка."
        )
    else:
        clear_auto_refresh_hold(hold_key)
        st.caption(
            "Убедитесь, что автообновление отключено во время ручного ввода чувствительных данных."
        )

    with st.form(form_key):
        symbol = st.text_input(
            "Торговая пара",
            value=defaults["symbol"],
            help="Введите символ, например BTCUSDT.",
        )
        side = st.radio(
            "Направление",
            ("Buy", "Sell"),
            horizontal=True,
            index=0 if str(defaults["side"]).lower() != "sell" else 1,
            help="Выберите действие: Buy для покупки, Sell для продажи.",
        )
        notional = st.number_input(
            "Сумма ордера (USDT)",
            min_value=0.0,
            value=defaults["notional"],
            step=1.0,
            help="Размер сделки в котируемой валюте (USDT).",
        )
        tolerance = st.slider(
            "Допустимый слиппедж (б.п.)",
            min_value=0,
            max_value=500,
            value=defaults["tolerance"],
            help="Максимальный слиппедж в базисных пунктах (1 б.п. = 0.01%).",
        )
        submitted = st.form_submit_button(submit_text)

    feedback_key = _state_key("feedback")
    if not submitted:
        feedback = state.get(feedback_key)
        if isinstance(feedback, Mapping):
            st.success(feedback.get("message", "Ордер подготовлен."))
            st.json(feedback.get("audit"), expanded=False)
        return

    cleaned_symbol = symbol.strip().upper()
    state[_state_key("symbol")] = cleaned_symbol or defaults["symbol"]
    state[_state_key("side")] = side
    state[_state_key("notional")] = notional
    state[_state_key("tolerance_bps")] = tolerance
    state[feedback_key] = None

    note_user_interaction("Отправлена форма ордера", cooldown=8.0)

    if not cleaned_symbol:
        show_error_banner("Укажите символ спот-торговли, например BTCUSDT.")
        return
    if notional <= 0:
        show_error_banner("Укажите положительный размер сделки в USDT.")
        return

    client = client_factory()
    if client is None:
        has_key = bool(active_api_key(settings))
        has_secret = bool(active_api_secret(settings))
        api_error = last_api_client_error()
        if not (has_key and has_secret):
            show_error_banner(
                "Не указан API-ключ или секрет — торговля недоступна, пока вы не добавите их.",
                title="Нужно подключить API",
            )
            navigation_link(
                "pages/00_✅_Подключение_и_Состояние.py",
                label="Открыть настройки API",
                icon="🔑",
                key=f"{key_prefix or 'trade'}_api_setup_link",
            )
        else:
            show_error_banner(
                "Нет связи с биржей Bybit. Проверьте интернет и попробуйте снова.",
                title="API клиент недоступен",
                details=api_error,
            )
        return

    optimistic_token: str | None = None
    try:
        snapshot = prepare_spot_trade_snapshot(client, cleaned_symbol)
        prepared = prepare_spot_market_order(
            client,
            cleaned_symbol,
            side,
            notional,
            unit="quoteCoin",
            tol_type="Bps",
            tol_value=float(tolerance),
            max_quote=notional,
            price_snapshot=snapshot.price,
            balances=snapshot.balances,
            limits=snapshot.limits,
            settings=settings,
        )
        optimistic_token = _record_optimistic_order(
            state=state,
            symbol=cleaned_symbol,
            side=side,
            qty=_to_float(prepared.audit.get("order_qty_base")),
            price=_to_float(
                prepared.audit.get("price_used") or prepared.audit.get("limit_price")
            ),
            notional=_to_float(prepared.audit.get("order_notional")),
        )
        response = place_spot_market_with_tolerance(
            client,
            cleaned_symbol,
            side,
            notional,
            unit="quoteCoin",
            tol_type="Bps",
            tol_value=float(tolerance),
            max_quote=notional,
            price_snapshot=snapshot.price,
            balances=snapshot.balances,
            limits=snapshot.limits,
            settings=settings,
        )
    except OrderValidationError as exc:
        if optimistic_token:
            _update_optimistic_order(
                optimistic_token, state=state, status="failed", message=str(exc)
            )
        show_error_banner(str(exc), details=_validation_details(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive
        if optimistic_token:
            _update_optimistic_order(
                optimistic_token, state=state, status="failed", message=str(exc)
            )
        show_error_banner("Не удалось разместить ордер.", details=str(exc))
        return

    if optimistic_token:
        _update_optimistic_order(
            optimistic_token,
            state=state,
            status="done",
            message="Маркет-ордер отправлен на биржу",
        )

    feedback = {
        "message": "Маркет-ордер отправлен на биржу.",
        "audit": prepared.audit,
        "response": response,
    }
    state[feedback_key] = feedback
    state[pause_checkbox_key] = False
    clear_auto_refresh_hold(hold_key)
    st.success(feedback["message"])
    with st.expander("Order audit", expanded=False):
        st.json(prepared.audit, expanded=False)
    if response:
        with st.expander("Exchange response", expanded=False):
            st.json(response, expanded=False)

    for callback in on_success:
        try:
            callback()
        except Exception:
            continue


def log_viewer(path: Path | str, *, default_limit: int = 400, state=None) -> None:
    """Show the tail of the application log file with lightweight filtering."""

    st.subheader("🪵 Logs")

    def _state_get(key: str, default: Any = None) -> Any:
        if state is None:
            return default
        getter = getattr(state, "get", None)
        if callable(getter):
            return getter(key, default)
        try:
            return state[key]
        except Exception:
            return default

    level_options = ("ALL", "INFO", "WARNING", "ERROR")
    stored_level = _state_get("logs_level")
    level_index = (
        level_options.index(stored_level) if stored_level in level_options else 0
    )
    level = st.selectbox("Log level", level_options, index=level_index)

    stored_limit = _state_get("logs_limit")
    limit = st.slider(
        "Количество строк",
        min_value=50,
        max_value=2000,
        step=50,
        value=int(stored_limit)
        if isinstance(stored_limit, (int, float))
        else default_limit,
    )
    stored_query = _state_get("logs_query", "")
    search_query = st.text_input("Поиск по журналу", value=str(stored_query or ""))

    if state is not None:
        state["logs_level"] = level
        state["logs_limit"] = limit
        state["logs_query"] = search_query

    path_obj = Path(path)
    if not path_obj.exists():
        st.caption("Файл логов пока не создан.")
        return

    try:
        raw_content = path_obj.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        show_error_banner("Не удалось прочитать журнал.", details=str(exc))
        return

    lines = raw_content.splitlines()

    tail_scope = lines[-2000:] if len(lines) > 2000 else lines
    if level != "ALL":
        needle = level.upper()
        filtered = [line for line in tail_scope if needle in line.upper()]
    else:
        filtered = tail_scope

    query = search_query.strip().lower()
    if query:
        filtered = [line for line in filtered if query in line.lower()]

    content_lines = filtered[-limit:]
    if not content_lines:
        st.info("Нет записей, удовлетворяющих выбранным фильтрам.")
        return

    content = "\n".join(content_lines)
    st.text_area("Последние события", value=content, height=320, key="logs_tail")
    display_query = search_query.strip() or "—"
    st.caption(
        f"Показано {len(content_lines)} из {len(filtered)} строк (уровень: {level}, поиск: '{display_query}')."
    )
    st.download_button(
        "💾 Скачать лог-файл",
        data=raw_content,
        file_name=path_obj.name,
        mime="text/plain",
        use_container_width=True,
    )
