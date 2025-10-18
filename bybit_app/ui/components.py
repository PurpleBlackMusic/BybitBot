"""Composable Streamlit components used across the dashboard tabs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, get_args

import pandas as pd
import streamlit as st

from bybit_app.utils.ai.kill_switch import KillSwitchState
from bybit_app.utils.envs import (
    active_api_key,
    active_api_secret,
    active_dry_run,
    last_api_client_error,
)
from bybit_app.ui.state import clear_auto_refresh_hold, set_auto_refresh_hold
from bybit_app.utils.ui import build_pill
from bybit_app.utils.spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    prepare_spot_market_order,
    prepare_spot_trade_snapshot,
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
            tone=_numeric_tone(self.value, warn_at=self.warn_at, danger_at=self.danger_at),
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
            tone=_age_tone(self.value, warn_after=self.warn_after, danger_after=self.danger_after),
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

        guardian = _ensure_mapping(guardian_snapshot)
        ws = _ensure_mapping(ws_snapshot)
        report_mapping = _ensure_mapping(report)
        portfolio = _ensure_mapping(report_mapping.get("portfolio"))
        totals = _ensure_mapping(portfolio.get("totals"))

        signal_state = _ensure_mapping(guardian.get("state"))
        signal_age_value = _coerce_float(signal_state.get("age_seconds"))
        signal_report = _ensure_mapping(signal_state.get("report"))
        signal_generated = signal_report.get("generated_at") or signal_report.get("timestamp")
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
        private_age_value = _coerce_float(_ensure_mapping(ws_status.get("private")).get("age_seconds"))
        public_age_value = _coerce_float(_ensure_mapping(ws_status.get("public")).get("age_seconds"))
        ws_age_candidates = [value for value in (private_age_value, public_age_value) if value is not None]
        ws_worst_age = max(ws_age_candidates) if ws_age_candidates else None

        ws_age_value = _coerce_float(ws.get("age_seconds"))
        ws_caption = f"обновл. {_format_age(ws_age_value)} назад" if ws_age_value is not None else ""

        kill_state = kill_switch or KillSwitchState(paused=False, until=None, reason=None)
        kill_caption = "Активен" if kill_state.paused else "Готов"
        if kill_state.paused:
            if kill_state.until:
                remaining = max(kill_state.until - now, 0.0)
                kill_caption = f"до {_format_age(remaining)}"
            if kill_state.reason:
                kill_caption += f" · {kill_state.reason}"

        return cls(
            testnet=bool(getattr(settings, "testnet", True)),
            dry_run=active_dry_run(settings),
            equity=_coerce_float(totals.get("total_equity") or totals.get("equity")),
            available=_coerce_float(totals.get("available_balance") or totals.get("available")),
            kill_switch=kill_state,
            kill_caption=kill_caption,
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
        ws_value = f"pub {_format_age(self.public_age)} · priv {_format_age(self.private_age)}"
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
                "WS",
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


def show_error_banner(message: str, *, title: str | None = None, details: Mapping[str, Any] | str | None = None) -> None:
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
    _render_badge_grid(context.badges(), columns=4)

    has_keys = bool(active_api_key(settings) and active_api_secret(settings))
    if not has_keys:
        show_error_banner(
            "Добавьте API ключ и секрет, чтобы бот мог размещать ордера.",
            title="API ключи отсутствуют",
        )
    if api_error:
        show_error_banner(api_error, title="API клиент недоступен")


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
    limit = stats.get("limit") if isinstance(stats, Mapping) else plan.get("limit") if isinstance(plan, Mapping) else 0
    equity = totals.get("total_equity") if isinstance(totals, Mapping) else totals.get("equity") if isinstance(totals, Mapping) else None
    pnl = totals.get("realized_pnl") if isinstance(totals, Mapping) else None

    cols = st.columns(4)
    cols[0].metric("Actionable", actionable)
    cols[1].metric("Ready", ready)
    cols[2].metric("Positions", positions, f"Limit {limit}" if limit else None)
    if equity is not None:
        cols[3].metric("Equity", f"{_num(equity):,.2f}")
    else:
        cols[3].metric("Realized PnL", f"{_num(pnl):,.2f}")


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


def orders_table(report: Mapping[str, Any]) -> None:
    """Show recent trades and current positions."""

    st.subheader("🧾 Orders & Trades")
    portfolio = report.get("portfolio") if isinstance(report, Mapping) else {}
    positions = portfolio.get("positions") if isinstance(portfolio, Mapping) else []
    recent_trades = report.get("recent_trades") if isinstance(report, Mapping) else []

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Open positions**")
        if positions:
            st.dataframe(pd.DataFrame(positions), hide_index=True, use_container_width=True)
        else:
            st.caption("Нет открытых позиций.")
    with cols[1]:
        st.markdown("**Recent trades**")
        if recent_trades:
            st.dataframe(pd.DataFrame(recent_trades), hide_index=True, use_container_width=True)
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
    cols[0].metric("Equity", f"{equity:,.2f}" if equity else "—")
    cols[1].metric("Available", f"{available:,.2f}" if available else "—")
    cols[2].metric("Reserve %", f"{reserve:.1f}%" if isinstance(reserve, (int, float)) else "—")

    if positions:
        st.markdown("**Positions**")
        st.dataframe(pd.DataFrame(positions), hide_index=True, use_container_width=True)
    else:
        st.caption("Активных позиций нет.")

    extras = {key: value for key, value in totals.items() if key not in {"total_equity", "equity", "available_balance", "available", "cash_reserve_pct", "reserve_pct"}}
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
) -> None:
    """Render an interactive trade ticket tied to ``place_spot_market_with_tolerance``."""

    heading = "⚡ Quick Ticket" if compact else "🛒 Trade Ticket"
    st.subheader(heading)
    if on_success is None:
        on_success = []

    def _state_key(suffix: str) -> str:
        suffix = suffix.lstrip("_")
        return f"{key_prefix}_{suffix}" if key_prefix else suffix

    defaults = {
        "symbol": state.get(_state_key("symbol"), "BTCUSDT"),
        "side": state.get(_state_key("side"), "Buy"),
        "notional": float(state.get(_state_key("notional"), 100.0) or 0.0),
        "tolerance": int(state.get(_state_key("tolerance_bps"), 50) or 0),
    }

    help_suffix = "" if compact else "Например BTCUSDT"
    form_key = f"{key_prefix}-ticket-form" if key_prefix else "trade-ticket-form"
    submit_text = submit_label or ("Отправить" if compact else "Place market order")

    hold_key = _state_key("auto_refresh_hold")
    pause_label = "При заполнении ордера приостановить автообновление"
    pause_help = (
        "Предотвращает автоматическую перезагрузку страницы, пока вы вводите параметры ордера. "
        "Снимите флажок, чтобы возобновить автообновление."
    )
    pause_checkbox_key = _state_key("auto_pause")
    pause_active = st.checkbox(pause_label, key=pause_checkbox_key, help=pause_help)
    pause_reason = "Форма быстрого ордера открыта" if compact else "Форма ордера открыта"
    if pause_active:
        set_auto_refresh_hold(hold_key, pause_reason)
        st.caption("Автообновление приостановлено до отправки формы или отключения флажка.")
    else:
        clear_auto_refresh_hold(hold_key)
        st.caption("Убедитесь, что автообновление отключено во время ручного ввода чувствительных данных.")

    with st.form(form_key):
        symbol = st.text_input("Symbol", value=defaults["symbol"], help=help_suffix or None)
        side = st.radio(
            "Side",
            ("Buy", "Sell"),
            horizontal=True,
            index=0 if str(defaults["side"]).lower() != "sell" else 1,
        )
        notional = st.number_input(
            "Notional (USDT)",
            min_value=0.0,
            value=defaults["notional"],
            step=1.0,
        )
        tolerance = st.slider(
            "Slippage guard (bps)",
            min_value=0,
            max_value=500,
            value=defaults["tolerance"],
            help="Максимально допустимый слиппедж в базисных пунктах",
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

    if not cleaned_symbol:
        show_error_banner("Укажите символ спот-торговли, например BTCUSDT.")
        return
    if notional <= 0:
        show_error_banner("Укажите положительный размер сделки в USDT.")
        return

    client = client_factory()
    if client is None:
        show_error_banner("API клиент не готов. Проверьте подключение и ключи.")
        return

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
        show_error_banner(str(exc), details=_validation_details(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive
        show_error_banner("Не удалось разместить ордер.", details=str(exc))
        return

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
    level_index = level_options.index(stored_level) if stored_level in level_options else 0
    level = st.selectbox("Log level", level_options, index=level_index)

    stored_limit = _state_get("logs_limit")
    limit = st.slider(
        "Количество строк",
        min_value=50,
        max_value=2000,
        step=50,
        value=int(stored_limit) if isinstance(stored_limit, (int, float)) else default_limit,
    )

    if state is not None:
        state["logs_level"] = level
        state["logs_limit"] = limit

    path_obj = Path(path)
    if not path_obj.exists():
        st.caption("Файл логов пока не создан.")
        return

    try:
        lines = path_obj.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        show_error_banner("Не удалось прочитать журнал.", details=str(exc))
        return

    tail_scope = lines[-2000:] if len(lines) > 2000 else lines
    if level != "ALL":
        needle = level.upper()
        filtered = [line for line in tail_scope if needle in line.upper()]
    else:
        filtered = tail_scope

    content_lines = filtered[-limit:]
    if not content_lines:
        st.info("Нет записей выбранного уровня за последние строки.")
        return

    content = "\n".join(content_lines)
    st.text_area("Последние события", value=content, height=320, key="logs_tail")
