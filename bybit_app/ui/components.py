"""Composable Streamlit components used across the dashboard tabs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from bybit_app.utils.envs import (
    active_api_key,
    active_api_secret,
    active_dry_run,
    last_api_client_error,
)
from bybit_app.utils.spot_market import (
    OrderValidationError,
    place_spot_market_with_tolerance,
    prepare_spot_market_order,
    prepare_spot_trade_snapshot,
)


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


def status_bar(settings: Any, *, guardian_snapshot: Mapping[str, Any], ws_snapshot: Mapping[str, Any]) -> None:
    """Display the high level status strip with connection and freshness hints."""

    st.subheader("⚙️ Status")

    api_error = last_api_client_error()
    cols = st.columns(4)
    with cols[0]:
        st.metric("Mode", "DRY-RUN" if active_dry_run(settings) else "Live")
    with cols[1]:
        st.metric("Network", "Testnet" if getattr(settings, "testnet", True) else "Mainnet")
    guardian_age = guardian_snapshot.get("age_seconds")
    with cols[2]:
        st.metric("Guardian", _format_age(guardian_age))
    ws_status = ws_snapshot.get("status") if isinstance(ws_snapshot, Mapping) else {}
    private_status = ws_status.get("private") if isinstance(ws_status, Mapping) else {}
    private_age = private_status.get("age_seconds") if isinstance(private_status, Mapping) else None
    with cols[3]:
        st.metric("Private WS", _format_age(private_age))

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


def signals_table(plan: Mapping[str, Any] | None) -> None:
    """Display the prioritised signal table with actionable context."""

    st.subheader("🚦 Signals")
    rows = _normalise_priority_table(plan)
    if not rows:
        st.caption("Пока нет активных кандидатов. Обновите данные позже.")
        return

    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
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
) -> None:
    """Render an interactive trade ticket tied to ``place_spot_market_with_tolerance``."""

    st.subheader("🛒 Trade Ticket")
    if on_success is None:
        on_success = []

    defaults = {
        "symbol": state.get("trade_symbol", "BTCUSDT"),
        "side": state.get("trade_side", "Buy"),
        "notional": float(state.get("trade_notional", 100.0) or 0.0),
        "tolerance": int(state.get("trade_tolerance_bps", 50) or 0),
    }

    with st.form("trade-ticket-form"):
        symbol = st.text_input("Symbol", value=defaults["symbol"], help="Например BTCUSDT")
        side = st.radio("Side", ("Buy", "Sell"), horizontal=True, index=0 if defaults["side"].lower() != "sell" else 1)
        notional = st.number_input("Notional (USDT)", min_value=0.0, value=defaults["notional"], step=1.0)
        tolerance = st.slider("Slippage guard (bps)", min_value=0, max_value=500, value=defaults["tolerance"], help="Максимально допустимый слиппедж в базисных пунктах")
        submitted = st.form_submit_button("Place market order")

    if not submitted:
        feedback = state.get("trade_feedback")
        if isinstance(feedback, Mapping):
            st.success(feedback.get("message", "Ордер подготовлен."))
            st.json(feedback.get("audit"), expanded=False)
        return

    cleaned_symbol = symbol.strip().upper()
    state["trade_symbol"] = cleaned_symbol or defaults["symbol"]
    state["trade_side"] = side
    state["trade_notional"] = notional
    state["trade_tolerance_bps"] = tolerance
    state["trade_feedback"] = None

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
    state["trade_feedback"] = feedback
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


def log_viewer(path: Path | str, *, limit: int = 400, state=None) -> None:
    """Show the tail of the application log file."""

    st.subheader("🪵 Logs")
    level_placeholder = st.selectbox(
        "Log level",
        ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
        index=2,
        disabled=True,
        help="Фильтр появится в следующих релизах",
    )
    if state is not None:
        state["logs_level"] = level_placeholder

    path_obj = Path(path)
    if not path_obj.exists():
        st.caption("Файл логов пока не создан.")
        return

    try:
        lines = path_obj.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        show_error_banner("Не удалось прочитать журнал.", details=str(exc))
        return

    content = "\n".join(lines)
    st.text_area("Последние события", value=content, height=320)
