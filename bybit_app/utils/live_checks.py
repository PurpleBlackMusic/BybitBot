from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple

from .bybit_api import BybitAPI, creds_from_settings, get_api
from .envs import Settings


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_numeric(source: dict, keys: Iterable[str]) -> Optional[float]:
    """Return the first meaningful numeric value for the provided keys."""

    fallback: Optional[float] = None
    for key in keys:
        value = _safe_float(source.get(key))
        if value is None:
            continue
        if value != 0.0:
            return value
        fallback = value
    return fallback


def _extract_wallet_totals(payload: Dict[str, object]) -> Tuple[float, float]:
    """Return total and available equity from wallet payload."""

    result = payload.get("result")
    if not isinstance(result, dict):
        return 0.0, 0.0

    accounts = result.get("list")
    if not isinstance(accounts, Iterable):
        return 0.0, 0.0

    total = 0.0
    available = 0.0
    for account in accounts:
        if not isinstance(account, dict):
            continue

        total_val = _first_numeric(account, ("totalEquity", "equity", "walletBalance"))
        if total_val is not None:
            total += total_val

        available_val = _first_numeric(
            account,
            (
                "availableBalance",
                "availableMargin",
                "availableToWithdraw",
                "cashBalance",
            ),
        )
        if available_val is not None:
            available += available_val
    return total, available


def _extract_wallet_assets(
    payload: Dict[str, object], *, limit: int = 5
) -> Tuple[Dict[str, object], ...]:
    """Return the heaviest wallet assets to prove balances are real."""

    result = payload.get("result")
    if not isinstance(result, dict):
        return tuple()

    accounts = result.get("list")
    if not isinstance(accounts, Iterable):
        return tuple()

    combined: Dict[str, Dict[str, float]] = {}

    for account in accounts:
        if not isinstance(account, dict):
            continue

        coins = account.get("coin") or account.get("coins")
        if not isinstance(coins, Iterable):
            continue

        for row in coins:
            if not isinstance(row, dict):
                continue

            raw_symbol = row.get("coin") or row.get("asset") or row.get("currency")
            if not isinstance(raw_symbol, str):
                continue
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue

            total = _first_numeric(
                row,
                (
                    "equity",
                    "walletBalance",
                    "balance",
                    "total",
                ),
            )
            available = _first_numeric(
                row,
                (
                    "availableToWithdraw",
                    "availableBalance",
                    "available",
                    "availableMargin",
                ),
            )

            if total is None and available is None:
                continue

            asset = combined.setdefault(symbol, {"coin": symbol, "total": 0.0, "available": 0.0})
            if total is not None:
                asset["total"] += float(total)
            if available is not None:
                asset["available"] += float(available)

    if not combined:
        return tuple()

    sorted_assets = sorted(
        combined.values(),
        key=lambda item: abs(item.get("total") or 0.0),
        reverse=True,
    )

    trimmed = []
    for asset in sorted_assets[: max(1, limit)]:
        total_val = float(asset.get("total") or 0.0)
        available_val = float(asset.get("available") or 0.0)
        reserved = max(0.0, total_val - available_val)
        trimmed.append(
            {
                "coin": asset.get("coin"),
                "total": total_val,
                "available": available_val,
                "reserved": reserved,
            }
        )

    return tuple(trimmed)


def _format_decimal(value: Optional[float], *, decimals: int = 8) -> Optional[str]:
    """Convert a numeric value to a trimmed string with fixed precision."""

    if value is None:
        return None

    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text or "0"


def _parse_epoch_seconds(value: object) -> Optional[float]:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    # Values from Bybit arrive as strings with millisecond timestamps.
    if numeric > 1e12:  # ms
        return numeric / 1000.0
    if numeric > 1e9:  # already seconds
        return numeric
    if numeric > 1e6:  # fail-safe for truncated ms timestamps
        return numeric / 1000.0
    return None


def _latest_timestamp_age_seconds(
    rows: Iterable[object], candidate_keys: Iterable[str]
) -> Optional[float]:
    latest: Optional[float] = None
    now = time.time()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in candidate_keys:
            ts = _parse_epoch_seconds(row.get(key))
            if ts is None:
                continue
            if latest is None or ts > latest:
                latest = ts
            break
    if latest is None:
        return None
    return max(0.0, now - latest)


def _latest_order_age_seconds(orders: Iterable[object]) -> Optional[float]:
    return _latest_timestamp_age_seconds(
        orders, ("updatedTime", "createdTime", "ts", "transactTime")
    )


def _latest_execution_age_seconds(executions: Iterable[object]) -> Optional[float]:
    return _latest_timestamp_age_seconds(
        executions,
        ("execTime", "tradeTime", "updatedTime", "createdTime", "ts", "transactTime"),
    )


def _latest_execution_details(executions: Iterable[object]) -> Optional[Dict[str, object]]:
    """Extract metadata about the freshest execution, if any."""

    latest_row: Optional[dict] = None
    latest_ts: Optional[float] = None
    timestamp_keys = (
        "execTime",
        "tradeTime",
        "updatedTime",
        "createdTime",
        "ts",
        "transactTime",
    )

    for row in executions:
        if not isinstance(row, dict):
            continue

        ts: Optional[float] = None
        for key in timestamp_keys:
            ts = _parse_epoch_seconds(row.get(key))
            if ts is not None:
                break

        if ts is None:
            continue

        if latest_ts is None or ts > latest_ts:
            latest_row = row
            latest_ts = ts

    if latest_row is None or latest_ts is None:
        return None

    raw_symbol = latest_row.get("symbol") or latest_row.get("s")
    symbol = str(raw_symbol).upper() if isinstance(raw_symbol, str) else None

    raw_side = latest_row.get("side") or latest_row.get("orderSide")
    side = str(raw_side).strip().capitalize() if isinstance(raw_side, str) else None

    price = _safe_float(
        latest_row.get("execPrice")
        or latest_row.get("price")
        or latest_row.get("orderPrice")
    )

    qty = _safe_float(
        latest_row.get("execQty")
        or latest_row.get("lastExecQty")
        or latest_row.get("orderQty")
        or latest_row.get("qty")
    )

    fee = _safe_float(latest_row.get("execFee") or latest_row.get("fee"))

    maker_raw = latest_row.get("isMaker")
    maker: Optional[bool]
    if isinstance(maker_raw, bool):
        maker = maker_raw
    elif isinstance(maker_raw, str):
        maker = maker_raw.strip().lower() in {"true", "1", "maker", "yes"}
    else:
        maker = None

    execution_id = latest_row.get("execId") or latest_row.get("tradeId")
    order_id = latest_row.get("orderId") or latest_row.get("orderID")

    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
        "fee": fee,
        "is_maker": maker,
        "exec_id": execution_id,
        "order_id": order_id,
        "ts": latest_ts,
    }


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1 сек"
    if seconds < 60:
        return f"{int(round(seconds))} сек"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(round(minutes))} мин"
    hours = minutes / 60.0
    if hours < 24:
        return f"{int(round(hours))} ч"
    days = hours / 24.0
    return f"{int(round(days))} д"


def _load_ws_status() -> Optional[Dict[str, object]]:
    try:
        # Import lazily to avoid circular dependencies when running tests.
        from . import ws_manager
    except Exception:  # pragma: no cover - import errors only in production misconfigurations
        return None

    try:
        status = ws_manager.manager.status()
    except Exception:  # pragma: no cover - runtime errors depend on WS state
        return None

    if isinstance(status, dict):
        return status
    return None


def bybit_realtime_status(
    settings: Settings,
    api: Optional[BybitAPI] = None,
    ws_status: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Check that Bybit responds with live balance and order data."""

    title = "Bybit реальное время"
    if not (settings.api_key and settings.api_secret):
        return {
            "title": title,
            "ok": False,
            "message": "API ключи не заданы — бот не может проверить реальные данные.",
            "details": "Добавьте ключ и секрет Bybit, чтобы читать баланс и ордера.",
        }

    if getattr(settings, "dry_run", True):
        return {
            "title": title,
            "ok": False,
            "message": "Включен DRY-RUN — сделки не затрагивают реальный баланс.",
            "details": "Отключите DRY-RUN в настройках, чтобы убедиться в реальной торговле.",
        }

    client: BybitAPI
    if api is None:
        client = get_api(
            creds_from_settings(settings),
            recv_window=getattr(settings, "recv_window_ms", 5000) or 5000,
            timeout=getattr(settings, "http_timeout_ms", 10000) or 10000,
            verify_ssl=getattr(settings, "verify_ssl", True),
        )
    else:
        client = api

    started = time.perf_counter()
    try:
        wallet = client.wallet_balance()
    except Exception as exc:  # pragma: no cover - network errors in production only
        return {
            "title": title,
            "ok": False,
            "message": "Не удалось запросить баланс Bybit.",
            "details": str(exc),
        }
    latency_ms = (time.perf_counter() - started) * 1000.0

    total_equity, available_equity = _extract_wallet_totals(wallet)
    wallet_assets = _extract_wallet_assets(wallet)

    try:
        orders_payload = client.open_orders(category="spot", openOnly=1)
        orders = ((orders_payload or {}).get("result") or {}).get("list") or []
    except Exception as exc:  # pragma: no cover - network errors in production only
        return {
            "title": title,
            "ok": False,
            "message": "Баланс получен, но не удалось запросить открытые ордера.",
            "details": str(exc),
            "latency_ms": round(latency_ms, 2),
            "balance_total": round(total_equity, 4),
            "balance_available": round(available_equity, 4),
        }

    order_count = len(orders)
    order_age = _latest_order_age_seconds(orders)
    order_age_human = _format_duration(order_age) if order_age is not None else None
    watchdog = float(getattr(settings, "ws_watchdog_max_age_sec", 90) or 90)
    execution_watchdog = float(
        getattr(settings, "execution_watchdog_max_age_sec", 600) or 600
    )

    try:
        executions_payload = client.execution_list(category="spot", limit=50)
        executions = ((executions_payload or {}).get("result") or {}).get("list") or []
    except Exception as exc:  # pragma: no cover - network errors in production only
        return {
            "title": title,
            "ok": False,
            "message": "Баланс и ордера получены, но журнал исполнений не открылся.",
            "details": str(exc),
            "latency_ms": round(latency_ms, 2),
            "balance_total": round(total_equity, 4),
            "balance_available": round(available_equity, 4),
            "order_count": order_count,
            "order_age_sec": order_age,
            "order_age_human": order_age_human,
        }

    execution_count = len(executions)
    execution_age = _latest_execution_age_seconds(executions)
    execution_age_human = (
        _format_duration(execution_age) if execution_age is not None else None
    )
    last_execution = _latest_execution_details(executions)
    last_execution_brief: Optional[str] = None
    last_execution_at: Optional[str] = None

    if last_execution:
        trade_bits: list[str] = []
        side_text = last_execution.get("side")
        if isinstance(side_text, str) and side_text:
            trade_bits.append(side_text.upper())

        symbol_text = last_execution.get("symbol")
        if isinstance(symbol_text, str) and symbol_text:
            trade_bits.append(symbol_text)

        qty_text = _format_decimal(_safe_float(last_execution.get("qty")))
        if qty_text:
            trade_bits.append(f"{qty_text}")

        price_text = _format_decimal(_safe_float(last_execution.get("price")))
        if price_text:
            trade_bits.append(f"по {price_text}")

        if trade_bits:
            last_execution_brief = " ".join(trade_bits)

        ts_value = last_execution.get("ts")
        if isinstance(ts_value, (int, float)):
            last_execution_at = (
                datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S UTC")
            )

    base_messages: list[str] = []
    warning_messages: list[str] = []

    if order_count:
        base_messages.append(
            "Биржа отвечает — ордера и баланс обновляются в реальном времени."
        )
    else:
        base_messages.append(
            "Баланс получен, открытых ордеров нет — все заявки исполнены."
        )

    if execution_count:
        if execution_age_human:
            trade_suffix = (
                f" ({last_execution_brief})" if last_execution_brief else ""
            )
            base_messages.append(
                f"Последняя сделка была {execution_age_human} назад{trade_suffix} — данные живые."
            )
        else:
            if last_execution_brief:
                base_messages.append(
                    f"Журнал исполнений доступен — последняя сделка {last_execution_brief}."
                )
            else:
                base_messages.append(
                    "Журнал исполнений доступен — биржа подтверждает сделки."
                )
    else:
        warning_messages.append(
            "Однако журнал исполнений пуст — бот ещё не подтвердил реальные сделки."
        )

    if order_age is not None and order_age > watchdog:
        warning_messages.append(
            "Однако обновления ордеров приходили слишком давно — проверьте WebSocket и исполнение."
        )

    if execution_age is not None and execution_age > execution_watchdog:
        warning_messages.append(
            "Однако последняя сделка была слишком давно — убедитесь, что бот продолжает торговать."
        )

    if ws_status is None:
        ws_status = _load_ws_status()

    ws_public_age: Optional[float] = None
    ws_private_age: Optional[float] = None
    ws_last_beat: Optional[float] = None

    if isinstance(ws_status, dict):
        public_info = ws_status.get("public")
        private_info = ws_status.get("private")

        if isinstance(public_info, dict):
            ws_public_age = _safe_float(public_info.get("age_seconds"))
            ws_last_beat = _safe_float(public_info.get("last_beat")) or ws_last_beat
            if not public_info.get("running"):
                warning_messages.append(
                    "Однако публичный WebSocket не запущен — данные маркетов не обновляются автоматически."
                )
        if isinstance(private_info, dict):
            ws_private_age = _safe_float(private_info.get("age_seconds"))
            ws_last_beat = _safe_float(private_info.get("last_beat")) or ws_last_beat
            if not private_info.get("running"):
                warning_messages.append(
                    "Однако приватный WebSocket не запущен — нет подтверждения сделок в реальном времени."
                )
            elif not private_info.get("connected"):
                warning_messages.append(
                    "Однако приватный WebSocket не подключён — перепроверьте ключи и соединение."
                )

    max_ws_age = max(watchdog, 30.0)
    if ws_private_age is not None and ws_private_age > watchdog:
        warning_messages.append(
            "Однако приватный WebSocket не присылал данные слишком долго — перезапустите соединение."
        )
    elif ws_public_age is not None and ws_public_age > max_ws_age:
        warning_messages.append(
            "Однако публичный WebSocket отстал — проверьте соединение."
        )

    detail_parts = [
        f"Суммарный баланс: {total_equity:.2f} USDT",
        f"Доступно: {available_equity:.2f} USDT",
        f"Открытых ордеров: {order_count}",
        f"Исполнений: {execution_count}",
    ]
    if order_age is not None:
        detail_parts.append(
            f"Последнее обновление ордеров: {_format_duration(order_age)} назад"
        )
    else:
        detail_parts.append("В ответе нет отметки времени — используйте мониторинг WS.")

    if execution_age is not None:
        detail_parts.append(
            f"Последнее исполнение: {_format_duration(execution_age)} назад"
        )
    else:
        detail_parts.append(
            "Исполнения без отметки времени — перепроверьте приватный WebSocket."
        )

    if last_execution_brief:
        detail_parts.append(f"Последняя сделка: {last_execution_brief}")

    last_execution_fee = _safe_float((last_execution or {}).get("fee"))
    if last_execution_fee is not None:
        fee_text = _format_decimal(last_execution_fee, decimals=6)
        detail_parts.append(f"Комиссия последней сделки: {fee_text}")

    if last_execution_at:
        detail_parts.append(f"Время сделки (UTC): {last_execution_at}")

    if ws_private_age is not None:
        detail_parts.append(
            f"Приватный WebSocket: {_format_duration(ws_private_age)} назад"
        )
    if ws_public_age is not None:
        detail_parts.append(
            f"Публичный WebSocket: {_format_duration(ws_public_age)} назад"
        )
    if ws_last_beat is not None:
        detail_parts.append(
            f"Последний WS heartbeat: {ws_last_beat:.0f}"
        )

    if wallet_assets:
        asset_bits = []
        for asset in wallet_assets:
            coin = asset.get("coin")
            total = asset.get("total")
            available = asset.get("available")
            if not isinstance(coin, str):
                continue
            try:
                total_val = float(total)
            except (TypeError, ValueError):
                continue
            available_val: Optional[float]
            try:
                available_val = float(available)
            except (TypeError, ValueError):
                available_val = None
            if available_val is not None:
                asset_bits.append(
                    f"{coin} {total_val:.4f} (доступно {available_val:.4f})"
                )
            else:
                asset_bits.append(f"{coin} {total_val:.4f}")
        if asset_bits:
            detail_parts.append("Основные активы: " + ", ".join(asset_bits))

    ws_private_age_human = (
        _format_duration(ws_private_age) if ws_private_age is not None else None
    )
    ws_public_age_human = (
        _format_duration(ws_public_age) if ws_public_age is not None else None
    )

    ok = not warning_messages
    if ok:
        message = " ".join(base_messages)
    else:
        base_text = " ".join(base_messages) if base_messages else ""
        warnings_text = " ".join(warning_messages)
        message = f"{base_text} {warnings_text}".strip()

    return {
        "title": title,
        "ok": ok,
        "message": message,
        "details": ". ".join(detail_parts),
        "latency_ms": round(latency_ms, 2),
        "balance_total": round(total_equity, 4),
        "balance_available": round(available_equity, 4),
        "order_count": order_count,
        "order_age_sec": order_age,
        "order_age_human": order_age_human,
        "execution_count": execution_count,
        "execution_age_sec": execution_age,
        "execution_age_human": execution_age_human,
        "last_execution": last_execution,
        "last_execution_brief": last_execution_brief,
        "last_execution_at": last_execution_at,
        "wallet_assets": wallet_assets,
        "ws_public_age_sec": ws_public_age,
        "ws_private_age_sec": ws_private_age,
        "ws_last_beat": ws_last_beat,
        "ws_public_age_human": ws_public_age_human,
        "ws_private_age_human": ws_private_age_human,
    }
