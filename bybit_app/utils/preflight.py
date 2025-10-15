"""Pre-flight health checks executed before automation starts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .bybit_api import BybitAPI
from .envs import (
    Settings,
    active_api_key,
    active_api_secret,
    get_api_client,
    get_settings,
)
from .live_checks import bybit_realtime_status
from .symbol_resolver import InstrumentMetadata, SymbolResolver


@dataclass
class _MetadataBundle:
    resolver: SymbolResolver
    resolved: dict[str, InstrumentMetadata]
    missing: list[str]


def _coerce_symbol_list(values: Sequence[str] | None) -> list[str]:
    seen: set[str] = set()
    normalised: list[str] = []
    for value in values or ():
        cleaned = str(value or "").strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalised.append(cleaned)
    return normalised


def _extract_symbols(settings: Settings, *, fallback: Sequence[str] = ("BTCUSDT",)) -> list[str]:
    raw = getattr(settings, "ai_symbols", "")
    symbols = []
    if isinstance(raw, str):
        symbols.extend(part.strip() for part in raw.split(","))
    elif isinstance(raw, Sequence):
        symbols.extend(raw)
    normalised = _coerce_symbol_list(symbols)
    if normalised:
        return normalised
    return _coerce_symbol_list(fallback)


def _collect_metadata(api: BybitAPI, symbols: Sequence[str]) -> _MetadataBundle:
    resolver = SymbolResolver(api, category="spot", refresh=True)
    resolved: dict[str, InstrumentMetadata] = {}
    missing: list[str] = []

    for symbol in symbols:
        metadata = resolver.resolve_symbol(symbol)
        if metadata is None:
            missing.append(symbol)
            continue
        resolved[symbol] = metadata

    return _MetadataBundle(resolver=resolver, resolved=resolved, missing=missing)


def _metadata_health(bundle: _MetadataBundle, *, requested: Sequence[str]) -> dict[str, object]:
    if bundle.resolver.is_ready and not bundle.missing:
        message = "Каталог инструментов загружен, все символы найдены."
        ok = True
    elif bundle.resolver.is_ready:
        missing = ", ".join(bundle.missing)
        message = f"Не удалось найти метаданные для: {missing}."
        ok = False
    else:
        message = "Каталог инструментов не загружен."
        ok = False

    return {
        "title": "Метаданные инструментов",
        "ok": ok,
        "message": message,
        "details": {
            "requested": list(requested),
            "resolved": sorted(bundle.resolved.keys()),
            "missing": list(bundle.missing),
            "last_refresh": bundle.resolver.last_refresh or None,
        },
    }


def _limits_health(bundle: _MetadataBundle, *, requested: Sequence[str]) -> dict[str, object]:
    problems: list[str] = []
    for symbol in requested:
        metadata = bundle.resolved.get(symbol)
        if metadata is None:
            continue
        tick = metadata.tick_size
        qty = metadata.qty_step
        notional = metadata.min_notional
        if (tick is None or tick <= 0) or (qty is None or qty <= 0) or (notional is None or notional <= 0):
            problems.append(symbol)

    if problems:
        message = "Найдены незаполненные торговые лимиты: " + ", ".join(problems)
        ok = False
    else:
        message = "Торговые лимиты загружены и валидны."
        ok = True

    details = {
        "checked": list(requested),
        "invalid": problems,
    }
    for symbol, metadata in bundle.resolved.items():
        details[symbol] = metadata.as_dict()

    return {
        "title": "Торговые лимиты",
        "ok": ok,
        "message": message,
        "details": details,
    }


def _websocket_health(
    ws_status: Mapping[str, object] | None,
    *,
    require_private: bool,
) -> dict[str, object]:
    public_info = ws_status.get("public") if isinstance(ws_status, Mapping) else None
    private_info = ws_status.get("private") if isinstance(ws_status, Mapping) else None

    def _channel_payload(info: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(info, Mapping):
            return {"running": False, "connected": None, "subscriptions": []}
        subscriptions: list[str] = []
        raw_subs = info.get("subscriptions")
        if isinstance(raw_subs, Iterable) and not isinstance(raw_subs, (str, bytes, bytearray)):
            for entry in raw_subs:
                cleaned = str(entry or "").strip()
                if cleaned:
                    subscriptions.append(cleaned)
        connected_field = info.get("connected")
        connected = None if connected_field is None else bool(connected_field)
        running = bool(info.get("running"))
        return {
            "running": running,
            "connected": connected,
            "subscriptions": subscriptions,
        }

    public_payload = _channel_payload(public_info)
    private_payload = _channel_payload(private_info)

    problems: list[str] = []
    if not public_payload["running"]:
        problems.append("публичный канал остановлен")
    elif not public_payload["subscriptions"]:
        problems.append("нет подписок публичного канала")

    if require_private:
        if not private_payload["running"]:
            problems.append("приватный канал остановлен")
        elif private_payload["connected"] is False:
            problems.append("приватный канал не подключён")

    ok = not problems
    message = "WebSocket запущен и подписки активны." if ok else "; ".join(problems)

    return {
        "title": "WebSocket подписки",
        "ok": ok,
        "message": message,
        "details": {
            "public": public_payload,
            "private": private_payload,
            "require_private": require_private,
        },
    }


def _quota_health(api: BybitAPI) -> dict[str, object]:
    snapshot = api.quota_snapshot
    if snapshot:
        message = "Квоты API загружены."
        ok = True
    else:
        message = "Биржа не вернула заголовки с квотами — мониторинг ограничен."
        ok = False
    return {
        "title": "API квоты",
        "ok": ok,
        "message": message,
        "details": snapshot,
    }


def collect_preflight_snapshot(
    settings: Settings | None = None,
    *,
    api: BybitAPI | None = None,
    ws_status: Mapping[str, object] | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, object]:
    """Gather readiness indicators used before automation loop starts."""

    if settings is None:
        settings = get_settings()
    if api is None:
        api = get_api_client()
    if ws_status is None:
        ws_status = {}

    checked_symbols = symbols if symbols is not None else _extract_symbols(settings)

    realtime = bybit_realtime_status(settings, api=api, ws_status=ws_status)

    bundle = _collect_metadata(api, checked_symbols)
    metadata_report = _metadata_health(bundle, requested=checked_symbols)
    limits_report = _limits_health(bundle, requested=checked_symbols)

    require_private = bool(active_api_key(settings) and active_api_secret(settings))
    ws_report = _websocket_health(ws_status, require_private=require_private)
    quota_report = _quota_health(api)

    components = {
        "realtime": realtime,
        "websocket": ws_report,
        "metadata": metadata_report,
        "limits": limits_report,
        "quotas": quota_report,
    }

    overall_ok = True
    for payload in components.values():
        ok_field = payload.get("ok") if isinstance(payload, Mapping) else None
        if not ok_field:
            overall_ok = False
            break

    return {
        "ok": overall_ok,
        "checked_at": time.time(),
        "symbols": list(checked_symbols),
        **components,
    }
