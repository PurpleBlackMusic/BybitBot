from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple, Union
import copy
import json
import threading

from .pnl import _ledger_path_for, read_ledger

_SPOT_CACHE_VERSION = 1
_CACHE_LOCK = threading.Lock()
_CACHE: dict[Path, dict[str, object]] = {}


def _resolve_ledger_path(
    ledger_path: Optional[Union[str, Path]],
    *,
    settings: object | None = None,
    network: object | None = None,
) -> Path:
    if ledger_path is None:
        return _ledger_path_for(settings, network=network)
    return Path(ledger_path)


def _spot_cache_path_for(ledger_path: Path) -> Path:
    return ledger_path.with_name(ledger_path.name + ".spot_cache.json")


def _clean_inventory(payload: Mapping[str, object]) -> Optional[dict[str, float]]:
    try:
        qty = float(payload.get("position_qty", 0.0))
        avg_cost = float(payload.get("avg_cost", 0.0))
        realized = float(payload.get("realized_pnl", 0.0))
    except (AttributeError, TypeError, ValueError):
        return None
    return {"position_qty": qty, "avg_cost": avg_cost, "realized_pnl": realized}


def _clean_layers(payload: Mapping[str, object]) -> Optional[dict[str, object]]:
    try:
        qty = float(payload.get("position_qty", 0.0))
    except (AttributeError, TypeError, ValueError):
        return None

    raw_layers = payload.get("layers")
    cleaned_layers = []
    if isinstance(raw_layers, Iterable) and not isinstance(
        raw_layers, (str, bytes, bytearray, memoryview)
    ):
        for entry in raw_layers:
            if not isinstance(entry, Mapping):
                continue
            try:
                layer_qty = float(entry.get("qty", 0.0))
                price = float(entry.get("price", 0.0))
            except (TypeError, ValueError):
                continue
            ts_value = entry.get("ts")
            try:
                timestamp = float(ts_value) if ts_value is not None else None
            except (TypeError, ValueError):
                timestamp = None
            cleaned_layers.append({"qty": layer_qty, "price": price, "ts": timestamp})

    return {"position_qty": qty, "layers": cleaned_layers}


def _load_cache_entry(ledger_path: Path) -> dict[str, object]:
    cache_path = _spot_cache_path_for(ledger_path)
    default = {
        "version": _SPOT_CACHE_VERSION,
        "last_exec_id": None,
        "inventory": {},
        "layers": {},
    }

    if not cache_path.exists():
        return default

    try:
        raw_data = cache_path.read_text(encoding="utf-8")
        payload = json.loads(raw_data)
    except Exception:
        return default

    if not isinstance(payload, Mapping):
        return default

    if int(payload.get("version", 0) or 0) != _SPOT_CACHE_VERSION:
        return default

    inventory_payload = payload.get("inventory")
    layers_payload = payload.get("layers")
    last_exec_id = payload.get("last_exec_id")

    inventory: dict[str, dict[str, float]] = {}
    if isinstance(inventory_payload, Mapping):
        for symbol, entry in inventory_payload.items():
            if not isinstance(symbol, str) or not isinstance(entry, Mapping):
                continue
            cleaned = _clean_inventory(entry)
            if cleaned is not None:
                inventory[symbol] = cleaned

    layers: dict[str, dict[str, object]] = {}
    if isinstance(layers_payload, Mapping):
        for symbol, entry in layers_payload.items():
            if not isinstance(symbol, str) or not isinstance(entry, Mapping):
                continue
            cleaned_layers = _clean_layers(entry)
            if cleaned_layers is not None:
                layers[symbol] = cleaned_layers

    if isinstance(last_exec_id, str):
        last_id: Optional[str] = last_exec_id or None
    else:
        last_id = None

    return {
        "version": _SPOT_CACHE_VERSION,
        "last_exec_id": last_id,
        "inventory": inventory,
        "layers": layers,
    }


def _save_cache_entry(ledger_path: Path, payload: Mapping[str, object]) -> None:
    cache_path = _spot_cache_path_for(ledger_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)


def _clone_inventory_state(inventory: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, float]]:
    cloned: dict[str, dict[str, float]] = {}
    for symbol, entry in inventory.items():
        if not isinstance(symbol, str) or not isinstance(entry, Mapping):
            continue
        cleaned = _clean_inventory(entry)
        if cleaned is not None:
            cloned[symbol] = cleaned
    return cloned


def _clone_layer_state(layers: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    cloned: dict[str, dict[str, object]] = {}
    for symbol, entry in layers.items():
        if not isinstance(symbol, str) or not isinstance(entry, Mapping):
            continue
        cleaned = _clean_layers(entry)
        if cleaned is not None:
            cleaned_layers = []
            for layer in cleaned["layers"]:
                cleaned_layers.append(dict(layer))
            cloned[symbol] = {
                "position_qty": float(cleaned.get("position_qty", 0.0)),
                "layers": cleaned_layers,
            }
    return cloned


def _extract_timestamp(event: Mapping[str, object]) -> Optional[float]:
    for key in (
        "execTimeNs",
        "execTime",
        "transactTime",
        "transactionTime",
        "ts",
        "created_at",
    ):
        value = event.get(key)
        if value is None:
            continue
        try:
            ts = float(value)
        except (TypeError, ValueError):
            continue
        if ts > 1e18:
            ts /= 1e9
        elif ts > 1e12:
            ts /= 1e3
        return ts
    return None


def _normalise_event(
    event: Mapping[str, object],
    idx: int,
) -> Optional[Tuple[float, int, str, str, float, float, float, Optional[float]]]:
    if not isinstance(event, Mapping):
        return None

    category = str(event.get("category") or "spot").lower()
    if category != "spot":
        return None

    symbol_value = event.get("symbol") or event.get("ticker")
    symbol = str(symbol_value or "").strip().upper()
    if not symbol:
        return None

    side = str(event.get("side") or "").lower()
    try:
        price = float(event.get("execPrice") or 0.0)
        qty = float(event.get("execQty") or 0.0)
    except (TypeError, ValueError):
        return None

    if price <= 0 or qty <= 0:
        return None

    try:
        fee = abs(float(event.get("execFee") or 0.0))
    except (TypeError, ValueError):
        fee = 0.0

    ts = _extract_timestamp(event)
    sort_key = ts if ts is not None else float(idx)
    return sort_key, idx, symbol, side, price, qty, fee, ts


def _replay_events(
    events: Iterable[Mapping[str, object]],
    inventory: dict[str, dict[str, float]],
    layers: dict[str, dict[str, object]],
) -> None:
    processed: list[Tuple[float, int, str, str, float, float, float, Optional[float]]] = []
    for idx, event in enumerate(events):
        normalised = _normalise_event(event, idx)
        if normalised is not None:
            processed.append(normalised)

    if not processed:
        return

    processed.sort(key=lambda item: (item[0], item[1]))

    for _, _, symbol, side, price, qty, fee, ts in processed:
        state = inventory.setdefault(
            symbol,
            {"position_qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0},
        )

        if side == "buy":
            total_cost = state["avg_cost"] * state["position_qty"] + price * qty + fee
            state["position_qty"] += qty
            if state["position_qty"] > 1e-12:
                state["avg_cost"] = total_cost / state["position_qty"]
            else:
                state["avg_cost"] = 0.0

            layer_state = layers.setdefault(
                symbol, {"position_qty": 0.0, "layers": []}
            )
            layer_state["position_qty"] = float(layer_state.get("position_qty", 0.0) + qty)
            raw_layers = layer_state.get("layers")
            if not isinstance(raw_layers, list):
                raw_layers = []
                layer_state["layers"] = raw_layers
            effective_price = (price * qty + fee) / qty
            raw_layers.append({"qty": float(qty), "price": float(effective_price), "ts": ts})
            continue

        if side != "sell":
            continue

        qty_to_close = min(qty, max(state["position_qty"], 0.0))
        if qty_to_close <= 0:
            continue

        proceeds = price * qty_to_close - fee
        cost = state["avg_cost"] * qty_to_close
        state["position_qty"] -= qty_to_close
        state["realized_pnl"] += proceeds - cost
        if state["position_qty"] <= 1e-12:
            state["position_qty"] = 0.0
            state["avg_cost"] = 0.0

        layer_state = layers.get(symbol)
        if not isinstance(layer_state, Mapping):
            continue

        remaining = max(0.0, float(layer_state.get("position_qty", 0.0)) - qty_to_close)
        layer_state = dict(layer_state)
        layer_state["position_qty"] = remaining
        raw_layers = layer_state.get("layers")
        if not isinstance(raw_layers, list):
            raw_layers = []
        updated_layers = []
        to_consume = qty_to_close
        for layer in raw_layers:
            if to_consume <= 1e-12:
                updated_layers.append(layer)
                continue
            if not isinstance(layer, Mapping):
                continue
            try:
                layer_qty = float(layer.get("qty", 0.0))
            except (TypeError, ValueError):
                continue
            consume = min(layer_qty, to_consume)
            layer_qty -= consume
            to_consume -= consume
            if layer_qty > 1e-12:
                updated_layer = dict(layer)
                updated_layer["qty"] = layer_qty
                updated_layers.append(updated_layer)
        layer_state["layers"] = updated_layers
        layers[symbol] = layer_state


def _inventory_from_events(
    events: Iterable[Mapping[str, object]]
) -> Tuple[dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    inventory: dict[str, dict[str, float]] = {}
    layers: dict[str, dict[str, object]] = {}
    _replay_events(events, inventory, layers)
    return inventory, layers


def spot_inventory_and_pnl(
    ledger_path: Optional[Union[str, Path]] = None,
    *,
    events: Optional[Iterable[Mapping[str, object]]] = None,
    settings: object | None = None,
    network: object | None = None,
    return_layers: bool = False,
):
    """Считает среднюю цену и реализованный PnL по споту на базе леджера исполнений.
    Метод: средняя стоимость (moving average). Комиссию учитываем в цене покупки/продажи.
    Возвращает dict: {symbol: {position_qty, avg_cost, realized_pnl}}.
    """

    if events is not None:
        inventory, layers = _inventory_from_events(events)
        return (inventory, layers) if return_layers else inventory

    path = _resolve_ledger_path(
        ledger_path,
        settings=settings,
        network=network,
    )

    if not path.exists():
        return ({}, {}) if return_layers else {}

    with _CACHE_LOCK:
        entry = _CACHE.get(path)
        if entry is None:
            entry = _load_cache_entry(path)
            _CACHE[path] = entry
        cached_inventory = _clone_inventory_state(entry.get("inventory", {}))
        cached_layers = _clone_layer_state(entry.get("layers", {}))
        last_exec_id = entry.get("last_exec_id")
        if not isinstance(last_exec_id, str) or not last_exec_id:
            last_exec_id = None

    rows, newest_exec_id, marker_found = read_ledger(
        None,
        settings=settings,
        network=network,
        ledger_path=path,
        last_exec_id=last_exec_id,
        return_meta=True,
    )

    inventory = cached_inventory
    layers = cached_layers

    if not marker_found:
        inventory = {}
        layers = {}

    if rows:
        _replay_events(rows, inventory, layers)

    if rows or not marker_found:
        payload = {
            "version": _SPOT_CACHE_VERSION,
            "last_exec_id": newest_exec_id or last_exec_id,
            "inventory": inventory,
            "layers": layers,
        }
        with _CACHE_LOCK:
            _CACHE[path] = payload
        try:
            _save_cache_entry(path, payload)
        except Exception:
            pass

    result_inventory = copy.deepcopy(inventory)
    if not return_layers:
        return result_inventory

    result_layers: dict[str, dict[str, object]] = {}
    for symbol, state in layers.items():
        if not isinstance(symbol, str) or not isinstance(state, Mapping):
            continue
        clean_state = {
            "position_qty": float(state.get("position_qty", 0.0)),
            "layers": [],
        }
        raw_layers = state.get("layers")
        if isinstance(raw_layers, Iterable) and not isinstance(
            raw_layers, (str, bytes, bytearray, memoryview)
        ):
            for layer in raw_layers:
                if not isinstance(layer, Mapping):
                    continue
                try:
                    layer_qty = float(layer.get("qty", 0.0))
                    layer_price = float(layer.get("price", 0.0))
                except (TypeError, ValueError):
                    continue
                ts_value = layer.get("ts")
                try:
                    layer_ts = float(ts_value) if ts_value is not None else None
                except (TypeError, ValueError):
                    layer_ts = None
                clean_state["layers"].append(
                    {"qty": layer_qty, "price": layer_price, "ts": layer_ts}
                )
        result_layers[symbol] = clean_state

    return result_inventory, result_layers
