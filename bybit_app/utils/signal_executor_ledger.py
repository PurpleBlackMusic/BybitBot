"""Ledger and execution reconciliation helpers for the signal executor."""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Dict, Mapping, Optional, Sequence

from .envs import Settings
from .signal_executor_models import _DECIMAL_ZERO
from .helpers import ensure_link_id


class SignalExecutorLedgerMixin:
    """Utility methods for reconciling orders with ledger and websocket data."""

    @staticmethod
    def _filter_symbol_ledger_rows(
        rows: Sequence[Mapping[str, object]],
        symbol: str,
    ) -> list[Mapping[str, object]]:
        symbol_upper = symbol.upper()
        filtered: list[Mapping[str, object]] = []
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("category") or "spot").lower() != "spot":
                continue
            if str(entry.get("symbol") or "").strip().upper() != symbol_upper:
                continue
            filtered.append(entry)
        return filtered

    @classmethod
    def _extract_new_symbol_rows(
        cls,
        rows_before: Sequence[Mapping[str, object]],
        rows_after: Sequence[Mapping[str, object]],
        symbol: str,
    ) -> list[Mapping[str, object]]:
        before_filtered = cls._filter_symbol_ledger_rows(rows_before, symbol)
        after_filtered = cls._filter_symbol_ledger_rows(rows_after, symbol)
        prefix = 0
        limit = min(len(before_filtered), len(after_filtered))
        while prefix < limit and cls._ledger_rows_equivalent(
            before_filtered[prefix], after_filtered[prefix]
        ):
            prefix += 1
        return after_filtered[prefix:]

    @staticmethod
    def _ledger_rows_equivalent(
        a: Mapping[str, object], b: Mapping[str, object]
    ) -> bool:
        if a is b:
            return True
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            return False
        try:
            return dict(a) == dict(b)
        except Exception:
            return a == b

    def _realized_delta(
        self,
        rows_before: Sequence[Mapping[str, object]],
        rows_after: Sequence[Mapping[str, object]],
        symbol: str,
        *,
        new_rows: Optional[Sequence[Mapping[str, object]]] = None,
        before_state: Optional[Mapping[str, object]] = None,
        before_layers: Optional[Mapping[str, object]] = None,
        settings: Optional[Settings] = None,
    ) -> Decimal:
        symbol_upper = symbol.upper()
        candidate_rows_source = (
            list(new_rows)
            if new_rows is not None
            else self._extract_new_symbol_rows(rows_before, rows_after, symbol_upper)
        )
        if not candidate_rows_source:
            return _DECIMAL_ZERO

        candidate_rows = self._filter_symbol_ledger_rows(
            candidate_rows_source, symbol_upper
        )
        if not candidate_rows:
            return _DECIMAL_ZERO

        has_sell_events = any(
            str(row.get("side") or "").strip().lower() == "sell"
            for row in candidate_rows
        )
        if not has_sell_events:
            return _DECIMAL_ZERO

        inventory: dict[str, dict[str, float]] = {}
        layers: dict[str, dict[str, object]] = {}

        resolved_state: Optional[Mapping[str, object]] = (
            before_state if isinstance(before_state, Mapping) else None
        )
        resolved_layers: Optional[Mapping[str, object]] = (
            before_layers if isinstance(before_layers, Mapping) else None
        )

        if resolved_state is None:
            try:
                inventory_snapshot, layer_snapshot = self._spot_inventory_and_pnl(
                    settings=settings, return_layers=True
                )
            except Exception:
                inventory_snapshot, layer_snapshot = {}, {}
            if isinstance(inventory_snapshot, Mapping):
                candidate_state = inventory_snapshot.get(symbol_upper)
                if isinstance(candidate_state, Mapping):
                    resolved_state = candidate_state
            if resolved_layers is None and isinstance(layer_snapshot, Mapping):
                candidate_layers = layer_snapshot.get(symbol_upper)
                if isinstance(candidate_layers, Mapping):
                    resolved_layers = candidate_layers

        realized_before = _DECIMAL_ZERO
        if isinstance(resolved_state, Mapping):
            realized_before = self._decimal_from(resolved_state.get("realized_pnl"))
            inventory[symbol_upper] = {
                "position_qty": float(
                    self._decimal_from(resolved_state.get("position_qty"))
                ),
                "avg_cost": float(
                    self._decimal_from(resolved_state.get("avg_cost"))
                ),
                "realized_pnl": float(realized_before),
            }
        else:
            inventory[symbol_upper] = {
                "position_qty": 0.0,
                "avg_cost": 0.0,
                "realized_pnl": 0.0,
            }

        if isinstance(resolved_layers, Mapping):
            layers[symbol_upper] = copy.deepcopy(resolved_layers)
        else:
            layers[symbol_upper] = {
                "position_qty": inventory[symbol_upper]["position_qty"],
                "layers": [],
            }

        self._replay_events(candidate_rows, inventory, layers)

        after_state = inventory.get(symbol_upper)
        realized_after = (
            self._decimal_from(after_state.get("realized_pnl"))
            if isinstance(after_state, Mapping)
            else _DECIMAL_ZERO
        )

        return realized_after - realized_before

    def _filled_base_from_private_ws(
        self,
        symbol: str,
        *,
        order_id: Optional[str],
        order_link_id: Optional[str],
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if not order_id and not order_link_id:
            return _DECIMAL_ZERO

        if rows is None:
            rows = self._ws_manager().realtime_private_rows("execution")
        if not rows:
            return _DECIMAL_ZERO

        symbol_upper = symbol.upper()
        total = _DECIMAL_ZERO
        seen_exec_ids: set[str] = set()

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            if not self._row_matches_order(row, order_id, order_link_id):
                continue
            side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
            if side and side != "buy":
                continue

            cum_qty = self._decimal_from(row.get("cumExecQty"))
            if cum_qty > 0:
                total = max(total, cum_qty)
                exec_id = self._normalise_exec_id(row)
                if exec_id:
                    seen_exec_ids.add(exec_id)
                continue

            exec_id = self._normalise_exec_id(row)
            if exec_id and exec_id in seen_exec_ids:
                continue
            if exec_id:
                seen_exec_ids.add(exec_id)
            qty = self._decimal_from(row.get("execQty"))
            if qty > 0:
                total += qty

        return total

    def _filled_base_from_ledger(
        self,
        symbol: str,
        *,
        settings: Settings,
        order_id: Optional[str],
        order_link_id: Optional[str],
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if not order_id and not order_link_id:
            return _DECIMAL_ZERO

        if rows is None:
            try:
                rows = self._read_ledger(2000, settings=settings)
            except Exception:
                return _DECIMAL_ZERO
        if not rows:
            return _DECIMAL_ZERO

        symbol_upper = symbol.upper()
        total = _DECIMAL_ZERO
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            if not self._row_matches_order(row, order_id, order_link_id):
                continue
            category = str(row.get("category") or "spot").strip().lower()
            if category and category != "spot":
                continue
            side = str(row.get("side") or "").strip().lower()
            if side and side != "buy":
                continue
            qty = self._decimal_from(row.get("execQty"))
            if qty > 0:
                total += qty

        return total

    def _resolve_open_sell_reserved(
        self,
        symbol: str,
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if rows is None:
            rows = self._ws_manager().realtime_private_rows("order")
        if not rows:
            return _DECIMAL_ZERO

        symbol_upper = symbol.upper()
        reserved: Dict[str, Decimal] = {}
        total_reserved = _DECIMAL_ZERO

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
            if side != "sell":
                continue
            order_type = str(row.get("orderType") or row.get("orderTypeV2") or "").strip().lower()
            if order_type and order_type != "limit":
                continue
            status = str(row.get("orderStatus") or row.get("status") or "").strip().lower()
            if status:
                closed_prefixes = (
                    "cancel",
                    "reject",
                    "filled",
                    "trigger",
                    "inactive",
                    "deactivate",
                    "expire",
                )
                if any(status.startswith(prefix) for prefix in closed_prefixes):
                    continue
            qty = self._decimal_from(row.get("leavesQty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("qty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("orderQty"))
            if qty <= 0:
                continue
            key: Optional[str] = None
            for candidate_key in ("orderId", "orderID", "orderLinkId", "orderLinkID"):
                candidate = row.get(candidate_key)
                if isinstance(candidate, str) and candidate.strip():
                    key = candidate.strip()
                    break
            if key is None:
                key = f"anon-{id(row)}"
            previous = reserved.get(key)
            if previous is not None:
                total_reserved -= previous
            reserved[key] = qty
            total_reserved += qty

        return total_reserved

    @staticmethod
    def _row_matches_order(
        row: Mapping[str, object],
        order_id: Optional[str],
        order_link_id: Optional[str],
    ) -> bool:
        normalised_link = ensure_link_id(order_link_id) if order_link_id else None
        if normalised_link:
            for key in ("orderLinkId", "orderLinkID"):
                candidate = row.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    if ensure_link_id(candidate.strip()) == normalised_link:
                        return True
        if order_id:
            for key in ("orderId", "orderID"):
                candidate = row.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    if candidate.strip() == order_id:
                        return True
        return not order_id and not order_link_id

    @staticmethod
    def _normalise_exec_id(row: Mapping[str, object]) -> Optional[str]:
        for key in ("execId", "execID", "executionId", "executionID", "fillId", "tradeId", "matchId"):
            candidate = row.get(key)
            if isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    return text
        return None

    def _ledger_rows_snapshot(
        self,
        limit: int = 2000,
        *,
        settings: Optional[Settings] = None,
        last_exec_id: Optional[str] = None,
    ) -> tuple[list[Mapping[str, object]], Optional[str]]:
        resolved_settings: Optional[Settings] = settings
        if resolved_settings is None:
            try:
                resolved_settings = self._resolve_settings()
            except Exception:
                resolved_settings = None
        try:
            rows, newest_exec_id, _ = self._read_ledger(
                limit,
                settings=resolved_settings,
                last_exec_id=last_exec_id,
                return_meta=True,
            )
        except Exception:
            return [], None
        clean_rows = [row for row in rows if isinstance(row, Mapping)]
        last_id: Optional[str]
        if isinstance(newest_exec_id, str) and newest_exec_id:
            last_id = newest_exec_id
        else:
            last_id = None
        return clean_rows, last_id
