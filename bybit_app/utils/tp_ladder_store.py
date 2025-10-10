"""Shared persistence for executor-provided take-profit ladders."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

from .file_io import atomic_write_text, ensure_directory
from .paths import CACHE_DIR

_DEFAULT_PATH = CACHE_DIR / "tp_ladders.json"
_STORE_LOCK = threading.Lock()
_SHARED_STORE: "TPLadderStore | None" = None


class TPLadderStore:
    """Lightweight JSON-backed key/value store for TP ladder metadata."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH
        ensure_directory(self._path.parent)
        if not self._path.exists():
            atomic_write_text(self._path, "{}", encoding="utf-8")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # basic properties
    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # public API
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of the persisted payload."""

        with self._lock:
            return json.loads(self._read_text() or "{}")

    def get(self, symbol: str) -> Dict[str, Any] | None:
        symbol_key = self._normalise_symbol(symbol)
        if not symbol_key:
            return None
        payload = self.snapshot()
        entry = payload.get(symbol_key)
        if isinstance(entry, dict):
            return entry
        return None

    def update(self, symbol: str, payload: Mapping[str, Any]) -> None:
        symbol_key = self._normalise_symbol(symbol)
        if not symbol_key:
            return
        normalised = self._normalise_payload(payload)
        with self._lock:
            state = json.loads(self._read_text() or "{}")
            state[symbol_key] = normalised
            self._write_state(state)

    def delete(self, symbol: str) -> None:
        symbol_key = self._normalise_symbol(symbol)
        if not symbol_key:
            return
        with self._lock:
            state = json.loads(self._read_text() or "{}")
            if symbol_key in state:
                state.pop(symbol_key, None)
                self._write_state(state)

    # ------------------------------------------------------------------
    # internals
    @staticmethod
    def _normalise_symbol(symbol: str | None) -> str:
        if not symbol:
            return ""
        return str(symbol).strip().upper()

    @staticmethod
    def _normalise_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        signature = payload.get("signature")
        if isinstance(signature, (list, tuple)):
            signature_out = []
            for pair in signature:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                price_text = "" if pair[0] is None else str(pair[0])
                qty_text = "" if pair[1] is None else str(pair[1])
                if price_text and qty_text:
                    signature_out.append([price_text, qty_text])
            if signature_out:
                out["signature"] = signature_out

        avg_cost = payload.get("avg_cost")
        if avg_cost is not None:
            out["avg_cost"] = str(avg_cost)

        qty = payload.get("qty")
        if qty is not None:
            out["qty"] = str(qty)

        updated_ts = payload.get("updated_ts")
        if isinstance(updated_ts, (int, float)):
            out["updated_ts"] = float(updated_ts)

        status = payload.get("status")
        if isinstance(status, str) and status:
            out["status"] = status

        source = payload.get("source")
        if isinstance(source, str) and source:
            out["source"] = source

        plan_entries = payload.get("plan")
        if isinstance(plan_entries, (list, tuple)):
            normalised_plan: list[Dict[str, Any]] = []
            for entry in plan_entries:
                if not isinstance(entry, Mapping):
                    continue
                price_text = str(entry.get("price_text") or "").strip()
                qty_text = str(entry.get("qty_text") or "").strip()
                if not price_text or not qty_text:
                    continue
                profit_labels = entry.get("profit_labels")
                labels: list[str] = []
                if isinstance(profit_labels, (list, tuple)):
                    for label in profit_labels:
                        label_text = str(label).strip()
                        if label_text:
                            labels.append(label_text)
                normalised_plan.append(
                    {
                        "price_text": price_text,
                        "qty_text": qty_text,
                        "profit_labels": labels,
                    }
                )
            if normalised_plan:
                out["plan"] = normalised_plan

        return out

    def _write_state(self, state: MutableMapping[str, Any]) -> None:
        text = json.dumps(state, ensure_ascii=False)
        atomic_write_text(self._path, text, encoding="utf-8")

    def _read_text(self) -> str:
        try:
            return self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "{}"


def get_tp_ladder_store(*, path: Path | None = None) -> TPLadderStore:
    """Return a module-wide singleton, optionally overriding the storage path."""

    global _SHARED_STORE
    with _STORE_LOCK:
        if _SHARED_STORE is None:
            _SHARED_STORE = TPLadderStore(path)
        elif path is not None and _SHARED_STORE.path != path:
            _SHARED_STORE = TPLadderStore(path)
        return _SHARED_STORE


def reset_tp_ladder_store() -> None:
    """Reset the shared singleton instance. Intended for tests."""

    global _SHARED_STORE
    with _STORE_LOCK:
        _SHARED_STORE = None

