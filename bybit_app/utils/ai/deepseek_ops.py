from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..file_io import atomic_write_text
from ..log import log
from ..paths import DATA_DIR
from .deepseek_adapter import DeepSeekAdapter
from .deepseek_utils import resolve_deepseek_drawdown_limit, resolve_deepseek_watchlist
from .models import MODEL_FILENAME


@dataclass(slots=True)
class RuntimeJournalEntry:
    """Structured journal payload describing an automation cycle."""

    ts: int
    status: str
    signature: Optional[str]
    settings_marker: Tuple[bool, bool, bool]
    reason: Optional[str]
    deepseek: Optional[Dict[str, object]]
    order: Optional[Dict[str, object]]

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "ts": self.ts,
            "status": self.status,
            "settings_marker": list(self.settings_marker),
        }
        if self.signature is not None:
            payload["signature"] = self.signature
        if self.reason:
            payload["reason"] = self.reason
        if self.deepseek:
            payload["deepseek"] = self.deepseek
        if self.order:
            payload["order"] = self.order
        return payload


def _detect_local_weight() -> Path | None:
    env_path = os.environ.get("DEEPSEEK_GGUF_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
    candidate = DATA_DIR / "ai" / "weights" / "DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
    if candidate.is_file():
        return candidate
    return None


class DeepSeekRuntimeSupervisor:
    """Coordinate DeepSeek runtime hygiene: caching, journaling and alerts."""

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], DeepSeekAdapter] | None = None,
        refresh_interval: float = 30.0 * 60.0,
        enable_refresh: bool = True,
        watchlist: Sequence[str] | None = None,
        trades_path: Path | None = None,
        metrics_path: Path | None = None,
        state_path: Path | None = None,
        status_path: Path | None = None,
        model_path: Path | None = None,
        model_stale_after: float = 14.0 * 24 * 3600,
        alert_cooldown: float = 3600.0,
    ) -> None:
        self._adapter_factory = adapter_factory or DeepSeekAdapter
        self._adapter: DeepSeekAdapter | None = None
        self._local_weight_path = _detect_local_weight()
        refresh_value = float(refresh_interval)
        if (
            enable_refresh
            and refresh_interval == 30.0 * 60.0
            and self._local_weight_path is not None
        ):
            refresh_value = max(refresh_value, 3600.0)
        self._refresh_interval = max(refresh_value, 0.0)
        self._enable_refresh = bool(enable_refresh)
        self._watchlist_override = tuple(str(s).strip().upper() for s in watchlist) if watchlist else None
        self._trades_path = Path(trades_path) if trades_path else DATA_DIR / "trades" / "deepseek_runtime.jsonl"
        self._metrics_path = Path(metrics_path) if metrics_path else DATA_DIR / "pnl" / "deepseek_runtime_metrics.jsonl"
        self._state_path = Path(state_path) if state_path else DATA_DIR / "ai" / "runtime_state.json"
        self._status_path = Path(status_path) if status_path else DATA_DIR / "ai" / "status.json"
        self._model_path = Path(model_path) if model_path else DATA_DIR / "ai" / MODEL_FILENAME
        self._model_stale_after = max(float(model_stale_after), 0.0)
        self._alert_cooldown = max(float(alert_cooldown), 60.0)
        self._last_refresh = 0.0
        self._last_model_alert = 0.0
        self._watchlist_cache: Tuple[float, Tuple[str, ...]] | None = None

    # ------------------------------------------------------------------
    def _get_adapter(self) -> DeepSeekAdapter | None:
        if self._adapter is None:
            try:
                self._adapter = self._adapter_factory()
            except Exception as exc:  # pragma: no cover - defensive guard
                log("deepseek.ops.adapter_error", err=str(exc))
                self._adapter = None
        return self._adapter

    def _resolve_watchlist(self) -> list[str]:
        if self._watchlist_override is not None:
            return [s for s in self._watchlist_override if s]
        now = time.time()
        cached = self._watchlist_cache
        if cached and now - cached[0] < 300.0:
            return list(cached[1])
        symbols = resolve_deepseek_watchlist(self._status_path)
        self._watchlist_cache = (now, tuple(symbols))
        return symbols

    def _resolve_drawdown_limit(self) -> Optional[float]:
        return resolve_deepseek_drawdown_limit(self._status_path)

    @staticmethod
    def _normalise(value: object) -> object:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, Mapping):
            return {str(k): DeepSeekRuntimeSupervisor._normalise(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [DeepSeekRuntimeSupervisor._normalise(v) for v in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.hex()
        return value

    def _append_jsonl(self, path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = json.dumps(payload, ensure_ascii=False)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(serialisable)
                handle.write("\n")
        except OSError as exc:  # pragma: no cover - defensive guard
            log("deepseek.ops.journal_write_error", path=str(path), err=str(exc))

    def _extract_deepseek_context(
        self,
        context: Mapping[str, object] | None,
        order: Mapping[str, object] | None,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {}
        sources: list[Mapping[str, object]] = []
        if isinstance(context, Mapping):
            sources.append(context)
        if isinstance(order, Mapping):
            sources.append(order)
        for payload in sources:
            signal = payload.get("deepseek_signal")
            if isinstance(signal, Mapping) and "signal" not in result:
                result["signal"] = self._normalise(signal)
            guidance = payload.get("deepseek_guidance")
            if isinstance(guidance, Mapping) and "guidance" not in result:
                result["guidance"] = self._normalise(guidance)
            stop_meta = payload.get("deepseek_stop")
            if isinstance(stop_meta, Mapping) and "stop" not in result:
                result["stop"] = self._normalise(stop_meta)
        return result

    def _record_metrics(
        self,
        result: object,
        signature: Optional[str],
        marker: Tuple[bool, bool, bool],
    ) -> None:
        status = getattr(result, "status", None) or "unknown"
        reason = getattr(result, "reason", None)
        order = getattr(result, "order", None)
        context = getattr(result, "context", None)
        deepseek_payload = self._extract_deepseek_context(context, order)
        entry = RuntimeJournalEntry(
            ts=int(time.time()),
            status=str(status),
            signature=signature,
            settings_marker=marker,
            reason=str(reason) if isinstance(reason, str) and reason else None,
            deepseek=deepseek_payload or None,
            order=self._normalise(order) if isinstance(order, Mapping) else None,
        )
        metrics = entry.as_dict()
        limit = self._resolve_drawdown_limit()
        if limit is not None:
            metrics["drawdown_limit_pct"] = float(limit)
        self._append_jsonl(self._metrics_path, metrics)

    def _record_trade(self, result: object, signature: Optional[str], marker: Tuple[bool, bool, bool]) -> None:
        order = getattr(result, "order", None)
        context = getattr(result, "context", None)
        reason = getattr(result, "reason", None)
        payload = {
            "ts": int(time.time()),
            "status": getattr(result, "status", "unknown"),
            "signature": signature,
            "settings_marker": list(marker),
        }
        if isinstance(reason, str) and reason:
            payload["reason"] = reason
        if isinstance(order, Mapping):
            payload["order"] = self._normalise(order)
        deepseek_payload = self._extract_deepseek_context(context, order)
        if deepseek_payload:
            payload["deepseek"] = deepseek_payload
        self._append_jsonl(self._trades_path, payload)

    def _store_state(self, **updates: object) -> None:
        state = {
            "last_refresh_ts": self._last_refresh,
            "model_alert_ts": self._last_model_alert,
            "watchlist": self._resolve_watchlist(),
        }
        state.update({k: v for k, v in updates.items() if v is not None})
        try:
            atomic_write_text(self._state_path, json.dumps(state, ensure_ascii=False, indent=2))
        except OSError as exc:  # pragma: no cover - defensive guard
            log("deepseek.ops.state_write_error", path=str(self._state_path), err=str(exc))

    def _maybe_refresh_deepseek(self) -> None:
        if not self._enable_refresh or self._refresh_interval <= 0.0:
            return
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return
        adapter = self._get_adapter()
        if adapter is None:
            return
        api_key = getattr(adapter, "api_key", None)
        local_model_path = getattr(adapter, "local_model_path", None)
        local_candidate: Path | None = None
        if local_model_path:
            try:
                candidate = Path(local_model_path)
            except TypeError:
                candidate = None
            else:
                if candidate.is_file():
                    local_candidate = candidate
        if not api_key and local_candidate is None:
            return
        symbols = self._resolve_watchlist()
        if local_candidate is not None:
            # Local inference is comparatively expensive; refresh only the
            # first symbol to keep cache warm without overwhelming the host.
            if symbols:
                symbols = symbols[:1]
            else:
                return
        if not symbols:
            return
        refreshed: list[str] = []
        for symbol in symbols:
            try:
                adapter.get_signal(symbol)
            except Exception as exc:  # pragma: no cover - defensive guard
                log("deepseek.ops.refresh_error", symbol=symbol, err=str(exc))
                continue
            refreshed.append(symbol)
        self._last_refresh = now
        if refreshed:
            log("deepseek.ops.cache_refreshed", symbols=refreshed, count=len(refreshed))
        self._store_state(refreshed_symbols=refreshed, last_refresh_ts=now)

    def _maybe_flag_model_refresh(self) -> None:
        if self._model_stale_after <= 0.0:
            return
        path = self._model_path
        if not path.exists():
            now = time.time()
            if now - self._last_model_alert >= self._alert_cooldown:
                log("deepseek.ops.model_missing", path=str(path))
                self._last_model_alert = now
                self._store_state(model_alert_ts=now)
            return
        age = time.time() - path.stat().st_mtime
        if age <= self._model_stale_after:
            return
        now = time.time()
        if now - self._last_model_alert < self._alert_cooldown:
            return
        days = age / 86400.0
        log("deepseek.ops.model_stale", path=str(path), age_days=round(days, 2))
        self._last_model_alert = now
        self._store_state(model_alert_ts=now)

    # ------------------------------------------------------------------
    def process_cycle(
        self,
        result: object,
        signature: Optional[str],
        marker: Sequence[bool],
    ) -> None:
        marker_tuple = tuple(bool(x) for x in (list(marker) + [False, False, False])[:3])
        self._record_metrics(result, signature, marker_tuple)
        status = getattr(result, "status", "")
        if status in {"filled", "dry_run"}:
            self._record_trade(result, signature, marker_tuple)
        self._maybe_refresh_deepseek()
        self._maybe_flag_model_refresh()


__all__ = ["DeepSeekRuntimeSupervisor"]
