"""User-friendly helper that summarises the spot AI signals for beginners."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from statistics import StatisticsError, median, quantiles


WARNING_SIGNAL_SECONDS = 300.0
STALE_SIGNAL_SECONDS = 900.0
LIVE_TICK_STALE_SECONDS = 3.0
LISTED_SYMBOLS_REFRESH_SECONDS = 600.0

DEFAULT_SYMBOL_UNIVERSE: Tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TONUSDT",
    "LINKUSDT",
    "LTCUSDT",
)

from .ai_thresholds import (
    min_change_from_ev_bps,
    resolve_min_ev_from_settings,
)
from .ai.deepseek_utils import extract_deepseek_snapshot, evaluate_deepseek_guidance
from .envs import (
    Settings,
    active_dry_run,
    get_settings,
    get_api_client,
    normalise_network_choice,
)
from .settings_loader import call_get_settings
from .paths import DATA_DIR
from .pnl import _ledger_path_for
from .trade_analytics import (
    ExecutionRecord,
    aggregate_execution_metrics,
    normalise_execution_payload,
)
from .trade_pairs import pair_trades, pair_trades_cache_signature
from .spot_pnl import spot_inventory_and_pnl
from .symbols import ensure_usdt_symbol
from .live_checks import api_key_status, bybit_realtime_status
from .live_signal import LiveSignalError, LiveSignalFetcher
from .market_scanner import MIN_EV_CHANGE_PCT_FLOOR, scan_market_opportunities
from .instruments import get_listed_spot_symbols
from .log import log
from .universe import build_universe, is_symbol_blacklisted, load_universe


@dataclass(frozen=True)
class GuardianBrief:
    """Human-readable snapshot of the current trading situation."""

    mode: str
    symbol: str
    headline: str
    action_text: str
    confidence_text: str
    ev_text: str
    caution: str
    updated_text: str
    analysis: str
    status_age: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        """Convert the brief to a JSON-serialisable payload."""

        return asdict(self)


@dataclass(frozen=True)
class GuardianLedgerView:
    """Cached aggregation of ledger-derived analytics."""

    portfolio: Dict[str, object]
    recent_trades: Tuple[Dict[str, object], ...]
    trade_stats: Dict[str, object]
    executions: Tuple[ExecutionRecord, ...]


@dataclass(frozen=True)
class GuardianSnapshot:
    """Aggregated view of bot state, built from disk once and reused."""

    status: Dict[str, object]
    brief: GuardianBrief
    status_summary: Dict[str, object]
    status_from_cache: bool
    portfolio: Dict[str, object]
    watchlist: List[Dict[str, object]]
    symbol_plan: Dict[str, object]
    recent_trades: List[Dict[str, object]]
    trade_stats: Dict[str, object]
    executions: Tuple[ExecutionRecord, ...]
    generated_at: float
    status_signature: Tuple[int, int, int]
    ledger_signature: int


class GuardianBot:
    """Transforms raw AI outputs into safe, beginner-friendly explanations."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._ensure_dirs()
        self._settings = settings
        self._custom_settings = settings is not None
        self._last_status: Dict[str, object] = {}
        self._last_raw_status: Dict[str, object] = {}
        self._snapshot: Optional[GuardianSnapshot] = None
        self._ledger_signature: Optional[int] = None
        self._ledger_view: Optional[GuardianLedgerView] = None
        self._status_fallback_used: bool = False
        self._status_read_error: Optional[str] = None
        self._status_content_hash: Optional[int] = None
        self._status_source: str = "missing"
        self._plan_cache_signature: Optional[int] = None
        self._plan_cache: Optional[Dict[str, object]] = None
        self._plan_cache_ledger_signature: Optional[int] = None
        self._digest_cache_signature: Optional[int] = None
        self._digest_cache: Optional[Dict[str, object]] = None
        self._watchlist_breakdown_cache_signature: Optional[int] = None
        self._watchlist_breakdown_cache: Optional[Dict[str, object]] = None
        self._pair_trades_signature: Optional[Tuple[int, int, int]] = None
        self._pair_trades_cache: Optional[Tuple[Dict[str, object], ...]] = None
        self._live_fetcher: Optional[LiveSignalFetcher] = None
        self._listed_spot_symbols: Optional[Set[str]] = None
        self._listed_spot_symbols_fetched_at: Optional[float] = None
        self._symbol_universe_cache: Optional[Tuple[float, Tuple[str, ...]]] = None
        self._signal_state: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # internal plumbing
    def _ensure_dirs(self) -> None:
        for sub in ("ai", "pnl"):
            (Path(self.data_dir) / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalise_network_marker(value: object) -> Optional[str]:
        flag = normalise_network_choice(value)
        if flag is None:
            return None
        return "testnet" if flag else "mainnet"

    @classmethod
    def status_filename(cls, *, network: object | None = None) -> str:
        marker = cls._normalise_network_marker(network)
        if marker:
            return f"status.{marker}.json"
        return "status.json"

    def _status_path(self, *, network: object | None = None) -> Path:
        marker: object | None = network
        if marker is None:
            try:
                marker = getattr(self.settings, "testnet", None)
            except Exception:
                marker = None

        filename = self.status_filename(network=marker)
        return Path(self.data_dir) / "ai" / filename

    def _status_file_hint(self) -> str:
        filename = self.status_filename(network=getattr(self.settings, "testnet", None))
        return f"ai/{filename}"

    def _ledger_path(self) -> Path:
        filename = _ledger_path_for(self.settings).name
        return Path(self.data_dir) / "pnl" / filename

    @staticmethod
    def _hash_bytes(payload: bytes) -> int:
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        return int.from_bytes(digest, "big")

    @staticmethod
    def _canonicalise_for_signature(value: object) -> object:
        if is_dataclass(value):
            value = asdict(value)

        if isinstance(value, Mapping):
            ordered_items = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                ordered_items.append(
                    (
                        str(key),
                        GuardianBot._canonicalise_for_signature(value[key]),
                    )
                )
            return {key: val for key, val in ordered_items}

        if isinstance(value, (list, tuple)):
            return [GuardianBot._canonicalise_for_signature(item) for item in value]

        if isinstance(value, (set, frozenset)):
            processed = [
                GuardianBot._canonicalise_for_signature(item) for item in value
            ]
            processed.sort(key=lambda item: repr(item))
            return processed

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, float):
            number = float(value)
            if not math.isfinite(number):
                return None
            return round(number, 12)

        if isinstance(value, (int, bool)) or value is None:
            return value

        if isinstance(value, str):
            return value

        return str(value)

    def _stable_signature(self, payload: object) -> Optional[int]:
        try:
            canonical = self._canonicalise_for_signature(payload)
            serialised = json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None

        return self._hash_bytes(serialised.encode("utf-8"))

    def _prefetch_status(self) -> Tuple[Optional[bytes], Optional[int], Optional[str]]:
        path = self._status_path()
        if not path.exists():
            return None, None, "status file not found"

        attempts = 0
        max_attempts = 3
        delay = 0.05
        last_error: Optional[str] = None

        while attempts < max_attempts:
            attempts += 1
            try:
                data = path.read_bytes()
            except OSError as exc:
                last_error = str(exc)
                data = b""

            if not data.strip():
                if attempts < max_attempts:
                    time.sleep(delay * attempts)
                    continue
                if last_error is None:
                    last_error = "status file is empty"
                return None, None, last_error

            return data, self._hash_bytes(data), None

        return None, None, last_error

    def _get_live_fetcher(self) -> LiveSignalFetcher:
        fetcher = self._live_fetcher
        if fetcher is None:
            fetcher = LiveSignalFetcher(settings=self.settings, data_dir=self.data_dir)
            self._live_fetcher = fetcher
        return fetcher

    def _should_use_live_status(self, status: Dict[str, object]) -> bool:
        live_only = bool(getattr(self.settings, "ai_live_only", False))
        if live_only:
            return True

        if not status:
            return False

        source = str(status.get("source") or "").lower()
        if source in {"demo", "seed", "demo_signal"}:
            return True

        mode = str(status.get("mode") or "").lower()
        if mode == "demo":
            return True

        ts_candidate = status.get("ts") or status.get("last_tick_ts")
        try:
            ts_value = float(ts_candidate)
        except (TypeError, ValueError):
            ts_value = None

        age_seconds: Optional[float] = None
        if ts_value is not None:
            now = time.time()
            if ts_value > now + 86_400:  # demo payloads often have dates in the future
                return True
            age_seconds = now - ts_value

        if age_seconds is None:
            raw_age = status.get("age_seconds")
            try:
                age_seconds = float(raw_age) if raw_age is not None else None
            except (TypeError, ValueError):
                age_seconds = None

        if age_seconds is not None and age_seconds >= STALE_SIGNAL_SECONDS:
            return True

        return False

    def _fetch_live_status(self) -> tuple[Dict[str, object], Optional[str]]:
        try:
            fetcher = self._get_live_fetcher()
        except Exception as exc:
            return {}, f"Не удалось инициализировать live-источник: {exc}"

        try:
            status = fetcher.fetch()
        except LiveSignalError as exc:
            return {}, str(exc)
        except Exception as exc:
            return {}, f"Неожиданная ошибка live-сканера: {exc}"

        if not isinstance(status, dict):
            return {}, "Live-источник вернул некорректный ответ"

        status = copy.deepcopy(status)
        if status:
            status.setdefault("status_source", "live")
        return status, None

    def _persist_live_status(self, status: Dict[str, object]) -> None:
        path = self._status_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(status, ensure_ascii=False, indent=2)
            path.write_text(payload, encoding="utf-8")
        except Exception:
            return

    @property
    def settings(self) -> Settings:
        if self._custom_settings:
            if self._settings is None:
                self._settings = Settings()
            return self._settings

        settings = get_settings()
        self._settings = settings
        return settings

    def reload_settings(self) -> None:
        if not self._custom_settings:
            self._settings = call_get_settings(get_settings, force_reload=True)
        self._snapshot = None
        self._ledger_signature = None
        self._ledger_view = None
        self._plan_cache_signature = None
        self._plan_cache = None
        self._plan_cache_ledger_signature = None
        self._digest_cache_signature = None
        self._digest_cache = None
        self._watchlist_breakdown_cache_signature = None
        self._watchlist_breakdown_cache = None
        self._live_fetcher = None
        self._status_source = "missing"
        self._listed_spot_symbols = None
        self._listed_spot_symbols_fetched_at = None

    def _fetch_listed_spot_symbols(self) -> Set[str]:
        cached = self._listed_spot_symbols
        cached_at = self._listed_spot_symbols_fetched_at
        now = time.time()
        should_refresh = (
            cached is None
            or cached_at is None
            or now - cached_at >= LISTED_SYMBOLS_REFRESH_SECONDS
        )

        if cached is not None and not should_refresh:
            return cached

        testnet = self.settings.testnet
        try:
            listed = get_listed_spot_symbols(
                testnet=testnet, force_refresh=should_refresh
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            log("guardian.listed_symbols.error", err=str(exc), testnet=testnet)
            listed = set()

        self._listed_spot_symbols = set(listed)
        self._listed_spot_symbols_fetched_at = now if listed else cached_at
        return self._listed_spot_symbols

    def _resolve_symbol_universe(
        self,
        settings: Optional[Settings] = None,
        *,
        refresh: bool = False,
    ) -> List[str]:
        settings = settings or self.settings

        cache = self._symbol_universe_cache
        now = time.time()
        cache_ttl = 600.0
        if (
            not refresh
            and cache is not None
            and now - cache[0] < cache_ttl
        ):
            return list(cache[1])

        try:
            concurrent = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        except Exception:
            concurrent = 0
        size_hint = concurrent * 6
        if size_hint <= 0:
            size_hint = 40
        size_hint = max(size_hint, len(DEFAULT_SYMBOL_UNIVERSE))
        size_hint = min(size_hint, 120)

        raw_symbols = self._parse_symbol_list(getattr(settings, "ai_symbols", ""))

        if not raw_symbols:
            try:
                universe_snapshot = load_universe(quote_assets=("USDT",))
            except Exception:
                universe_snapshot = []
            else:
                raw_symbols = list(universe_snapshot)

        if not raw_symbols:
            try:
                api = get_api_client()
            except Exception:
                api = None
            if api is not None:
                try:
                    raw_symbols = list(
                        build_universe(
                            api,
                            size=size_hint,
                            quote_assets=("USDT",),
                            persist=True,
                        )
                    )
                except Exception:
                    raw_symbols = []

        if not raw_symbols:
            raw_symbols = list(DEFAULT_SYMBOL_UNIVERSE)

        cleaned: List[str] = []
        seen: Set[str] = set()
        for item in raw_symbols:
            symbol = str(item).strip().upper()
            if not symbol or symbol in seen:
                continue
            if is_symbol_blacklisted(symbol):
                continue
            seen.add(symbol)
            cleaned.append(symbol)
            if len(cleaned) >= size_hint:
                break

        if not cleaned:
            cleaned = list(DEFAULT_SYMBOL_UNIVERSE)

        if not getattr(settings, "ai_symbols", ""):
            try:
                settings.ai_symbols = ",".join(cleaned)
            except Exception:
                pass

        self._symbol_universe_cache = (now, tuple(cleaned))
        return list(cleaned)

    @staticmethod
    def _parse_symbol_list(raw: object) -> List[str]:
        if raw is None:
            return []

        if isinstance(raw, (list, tuple, set, frozenset)):
            items = list(raw)
        elif isinstance(raw, str):
            items = raw.split(",")
        else:
            items = str(raw).split(",")

        symbols: List[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            symbol = text.upper()
            if symbol not in symbols:
                symbols.append(symbol)

        return symbols

    def _load_status(
        self,
        prefetched: Optional[bytes] = None,
        prefetched_hash: Optional[int] = None,
    ) -> Dict[str, object]:
        path = self._status_path()
        fallback_used = False

        raw: Dict[str, object] = {}
        read_error: Optional[str] = None
        content_hash: Optional[int] = prefetched_hash
        status_source: str = "missing"
        live_only = bool(getattr(self.settings, "ai_live_only", False))

        def _decode_and_parse(data: bytes) -> Dict[str, object]:
            nonlocal read_error, content_hash
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                read_error = str(exc)
                return {}

            if not text.strip():
                if read_error is None:
                    read_error = "status file is empty"
                return {}

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                read_error = str(exc)
                return {}
            except Exception as exc:  # pragma: no cover - unexpected decoder errors
                read_error = str(exc)
                return {}

            if isinstance(parsed, dict):
                return parsed

            read_error = f"status payload is {type(parsed).__name__}, expected dict"
            return {}

        if not live_only:
            if prefetched is not None:
                data = prefetched
                if content_hash is None:
                    content_hash = self._hash_bytes(data)
                raw = _decode_and_parse(data)
                if raw:
                    status_source = "file"
            elif path.exists():
                attempts = 0
                max_attempts = 3
                delay = 0.05
                while attempts < max_attempts:
                    attempts += 1
                    try:
                        data = path.read_bytes()
                    except OSError as exc:
                        read_error = str(exc)
                        data = b""

                    if not data.strip():
                        if attempts < max_attempts:
                            time.sleep(delay * attempts)
                            continue
                        if read_error is None:
                            read_error = "status file is empty"
                        break

                    content_hash = self._hash_bytes(data)
                    raw = _decode_and_parse(data)
                    if raw:
                        status_source = "file"
                        read_error = None
                        break

                    if attempts < max_attempts:
                        time.sleep(delay * attempts)

                if not raw and read_error is None and not path.exists():
                    read_error = "status file disappeared during read"
            else:
                read_error = "status file not found"

            if not raw and self._last_raw_status:
                raw = copy.deepcopy(self._last_raw_status)
                fallback_used = True
                status_source = "cached"
                if content_hash is None:
                    content_hash = self._status_content_hash

        status = dict(raw)
        live_status_used = False

        live_error: Optional[str] = None

        if live_only or self._should_use_live_status(status):
            live_candidate, live_error = self._fetch_live_status()
            if live_candidate:
                status = live_candidate
                fallback_used = False
                read_error = None
                live_status_used = True
                content_hash = None
                status_source = "live"
            elif live_only:
                status = {}
                read_error = live_error or "live status unavailable"
                status_source = "missing"
                fallback_used = False

        if live_error and not live_status_used:
            if read_error:
                read_error = f"{live_error}; {read_error}"
            else:
                read_error = live_error

        if status:
            try:
                ts_value = float(
                    status.get("last_tick_ts") or status.get("ts") or 0.0
                )
            except Exception:
                ts_value = 0.0

            if ts_value > 0:
                status["age_seconds"] = time.time() - ts_value
            else:
                status.setdefault("age_seconds", None)

        if status:
            if status_source == "missing":
                existing_source = status.get("status_source")
                if isinstance(existing_source, str) and existing_source.strip():
                    status_source = existing_source.strip().lower()
            status = dict(status)
            status["status_source"] = status_source

        if status and content_hash is None:
            try:
                canonical = json.dumps(status, sort_keys=True).encode("utf-8")
            except Exception:
                content_hash = self._status_content_hash
            else:
                content_hash = self._hash_bytes(canonical)

        if live_status_used and status:
            self._persist_live_status(status)

        if content_hash is None:
            content_hash = 0

        self._status_content_hash = content_hash
        self._status_fallback_used = fallback_used and not live_only
        self._status_read_error = read_error if read_error else None
        self._status_source = status_source
        return status

    @staticmethod
    def _extract_text(status: Dict[str, object], keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            value = status.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalise_mode_hint(value: object) -> Optional[str]:
        """Convert heterogeneous directional hints into canonical modes."""

        if value is None:
            return None

        if isinstance(value, bool):
            return "buy" if value else "sell"

        if isinstance(value, (int, float)):
            if value > 0:
                return "buy"
            if value < 0:
                return "sell"
            return "wait"

        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower()
        if not cleaned:
            return None

        # Normalise delimiters and remove apostrophes for "don't" like phrases.
        cleaned = cleaned.replace("-", " ").replace("_", " ").replace("'", "")

        # Try to interpret numeric or boolean-looking strings early.
        if cleaned in {"true", "yes"}:
            return "buy"
        if cleaned in {"false", "no"}:
            return "sell"
        try:
            numeric = float(cleaned)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric > 0:
                return "buy"
            if numeric < 0:
                return "sell"
            return "wait"

        tokens = [token for token in cleaned.split() if token]
        joined = " ".join(tokens)

        phrase_map = {
            "no trade": "wait",
            "do not trade": "wait",
            "dont trade": "wait",
            "no signal": "wait",
            "stand aside": "wait",
            "stay aside": "wait",
            "не торгуй": "wait",
            "не торгуем": "wait",
            "не торговать": "wait",
            "ничего не делаем": "wait",
            "без сделки": "wait",
            "take profit": "sell",
            "take profits": "sell",
            "trim position": "sell",
            "фиксиру": "sell",
            "зафиксиру": "sell",
        }
        for phrase, mode in phrase_map.items():
            if phrase in joined:
                return mode

        buy_tokens = {
            "buy",
            "long",
            "bull",
            "bullish",
            "accumulate",
            "accumulation",
            "bid",
            "покупай",
            "покупаем",
            "покупка",
            "покупать",
            "лонг",
        }
        sell_tokens = {
            "sell",
            "short",
            "bear",
            "bearish",
            "distribute",
            "distribution",
            "exit",
            "reduce",
            "trim",
            "продавай",
            "продаем",
            "продаём",
            "продажа",
            "продавать",
            "фиксиру",
            "зафиксиру",
            "сократи",
            "сокращаем",
        }
        wait_tokens = {
            "wait",
            "hold",
            "holding",
            "flat",
            "neutral",
            "idle",
            "none",
            "stay",
            "pause",
            "observe",
            "sideline",
            "watch",
            "ждать",
            "ждём",
            "ждем",
            "ждите",
            "ждемс",
            "ожидаем",
            "ожидай",
            "держим",
            "держи",
            "держать",
            "пауза",
            "паузы",
            "ничего",
            "сидим",
            "выжидаем",
        }

        for token in tokens:
            if token in buy_tokens:
                return "buy"
            if token in sell_tokens:
                return "sell"
            if token in wait_tokens:
                return "wait"

        prefixes = {
            "buy": "buy",
            "long": "buy",
            "bull": "buy",
            "accum": "buy",
            "покуп": "buy",
            "лонг": "buy",
            "sell": "sell",
            "short": "sell",
            "bear": "sell",
            "distrib": "sell",
            "exit": "sell",
            "trim": "sell",
            "прода": "sell",
            "фикс": "sell",
            "сократ": "sell",
            "reduc": "sell",
            "wait": "wait",
            "hold": "wait",
            "flat": "wait",
            "neutral": "wait",
            "pause": "wait",
            "idle": "wait",
            "stay": "wait",
            "watch": "wait",
            "side": "wait",
            "жд": "wait",
            "ожид": "wait",
            "держ": "wait",
            "пауз": "wait",
            "ничег": "wait",
            "сид": "wait",
            "выжид": "wait",
        }
        for token in tokens:
            for prefix, mode in prefixes.items():
                if token.startswith(prefix):
                    return mode

        if tokens:
            first = tokens[0]
            if first in {"up", "increase", "add"}:
                return "buy"
            if first in {"down", "decrease", "cut"}:
                return "sell"

        return None

    def _narrative_from_status(self, status: Dict[str, object], mode: str, symbol: str) -> str:
        narrative = self._extract_text(
            status,
            (
                "explanation",
                "narrative",
                "commentary",
                "context",
                "reason",
                "summary",
            ),
        )
        if narrative:
            return narrative

        if mode == "buy":
            return (
                f"Модель заметила усиливающийся спрос на {symbol}. "
                "Мы готовимся покупать только ту долю, которая вписывается в лимиты по риску."
            )
        if mode == "sell":
            return (
                f"Цена по {symbol} дошла до зоны, где выгодно зафиксировать часть прибыли. "
                "Решение сопровождаем проверкой позиции и комиссии."
            )
        return (
            f"По {symbol} нет чёткой тенденции. Мы сохраняем капитал и ждём, пока данные дадут явное преимущество."
        )

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _resolve_thresholds(
        self, settings: Settings
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Normalise user-provided thresholds and derive safe fallbacks."""

        def _clamp_threshold(value: object, default: float) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = default
            return max(0.0, min(numeric, 1.0))

        buy_threshold = _clamp_threshold(
            getattr(settings, "ai_buy_threshold", None),
            0.52,
        )
        sell_threshold = _clamp_threshold(
            getattr(settings, "ai_sell_threshold", None),
            0.42,
        )
        min_ev = resolve_min_ev_from_settings(settings, default_bps=12.0)

        effective_buy_threshold = buy_threshold
        effective_sell_threshold = max(0.0, min(sell_threshold, effective_buy_threshold))

        hysteresis = getattr(settings, "ai_signal_hysteresis", None)
        try:
            hysteresis_margin = float(hysteresis)
        except (TypeError, ValueError):
            hysteresis_margin = 0.015
        if not math.isfinite(hysteresis_margin):
            hysteresis_margin = 0.015
        hysteresis_margin = max(0.0, min(hysteresis_margin, 0.25))

        exit_buy_threshold = min(1.0, effective_buy_threshold + hysteresis_margin)
        exit_sell_threshold = max(0.0, effective_sell_threshold - hysteresis_margin)

        return (
            buy_threshold,
            sell_threshold,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            exit_buy_threshold,
            exit_sell_threshold,
        )

    @staticmethod
    def _age_to_text(age: Optional[float]) -> str:
        if age is None:
            return "Данные обновлены только что."
        if age < 60:
            return "Данные обновлены менее минуты назад."
        minutes = int(age // 60)
        if minutes == 0:
            return "Данные обновлены около минуты назад."
        if minutes < 5:
            return f"Данные обновлены {minutes} мин назад."
        return "Данные не поступали более пяти минут — убедитесь, что соединение с ботом активно."

    @staticmethod
    def _format_timestamp(value: object) -> str:
        """Convert diverse timestamp values to a short human string."""

        if value in (None, ""):
            return "—"

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        # heuristics: timestamps may come in seconds, milliseconds or nanoseconds
        if numeric > 1e18:
            numeric /= 1e9
        elif numeric > 1e12:
            numeric /= 1e3

        try:
            dt = datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return "—"
        return dt.strftime("%d.%m %H:%M")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return "менее минуты"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes} мин"
        hours = minutes // 60
        minutes = minutes % 60
        if hours < 24:
            if minutes == 0:
                return f"{hours} ч"
            return f"{hours} ч {minutes} мин"
        days = hours // 24
        hours = hours % 24
        if hours == 0:
            return f"{days} дн"
        return f"{days} дн {hours} ч"

    def _status_staleness(self, age: Optional[float]) -> Tuple[str, str]:
        """Categorise the freshness of the AI status file."""

        live_only = bool(getattr(self.settings, "ai_live_only", False))
        subject = "Live-сигнал" if live_only else "AI сигнал"
        suffix = (
            "проверьте подключение к прямому каналу Bybit."
            if live_only
            else f"убедитесь, что сервис записи {self._status_file_hint()} активен."
        )

        if age is None:
            return "fresh", f"{subject} обновился только что."
        if age < 60:
            return "fresh", f"{subject} обновился менее минуты назад."
        if age < WARNING_SIGNAL_SECONDS:
            return "fresh", f"{subject} обновился {int(age // 60)} мин назад."
        if age < STALE_SIGNAL_SECONDS:
            return (
                "warning",
                f"{subject} обновлялся {self._format_duration(age)} назад — дождитесь скорого обновления.",
            )
        return (
            "stale",
            f"{subject} не обновлялся более 15 минут — {suffix}",
        )

    # snapshot helpers -------------------------------------------------
    def _snapshot_signature(self) -> Tuple[Tuple[int, int], int]:
        status_path = self._status_path()
        ledger_path = self._ledger_path()

        def _stat_signature(path: Path) -> Tuple[int, int]:
            try:
                stat = path.stat()
            except FileNotFoundError:
                return 0, 0

            mtime_ns = getattr(stat, "st_mtime_ns", None)
            if mtime_ns is None:
                mtime_ns = int(stat.st_mtime * 1_000_000_000)
            size = getattr(stat, "st_size", 0)
            return int(mtime_ns), int(size)

        status_sig = _stat_signature(status_path)
        ledger_mtime, ledger_size = _stat_signature(ledger_path)

        if ledger_mtime == 0 and ledger_size == 0:
            ledger_sig = 0
        else:
            ledger_sig = self._hash_bytes(f"{ledger_mtime}:{ledger_size}".encode("utf-8"))

        return status_sig, ledger_sig

    def _load_ledger_events(self) -> List[Dict[str, object]]:
        path = self._ledger_path()
        if not path.exists():
            return []

        events: List[Dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        events.append(payload)
        except OSError:
            return []
        return events

    @staticmethod
    def _spot_events(events: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        spot: List[Dict[str, object]] = []
        for event in events:
            category = str(event.get("category") or "spot").lower()
            if category == "spot":
                spot.append(event)
        return spot

    def _ledger_view_for_signature(self, signature: int) -> GuardianLedgerView:
        cached_signature = self._ledger_signature
        cached_view = self._ledger_view
        if cached_signature == signature and cached_view is not None:
            return cached_view

        events = self._load_ledger_events()
        spot_events = self._spot_events(events)
        portfolio = self._build_portfolio(spot_events)
        recent_trades = tuple(self._build_recent_trades(spot_events))
        executions = self._build_execution_records(events)
        trade_stats = aggregate_execution_metrics(executions)

        view = GuardianLedgerView(
            portfolio=portfolio,
            recent_trades=recent_trades,
            trade_stats=trade_stats,
            executions=executions,
        )

        self._ledger_signature = signature
        self._ledger_view = view
        return view

    def _build_portfolio(self, spot_events: List[Dict[str, object]]) -> Dict[str, object]:
        inventory = spot_inventory_and_pnl(events=spot_events)
        positions: List[Dict[str, object]] = []
        total_realized = 0.0
        total_notional = 0.0
        open_positions = 0

        for symbol in sorted(inventory.keys()):
            rec = inventory[symbol]
            qty = float(rec.get("position_qty") or 0.0)
            avg_cost = float(rec.get("avg_cost") or 0.0)
            realized = float(rec.get("realized_pnl") or 0.0)
            notional = qty * avg_cost

            total_realized += realized
            total_notional += notional
            if qty > 0:
                open_positions += 1

            positions.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_cost": avg_cost,
                    "notional": notional,
                    "realized_pnl": realized,
                }
            )

        human_totals = {
            "realized": f"{total_realized:.2f} USDT",
            "open_notional": f"{total_notional:.2f} USDT",
            "open_positions": f"{open_positions}",
        }

        return {
            "positions": positions,
            "totals": {
                "realized_pnl": total_realized,
                "open_notional": total_notional,
                "open_positions": open_positions,
            },
            "human_totals": human_totals,
        }

    def _build_recent_trades(
        self, spot_events: List[Dict[str, object]], limit: int = 50
    ) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for event in reversed(spot_events):
            if len(records) >= limit:
                break
            symbol = str(event.get("symbol") or event.get("ticker") or "?")
            if not symbol:
                continue
            side = str(event.get("side") or event.get("direction") or "").capitalize()
            qty = float(event.get("execQty") or event.get("qty") or event.get("size") or 0.0)
            price = float(event.get("execPrice") or event.get("price") or 0.0)
            fee = float(event.get("execFee") or event.get("fee") or 0.0)
            ts = (
                event.get("execTime")
                or event.get("execTimeNs")
                or event.get("transactTime")
                or event.get("created_at")
                or event.get("tradeTime")
            )

            records.append(
                {
                    "symbol": symbol.upper(),
                    "side": side or "—",
                    "price": round(price, 6) if price else None,
                    "qty": round(qty, 6) if qty else None,
                    "fee": round(fee, 6) if fee else None,
                    "when": self._format_timestamp(ts),
                }
            )
        return records

    @staticmethod
    def _build_execution_records(
        events: Iterable[Dict[str, object]]
    ) -> Tuple[ExecutionRecord, ...]:
        records: List[ExecutionRecord] = []
        for event in events:
            record = normalise_execution_payload(event)
            if record is not None:
                records.append(record)
        return tuple(records)

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    def _normalise_probability_input(self, value: object) -> Optional[float]:
        numeric = self._coerce_float(value)
        if numeric is None:
            return None
        if numeric > 1.0:
            if numeric <= 100.0:
                numeric /= 100.0
            else:
                numeric = 1.0
        elif numeric < 0.0:
            numeric = 0.0
        return self._clamp_probability(numeric)

    def _market_scanner_watchlist(
        self, settings: Optional[Settings] = None
    ) -> List[Dict[str, object]]:
        settings = settings or self.settings
        enabled = bool(getattr(settings, "ai_market_scan_enabled", False))
        if not enabled:
            return []

        try:
            min_turnover = float(getattr(settings, "ai_min_turnover_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_turnover = 0.0

        try:
            max_spread = float(getattr(settings, "ai_max_spread_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_spread = 0.0

        min_ev_bps = resolve_min_ev_from_settings(settings, default_bps=12.0)
        min_change_pct = min_change_from_ev_bps(
            min_ev_bps, floor=MIN_EV_CHANGE_PCT_FLOOR
        )

        limit_hint = int(getattr(settings, "ai_max_concurrent", 0) or 0) * 4
        if limit_hint <= 0:
            limit_hint = 25

        raw_whitelist = self._parse_symbol_list(getattr(settings, "ai_whitelist", ""))
        symbol_universe = self._resolve_symbol_universe(settings)

        whitelist_sources: List[str] = []
        if raw_whitelist:
            whitelist_sources.extend(raw_whitelist)

        if symbol_universe:
            limit_slice = max(limit_hint, len(DEFAULT_SYMBOL_UNIVERSE))
            whitelist_sources.extend(symbol_universe[:limit_slice])

        combined_whitelist: List[str] = []
        whitelist_seen: Set[str] = set()
        for candidate in whitelist_sources:
            cleaned = str(candidate).strip().upper()
            if not cleaned or cleaned in whitelist_seen:
                continue
            whitelist_seen.add(cleaned)
            combined_whitelist.append(cleaned)

        whitelist: Optional[Sequence[str]] = combined_whitelist or None

        universe_set: Set[str] = {sym.strip().upper() for sym in symbol_universe if sym}

        blacklist = self._parse_symbol_list(getattr(settings, "ai_blacklist", ""))

        try:
            api = get_api_client()
        except Exception:
            api = None

        testnet = bool(getattr(settings, "testnet", False))

        try:
            min_top_quote = float(getattr(settings, "ai_min_top_quote_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_top_quote = 0.0

        try:
            opportunities = scan_market_opportunities(
                api,
                data_dir=self.data_dir,
                limit=limit_hint,
                min_turnover=min_turnover,
                min_change_pct=min_change_pct,
                max_spread_bps=max_spread,
                whitelist=whitelist,
                blacklist=blacklist or None,
                settings=settings,
                testnet=testnet,
                min_top_quote=min_top_quote,
            )
        except Exception:
            return []

        if universe_set:
            allowed_symbols = set(universe_set)
            allowed_symbols.update(whitelist_seen)
            filtered_entries: List[Dict[str, object]] = []
            removed_entries: List[Dict[str, object]] = []
            for entry in opportunities:
                symbol_value = entry.get("symbol")
                if not isinstance(symbol_value, str):
                    continue
                symbol_key = symbol_value.strip().upper()
                if symbol_key in allowed_symbols:
                    filtered_entries.append(entry)
                else:
                    removed_entries.append(entry)
            if filtered_entries:
                if removed_entries and len(filtered_entries) < limit_hint:
                    extras = [
                        entry
                        for entry in removed_entries
                        if bool(entry.get("actionable"))
                        or bool(entry.get("source"))
                    ]
                    if extras:
                        room = max(limit_hint - len(filtered_entries), 0)
                        filtered_entries.extend(extras[:room])
                opportunities = filtered_entries
        return opportunities

    def _symbol_plan_signature(
        self,
        watchlist: Sequence[Dict[str, object]],
        settings: Settings,
        portfolio: Optional[Dict[str, object]] = None,
    ) -> Optional[int]:
        try:
            limit_hint = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        except Exception:
            limit_hint = 0

        def _float_component(value: object, decimals: int) -> str:
            numeric = self._coerce_float(value)
            if numeric is None or not math.isfinite(numeric):
                return ""
            return f"{numeric:.{decimals}f}"

        def _text_component(value: object) -> str:
            if value is None:
                return ""
            text = str(value).strip()
            return text

        watchlist_fingerprint: List[List[str]] = []
        for idx, raw_entry in enumerate(watchlist):
            if not isinstance(raw_entry, Mapping):
                symbol = self._normalise_symbol_value(raw_entry) or _text_component(raw_entry)
                watchlist_fingerprint.append(
                    [
                        str(idx),
                        symbol,
                        "",
                        "0",
                        "0",
                        "0",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue

            symbol = self._normalise_symbol_value(raw_entry.get("symbol")) or ""
            trend = (
                self._normalise_mode_hint(
                    raw_entry.get("trend_hint") or raw_entry.get("trend")
                )
                or ""
            )
            note = _text_component(raw_entry.get("note"))
            source = _text_component(raw_entry.get("source"))

            watchlist_fingerprint.append(
                [
                    str(idx),
                    symbol,
                    trend,
                    "1" if raw_entry.get("actionable") else "0",
                    "1" if raw_entry.get("probability_ready") else "0",
                    "1" if raw_entry.get("ev_ready") else "0",
                    _float_component(raw_entry.get("probability"), 6),
                    _float_component(raw_entry.get("ev_bps"), 6),
                    _float_component(raw_entry.get("score"), 6),
                    _float_component(raw_entry.get("edge_score"), 6),
                    note,
                    source,
                ]
            )

        portfolio_fingerprint: List[Tuple[str, str, str, str]] = []
        if isinstance(portfolio, Mapping):
            raw_positions = portfolio.get("positions") or []
            for raw_position in raw_positions:
                if not isinstance(raw_position, Mapping):
                    continue
                symbol = self._normalise_symbol_value(raw_position.get("symbol"))
                if not symbol:
                    continue
                qty = _float_component(raw_position.get("qty"), 8)
                notional = _float_component(raw_position.get("notional"), 6)
                avg_cost = _float_component(raw_position.get("avg_cost"), 8)
                portfolio_fingerprint.append((symbol, qty, notional, avg_cost))

        portfolio_fingerprint.sort()

        payload = {
            "limit": limit_hint,
            "watchlist": watchlist_fingerprint,
            "positions": portfolio_fingerprint,
        }

        try:
            serialised = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None

        return self._hash_bytes(serialised.encode("utf-8"))

    def _build_symbol_plan(
        self,
        watchlist: Sequence[Dict[str, object]],
        settings: Optional[Settings] = None,
        portfolio: Optional[Dict[str, object]] = None,
        *,
        ledger_signature: Optional[int] = None,
    ) -> Dict[str, object]:
        settings = settings or self.settings

        signature: Optional[int]
        try:
            signature = self._symbol_plan_signature(watchlist, settings, portfolio)
        except Exception:
            signature = None

        if (
            signature is not None
            and signature == self._plan_cache_signature
            and self._plan_cache is not None
            and (
                ledger_signature is None
                or ledger_signature == self._plan_cache_ledger_signature
            )
        ):
            return copy.deepcopy(self._plan_cache)

        actionable: List[str] = []
        ready: List[str] = []
        watchlist_symbols: List[str] = []
        position_symbols: List[str] = []
        position_payload: Dict[str, Dict[str, float]] = {}
        total_notional = 0.0

        def _coerce_metric(value: object) -> Optional[float]:
            if isinstance(value, bool):
                return None
            return self._coerce_float(value)

        symbol_details: Dict[str, Dict[str, object]] = {}

        def _record(symbol: str, source: Optional[str] = None, **fields: object) -> None:
            if not symbol:
                return
            detail = symbol_details.setdefault(symbol, {"sources": []})
            sources: List[str] = detail.setdefault("sources", [])  # type: ignore[assignment]
            if source and source not in sources:
                sources.append(source)

            if source == "actionable":
                detail["actionable"] = True
            elif source == "ready":
                detail["ready"] = True
            elif source == "watchlist":
                detail["watchlist"] = True
            elif source == "holding":
                detail["holding"] = True
            elif source == "positions_only":
                detail["positions_only"] = True

            for key, value in fields.items():
                if value is None:
                    continue
                if key == "note":
                    value = str(value).strip()
                    if not value:
                        continue
                if key not in detail:
                    detail[key] = value

        if portfolio:
            raw_positions = portfolio.get("positions") or []
            for entry in raw_positions:
                symbol_value = entry.get("symbol")
                if not isinstance(symbol_value, str):
                    continue
                symbol = symbol_value.strip().upper()
                if not symbol:
                    continue

                qty = self._coerce_float(entry.get("qty")) or 0.0
                notional = self._coerce_float(entry.get("notional"))
                avg_cost = self._coerce_float(entry.get("avg_cost")) or 0.0
                if notional is None:
                    notional = qty * avg_cost

                if qty <= 0 and (notional is None or notional <= 0):
                    continue

                notional = float(notional or 0.0)
                position_symbols.append(symbol)
                position_payload[symbol] = {
                    "qty": float(qty),
                    "notional": notional,
                    "avg_cost": avg_cost,
                }
                if notional > 0:
                    total_notional += notional

            for symbol, info in position_payload.items():
                exposure_pct: Optional[float] = None
                notional = info.get("notional") or 0.0
                if total_notional > 0:
                    exposure_pct = round((notional / total_notional) * 100.0, 2)

                _record(
                    symbol,
                    source="holding",
                    position_qty=info.get("qty"),
                    position_notional=notional,
                    position_avg_cost=info.get("avg_cost"),
                    exposure_pct=exposure_pct,
                )

        for watch_idx, entry in enumerate(watchlist):
            symbol_value = entry.get("symbol")
            if not isinstance(symbol_value, str):
                continue
            symbol = symbol_value.strip().upper()
            if not symbol:
                continue

            if symbol not in watchlist_symbols:
                watchlist_symbols.append(symbol)

            probability_value = self._normalise_probability_input(entry.get("probability"))
            probability_pct = (
                round(probability_value * 100.0, 2) if probability_value is not None else None
            )
            ev_value = _coerce_metric(entry.get("ev_bps"))
            score_value = _coerce_metric(entry.get("score"))
            edge_score = _coerce_metric(entry.get("edge_score"))
            gap_pct = _coerce_metric(entry.get("probability_gap_pct"))
            trend_hint = str(entry.get("trend_hint") or entry.get("trend") or "").lower()

            _record(
                symbol,
                source="watchlist",
                watchlist_rank=watch_idx,
                watchlist_order=watch_idx + 1,
                trend=trend_hint or None,
                score=score_value,
                probability=probability_value,
                probability_pct=probability_pct,
                ev_bps=ev_value,
                note=entry.get("note"),
                probability_ready=(
                    bool(entry.get("probability_ready"))
                    if entry.get("probability_ready") is not None
                    else None
                ),
                ev_ready=(
                    bool(entry.get("ev_ready")) if entry.get("ev_ready") is not None else None
                ),
                edge_score=edge_score,
                probability_gap_pct=gap_pct,
            )

            if symbol not in actionable and bool(entry.get("actionable")):
                actionable.append(symbol)
                _record(symbol, source="actionable")
                continue

            probability_ready = bool(entry.get("probability_ready"))
            ev_ready = bool(entry.get("ev_ready"))

            if (
                symbol not in actionable
                and symbol not in ready
                and trend_hint in {"buy", "sell"}
                and (probability_ready or ev_ready)
            ):
                ready.append(symbol)
                _record(symbol, source="ready")

        positions_only = [
            sym for sym in position_symbols if sym not in watchlist_symbols
        ]
        for symbol in positions_only:
            _record(symbol, source="positions_only")

        performance_metrics: Dict[str, Dict[str, object]] = {}
        try:
            paired_trades = pair_trades(settings=settings)
        except Exception as exc:  # pragma: no cover - defensive logging
            log(
                "guardian.symbol_plan.trade_pairs_error",
                error=str(exc),
            )
            paired_trades = []

        if paired_trades:
            buckets: Dict[str, Dict[str, object]] = {}
            for trade in paired_trades:
                symbol_value = trade.get("symbol")
                if not isinstance(symbol_value, str):
                    continue
                symbol = symbol_value.strip().upper()
                if not symbol:
                    continue

                bucket = buckets.setdefault(
                    symbol,
                    {
                        "bps": [],
                        "hold": [],
                        "wins": 0,
                        "count": 0,
                    },
                )

                bps_value = self._coerce_float(trade.get("bps_realized"))
                if bps_value is not None:
                    bucket["bps"].append(bps_value)
                    bucket["count"] = int(bucket.get("count", 0)) + 1
                    if bps_value > 0:
                        bucket["wins"] = int(bucket.get("wins", 0)) + 1

                hold_value = self._coerce_float(trade.get("hold_sec"))
                if hold_value is not None:
                    bucket.setdefault("hold", []).append(hold_value)

            for symbol, bucket in buckets.items():
                trade_count = int(bucket.get("count", 0))
                if trade_count <= 0:
                    continue

                bps_values = [float(val) for val in bucket.get("bps", []) if isinstance(val, (int, float))]
                if not bps_values:
                    continue

                realised_avg = round(sum(bps_values) / len(bps_values), 3)
                wins = int(bucket.get("wins", 0))
                win_rate_pct: Optional[float]
                if trade_count > 0:
                    win_rate_pct = round((wins / trade_count) * 100.0, 2)
                else:
                    win_rate_pct = None

                hold_samples = [
                    float(val)
                    for val in bucket.get("hold", [])
                    if isinstance(val, (int, float))
                ]
                median_hold: Optional[float]
                if hold_samples:
                    try:
                        median_hold = float(median(hold_samples))
                    except StatisticsError:
                        median_hold = None
                else:
                    median_hold = None

                metrics_payload: Dict[str, object] = {
                    "realized_bps_avg": realised_avg,
                    "win_rate_pct": win_rate_pct,
                    "median_hold_sec": round(median_hold, 2)
                    if isinstance(median_hold, float)
                    else None,
                    "trade_sample_count": trade_count,
                }
                performance_metrics[symbol] = metrics_payload

        for symbol, metrics_payload in performance_metrics.items():
            if symbol in symbol_details:
                _record(symbol, **metrics_payload)

        dynamic_sequence: List[str] = []
        for bucket in (position_symbols, actionable, ready, watchlist_symbols):
            for symbol in bucket:
                if symbol not in dynamic_sequence:
                    dynamic_sequence.append(symbol)

        limit_hint = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        if len(position_symbols) > limit_hint:
            limit_hint = len(position_symbols)

        base_order = {symbol: idx for idx, symbol in enumerate(dynamic_sequence)}
        priority_adjustments: Dict[str, float] = {}
        for symbol in dynamic_sequence:
            info = symbol_details.get(symbol, {})
            sample_raw = info.get("trade_sample_count")
            sample_size = int(sample_raw) if isinstance(sample_raw, (int, float)) else 0
            if sample_size < 3:
                continue

            realised = self._coerce_float(info.get("realized_bps_avg"))
            win_rate_pct = self._coerce_float(info.get("win_rate_pct"))
            win_rate = None
            if win_rate_pct is not None:
                win_rate = win_rate_pct / 100.0

            adjustment = 0.0
            if realised is not None:
                if realised >= 10.0:
                    adjustment -= 2.0
                elif realised >= 3.0:
                    adjustment -= 1.0
                elif realised <= -10.0:
                    adjustment += 2.0
                elif realised <= -3.0:
                    adjustment += 1.0

            if win_rate is not None:
                if win_rate >= 0.65:
                    adjustment -= 1.0
                elif win_rate >= 0.55:
                    adjustment -= 0.5
                elif win_rate <= 0.35:
                    adjustment += 1.0
                elif win_rate <= 0.45:
                    adjustment += 0.5

            if abs(adjustment) < 1e-6:
                continue

            priority_adjustments[symbol] = adjustment

        if priority_adjustments:
            dynamic_sequence = sorted(
                dynamic_sequence,
                key=lambda sym: (
                    base_order[sym] + priority_adjustments.get(sym, 0.0),
                    base_order[sym],
                    sym,
                ),
            )

            for symbol, adjustment in priority_adjustments.items():
                if adjustment < 0:
                    bias = "positive"
                elif adjustment > 0:
                    bias = "negative"
                else:
                    bias = "neutral"
                _record(
                    symbol,
                    priority_adjustment=round(adjustment, 3),
                    performance_bias=bias,
                )

        for idx, symbol in enumerate(dynamic_sequence):
            _record(symbol, priority_rank=idx, priority_order=idx + 1)

        details: Dict[str, Dict[str, object]] = {}
        for symbol, payload in symbol_details.items():
            sources = tuple(payload.get("sources") or [])
            payload["sources"] = sources
            details[symbol] = payload

        def _best_of(
            symbols: Sequence[str],
            metric: str,
        ) -> Optional[Dict[str, object]]:
            best_symbol: Optional[str] = None
            best_value: Optional[float] = None
            for symbol in symbols:
                info = details.get(symbol) or {}
                raw_value = info.get(metric)
                if isinstance(raw_value, bool):
                    continue
                value = self._coerce_float(raw_value)
                if value is None:
                    continue
                if best_value is None or value > best_value:
                    best_value = value
                    best_symbol = symbol

            if best_symbol is None or best_value is None:
                return None

            return {"symbol": best_symbol, metric: best_value}

        stats = {
            "position_count": len(position_symbols),
            "actionable_count": len(actionable),
            "ready_count": len(ready),
            "watchlist_count": len(watchlist_symbols),
            "positions_only_count": len(positions_only),
            "dynamic_count": len(dynamic_sequence),
            "combined_count": len(dynamic_sequence),
        }

        open_slots = limit_hint - len(position_symbols)
        stats["limit"] = limit_hint
        stats["open_slots"] = open_slots if open_slots > 0 else 0

        actionable_summary = {
            "count": len(actionable),
            "top_probability": _best_of(actionable, "probability_pct"),
            "top_ev": _best_of(actionable, "ev_bps"),
            "top_edge_score": _best_of(actionable, "edge_score"),
            "top_realized_bps": _best_of(actionable, "realized_bps_avg"),
            "top_win_rate": _best_of(actionable, "win_rate_pct"),
        }

        ready_summary = {
            "count": len(ready),
            "top_probability": _best_of(ready, "probability_pct"),
            "top_ev": _best_of(ready, "ev_bps"),
            "top_edge_score": _best_of(ready, "edge_score"),
            "top_realized_bps": _best_of(ready, "realized_bps_avg"),
            "top_win_rate": _best_of(ready, "win_rate_pct"),
        }

        total_notional = float(total_notional)
        sorted_positions = sorted(
            (
                (
                    symbol,
                    float(position_payload.get(symbol, {}).get("notional") or 0.0),
                )
                for symbol in position_symbols
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        position_summary: Dict[str, object] = {
            "total_notional": round(total_notional, 4),
            "positions": tuple(
                {
                    "symbol": symbol,
                    "notional": round(notional, 4),
                    "exposure_pct": details.get(symbol, {}).get("exposure_pct"),
                }
                for symbol, notional in sorted_positions
            ),
        }

        if sorted_positions:
            largest_symbol, largest_notional = sorted_positions[0]
            position_summary["largest"] = {
                "symbol": largest_symbol,
                "notional": round(largest_notional, 4),
                "exposure_pct": details.get(largest_symbol, {}).get("exposure_pct"),
            }

        source_breakdown: Dict[str, int] = {}
        for source_key in (
            "holding",
            "actionable",
            "ready",
            "watchlist",
            "positions_only",
        ):
            count = sum(1 for info in details.values() if info.get(source_key))
            if count:
                source_breakdown[source_key] = count

        multi_source = sum(
            1 for info in details.values() if len(info.get("sources", ())) > 1
        )
        if multi_source:
            source_breakdown["multi_source"] = multi_source

        capacity_summary: Dict[str, object] = {
            "positions": len(position_symbols),
            "actionable": len(actionable),
            "ready": len(ready),
            "open_slots": stats.get("open_slots", 0),
        }

        effective_limit = limit_hint if limit_hint > 0 else None
        capacity_summary["limit"] = effective_limit

        if effective_limit is not None:
            remaining_capacity = max(effective_limit - len(position_symbols), 0)
            utilisation_pct: Optional[float]
            if effective_limit > 0:
                utilisation_pct = round(
                    (len(position_symbols) / effective_limit) * 100.0, 2
                )
            else:
                utilisation_pct = None
            capacity_summary["remaining_capacity"] = remaining_capacity
            capacity_summary["utilisation_pct"] = utilisation_pct
        else:
            capacity_summary["remaining_capacity"] = None
            capacity_summary["utilisation_pct"] = None

        backlog = 0
        open_slots_value = stats.get("open_slots", 0)
        if effective_limit is not None:
            backlog = max(len(actionable) - open_slots_value, 0)
        capacity_summary["backlog"] = backlog
        capacity_summary["can_take_all_actionable"] = backlog == 0

        if len(actionable):
            if open_slots_value > 0:
                slot_pressure = round(len(actionable) / open_slots_value, 2)
            else:
                slot_pressure = float(len(actionable))
            capacity_summary["slot_pressure"] = slot_pressure
        else:
            capacity_summary["slot_pressure"] = 0.0

        capacity_summary["needs_attention"] = backlog > 0 or (
            effective_limit is not None and len(position_symbols) >= effective_limit
        )

        exposures: List[float] = []
        if total_notional > 0:
            for symbol in position_symbols:
                notional_value = float(
                    position_payload.get(symbol, {}).get("notional") or 0.0
                )
                if notional_value <= 0:
                    continue
                exposures.append(notional_value / total_notional)

        if exposures:
            hhi = sum(fraction * fraction for fraction in exposures)
            max_fraction = max(exposures)
            diversification: Dict[str, object] = {
                "hhi": round(hhi, 6),
                "largest_share_pct": round(max_fraction * 100.0, 2),
                "diversification_score": round((1.0 - hhi) * 100.0, 2),
            }
            if hhi > 0:
                diversification["effective_positions"] = round(1.0 / hhi, 2)

            if diversification["largest_share_pct"] >= 60.0:
                concentration_level = "high"
            elif diversification["largest_share_pct"] >= 40.0:
                concentration_level = "medium"
            else:
                concentration_level = "low"
            diversification["concentration_level"] = concentration_level

            position_summary["diversification"] = diversification

        def _skip_reason(info: Mapping[str, object]) -> Optional[str]:
            if not isinstance(info, Mapping):
                return None
            if info.get("actionable"):
                return None
            reasons: list[str] = []
            if info.get("holding") and not info.get("actionable"):
                reasons.append("Позиция уже открыта")
            probability_ready = info.get("probability_ready")
            if probability_ready is False:
                reasons.append("Нет подтверждения вероятности")
            ev_ready = info.get("ev_ready")
            if ev_ready is False:
                reasons.append("EV ниже порога")
            trend_hint = str(info.get("trend") or "").strip().lower()
            if not trend_hint or trend_hint == "wait":
                reasons.append("Нет направления")
            note_value = info.get("note")
            if isinstance(note_value, str) and note_value.strip():
                reasons.append(note_value.strip())
            if info.get("watchlist") and not reasons:
                reasons.append("В списке наблюдения")
            if not reasons:
                return None
            unique_reasons: list[str] = []
            for item in reasons:
                if item not in unique_reasons:
                    unique_reasons.append(item)
            return "; ".join(unique_reasons)

        priority_table: List[Dict[str, object]] = []
        for idx, symbol in enumerate(dynamic_sequence):
            info = details.get(symbol, {})
            sources = tuple(info.get("sources") or ())
            reasons: List[str] = []
            if info.get("holding"):
                reasons.append("holding")
            if info.get("actionable"):
                reasons.append("actionable")
            if info.get("ready") and "actionable" not in reasons:
                reasons.append("ready")
            if info.get("watchlist") and "actionable" not in reasons:
                reasons.append("watchlist")

            priority_table.append(
                {
                    "symbol": symbol,
                    "priority": idx + 1,
                    "sources": sources,
                    "source_count": len(sources),
                    "reasons": tuple(reasons),
                    "holding": bool(info.get("holding")),
                    "actionable": bool(info.get("actionable")),
                    "ready": bool(info.get("ready")),
                    "watchlist": bool(info.get("watchlist")),
                    "watchlist_order": info.get("watchlist_order"),
                    "watchlist_rank": info.get("watchlist_rank"),
                    "probability_pct": info.get("probability_pct"),
                    "probability": info.get("probability"),
                    "ev_bps": info.get("ev_bps"),
                    "edge_score": info.get("edge_score"),
                    "probability_gap_pct": info.get("probability_gap_pct"),
                    "score": info.get("score"),
                    "note": info.get("note"),
                    "trend": info.get("trend"),
                    "position_qty": info.get("position_qty"),
                    "position_notional": info.get("position_notional"),
                    "position_avg_cost": info.get("position_avg_cost"),
                    "exposure_pct": info.get("exposure_pct"),
                    "realized_bps_avg": info.get("realized_bps_avg"),
                    "win_rate_pct": info.get("win_rate_pct"),
                    "median_hold_sec": info.get("median_hold_sec"),
                    "trade_sample_count": info.get("trade_sample_count"),
                    "priority_adjustment": info.get("priority_adjustment"),
                    "performance_bias": info.get("performance_bias"),
                    "skip_reason": _skip_reason(info),
                }
            )

        plan = {
            "actionable": tuple(actionable),
            "ready": tuple(sym for sym in ready if sym not in actionable),
            "watchlist": tuple(watchlist_symbols),
            "positions": tuple(position_symbols),
            "positions_only": tuple(positions_only),
            "dynamic": tuple(dynamic_sequence),
            "combined": tuple(dynamic_sequence),
            "details": details,
            "limit": limit_hint,
            "stats": stats,
            "actionable_summary": actionable_summary,
            "ready_summary": ready_summary,
            "position_summary": position_summary,
            "capacity_summary": capacity_summary,
            "source_breakdown": source_breakdown,
            "priority_table": tuple(priority_table),
        }

        combined_pool = self._compose_symbol_pool(plan, only_actionable=False)
        plan["combined"] = tuple(combined_pool)
        stats["combined_count"] = len(combined_pool)
        actionable_pool = self._compose_symbol_pool(plan, only_actionable=True)
        plan["actionable_combined"] = tuple(actionable_pool)
        stats["actionable_combined_count"] = len(actionable_pool)

        if signature is not None:
            self._plan_cache_signature = signature
            self._plan_cache = copy.deepcopy(plan)
            self._plan_cache_ledger_signature = ledger_signature
        else:
            self._plan_cache_signature = None
            self._plan_cache = None
            self._plan_cache_ledger_signature = None
        return plan

    def _build_watchlist(
        self, status: Dict[str, object], settings: Optional[Settings] = None
    ) -> List[Dict[str, object]]:
        candidates = (
            status.get("watchlist")
            or status.get("heatmap")
            or status.get("opportunities")
            or status.get("signals")
        )
        entries: List[Dict[str, object]] = []

        if isinstance(candidates, dict):
            for symbol, payload in candidates.items():
                entries.append(self._normalise_watchlist_entry(str(symbol), payload))
        elif isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    symbol = item.get("symbol") or item.get("ticker") or "?"
                    entries.append(self._normalise_watchlist_entry(str(symbol), item))
                elif isinstance(item, (list, tuple)) and item:
                    symbol = str(item[0])
                    payload = item[1] if len(item) > 1 else None
                    entries.append(self._normalise_watchlist_entry(symbol, payload))
                elif isinstance(item, str):
                    entries.append(self._normalise_watchlist_entry(item, None))

        filtered = [entry for entry in entries if any(entry.values())]

        settings = settings or self.settings

        dynamic_entries = self._market_scanner_watchlist(settings)
        if dynamic_entries:
            seen_symbols = {
                str(entry.get("symbol")).upper()
                for entry in filtered
                if isinstance(entry.get("symbol"), str)
            }
            for dynamic_entry in dynamic_entries:
                symbol = dynamic_entry.get("symbol")
                if not isinstance(symbol, str):
                    continue
                upper_symbol = symbol.upper()
                if upper_symbol in seen_symbols:
                    continue
                entry_copy = dict(dynamic_entry)
                entry_copy.setdefault("source", "market_scanner")
                filtered.append(entry_copy)
                seen_symbols.add(upper_symbol)

        listed_symbols = set()
        testnet_mode = settings.testnet
        if filtered:
            listed_symbols = self._fetch_listed_spot_symbols()
            if listed_symbols:
                before = len(filtered)
                cleaned: List[Dict[str, object]] = []
                removed_symbols: List[str] = []
                for entry in filtered:
                    symbol_value = entry.get("symbol")
                    symbol_cleaned = (
                        symbol_value.strip().upper()
                        if isinstance(symbol_value, str)
                        else ""
                    )
                    if symbol_cleaned and symbol_cleaned in listed_symbols:
                        cleaned.append(entry)
                    else:
                        if symbol_cleaned:
                            removed_symbols.append(symbol_cleaned)
                filtered_candidates = cleaned
                removed = before - len(filtered_candidates)
                if removed > 0 and testnet_mode and len(filtered_candidates) <= max(5, before // 2):
                    log(
                        "guardian.watchlist.listing_filter.skipped",
                        removed=removed,
                        remaining=len(filtered_candidates),
                        reason="testnet",
                    )
                else:
                    filtered = filtered_candidates
                    if removed > 0:
                        log(
                            "guardian.watchlist.filtered_unlisted",
                            removed=removed,
                            remaining=len(filtered),
                            symbols=sorted(removed_symbols),
                        )
            elif testnet_mode:
                log(
                    "guardian.watchlist.listing_filter.skipped",
                    removed=0,
                    remaining=len(filtered),
                    reason="empty_catalog",
                )

        if not filtered:
            status["watchlist"] = []
            return filtered
        (
            _,
            _,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            _,
            _,
        ) = self._resolve_thresholds(settings)

        def annotate(entry: Dict[str, object]) -> None:
            trend_hint = self._normalise_mode_hint(entry.get("trend"))
            probability = self._normalise_probability_input(entry.get("probability"))
            ev_value = self._coerce_float(entry.get("ev_bps"))
            score_value = entry.get("score")
            numeric_score = float(score_value) if isinstance(score_value, (int, float)) else None

            entry["trend_hint"] = trend_hint
            entry["probability"] = probability
            entry["ev_bps"] = ev_value
            if probability is not None:
                entry["probability_pct"] = round(probability * 100.0, 2)
            else:
                entry["probability_pct"] = None

            if numeric_score is not None:
                entry["score"] = round(numeric_score, 2)

            if ev_value is not None:
                entry["ev_bps"] = round(ev_value, 3)

            if trend_hint in {"buy", "sell"}:
                if probability is None:
                    probability_ready = True
                    probability_gap = None
                elif trend_hint == "sell":
                    probability_ready = probability <= effective_sell_threshold
                    probability_gap = effective_sell_threshold - probability
                else:
                    probability_ready = probability >= effective_buy_threshold
                    probability_gap = probability - effective_buy_threshold
            else:
                if probability is None:
                    probability_ready = False
                    probability_gap = None
                else:
                    probability_ready = probability >= 0.5
                    probability_gap = probability - 0.5

            ev_ready = ev_value is None or ev_value >= min_ev
            actionable = trend_hint in {"buy", "sell"} and probability_ready and ev_ready

            entry["ev_ready"] = bool(ev_ready)
            entry["probability_ready"] = probability_ready
            entry["probability_gap_pct"] = (
                round(probability_gap * 100.0, 2) if probability_gap is not None else None
            )
            entry["actionable"] = actionable
            entry["trend_category"] = trend_hint if trend_hint in {"buy", "sell"} else "wait"

            edge_components: List[float] = []
            if actionable:
                edge_components.append(5.0)
            elif trend_hint in {"buy", "sell"} and probability_ready:
                edge_components.append(2.5)

            if probability_gap is not None:
                edge_components.append(probability_gap * 100.0)

            if ev_value is not None and min_ev > 0:
                edge_components.append(ev_value / min_ev)
            elif ev_value is not None:
                edge_components.append(ev_value)

            if numeric_score is not None:
                edge_components.append(numeric_score)

            entry["edge_score"] = (
                round(sum(edge_components), 4) if edge_components else None
            )

        for item in filtered:
            annotate(item)

        def best_metric(item: Dict[str, object]) -> Optional[float]:
            for key in ("edge_score", "score", "probability", "ev_bps"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            return None

        def sort_key(item: Dict[str, object]) -> tuple:
            trend_hint = item.get("trend_hint")
            actionable = bool(item.get("actionable"))
            probability_ready = bool(item.get("probability_ready"))
            ev_ready = bool(item.get("ev_ready"))
            metric = best_metric(item)
            if isinstance(item.get("edge_score"), (int, float)):
                metric_value = -float(item["edge_score"])
            elif metric is not None:
                metric_value = -metric
            else:
                metric_value = float("inf")

            return (
                0 if actionable else 1,
                0 if trend_hint in {"buy", "sell"} and probability_ready else 1,
                0 if ev_ready else 1,
                metric_value,
                item["symbol"],
            )

        filtered.sort(key=sort_key)
        status["watchlist"] = [dict(item) for item in filtered]
        return filtered

    def _summarise_trade_candidates(
        self,
        plan: Mapping[str, object],
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Build a compact, prioritised view of trade candidates."""

        priority_table = plan.get("priority_table") if isinstance(plan, Mapping) else None
        if not isinstance(priority_table, Sequence):
            return []

        position_symbols = {
            str(symbol).strip().upper()
            for symbol in (plan.get("positions") or [])
            if str(symbol).strip()
        }

        summary: List[Dict[str, object]] = []

        for idx, raw_entry in enumerate(priority_table):
            if not isinstance(raw_entry, Mapping):
                continue

            symbol_value = raw_entry.get("symbol")
            if not isinstance(symbol_value, str) or not symbol_value.strip():
                continue

            symbol = symbol_value.strip().upper()
            probability_pct = self._coerce_float(raw_entry.get("probability_pct"))
            probability = self._coerce_float(raw_entry.get("probability"))
            ev_bps = self._coerce_float(raw_entry.get("ev_bps"))
            edge_score = self._coerce_float(raw_entry.get("edge_score"))
            score = self._coerce_float(raw_entry.get("score"))
            exposure_pct = self._coerce_float(raw_entry.get("exposure_pct"))
            position_qty = self._coerce_float(raw_entry.get("position_qty"))
            position_notional = self._coerce_float(raw_entry.get("position_notional"))
            realized_bps_avg = self._coerce_float(raw_entry.get("realized_bps_avg"))
            win_rate_pct = self._coerce_float(raw_entry.get("win_rate_pct"))
            median_hold_sec = self._coerce_float(raw_entry.get("median_hold_sec"))
            priority_adjustment = self._coerce_float(raw_entry.get("priority_adjustment"))

            entry = {
                "symbol": symbol,
                "priority": int(raw_entry.get("priority") or idx + 1),
                "sources": tuple(raw_entry.get("sources") or ()),
                "reasons": tuple(raw_entry.get("reasons") or ()),
                "actionable": bool(raw_entry.get("actionable")),
                "ready": bool(raw_entry.get("ready")),
                "holding": bool(raw_entry.get("holding")),
                "trend": raw_entry.get("trend"),
                "note": raw_entry.get("note"),
                "watchlist_rank": raw_entry.get("watchlist_rank"),
                "probability_pct": round(probability_pct, 2) if probability_pct is not None else None,
                "probability": round(probability, 4) if probability is not None else None,
                "ev_bps": round(ev_bps, 3) if ev_bps is not None else None,
                "edge_score": round(edge_score, 3) if edge_score is not None else None,
                "score": round(score, 3) if score is not None else None,
                "exposure_pct": round(exposure_pct, 3) if exposure_pct is not None else None,
                "position_qty": position_qty,
                "position_notional": position_notional,
                "realized_bps_avg": round(realized_bps_avg, 3)
                if realized_bps_avg is not None
                else None,
                "win_rate_pct": round(win_rate_pct, 2) if win_rate_pct is not None else None,
                "median_hold_sec": round(median_hold_sec, 2)
                if median_hold_sec is not None
                else None,
                "trade_sample_count": raw_entry.get("trade_sample_count"),
                "priority_adjustment": round(priority_adjustment, 3)
                if priority_adjustment is not None
                else None,
                "performance_bias": raw_entry.get("performance_bias"),
            }

            summary.append(entry)

            if limit is not None and limit > 0 and len(summary) >= limit:
                break

        return summary

    @staticmethod
    def _compose_performance_overview(
        plan: Mapping[str, object]
    ) -> Optional[Dict[str, object]]:
        priority_table = plan.get("priority_table") if isinstance(plan, Mapping) else None
        if not isinstance(priority_table, Sequence):
            return None

        buckets: Dict[str, List[Dict[str, object]]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
        }

        for raw_entry in priority_table:
            if not isinstance(raw_entry, Mapping):
                continue

            symbol_value = raw_entry.get("symbol")
            if not isinstance(symbol_value, str):
                continue
            symbol = symbol_value.strip().upper()
            if not symbol:
                continue

            bias_value = str(raw_entry.get("performance_bias") or "neutral").lower()
            bucket_key = bias_value if bias_value in buckets else "neutral"

            payload = {
                "symbol": symbol,
                "realized_bps_avg": raw_entry.get("realized_bps_avg"),
                "win_rate_pct": raw_entry.get("win_rate_pct"),
                "median_hold_sec": raw_entry.get("median_hold_sec"),
                "trade_sample_count": raw_entry.get("trade_sample_count"),
                "priority_adjustment": raw_entry.get("priority_adjustment"),
            }
            buckets[bucket_key].append(payload)

        if not any(buckets.values()):
            return None

        overview: Dict[str, object] = {}
        for key, items in buckets.items():
            overview[f"{key}_count"] = len(items)
            overview[key] = tuple(items)
            if items:
                overview[f"top_{key}"] = tuple(items[:3])

        return overview

    @staticmethod
    def _condense_watchlist_entry(entry: Dict[str, object]) -> Dict[str, object]:
        """Reduce a raw watchlist entry to UI-friendly fields."""

        trend_hint = entry.get("trend_hint") or entry.get("trend")
        compact = {
            "symbol": entry.get("symbol"),
            "trend": trend_hint,
            "actionable": bool(entry.get("actionable")),
            "probability_pct": entry.get("probability_pct"),
            "probability_ready": entry.get("probability_ready"),
            "probability_gap_pct": entry.get("probability_gap_pct"),
            "ev_bps": entry.get("ev_bps"),
            "ev_ready": entry.get("ev_ready"),
            "edge_score": entry.get("edge_score"),
            "score": entry.get("score"),
            "note": entry.get("note"),
            "realized_bps_avg": entry.get("realized_bps_avg"),
            "win_rate_pct": entry.get("win_rate_pct"),
            "median_hold_sec": entry.get("median_hold_sec"),
            "trade_sample_count": entry.get("trade_sample_count"),
            "priority_adjustment": entry.get("priority_adjustment"),
            "performance_bias": entry.get("performance_bias"),
        }

        return compact

    def _watchlist_breakdown(
        self, entries: Sequence[Dict[str, object]]
    ) -> Dict[str, object]:
        """Provide aggregated perspective for dashboards and chat replies."""

        condensed_entries: List[Dict[str, object]] = []
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                raw_entry = {"symbol": raw_entry}
            condensed_entries.append(self._condense_watchlist_entry(raw_entry))

        signature = self._stable_signature(condensed_entries)
        if (
            signature is not None
            and signature == self._watchlist_breakdown_cache_signature
            and self._watchlist_breakdown_cache is not None
        ):
            return copy.deepcopy(self._watchlist_breakdown_cache)

        total = len(condensed_entries)
        buys: List[Dict[str, object]] = []
        sells: List[Dict[str, object]] = []
        neutral: List[Dict[str, object]] = []
        actionable: List[Dict[str, object]] = []
        actionable_buys = 0
        actionable_sells = 0

        overall_probabilities: List[float] = []
        overall_expectancies: List[float] = []
        actionable_probabilities: List[float] = []
        actionable_expectancies: List[float] = []
        overall_realised: List[float] = []
        actionable_realised: List[float] = []
        overall_win_rates: List[float] = []
        actionable_win_rates: List[float] = []
        overall_holds: List[float] = []
        actionable_holds: List[float] = []
        positive_bias = 0
        negative_bias = 0

        for compact in condensed_entries:
            trend = str(compact.get("trend") or "wait").lower()

            if trend == "buy":
                buys.append(copy.deepcopy(compact))
                if compact["actionable"]:
                    actionable_buys += 1
            elif trend == "sell":
                sells.append(copy.deepcopy(compact))
                if compact["actionable"]:
                    actionable_sells += 1
            else:
                neutral.append(copy.deepcopy(compact))

            if compact["actionable"]:
                actionable.append(copy.deepcopy(compact))

            probability_value = compact.get("probability_pct")
            if isinstance(probability_value, (int, float)):
                probability_number = float(probability_value)
                overall_probabilities.append(probability_number)
                if compact["actionable"]:
                    actionable_probabilities.append(probability_number)

            expectancy_value = compact.get("ev_bps")
            if isinstance(expectancy_value, (int, float)):
                expectancy_number = float(expectancy_value)
                overall_expectancies.append(expectancy_number)
                if compact["actionable"]:
                    actionable_expectancies.append(expectancy_number)

            realised_value = compact.get("realized_bps_avg")
            if isinstance(realised_value, (int, float)):
                realised_number = float(realised_value)
                overall_realised.append(realised_number)
                if compact["actionable"]:
                    actionable_realised.append(realised_number)

            win_rate_value = compact.get("win_rate_pct")
            if isinstance(win_rate_value, (int, float)):
                win_rate_number = float(win_rate_value)
                overall_win_rates.append(win_rate_number)
                if compact["actionable"]:
                    actionable_win_rates.append(win_rate_number)

            hold_value = compact.get("median_hold_sec")
            if isinstance(hold_value, (int, float)):
                hold_number = float(hold_value)
                overall_holds.append(hold_number)
                if compact["actionable"]:
                    actionable_holds.append(hold_number)

            bias = compact.get("performance_bias")
            if bias == "positive":
                positive_bias += 1
            elif bias == "negative":
                negative_bias += 1

        counts = {
            "total": total,
            "actionable": len(actionable),
            "actionable_buys": actionable_buys,
            "actionable_sells": actionable_sells,
            "buys": len(buys),
            "sells": len(sells),
            "neutral": len(neutral),
        }
        if positive_bias:
            counts["positive_bias"] = positive_bias
        if negative_bias:
            counts["negative_bias"] = negative_bias

        dominant_trend: Optional[str]
        if actionable_buys or actionable_sells:
            dominant_trend = "buy" if actionable_buys >= actionable_sells else "sell"
        elif len(buys) > len(sells):
            dominant_trend = "buy"
        elif len(sells) > len(buys):
            dominant_trend = "sell"
        elif total:
            dominant_trend = "wait"
        else:
            dominant_trend = None

        def _take(items: Sequence[Dict[str, object]], limit: int = 3) -> List[Dict[str, object]]:
            return [copy.deepcopy(item) for item in list(items)[:limit]]

        def _average(values: Sequence[float]) -> Optional[float]:
            if not values:
                return None
            return round(sum(values) / len(values), 2)

        def _median(values: Sequence[float]) -> Optional[float]:
            if not values:
                return None
            try:
                return float(median(values))
            except StatisticsError:
                return None

        overall_metrics: Dict[str, float] = {}
        overall_probability_avg = _average(overall_probabilities)
        if overall_probability_avg is not None:
            overall_metrics["probability_avg_pct"] = overall_probability_avg
        overall_expectancy_avg = _average(overall_expectancies)
        if overall_expectancy_avg is not None:
            overall_metrics["ev_avg_bps"] = overall_expectancy_avg
        overall_realised_avg = _average(overall_realised)
        if overall_realised_avg is not None:
            overall_metrics["realized_bps_avg"] = overall_realised_avg
        overall_win_rate_avg = _average(overall_win_rates)
        if overall_win_rate_avg is not None:
            overall_metrics["win_rate_avg_pct"] = overall_win_rate_avg
        overall_hold_median = _median(overall_holds)
        if overall_hold_median is not None:
            overall_metrics["median_hold_sec"] = round(overall_hold_median, 2)

        actionable_metrics: Dict[str, float] = {}
        actionable_probability_avg = _average(actionable_probabilities)
        if actionable_probability_avg is not None:
            actionable_metrics["probability_avg_pct"] = actionable_probability_avg
        actionable_expectancy_avg = _average(actionable_expectancies)
        if actionable_expectancy_avg is not None:
            actionable_metrics["ev_avg_bps"] = actionable_expectancy_avg
        actionable_realised_avg = _average(actionable_realised)
        if actionable_realised_avg is not None:
            actionable_metrics["realized_bps_avg"] = actionable_realised_avg
        actionable_win_rate_avg = _average(actionable_win_rates)
        if actionable_win_rate_avg is not None:
            actionable_metrics["win_rate_avg_pct"] = actionable_win_rate_avg
        actionable_hold_median = _median(actionable_holds)
        if actionable_hold_median is not None:
            actionable_metrics["median_hold_sec"] = round(actionable_hold_median, 2)

        metrics: Dict[str, Dict[str, float]] = {}
        if overall_metrics:
            metrics["overall"] = overall_metrics
        if actionable_metrics:
            metrics["actionable"] = actionable_metrics

        breakdown = {
            "counts": counts,
            "dominant_trend": dominant_trend,
            "actionable": _take(actionable, limit=5),
            "top_buys": _take(buys),
            "top_sells": _take(sells),
            "top_neutral": _take(neutral),
            "metrics": metrics,
        }

        if signature is not None:
            self._watchlist_breakdown_cache_signature = signature
            self._watchlist_breakdown_cache = copy.deepcopy(breakdown)
        else:
            self._watchlist_breakdown_cache_signature = None
            self._watchlist_breakdown_cache = None

        return breakdown

    @staticmethod
    def _format_watchlist_detail(entry: Dict[str, object]) -> str:
        symbol = str(entry.get("symbol") or "").upper() or "—"
        trend = str(entry.get("trend") or "wait").lower()
        trend_text = {
            "buy": "покупка",
            "sell": "продажа",
        }.get(trend, "наблюдение")

        parts = [symbol, trend_text]

        probability = entry.get("probability_pct")
        if isinstance(probability, (int, float)):
            parts.append(f"{float(probability):.1f}%")

        ev_bps = entry.get("ev_bps")
        if isinstance(ev_bps, (int, float)):
            parts.append(f"EV {float(ev_bps):.1f} б.п.")

        realized = entry.get("realized_bps_avg")
        if isinstance(realized, (int, float)):
            parts.append(f"реал {float(realized):.1f} б.п.")

        win_rate = entry.get("win_rate_pct")
        if isinstance(win_rate, (int, float)):
            parts.append(f"win {float(win_rate):.1f}%")

        note = entry.get("note")
        if isinstance(note, str):
            note_text = note.strip()
            if note_text:
                parts.append(note_text)

        return ", ".join(parts)

    def _watchlist_digest(self, breakdown: Dict[str, object]) -> Dict[str, object]:
        signature = self._stable_signature(breakdown)
        if (
            signature is not None
            and signature == self._digest_cache_signature
            and self._digest_cache is not None
        ):
            return copy.deepcopy(self._digest_cache)

        counts = copy.deepcopy(breakdown.get("counts") or {})
        total = int(counts.get("total") or 0)
        actionable = int(counts.get("actionable") or 0)
        actionable_buys = int(counts.get("actionable_buys") or 0)
        actionable_sells = int(counts.get("actionable_sells") or 0)
        buys = int(counts.get("buys") or 0)
        sells = int(counts.get("sells") or 0)
        neutral = int(counts.get("neutral") or 0)

        if total == 0:
            headline = "Список наблюдения пуст — ждём новые сигналы."
        elif actionable:
            headline = (
                f"{total} инструментов в наблюдении, {actionable} готовы к сделке."
            )
        else:
            headline = f"{total} инструментов в наблюдении, пока без готовых сделок."

        dominant = breakdown.get("dominant_trend")
        dominant_note = None
        if dominant == "buy":
            dominant_note = "Преобладает спрос — бычьи сигналы ведут список."
        elif dominant == "sell":
            dominant_note = "Преобладает давление продавцов — важен контроль риска."
        elif dominant == "wait":
            dominant_note = "Преобладают нейтральные сигналы — наблюдаем и ждём подтверждений."

        actionable_entries = breakdown.get("actionable") or []
        metrics = copy.deepcopy(breakdown.get("metrics") or {})
        actionable_metrics = metrics.get("actionable") if isinstance(metrics, dict) else {}
        overall_metrics = metrics.get("overall") if isinstance(metrics, dict) else {}

        detail_lines: List[str] = []
        if actionable_entries:
            formatted = "; ".join(
                self._format_watchlist_detail(entry)
                for entry in actionable_entries[:3]
            )
            detail_lines.append(f"Активные идеи: {formatted}.")
            metric_bits: List[str] = []
            if isinstance(actionable_metrics, dict):
                probability_avg = actionable_metrics.get("probability_avg_pct")
                if isinstance(probability_avg, (int, float)):
                    metric_bits.append(f"уверенность {float(probability_avg):.1f}%")
                ev_avg = actionable_metrics.get("ev_avg_bps")
                if isinstance(ev_avg, (int, float)):
                    metric_bits.append(f"EV {float(ev_avg):.1f} б.п.")
            if metric_bits:
                detail_lines.append(
                    "Средние показатели активных идей: " + ", ".join(metric_bits) + "."
                )
        elif total:
            leaders = breakdown.get("top_buys") or breakdown.get("top_neutral") or []
            if leaders:
                formatted = "; ".join(
                    self._format_watchlist_detail(entry) for entry in leaders[:3]
                )
                detail_lines.append(f"На радаре: {formatted}.")
            metric_bits: List[str] = []
            if isinstance(overall_metrics, dict):
                probability_avg = overall_metrics.get("probability_avg_pct")
                if isinstance(probability_avg, (int, float)):
                    metric_bits.append(f"средняя уверенность {float(probability_avg):.1f}%")
                ev_avg = overall_metrics.get("ev_avg_bps")
                if isinstance(ev_avg, (int, float)):
                    metric_bits.append(f"средний EV {float(ev_avg):.1f} б.п.")
            if metric_bits:
                detail_lines.append(
                    "Пульс наблюдения: " + ", ".join(metric_bits) + "."
                )

        if total:
            detail_lines.append(
                "Структура: "
                f"{buys} buy / {sells} sell / {neutral} нейтральных сигналов."
            )

        if dominant_note:
            detail_lines.append(dominant_note)

        if actionable and (actionable_buys or actionable_sells):
            detail_lines.append(
                f"Готовые сделки: {actionable_buys} на покупку и {actionable_sells} на продажу."
            )

        digest = {
            "headline": headline,
            "details": detail_lines,
            "dominant_trend": dominant,
            "counts": counts,
            "metrics": metrics,
        }

        if actionable_entries:
            digest["top_actionable"] = [
                copy.deepcopy(entry) for entry in actionable_entries[:3]
            ]

        if signature is not None:
            self._digest_cache_signature = signature
            self._digest_cache = copy.deepcopy(digest)
        else:
            self._digest_cache_signature = None
            self._digest_cache = None

        return digest

    def _build_snapshot(
        self,
        signature: Tuple[Tuple[int, int], int, int],
        prefetched_status: Optional[bytes] = None,
        prefetched_hash: Optional[int] = None,
    ) -> GuardianSnapshot:
        previous_snapshot = self._snapshot
        status = self._load_status(
            prefetched=prefetched_status, prefetched_hash=prefetched_hash
        )
        has_raw_status = bool(status)
        self._last_raw_status = copy.deepcopy(status) if has_raw_status else {}
        ledger_view = self._ledger_view_for_signature(signature[1])
        portfolio = ledger_view.portfolio
        settings = self.settings
        context = self._derive_signal_context(status, settings, portfolio=portfolio)
        raw_watchlist = context.get("watchlist", [])
        plan_from_context = context.get("symbol_plan") if isinstance(context, Mapping) else None
        if isinstance(plan_from_context, Mapping):
            symbol_plan = dict(plan_from_context)
        else:
            symbol_plan = self._build_symbol_plan(
                raw_watchlist,
                settings,
                portfolio=portfolio,
                ledger_signature=signature[1],
            )

        plan_copy = copy.deepcopy(symbol_plan)
        context["symbol_plan"] = plan_copy

        plan_details = plan_copy.get("details") if isinstance(plan_copy, Mapping) else None
        if isinstance(plan_details, Mapping) and isinstance(raw_watchlist, Sequence):
            enriched_watchlist: List[Dict[str, object]] = []
            for entry in raw_watchlist:
                if isinstance(entry, Mapping):
                    updated_entry = dict(entry)
                else:
                    updated_entry = {"symbol": entry}
                symbol_value = updated_entry.get("symbol")
                symbol_key = (
                    str(symbol_value).strip().upper()
                    if isinstance(symbol_value, str)
                    else None
                )
                if symbol_key and symbol_key in plan_details:
                    detail = plan_details[symbol_key]
                    if isinstance(detail, Mapping):
                        for key in (
                            "realized_bps_avg",
                            "win_rate_pct",
                            "median_hold_sec",
                            "trade_sample_count",
                            "priority_adjustment",
                            "performance_bias",
                        ):
                            value = detail.get(key)
                            if value is not None:
                                updated_entry[key] = value
                enriched_watchlist.append(updated_entry)

            context["watchlist"] = enriched_watchlist
            try:
                enriched_breakdown = self._watchlist_breakdown(enriched_watchlist)
            except Exception:
                enriched_breakdown = None
            else:
                context["watchlist_breakdown"] = copy.deepcopy(enriched_breakdown)
                context["watchlist_digest"] = self._watchlist_digest(enriched_breakdown)
            watchlist = enriched_watchlist
        else:
            watchlist = raw_watchlist

        brief = self._brief_from_status(status, settings, context)
        status_summary = self._build_status_summary(
            status,
            brief,
            settings,
            self._status_fallback_used,
            context,
            has_status=has_raw_status,
        )

        recent_trades = list(ledger_view.recent_trades)
        execution_records = ledger_view.executions
        trade_stats = ledger_view.trade_stats

        summary_plan = copy.deepcopy(plan_copy)
        status_summary["symbol_plan"] = summary_plan

        performance_overview = self._compose_performance_overview(summary_plan)
        if performance_overview:
            status_summary["performance_overview"] = performance_overview

        limit_hint = int(symbol_plan.get("limit") or 0)
        candidate_pool = list(symbol_plan.get("combined") or [])
        fallback_symbol = self._normalise_symbol_value(brief.symbol)
        if fallback_symbol and fallback_symbol not in candidate_pool:
            candidate_pool.append(fallback_symbol)
        if limit_hint > 0:
            status_summary["candidate_symbols"] = candidate_pool[:limit_hint]
        else:
            status_summary["candidate_symbols"] = candidate_pool

        candidate_limit = max(limit_hint, 5) if limit_hint > 0 else 5
        status_summary["trade_candidates"] = self._summarise_trade_candidates(
            symbol_plan,
            limit=candidate_limit,
        )

        status_signature = (
            signature[0][0],
            signature[0][1],
            int(self._status_content_hash or 0),
        )
        if self._status_fallback_used and previous_snapshot is not None:
            status_signature = previous_snapshot.status_signature

        self._last_status = copy.deepcopy(status) if status else {}

        return GuardianSnapshot(
            status=status,
            brief=brief,
            status_summary=status_summary,
            status_from_cache=self._status_fallback_used,
            portfolio=portfolio,
            watchlist=watchlist,
            symbol_plan=copy.deepcopy(plan_copy),
            recent_trades=recent_trades,
            trade_stats=trade_stats,
            executions=execution_records,
            generated_at=time.time(),
            status_signature=status_signature,
            ledger_signature=signature[1],
        )

    @staticmethod
    def _copy_dict(payload: Dict[str, object]) -> Dict[str, object]:
        return copy.deepcopy(payload)

    @staticmethod
    def _copy_list(payload: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return copy.deepcopy(payload)

    def _get_snapshot(self, force: bool = False) -> GuardianSnapshot:
        signature = self._snapshot_signature()
        snapshot = self._snapshot
        prefetched_status: Optional[bytes] = None
        prefetched_hash: Optional[int] = None

        needs_rebuild = (
            force
            or snapshot is None
            or snapshot.ledger_signature != signature[1]
        )

        if not needs_rebuild and snapshot is not None:
            if snapshot.status_signature[:2] != signature[0]:
                needs_rebuild = True
            else:
                data, data_hash, prefetch_error = self._prefetch_status()
                if data is None:
                    if prefetch_error is not None:
                        needs_rebuild = True
                else:
                    if data_hash is None:
                        needs_rebuild = True
                    elif snapshot.status_signature[2] != data_hash:
                        prefetched_status = data
                        prefetched_hash = data_hash
                        needs_rebuild = True

        if needs_rebuild:
            snapshot = self._build_snapshot(
                signature,
                prefetched_status=prefetched_status,
                prefetched_hash=prefetched_hash,
            )
            self._snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # public analytics helpers
    def _build_status_summary(
        self,
        status: Dict[str, object],
        brief: GuardianBrief,
        settings: Settings,
        fallback_used: bool,
        context: Optional[Dict[str, object]] = None,
        *,
        has_status: Optional[bool] = None,
    ) -> Dict[str, object]:
        context = context or self._derive_signal_context(status, settings)
        probability = float(context.get("probability") or 0.0)
        ev_bps = float(context.get("ev_bps") or 0.0)
        last_tick = status.get("last_tick_ts") or status.get("timestamp")

        now = time.time()
        tick_ts = self._coerce_float(last_tick)
        tick_age = None
        if tick_ts is not None and tick_ts > 0:
            tick_age = max(0.0, now - tick_ts)
        elif brief.status_age is not None:
            tick_age = max(0.0, float(brief.status_age))

        try:
            tick_guard = float(getattr(settings, "ai_tick_stale_seconds", LIVE_TICK_STALE_SECONDS))
        except (TypeError, ValueError):
            tick_guard = LIVE_TICK_STALE_SECONDS
        if not math.isfinite(tick_guard):
            tick_guard = LIVE_TICK_STALE_SECONDS
        tick_guard = max(tick_guard, 0.0)
        tick_fresh = True
        if tick_age is not None and tick_guard > 0:
            tick_fresh = tick_age <= tick_guard

        (
            buy_threshold,
            sell_threshold,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            exit_buy_threshold,
            exit_sell_threshold,
        ) = self._resolve_thresholds(settings)

        buy_decision_threshold = effective_buy_threshold
        sell_decision_threshold = effective_sell_threshold

        staleness_state, staleness_message = self._status_staleness(brief.status_age)
        actionable, reasons = self._evaluate_actionability(
            brief.mode,
            probability,
            ev_bps,
            effective_buy_threshold,
            effective_sell_threshold,
            min_ev,
            staleness_state,
            tick_age,
            tick_guard,
        )

        if not getattr(settings, "ai_enabled", False):
            reasons.append(
                "AI сигналы выключены настройками — автоматические сделки не запускаются."
            )
            actionable = False

        summary = {
            "symbol": brief.symbol,
            "mode": brief.mode,
            "headline": brief.headline,
            "probability": probability,
            "probability_pct": round(probability * 100.0, 2),
            "ev_bps": round(ev_bps, 2),
            "ev_text": brief.ev_text,
            "action_text": brief.action_text,
            "confidence_text": brief.confidence_text,
            "caution": brief.caution,
            "analysis": brief.analysis,
            "updated_text": brief.updated_text,
            "age_seconds": brief.status_age,
            "last_update": self._format_timestamp(last_tick),
            "tick_age_seconds": round(tick_age, 3) if tick_age is not None else None,
            "tick_stale_after": tick_guard if tick_guard > 0 else None,
            "tick_fresh": tick_fresh,
            "actionable": actionable,
            "actionable_reasons": reasons,
            "symbol_source": context.get("symbol_source"),
            "probability_source": context.get("probability_source"),
            "ev_source": context.get("ev_source"),
            "thresholds": {
                "buy_probability_pct": round(buy_threshold * 100.0, 2),
                "sell_probability_pct": round(sell_threshold * 100.0, 2),
                "effective_buy_probability_pct": round(
                    effective_buy_threshold * 100.0, 2
                ),
                "effective_sell_probability_pct": round(
                    effective_sell_threshold * 100.0, 2
                ),
                "exit_buy_probability_pct": round(exit_buy_threshold * 100.0, 2),
                "exit_sell_probability_pct": round(exit_sell_threshold * 100.0, 2),
                "min_ev_bps": round(min_ev, 2),
            },
            "has_status": bool(status) if has_status is None else bool(has_status),
            "fallback_used": bool(fallback_used),
            "status_source": self._status_source
            if self._status_source
            else ("cached" if fallback_used else "live"),
            "status_error": self._status_read_error,
            "staleness": {
                "state": staleness_state,
                "message": staleness_message,
            },
            "operation_mode": "auto",
        }

        watchlist_entries = context.get("watchlist") or []
        watchlist_breakdown = context.get("watchlist_breakdown") if context else None
        if watchlist_entries:
            summary["watchlist_total"] = len(watchlist_entries)
            summary["watchlist_actionable"] = sum(
                1 for item in watchlist_entries if item.get("actionable")
            )
            highlights: List[Dict[str, object]] = []
            for idx, entry in enumerate(watchlist_entries[:3]):
                highlight = {
                    "symbol": entry.get("symbol"),
                    "trend": entry.get("trend_hint") or entry.get("trend"),
                    "probability_pct": entry.get("probability_pct"),
                    "probability_ready": entry.get("probability_ready"),
                    "probability_gap_pct": entry.get("probability_gap_pct"),
                    "ev_bps": entry.get("ev_bps"),
                    "ev_ready": entry.get("ev_ready"),
                    "score": entry.get("score"),
                    "note": entry.get("note"),
                    "actionable": entry.get("actionable"),
                    "edge_score": entry.get("edge_score"),
                    "realized_bps_avg": entry.get("realized_bps_avg"),
                    "win_rate_pct": entry.get("win_rate_pct"),
                    "median_hold_sec": entry.get("median_hold_sec"),
                    "trade_sample_count": entry.get("trade_sample_count"),
                    "priority_adjustment": entry.get("priority_adjustment"),
                    "performance_bias": entry.get("performance_bias"),
                }
                if idx == 0:
                    highlight["primary"] = True
                highlights.append(highlight)

            summary["watchlist_highlights"] = highlights
            summary["primary_watch"] = copy.deepcopy(watchlist_entries[0])
            breakdown = (
                copy.deepcopy(watchlist_breakdown)
                if watchlist_breakdown
                else self._watchlist_breakdown(watchlist_entries)
            )
            summary["watchlist_breakdown"] = breakdown
            digest = context.get("watchlist_digest") if context else None
            if not digest:
                digest = self._watchlist_digest(breakdown)
            summary["watchlist_digest"] = digest

        if status:
            summary["raw_keys"] = sorted(status.keys())

            for source_key in ("mode", "signal", "action", "bias", "side"):
                hint = self._normalise_mode_hint(status.get(source_key))
                if hint:
                    summary["mode_hint"] = hint
                    summary["mode_hint_source"] = source_key
                    break

            if "mode_hint" not in summary and context.get("side_hint"):
                summary["mode_hint"] = context.get("side_hint")
                summary["mode_hint_source"] = "watchlist"

        return summary

    def _evaluate_actionability(
        self,
        mode: str,
        probability: float,
        ev_bps: float,
        buy_threshold: float,
        sell_threshold: float,
        min_ev: float,
        staleness_state: str,
        tick_age: Optional[float],
        tick_guard: float,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        def _pct(value: float) -> str:
            return f"{value * 100.0:.2f}%"

        def _bps(value: float) -> str:
            return f"{value:.2f} б.п."

        if staleness_state == "stale":
            reasons.append(
                f"Сигнал устарел — обновите {self._status_file_hint()} перед тем, как открывать сделки."
            )

        if (
            tick_guard > 0
            and tick_age is not None
            and tick_age > tick_guard
        ):
            reasons.append(
                (
                    "Котировка устарела — последний тик был {age:.1f} с назад, "
                    "ждём обновление стакана (лимит {limit:.1f} с)."
                ).format(age=tick_age, limit=tick_guard)
            )

        if mode == "buy":
            if probability < max(buy_threshold, 0.0):
                reasons.append(
                    (
                        "Уверенность модели {prob} ниже порога покупки {threshold} — "
                        "дождитесь более сильного сигнала."
                    ).format(
                        prob=_pct(probability),
                        threshold=_pct(max(buy_threshold, 0.0)),
                    )
                )
            if ev_bps < min_ev:
                reasons.append(
                    (
                        "Ожидаемая выгода {value} ниже безопасного минимума {minimum} — "
                        "риск/прибыль не в нашу пользу."
                    ).format(
                        value=_bps(ev_bps),
                        minimum=_bps(min_ev),
                    )
                )
        elif mode == "sell":
            if probability < max(sell_threshold, 0.0):
                reasons.append(
                    (
                        "Уверенность продажи {prob} ниже заданного порога {threshold} — "
                        "можно не спешить с фиксацией."
                    ).format(
                        prob=_pct(probability),
                        threshold=_pct(max(sell_threshold, 0.0)),
                    )
                )
            if ev_bps < min_ev:
                reasons.append(
                    (
                        "Ожидаемая выгода по продаже {value} ниже минимума {minimum} — "
                        "сделка может быть невыгодной."
                    ).format(
                        value=_bps(ev_bps),
                        minimum=_bps(min_ev),
                    )
                )
        else:
            reasons.append("Нет активного сигнала — бот предпочитает выжидать.")

        actionable = len(reasons) == 0
        return actionable, reasons

    def _derive_signal_context(
        self,
        status: Dict[str, object],
        settings: Settings,
        portfolio: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        raw_symbol = status.get("symbol") or status.get("ticker") or status.get("pair")
        symbol: Optional[str] = None
        symbol_source = "status"

        if isinstance(raw_symbol, str) and raw_symbol.strip():
            symbol = raw_symbol.strip().upper()
        else:
            symbol_source = "settings"
            for candidate in (settings.ai_symbols or "").split(","):
                candidate = candidate.strip()
                if candidate:
                    symbol = candidate.upper()
                    break

        if not symbol:
            universe = self._resolve_symbol_universe(settings)
            if universe:
                symbol = universe[0]
                symbol_source = "universe"

        if not symbol:
            symbol = "BTCUSDT"
            symbol_source = "default"

        holding_symbols: Set[str] = set()
        if isinstance(portfolio, Mapping):
            raw_positions = portfolio.get("positions") or []
            for entry in raw_positions:
                if not isinstance(entry, Mapping):
                    continue
                raw_position_symbol = entry.get("symbol")
                if not isinstance(raw_position_symbol, str):
                    continue
                position_symbol = raw_position_symbol.strip().upper()
                if not position_symbol:
                    continue
                qty = self._coerce_float(entry.get("qty")) or 0.0
                notional = self._coerce_float(entry.get("notional")) or 0.0
                if qty > 0.0 or notional > 0.0:
                    holding_symbols.add(position_symbol)

        holding_symbol = symbol in holding_symbols if symbol else False

        probability_value = self._normalise_probability_input(status.get("probability"))
        probability_source: Optional[str] = None
        if probability_value is not None and probability_value > 0.0:
            probability_source = "status"

        ev_value = self._coerce_float(status.get("ev_bps"))
        ev_source: Optional[str] = None
        if ev_value is not None and ev_value != 0.0:
            ev_source = "status"

        side_hint = self._normalise_mode_hint(status.get("side"))
        status_side_hint = side_hint

        (
            buy_threshold,
            sell_threshold,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            exit_buy_threshold,
            exit_sell_threshold,
        ) = self._resolve_thresholds(settings)

        buy_decision_threshold = effective_buy_threshold
        sell_decision_threshold = effective_sell_threshold

        watchlist = self._build_watchlist(status, settings)
        watchlist_breakdown = self._watchlist_breakdown(watchlist) if watchlist else None
        primary_entry: Optional[Dict[str, object]] = None
        fallback_entry: Optional[Tuple[Dict[str, object], Optional[str]]] = None
        watchlist_hint: Optional[str] = None

        for entry in watchlist:
            trend_hint = entry.get("trend_hint") or self._normalise_mode_hint(entry.get("trend"))

            if bool(entry.get("actionable")):
                primary_entry = entry
                watchlist_hint = trend_hint
                break

            if fallback_entry is None:
                fallback_entry = (entry, trend_hint)
            elif fallback_entry[1] not in {"buy", "sell"} and trend_hint in {"buy", "sell"}:
                fallback_entry = (entry, trend_hint)

        if primary_entry is None and fallback_entry is not None:
            primary_entry, trend_hint = fallback_entry
            watchlist_hint = trend_hint

        if primary_entry is None and watchlist:
            primary_entry = watchlist[0]
            trend_hint = self._normalise_mode_hint(primary_entry.get("trend"))
            watchlist_hint = trend_hint

        status_actionable = False
        if symbol_source == "status":
            if status_side_hint == "sell":
                status_actionable = (
                    probability_value is not None
                    and probability_value <= sell_decision_threshold
                    and (ev_value is None or ev_value >= min_ev)
                )
            else:
                status_actionable = (
                    probability_value is not None
                    and probability_value >= buy_decision_threshold
                    and (ev_value is None or ev_value >= min_ev)
                )

        if primary_entry:
            same_symbol_as_status = (
                symbol_source == "status"
                and isinstance(primary_entry.get("symbol"), str)
                and primary_entry.get("symbol") == symbol
            )
            primary_actionable = bool(primary_entry.get("actionable"))
            primary_source = str(primary_entry.get("source") or "").strip().lower()
            primary_dynamic = bool(primary_source and primary_source != "status")
            allow_watchlist_symbol = (
                symbol_source != "status"
                or not status_actionable
                or (primary_actionable and primary_dynamic)
            )
            using_watchlist_symbol = allow_watchlist_symbol or same_symbol_as_status

            if allow_watchlist_symbol and isinstance(primary_entry.get("symbol"), str):
                candidate_symbol = primary_entry["symbol"].strip().upper()
                if candidate_symbol:
                    symbol = candidate_symbol
                    symbol_source = "watchlist"

            if using_watchlist_symbol and watchlist_hint and (
                not side_hint or side_hint == "wait"
            ):
                side_hint = watchlist_hint

            if using_watchlist_symbol:
                for key in ("probability", "score"):
                    candidate_prob = self._normalise_probability_input(
                        primary_entry.get(key)
                    )
                    if candidate_prob is not None and candidate_prob > 0.0:
                        probability_value = candidate_prob
                        probability_source = "watchlist"
                        break

                candidate_ev = self._coerce_float(primary_entry.get("ev_bps"))
                if candidate_ev is not None and candidate_ev != 0.0:
                    ev_value = candidate_ev
                    ev_source = "watchlist"

            if using_watchlist_symbol:
                holding_symbol = symbol in holding_symbols if symbol else False
                buy_decision_threshold = effective_buy_threshold
                sell_decision_threshold = effective_sell_threshold
        else:
            side_hint = status_side_hint

        probability = self._clamp_probability(probability_value or 0.0)
        ev_bps = float(ev_value or 0.0)

        context = {
            "symbol": symbol,
            "symbol_source": symbol_source,
            "probability": probability,
            "probability_source": probability_source or symbol_source,
            "ev_bps": ev_bps,
            "ev_source": ev_source or symbol_source,
            "side_hint": side_hint,
            "watchlist": watchlist,
            "primary_watch": primary_entry,
        }

        if watchlist_breakdown:
            context["watchlist_breakdown"] = watchlist_breakdown
            context["watchlist_digest"] = self._watchlist_digest(watchlist_breakdown)

        context["symbol_plan"] = self._build_symbol_plan(
            watchlist,
            settings,
            portfolio=portfolio,
            ledger_signature=self._ledger_signature,
        )
        
        return context

    def _brief_from_status(
        self,
        status: Dict[str, object],
        settings: Settings,
        context: Optional[Dict[str, object]] = None,
    ) -> GuardianBrief:
        context = context or self._derive_signal_context(status, settings)
        symbol = context.get("symbol") or "BTCUSDT"
        probability = float(context.get("probability") or 0.0)
        ev_bps = float(context.get("ev_bps") or 0.0)

        mode_hint: Optional[str] = None
        for key in ("mode", "signal", "action", "bias"):
            mode_hint = self._normalise_mode_hint(status.get(key))
            if mode_hint:
                break
        side_hint = context.get("side_hint") or self._normalise_mode_hint(status.get("side"))

        (
            buy_threshold,
            sell_threshold,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            exit_buy_threshold,
            exit_sell_threshold,
        ) = self._resolve_thresholds(settings)

        if mode_hint:
            mode = mode_hint
        else:
            candidate_mode: str
            if side_hint == "wait":
                candidate_mode = "wait"
            elif probability >= effective_buy_threshold and ev_bps >= min_ev:
                candidate_mode = "buy"
            elif (
                side_hint == "sell"
                and probability <= effective_sell_threshold
                and ev_bps >= min_ev
            ):
                candidate_mode = "sell"
            elif probability <= effective_sell_threshold and ev_bps >= min_ev:
                candidate_mode = "sell"
            else:
                candidate_mode = "wait"

            previous_mode = self._signal_state.get(symbol)
            mode = candidate_mode

            if previous_mode == "buy":
                if candidate_mode == "sell":
                    if probability <= exit_sell_threshold and ev_bps >= min_ev:
                        mode = "sell"
                    else:
                        mode = "buy"
                elif candidate_mode == "wait":
                    if ev_bps < min_ev or probability <= exit_sell_threshold:
                        mode = "wait"
                    else:
                        mode = "buy"
            elif previous_mode == "sell":
                if candidate_mode == "buy":
                    if probability >= exit_buy_threshold and ev_bps >= min_ev:
                        mode = "buy"
                    else:
                        mode = "sell"
                elif candidate_mode == "wait":
                    if ev_bps < min_ev or probability >= exit_buy_threshold:
                        mode = "wait"
                    else:
                        mode = "sell"

        confidence_pct = probability * 100.0
        if confidence_pct >= 70:
            level = "высокая"
        elif confidence_pct >= 55:
            level = "средняя"
        else:
            level = "низкая"

        if symbol:
            self._signal_state[symbol] = mode

        confidence_text = (
            f"Уверенность модели: {confidence_pct:.1f}% — {level}. Чем выше процент, тем спокойнее можно действовать."
        )

        if ev_bps >= min_ev:
            ev_text = f"Потенциальная выгода: {ev_bps:.1f} б.п. Цель по безопасности — не ниже {min_ev:.1f} б.п."
        else:
            ev_text = (
                f"Потенциальная выгода: {ev_bps:.1f} б.п., это ниже безопасного порога {min_ev:.1f} б.п. — торгуем осторожно."
            )

        if mode == "buy":
            headline = f"{symbol}: появился шанс аккуратно купить."
            action_text = "Действие: готовим небольшую покупку. Бот рассчитает безопасный объём сам."
            caution = (
                "Покупаем только на ту сумму, которую готовы спокойно удерживать. Без плеча, без импульсивных решений."
            )
        elif mode == "sell":
            headline = f"{symbol}: бот предлагает зафиксировать прибыль."
            action_text = "Действие: выставляем продажу по рыночной цене, но без суеты и с учётом комиссии."
            caution = "Перед продажей убедитесь, что позиция действительно открыта и нет незакрытых заявок."
        else:
            headline = f"{symbol}: спокойный режим, явного преимущества нет."
            action_text = "Действие: ждём. Сохраняем депозит и не открываем новые сделки."
            caution = "Дисциплина важнее сделок. Пауза — это тоже стратегия."

        updated_text = self._age_to_text(status.get("age_seconds"))
        analysis = self._narrative_from_status(status, mode, symbol)

        return GuardianBrief(
            mode=mode,
            symbol=symbol,
            headline=headline,
            action_text=action_text,
            confidence_text=confidence_text,
            ev_text=ev_text,
            caution=caution,
            updated_text=updated_text,
            analysis=analysis,
            status_age=status.get("age_seconds"),
        )

    def generate_brief(self) -> GuardianBrief:
        snapshot = self._get_snapshot()
        return snapshot.brief

    def portfolio_overview(self) -> Dict[str, object]:
        snapshot = self._get_snapshot()
        return self._copy_dict(snapshot.portfolio)

    def risk_summary(self) -> str:
        s = self.settings
        mode = "учебный режим (демо)" if s.dry_run else "работаем с реальными деньгами"
        reserve = float(getattr(s, "spot_cash_reserve_pct", 0.0))
        per_trade = float(getattr(s, "ai_risk_per_trade_pct", 0.0))
        loss_limit = float(getattr(s, "ai_daily_loss_limit_pct", 0.0))
        concurrent = int(getattr(s, "ai_max_concurrent", 0))
        cash_only = bool(getattr(s, "spot_cash_only", True))
        slippage = int(getattr(s, "ai_max_slippage_bps", 0) or 0)

        def _strip_trailing(value: str) -> str:
            if "." in value:
                value = value.rstrip("0").rstrip(".")
            return value

        def fmt_pct(
            value: float, decimals: int = 2, *, keep_trailing: bool = False
        ) -> str:
            formatted = f"{value:.{decimals}f}"
            if not keep_trailing:
                formatted = _strip_trailing(formatted)
            return formatted

        def fmt_bps(value: int) -> str:
            return f"{value:,}".replace(",", " ") if value else "0"

        slip_low, slip_high = (300, 500) if s.testnet else (100, 200)
        slip_env = "тестнет" if s.testnet else "мейннет"
        slip_target = f"{slip_low}–{slip_high}"
        current_slip = fmt_bps(slippage)
        slip_line = (
            f"• Слиппедж: ориентир {slip_target} б.п. для {slip_env}"
            f" (сейчас {current_slip} б.п.)."
        )
        if slippage == 0:
            slip_line += " Настрой диапазон, чтобы бот учитывал рыночное проскальзывание."
        elif not (slip_low <= slippage <= slip_high):
            slip_line += " Подстрой значение, чтобы не переплачивать за вход."

        adaptive_floor = 0.5
        adaptive_ceiling = 2.0
        adaptive_range = f"{fmt_pct(adaptive_floor, keep_trailing=True)}–{fmt_pct(adaptive_ceiling, keep_trailing=True)}"
        risk_line = (
            "• Риск на сделку: бот подстраивает долю между "
            f"{adaptive_range}% капитала в зависимости от уверенности сигнала."
        )
        if per_trade > 0:
            per_trade_txt = fmt_pct(per_trade)
            risk_line += f" Минимум из настроек: {per_trade_txt}%."
            if per_trade > adaptive_ceiling:
                risk_line += " Лимит выше авто-диапазона — держи руку на пульсе."
        else:
            risk_line += " Дополнительный ручной лимит пока не задан."

        if loss_limit > 0:
            risk_line += f" Дневной лимит убытка: {fmt_pct(loss_limit)}%."
        else:
            risk_line += " Добавь дневной лимит убытка, чтобы бот умел вовремя остановиться."

        if concurrent <= 0:
            concurrent_line = (
                "• Одновременно открывается 0 сделок — установи лимит 1–2 для постепенного разгона."
            )
        else:
            concurrent_line = f"• Одновременно открывается не более {concurrent} сделок."
            if concurrent > 2:
                concurrent_line += " Для стабилизации придерживайся лимита 1–2 позиций."
            elif concurrent in (1, 2):
                concurrent_line += " Это соответствует рекомендуемому диапазону 1–2."

        lines = [
            f"• Режим: {mode}.",
            f"• Резерв безопасности: {fmt_pct(reserve, 1, keep_trailing=True)}% депозита хранится в кэше.",
            risk_line,
            slip_line,
            concurrent_line,
        ]
        if cash_only:
            lines.append("• Используем только собственные средства, без кредитного плеча.")
        else:
            lines.append("• Разрешено использовать заёмные средства — контролируйте плечо самостоятельно.")

        return "\n".join(lines)

    def plan_steps(self, brief: Optional[GuardianBrief] = None) -> List[str]:
        brief = brief or self.generate_brief()
        steps = [
            "Проверяем баланс USDT и убеждаемся, что на бирже достаточно средств без плеча.",
        ]
        if brief.mode == "buy":
            steps.append("Бот рассчитает размер покупки и предложит цену. Мы подтверждаем только если готовы держать позицию.")
        elif brief.mode == "sell":
            steps.append("Проверяем открытую позицию и подтверждаем продажу, чтобы зафиксировать результат.")
        else:
            steps.append("Наблюдаем за рынком и не торопимся. Пауза защищает капитал.")
        steps.append("После сделки сверяем PnL и обновляем заметки: что получилось и почему.")
        return steps

    def safety_notes(self) -> List[str]:
        notes = [
            "Бот не открывает сделки без ваших API-ключей и всегда уважает лимиты риска.",
            "Перед реальной торговлей протестируйте логику на учебном аккаунте или с маленькой суммой.",
            "Следите за обновлениями данных: если сигнал устарел, не спешите входить в рынок.",
        ]
        if active_dry_run(self.settings):
            notes.insert(0, "Сейчас включен учебный режим: сделки не затрагивают реальные средства.")
        else:
            notes.insert(0, "Работаем с реальными средствами — подтверждайте только понятные сделки.")
        return notes

    def signal_scorecard(self, brief: Optional[GuardianBrief] = None) -> Dict[str, object]:
        snapshot = self._get_snapshot()
        brief = brief or snapshot.brief
        summary = snapshot.status_summary
        probability = float(summary.get("probability") or 0.0)
        ev_bps = float(summary.get("ev_bps") or 0.0)
        (
            configured_buy_threshold,
            configured_sell_threshold,
            min_ev,
            effective_buy_threshold,
            effective_sell_threshold,
            exit_buy_threshold,
            exit_sell_threshold,
        ) = self._resolve_thresholds(self.settings)
        return {
            "symbol": brief.symbol,
            "mode": brief.mode,
            "probability_pct": round(self._clamp_probability(probability) * 100.0, 2),
            "ev_bps": round(ev_bps, 2),
            "buy_threshold": round(effective_buy_threshold * 100.0, 2),
            "sell_threshold": round(effective_sell_threshold * 100.0, 2),
            "exit_buy_threshold": round(exit_buy_threshold * 100.0, 2),
            "exit_sell_threshold": round(exit_sell_threshold * 100.0, 2),
            "configured_buy_threshold": round(configured_buy_threshold * 100.0, 2),
            "configured_sell_threshold": round(configured_sell_threshold * 100.0, 2),
            "min_ev_bps": round(min_ev, 2),
            "last_update": brief.updated_text,
        }

    def _normalise_watchlist_entry(self, symbol: str, payload: object) -> Dict[str, object]:
        if isinstance(payload, dict):
            score_candidate = payload.get("score")
            trend = payload.get("trend") or payload.get("direction") or payload.get("side")
            note = payload.get("note") or payload.get("comment") or payload.get("reason")
            source_value = payload.get("source")

            probability_value: Optional[float] = None
            for key in ("probability", "confidence", "p", "prob"):
                probability_value = self._normalise_probability_input(payload.get(key))
                if probability_value is not None:
                    break

            ev_value: Optional[float] = None
            for key in ("ev_bps", "ev", "edge", "alpha", "expected_value"):
                ev_value = self._coerce_float(payload.get(key))
                if ev_value is not None:
                    break
        else:
            score_candidate = payload
            trend = None
            note = None
            probability_value = None
            ev_value = None
            source_value = None

        numeric_score: Optional[float] = None
        if isinstance(score_candidate, (int, float)):
            numeric_score = round(float(score_candidate), 2)
        elif isinstance(score_candidate, str):
            stripped = score_candidate.strip()
            if stripped:
                try:
                    numeric_score = round(float(stripped), 2)
                except ValueError:
                    numeric_score = None

        if probability_value is None and numeric_score is not None:
            probability_value = self._normalise_probability_input(numeric_score)

        entry = {
            "symbol": symbol.upper(),
            "score": numeric_score,
            "trend": str(trend) if trend not in (None, "") else None,
            "note": str(note) if note not in (None, "") else None,
            "probability": probability_value,
            "ev_bps": ev_value,
        }
        if isinstance(payload, Mapping):
            deepseek_snapshot = extract_deepseek_snapshot(payload)
        else:
            deepseek_snapshot = {}
        if deepseek_snapshot:
            deepseek_data = dict(deepseek_snapshot)
            mode_hint = entry.get("trend")
            if not isinstance(mode_hint, str):
                mode_hint = str(mode_hint or "")
            guidance = evaluate_deepseek_guidance(deepseek_snapshot, mode_hint)
            if guidance:
                deepseek_data["guidance"] = guidance
            entry["deepseek"] = deepseek_data
        if isinstance(source_value, str) and source_value.strip():
            entry["source"] = source_value.strip()
        return entry

    @staticmethod
    def _normalise_symbol_value(symbol: object) -> Optional[str]:
        if isinstance(symbol, Mapping):
            return None
        if isinstance(symbol, (list, tuple, set, frozenset)):
            return None
        normalised, _ = ensure_usdt_symbol(symbol)
        return normalised

    @classmethod
    def _extend_unique_symbols(
        cls, target: List[str], symbols: Iterable[object]
    ) -> None:
        for symbol in symbols:
            cleaned = cls._normalise_symbol_value(symbol)
            if cleaned and cleaned not in target:
                target.append(cleaned)

    @classmethod
    def _priority_symbols_from_plan(
        cls, plan: Mapping[str, object]
    ) -> List[str]:
        priority_table = plan.get("priority_table") if isinstance(plan, Mapping) else None
        if not isinstance(priority_table, Sequence):
            return []

        symbols: List[str] = []
        for entry in priority_table:
            if not isinstance(entry, Mapping):
                continue
            cleaned = cls._normalise_symbol_value(entry.get("symbol"))
            if cleaned and cleaned not in symbols:
                symbols.append(cleaned)
        return symbols

    def _compose_symbol_pool(
        self,
        plan: Mapping[str, object],
        *,
        only_actionable: bool,
        fallback: Optional[object] = None,
    ) -> List[str]:
        if not isinstance(plan, Mapping):
            plan = {}

        def _items(key: str) -> List[object]:
            value = plan.get(key)
            if value is None:
                return []
            if isinstance(value, (list, tuple, set, frozenset)):
                return list(value)
            return [value]

        pool: List[str] = []
        priority_symbols = self._priority_symbols_from_plan(plan)

        if only_actionable:
            self._extend_unique_symbols(pool, _items("positions"))
            self._extend_unique_symbols(pool, _items("actionable"))
            if not pool:
                self._extend_unique_symbols(pool, _items("ready"))
            if not pool and priority_symbols:
                self._extend_unique_symbols(pool, priority_symbols)
        else:
            self._extend_unique_symbols(pool, _items("dynamic"))
            if not pool:
                for key in ("positions", "actionable", "ready", "watchlist"):
                    self._extend_unique_symbols(pool, _items(key))
            if not pool and priority_symbols:
                self._extend_unique_symbols(pool, priority_symbols)

        if not pool and fallback is not None:
            cleaned = self._normalise_symbol_value(fallback)
            if cleaned and cleaned not in pool:
                pool.append(cleaned)

        return pool

    def market_watchlist(self) -> List[Dict[str, object]]:
        snapshot = self._get_snapshot()
        return self._copy_list(snapshot.watchlist)

    def dynamic_symbols(
        self,
        limit: Optional[int] = None,
        *,
        only_actionable: bool = False,
    ) -> List[str]:
        """Return a prioritised list of symbols to trade based on the watchlist."""

        snapshot = self._get_snapshot()
        plan = snapshot.symbol_plan or {}
        if not isinstance(plan, Mapping):
            plan = {}

        if limit is None:
            limit = int(plan.get("limit") or 0)
            if limit <= 0:
                limit = None
        elif limit <= 0:
            limit = None

        if only_actionable:
            pool = list(plan.get("actionable_combined") or [])
        else:
            pool = list(plan.get("combined") or [])

        if not pool:
            pool = self._compose_symbol_pool(
                plan,
                only_actionable=only_actionable,
            )

        fallback_symbol = self._normalise_symbol_value(snapshot.brief.symbol)
        if fallback_symbol and fallback_symbol not in pool:
            pool.append(fallback_symbol)

        if limit is not None:
            pool = pool[:limit]

        return pool

    def trade_candidates(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Expose prioritised trade candidates with context for execution logic."""

        snapshot = self._get_snapshot()
        plan = snapshot.symbol_plan or {}

        if limit is not None and limit <= 0:
            limit = None

        return self._summarise_trade_candidates(
            plan,
            limit=limit,
        )

    def actionable_opportunities(
        self, limit: int = 5, include_neutral: bool = False
    ) -> List[Dict[str, object]]:
        """Return the strongest opportunities ranked by the guardian bot."""

        snapshot = self._get_snapshot()
        ranked: List[Dict[str, object]] = []

        for entry in snapshot.watchlist:
            trend = str(entry.get("trend_hint") or entry.get("trend") or "").lower()
            if entry.get("actionable"):
                ranked.append(copy.deepcopy(entry))
            elif include_neutral and trend not in {"buy", "sell"}:
                ranked.append(copy.deepcopy(entry))

        if limit and limit > 0:
            return ranked[:limit]
        return ranked

    def watchlist_breakdown(self) -> Dict[str, object]:
        """Expose aggregated watchlist stats for dashboards or APIs."""

        snapshot = self._get_snapshot()
        return self._watchlist_breakdown(snapshot.watchlist)

    def watchlist_digest(self) -> Dict[str, object]:
        """Return ready-to-use text snippets summarising the watchlist."""

        snapshot = self._get_snapshot()
        breakdown = self._watchlist_breakdown(snapshot.watchlist)
        return self._watchlist_digest(breakdown)

    def recent_trades(self, limit: int = 10) -> List[Dict[str, object]]:
        snapshot = self._get_snapshot()
        trades = snapshot.recent_trades[:limit] if limit else snapshot.recent_trades
        return self._copy_list(list(trades))

    def trade_statistics(self, limit: Optional[int] = None) -> Dict[str, object]:
        """Return aggregated execution metrics for dashboards."""

        snapshot = self._get_snapshot()
        if limit is None or limit <= 0:
            stats = copy.deepcopy(snapshot.trade_stats)
            defaults = self._auto_exit_defaults()
            if defaults:
                stats = dict(stats)
                stats["auto_exit_defaults"] = defaults
            return stats

        records: Tuple[ExecutionRecord, ...] = snapshot.executions[-limit:]
        return aggregate_execution_metrics(records)

    def _auto_exit_defaults(self) -> Dict[str, object]:
        """Derive adaptive exit thresholds from paired trade statistics."""

        try:
            signature = pair_trades_cache_signature(settings=self.settings)
        except Exception:
            signature = None

        trades: Optional[List[Dict[str, object]]] = None

        if (
            signature is not None
            and self._pair_trades_signature == signature
            and self._pair_trades_cache is not None
        ):
            trades = [copy.deepcopy(item) for item in self._pair_trades_cache]
        else:
            try:
                trades = pair_trades(settings=self.settings)
            except Exception:
                self._pair_trades_signature = None
                self._pair_trades_cache = None
                return {}

            if signature is not None:
                self._pair_trades_signature = signature
                self._pair_trades_cache = tuple(copy.deepcopy(item) for item in trades)
            else:
                self._pair_trades_signature = None
                self._pair_trades_cache = None

        if not trades:
            return {}

        hold_samples: List[float] = []
        loss_samples: List[float] = []

        for trade in trades:
            hold_value = self._coerce_float(trade.get("hold_sec"))
            if hold_value is not None and math.isfinite(hold_value) and hold_value > 0:
                hold_samples.append(float(hold_value))

            pnl_value = self._coerce_float(trade.get("bps_realized"))
            if pnl_value is not None and math.isfinite(pnl_value):
                pnl_float = float(pnl_value)
                if pnl_float < 0:
                    loss_samples.append(pnl_float)

        defaults: Dict[str, object] = {}

        if hold_samples:
            try:
                hold_median = float(median(hold_samples))
            except StatisticsError:
                hold_median = None
            if hold_median is not None and math.isfinite(hold_median) and hold_median > 0:
                defaults["hold_seconds"] = round(hold_median, 2)
                defaults["hold_minutes"] = round(hold_median / 60.0, 4)
                defaults["hold_sample_count"] = len(hold_samples)

        if loss_samples:
            try:
                quartiles = quantiles(loss_samples, n=4, method="inclusive")
            except (StatisticsError, ValueError):
                quartiles = []
            exit_threshold = quartiles[0] if quartiles else None
            if (
                exit_threshold is not None
                and math.isfinite(exit_threshold)
                and exit_threshold < 0
            ):
                defaults["exit_bps"] = round(float(exit_threshold), 4)
                defaults["bps_sample_count"] = len(loss_samples)

        return defaults

    def data_health(self) -> Dict[str, Dict[str, object]]:
        """High level diagnostics for the UI to highlight stale inputs."""

        snapshot = self._get_snapshot()
        status = snapshot.status
        summary = snapshot.status_summary
        age = snapshot.brief.status_age
        status_hint = self._status_file_hint()

        staleness_state, staleness_message = self._status_staleness(age)

        if snapshot.status_from_cache:
            ai_ok = False
            ai_message = (
                f"Используем сохранённый сигнал — не удалось прочитать актуальный {status_hint}."
            )
            ai_details = (
                f"Последнее обновление: {summary.get('last_update', '—')}. "
                f"Проверьте сервис, который пишет файл {status_hint}."
            )
            if self._status_read_error:
                error_text = str(self._status_read_error).strip()
                if error_text:
                    ai_details = f"{ai_details} Ошибка чтения: {error_text}."
        elif status:
            if staleness_state == "stale":
                ai_ok = False
            else:
                ai_ok = True
            ai_message = staleness_message
            symbol = str(summary.get("symbol") or status.get("symbol") or "?").upper()
            probability = float(summary.get("probability") or 0.0) * 100.0
            ai_details = f"Текущий символ: {symbol}, уверенность {probability:.1f}%."
        else:
            ai_ok = False
            ai_message = "AI сигнал ещё не поступал — запустите Guardian Bot или загрузите демо-данные."
            ai_details = f"Файл {status_hint} не найден."

        stats = snapshot.trade_stats
        trades = int(stats.get("trades", 0) or 0)
        last_trade_ts = stats.get("last_trade_ts")
        if trades == 0 or not last_trade_ts:
            exec_ok = False
            exec_message = "Журнал исполнений пуст — бот ещё не записывал сделки."
            exec_details = "Добавьте записи в pnl/executions.<network>.jsonl для проверки."
        else:
            last_trade_dt = datetime.fromtimestamp(float(last_trade_ts), tz=timezone.utc)
            exec_age = (datetime.now(timezone.utc) - last_trade_dt).total_seconds()
            if exec_age < 900:
                exec_ok = True
                exec_message = "Журнал сделок обновлялся менее 15 минут назад."
            elif exec_age < 3600:
                exec_ok = True
                exec_message = f"Журнал сделок обновлялся {self._format_duration(exec_age)} назад."
            else:
                exec_ok = False
                exec_message = (
                    "Журнал сделок не обновлялся более часа — проверьте исполнение ордеров и соединение."
                )
            exec_details = (
                f"Записей: {trades}, последняя сделка: {stats.get('last_trade_at', '—')}"
            )

        settings = self.settings
        api_info = api_key_status(settings)
        has_keys = bool(api_info.get("ok"))
        api_message = str(api_info.get("message") or "")
        api_details = api_info.get("details")
        api_title = str(api_info.get("title") or "Подключение API")

        automation_enabled = bool(getattr(settings, "ai_enabled", False))
        actionable = bool(summary.get("actionable"))
        automation_reasons = list(summary.get("actionable_reasons") or [])
        current_mode = str(summary.get("mode") or "wait").lower()

        if not automation_enabled:
            automation_ok = False
            automation_message = (
                "Автоматическая торговля выключена — активируйте AI в настройках."
            )
            automation_details = (
                "Настройка AI_ENABLED=False. Включите автоматизацию, чтобы бот мог исполнять сделки."
            )
        elif actionable:
            automation_ok = True
            automation_message = "AI готов к автоматическим сделкам."
            automation_details = (
                f"Режим: {current_mode.upper()} · Уверенность {summary.get('probability_pct', 0):.1f}% · "
                f"EV {summary.get('ev_bps', 0):.1f} б.п."
            )
        else:
            automation_ok = False
            if automation_reasons:
                reasons_text = "; ".join(str(reason) for reason in automation_reasons)
                automation_message = "Автоматизация приостановлена: " + reasons_text
            else:
                automation_message = "Автоматизация приостановлена — условия сигнала не выполнены."
            automation_details = (
                f"Текущий режим {current_mode.upper()}, вероятность {summary.get('probability_pct', 0):.1f}%"
            )

        realtime = bybit_realtime_status(settings)

        return {
            "ai_signal": {
                "title": "AI сигнал",
                "ok": ai_ok,
                "message": ai_message,
                "details": ai_details,
                "age_seconds": age,
            },
            "automation": {
                "title": "Автоматизация",
                "ok": automation_ok,
                "message": automation_message,
                "details": automation_details,
                "actionable": actionable,
            },
            "executions": {
                "title": "Журнал исполнений",
                "ok": exec_ok,
                "message": exec_message,
                "details": exec_details,
                "trades": trades,
                "last_trade_at": stats.get("last_trade_at"),
            },
            "api_keys": {
                "title": api_title,
                "ok": has_keys,
                "message": api_message,
                "details": api_details,
            },
            "realtime_trading": realtime,
        }

    def refresh(self) -> GuardianSnapshot:
        """Drop cached aggregates and rebuild them on next access."""

        self._snapshot = None
        self._ledger_signature = None
        self._ledger_view = None
        self._plan_cache_signature = None
        self._plan_cache = None
        self._plan_cache_ledger_signature = None
        self._digest_cache_signature = None
        self._digest_cache = None
        self._watchlist_breakdown_cache_signature = None
        self._watchlist_breakdown_cache = None
        self._listed_spot_symbols = None
        self._listed_spot_symbols_fetched_at = None
        return self._get_snapshot(force=True)

    def unified_report(self) -> Dict[str, object]:
        """Return a merged view of spot, risk and execution data."""

        snapshot = self._get_snapshot()
        return {
            "generated_at": snapshot.generated_at,
            "status": self._copy_dict(snapshot.status_summary),
            "brief": snapshot.brief.to_dict(),
            "portfolio": self._copy_dict(snapshot.portfolio),
            "watchlist": self._copy_list(snapshot.watchlist),
            "symbol_plan": copy.deepcopy(snapshot.symbol_plan),
            "recent_trades": self._copy_list(snapshot.recent_trades),
            "statistics": copy.deepcopy(snapshot.trade_stats),
            "health": self.data_health(),
        }

    def brief_payload(self) -> Dict[str, object]:
        """Return the cached brief as a serialisable dict."""

        snapshot = self._get_snapshot()
        return snapshot.brief.to_dict()

    def status_summary(self) -> Dict[str, object]:
        """Return a user-friendly dictionary of the latest signal."""

        snapshot = self._get_snapshot()
        return self._copy_dict(snapshot.status_summary)

    def status_fingerprint(self) -> Optional[str]:
        """Return a stable identifier for the currently cached signal."""

        snapshot = self._get_snapshot()
        signature = getattr(snapshot, "status_signature", None)
        if isinstance(signature, tuple):
            return ":".join(str(part) for part in signature)
        if signature is None:
            return None
        return str(signature)

    def auto_executor(self):  # pragma: no cover - simple factory wrapper
        from .signal_executor import SignalExecutor

        return SignalExecutor(self)

    def automation_loop(
        self,
        *,
        poll_interval: float = 15.0,
        success_cooldown: float = 120.0,
        error_backoff: float = 5.0,
    ):
        """Return a continuous executor that keeps trading until stopped."""

        from .signal_executor import AutomationLoop, SignalExecutor

        executor = SignalExecutor(self)
        return AutomationLoop(
            executor,
            poll_interval=poll_interval,
            success_cooldown=success_cooldown,
            error_backoff=error_backoff,
        )

    # ------------------------------------------------------------------
    # conversation helpers
    @staticmethod
    def _contains_any(text: str, needles: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

    def _format_profit_answer(self, portfolio: Dict[str, object]) -> str:
        totals = portfolio.get("totals", {})
        realized = float(totals.get("realized_pnl", 0.0))
        open_notional = float(totals.get("open_notional", 0.0))
        open_positions = int(totals.get("open_positions", 0))
        return (
            f"Зафиксированная прибыль: {realized:.2f} USDT. "
            f"В позициях работает около {open_notional:.2f} USDT, активных сделок: {open_positions}. "
            "Плавный прирост капитала важнее быстрых прыжков, поэтому бот закрывает сделки только при понятном плюсе."
        )

    def _format_signal_quality_answer(self, summary: Dict[str, object]) -> str:
        status_hint = self._status_file_hint()
        if not summary.get("has_status"):
            return (
                f"Живой сигнал пока не загружен — обновите {status_hint}, "
                "чтобы бот смог рассказать про вероятность и выгоду."
            )

        mode = str(summary.get("mode") or "wait")
        probability_pct = float(summary.get("probability_pct") or 0.0)
        ev_bps = float(summary.get("ev_bps") or 0.0)
        thresholds = summary.get("thresholds") or {}
        buy_threshold = float(thresholds.get("buy_probability_pct") or 0.0)
        sell_threshold = float(thresholds.get("sell_probability_pct") or 0.0)
        effective_buy_threshold = float(
            thresholds.get("effective_buy_probability_pct") or buy_threshold
        )
        effective_sell_threshold = float(
            thresholds.get("effective_sell_probability_pct") or sell_threshold
        )
        min_ev = float(thresholds.get("min_ev_bps") or 0.0)
        status_source = str(summary.get("status_source") or "live")
        actionable = bool(summary.get("actionable"))
        reasons = summary.get("actionable_reasons") or []

        lines = [
            (
                "Сигнал {mode} с уверенностью {prob:.2f}% и ожидаемой выгодой {ev:.2f} б.п.".format(
                    mode="на покупку" if mode == "buy" else "на продажу" if mode == "sell" else "в режиме ожидания",
                    prob=probability_pct,
                    ev=ev_bps,
                )
            )
        ]

        threshold_line = (
            "Рабочие пороги: покупка от {buy:.2f}%, продажа от {sell:.2f}%, минимальная выгода {ev:.2f} б.п.".format(
                buy=effective_buy_threshold or 0.0,
                sell=effective_sell_threshold or 0.0,
                ev=min_ev,
            )
        )
        lines.append(threshold_line)

        configured_notes: List[str] = []
        if abs(buy_threshold - effective_buy_threshold) >= 0.01:
            configured_notes.append(
                "⚙️ Настройка покупки была {configured:.2f}%, но фактически применяется {effective:.2f}% после авто-коррекции.".format(
                    configured=buy_threshold,
                    effective=effective_buy_threshold,
                )
            )
        if abs(sell_threshold - effective_sell_threshold) >= 0.01:
            configured_notes.append(
                "⚙️ Порог продажи из настроек {configured:.2f}%, однако для безопасности используется {effective:.2f}%.".format(
                    configured=sell_threshold,
                    effective=effective_sell_threshold,
                )
            )
        lines.extend(configured_notes)

        if actionable:
            lines.append("Показатели проходят контроль, сигнал можно исполнять при соблюдении риск-плана.")
        else:
            if reasons:
                for reason in reasons:
                    lines.append(f"⚠️ {reason}")
            else:
                lines.append("Сигнал пока наблюдаем — ждём совпадения с порогами стратегии.")

        staleness = summary.get("staleness") or {}
        staleness_message = staleness.get("message")
        if isinstance(staleness_message, str) and staleness_message.strip():
            lines.append(staleness_message.strip())

        if status_source == "file":
            lines.append(
                f"Данные прочитаны из локального {status_hint} — убедитесь, что файл обновляется автоматически."
            )
        elif status_source == "cached" or summary.get("fallback_used"):
            lines.append(
                "Данные получены из последнего сохранённого файла — проверьте, что бот обновляет статус в реальном времени."
            )

        last_update = summary.get("last_update")
        if isinstance(last_update, str) and last_update.strip():
            lines.append(f"Последнее обновление: {last_update.strip()}.")

        return "\n".join(lines)

    def _format_update_answer(self, summary: Dict[str, object]) -> str:
        status_hint = self._status_file_hint()
        if not summary.get("has_status"):
            return (
                f"Свежие данные ещё не поступали — файл {status_hint} отсутствует или пуст. "
                "Запустите сервис генерации сигнала или загрузите демо-статус, чтобы бот видел обновления."
            )

        lines: List[str] = []

        staleness = summary.get("staleness") or {}
        staleness_message = staleness.get("message")
        if isinstance(staleness_message, str) and staleness_message.strip():
            lines.append(staleness_message.strip())

        updated_text = summary.get("updated_text")
        if isinstance(updated_text, str) and updated_text.strip():
            lines.append(updated_text.strip())

        last_update = summary.get("last_update")
        if isinstance(last_update, str) and last_update.strip() and last_update.strip() != "—":
            lines.append(f"Последняя отметка обновления: {last_update.strip()} по UTC.")

        age_seconds = summary.get("age_seconds")
        if isinstance(age_seconds, (int, float)) and age_seconds is not None and age_seconds >= 0:
            lines.append(
                f"Текущий возраст сигнала: {self._format_duration(float(age_seconds))}."
            )

        if summary.get("fallback_used"):
            lines.append(
                "Ответ собран из кэшированной копии — убедитесь, что статус обновляется автоматически."
            )

        retrain_minutes = int(getattr(self.settings, "ai_retrain_minutes", 0) or 0)
        if retrain_minutes > 0:
            lines.append(
                f"Модель пересматривает веса примерно каждые {retrain_minutes} мин — держите статус свежим."
            )

        health = self.data_health()
        ai_health = health.get("ai_signal") if isinstance(health, dict) else None
        ai_details = ai_health.get("details") if isinstance(ai_health, dict) else None
        if isinstance(ai_details, str) and ai_details.strip():
            lines.append(f"Диагностика: {ai_details.strip()}")

        if not lines:
            lines.append(
                "Статус обновляется без предупреждений — можно продолжать следить за сигналом."
            )

        return "\n".join(lines)

    def _format_exposure_answer(self, portfolio: Dict[str, object]) -> str:
        settings = self.settings
        totals = portfolio.get("totals", {})
        positions = portfolio.get("positions", [])

        open_notional = float(totals.get("open_notional", 0.0))
        realized = float(totals.get("realized_pnl", 0.0))
        open_positions = int(totals.get("open_positions", 0))
        reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0))
        risk_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0))
        cash_only = bool(getattr(settings, "spot_cash_only", True))

        adaptive_floor = 0.5
        adaptive_ceiling = 2.0
        adaptive_range_txt = f"{adaptive_floor:.2f}–{adaptive_ceiling:.2f}%"

        if open_notional <= 0:
            reserve_line = (
                "Капитал свободен — сделки не открыты. "
                f"По плану держим резерв не менее {reserve_pct:.1f}% в кэше"
            )
            if cash_only:
                reserve_line += ", работаем без плеча."
            else:
                reserve_line += ", при необходимости оператор может добавить плечо."
            allocation_hint = (
                f" Авто-аллокатор выдаёт {adaptive_range_txt} капитала на новую идею."
            )
            if risk_pct > 0:
                allocation_hint += f" Минимум из настроек: {risk_pct:.2f}%."
            return (
                reserve_line
                + allocation_hint
                + " Зафиксированная прибыль с начала сессии: "
                + f"{realized:.2f} USDT."
            )

        engaged_line = (
            f"В работе {open_notional:.2f} USDT через {open_positions} активных позиций. "
            f"Резерв безопасности по настройкам — {reserve_pct:.1f}% капитала."
        )

        leaders: List[str] = []
        for record in sorted(positions, key=lambda item: float(item.get("notional") or 0.0), reverse=True):
            notional = float(record.get("notional") or 0.0)
            if notional <= 0:
                continue
            symbol = str(record.get("symbol") or "?")
            qty = float(record.get("qty") or 0.0)
            share = (notional / open_notional * 100.0) if open_notional else 0.0
            leaders.append(
                f"{symbol}: {qty:.6g} шт ≈ {notional:.2f} USDT ({share:.1f}%)"
            )
            if len(leaders) == 3:
                break

        if leaders:
            leaders_line = "Крупнейшие позиции: " + "; ".join(leaders) + "."
        else:
            leaders_line = "Позиции с ненулевой стоимостью не найдены — проверьте журнал сделок."

        sizing_line = (
            "Новые сделки открываем небольшими частями:"
            f" авто-аллокатор выделяет {adaptive_range_txt} капитала на идею."
        )
        if risk_pct > 0:
            sizing_line += f" Ручной минимум из настроек: {risk_pct:.2f}%."
        if cash_only:
            sizing_line += " работаем только на собственные средства."
        else:
            sizing_line += " допускается аккуратное использование плеча оператором."

        return " ".join([engaged_line, leaders_line, sizing_line, f"Фиксированный результат: {realized:.2f} USDT."])

    def _format_plan_answer(self, brief: GuardianBrief) -> str:
        steps = self.plan_steps(brief)
        formatted = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
        return f"План действий:\n{formatted}"

    def _format_settings_answer(self) -> str:
        settings = self.settings

        lines: List[str] = []

        if settings.testnet:
            lines.append(
                "Работаем на тестнете Bybit — все сделки учебные, можно проверять логику без риска."
            )
        else:
            env_line = "Бот подключён к боевому счёту."
            if active_dry_run(settings):
                env_line += " Сделки подтверждаются в dry-run, исполнение нужно запускать вручную."
            else:
                env_line += " Реальные сделки включены — контролируйте лимиты тщательно."
            lines.append(env_line)

        if settings.testnet and not active_dry_run(settings):
            lines.append(
                "Несмотря на тестнет, dry-run выключен — ордера будут отправляться в симулятор биржи."
            )
        elif active_dry_run(settings) and not settings.testnet:
            lines.append(
                "Dry-run включён: заявки не отправляются на биржу, но журнал и сигналы записываются."
            )

        ai_enabled = getattr(settings, "ai_enabled", False)
        ai_symbols = str(getattr(settings, "ai_symbols", "") or "").strip()
        if ai_enabled:
            if ai_symbols:
                raw_symbols = [symbol.strip().upper() for symbol in ai_symbols.split(",") if symbol.strip()]
                if raw_symbols:
                    symbol_text = ", ".join(sorted(raw_symbols))
                else:
                    symbol_text = "все доступные символы пресета"
            else:
                symbol_text = "все доступные символы пресета"
            lines.append(
                f"AI сигналы активны в категории {getattr(settings, 'ai_category', 'spot')} для: {symbol_text}."
            )
        else:
            lines.append("AI сигналы выключены — бот может действовать только по ручным командам.")

        buy_threshold = float(getattr(settings, "ai_buy_threshold", 0.0) or 0.0) * 100.0
        sell_threshold = float(getattr(settings, "ai_sell_threshold", 0.0) or 0.0) * 100.0
        min_ev = resolve_min_ev_from_settings(settings, default_bps=12.0)
        lines.append(
            "Порог входа: покупка от {buy:.2f}%, продажа от {sell:.2f}%, ожидаемая выгода не ниже {ev:.2f} б.п.".format(
                buy=buy_threshold,
                sell=sell_threshold,
                ev=min_ev,
            )
        )

        risk_per_trade = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0) or 0.0)
        daily_loss_limit = float(getattr(settings, "ai_daily_loss_limit_pct", 0.0) or 0.0)
        trade_loss_limit = float(getattr(settings, "ai_max_trade_loss_pct", 0.0) or 0.0)
        portfolio_stop = float(getattr(settings, "ai_portfolio_loss_limit_pct", 0.0) or 0.0)
        kill_switch_pause = float(
            getattr(settings, "ai_kill_switch_cooldown_min", 0.0) or 0.0
        )
        cash_only = bool(getattr(settings, "spot_cash_only", True))
        adaptive_floor = 0.5
        adaptive_ceiling = 2.0
        adaptive_range_txt = f"{adaptive_floor:.2f}–{adaptive_ceiling:.2f}%"
        risk_line_parts = [
            f"бот варьирует риск {adaptive_range_txt} капитала на сделку"
        ]
        if risk_per_trade > 0:
            risk_line_parts.append(f"минимум по настройкам {risk_per_trade:.2f}%")
        risk_line_parts.append(f"резервируем в кэше не менее {reserve_pct:.1f}%")
        if daily_loss_limit > 0:
            risk_line_parts.append(f"дневной стоп по убытку {daily_loss_limit:.2f}%")
        if trade_loss_limit > 0:
            risk_line_parts.append(
                f"максимальный убыток на позицию {trade_loss_limit:.2f}%"
            )
        if portfolio_stop > 0:
            risk_line_parts.append(f"портфельный стоп {portfolio_stop:.2f}%")
        if cash_only:
            risk_line_parts.append("работаем без заимствований")
        else:
            risk_line_parts.append("допускается использование плеча по усмотрению оператора")
        lines.append(
            ", ".join(risk_line_parts) + "."
        )

        if portfolio_stop > 0 and kill_switch_pause > 0:
            lines.append(
                f"При срабатывании портфельного стопа включается пауза автоматики на {kill_switch_pause:.0f} мин."
            )

        max_concurrent = int(getattr(settings, "ai_max_concurrent", 0) or 0)
        if max_concurrent > 0:
            lines.append(
                f"Одновременно AI ведёт до {max_concurrent} активных идей, чтобы не распылять капитал."
            )

        retrain_minutes = int(getattr(settings, "ai_retrain_minutes", 0) or 0)
        if retrain_minutes > 0:
            lines.append(
                f"Модель обновляет весы примерно каждые {retrain_minutes} минут для актуальности статистики."
            )

        watchdog_enabled = bool(getattr(settings, "ws_watchdog_enabled", False))
        execution_guard = int(getattr(settings, "execution_watchdog_max_age_sec", 0) or 0)
        if watchdog_enabled or execution_guard:
            guard_parts: List[str] = []
            if watchdog_enabled:
                guard_parts.append("веб-сокет сторож активен")
            if execution_guard:
                guard_parts.append(
                    f"за обновлением сделок следим, предел задержки {execution_guard} с"
                )
            lines.append("Сторожа соединений: " + ", ".join(guard_parts) + ".")

        return "\n".join(lines)

    def _format_positions_answer(self, portfolio: Dict[str, object]) -> str:
        positions = portfolio.get("positions") or []
        totals = portfolio.get("totals") or {}

        open_notional = float(totals.get("open_notional", 0.0))
        realized = float(totals.get("realized_pnl", 0.0))
        open_positions = int(totals.get("open_positions", 0))

        meaningful_positions: List[Dict[str, object]] = []
        for entry in positions:
            try:
                qty = float(entry.get("qty", 0.0))
                notional = float(entry.get("notional", 0.0))
            except (TypeError, ValueError):
                continue
            if qty <= 0 and notional <= 0:
                continue
            meaningful_positions.append(entry)

        if not meaningful_positions:
            return (
                "Открытых позиций нет — капитал ждёт нового сигнала. "
                f"Зафиксированная прибыль сейчас {realized:.2f} USDT."
            )

        sorted_positions = sorted(
            meaningful_positions,
            key=lambda item: float(item.get("notional") or 0.0),
            reverse=True,
        )

        lines: List[str] = []
        for entry in sorted_positions[:3]:
            symbol = str(entry.get("symbol") or "?").upper()
            qty = float(entry.get("qty") or 0.0)
            avg_cost = float(entry.get("avg_cost") or 0.0)
            notional = float(entry.get("notional") or 0.0)
            lines.append(
                (
                    f"{symbol}: {qty:.4f} шт по {avg_cost:.2f} USDT "
                    f"→ ~{notional:.2f} USDT"
                )
            )

        if len(sorted_positions) > 3:
            lines.append(
                f"Ещё {len(sorted_positions) - 3} позиция(и) с меньшим весом остаются под контролем."
            )

        header = (
            f"В портфеле {open_positions} актив(а) на сумму около {open_notional:.2f} USDT."
        )
        footer = (
            "Дисциплина важна: каждая позиция укладывается в риск-профиль,"
            " а свободные USDT страхуют волатильность."
        )

        return "\n".join([header, *lines, footer, f"Зафиксировано прибыли: {realized:.2f} USDT."])

    def _format_watchlist_answer(self) -> str:
        watchlist = self.market_watchlist()
        if not watchlist:
            return "Список наблюдения пуст — бот ждёт свежих сигналов, чтобы предложить идеи."

        lines: List[str] = []
        for entry in watchlist[:3]:
            symbol = str(entry.get("symbol") or "?").upper()
            bits: List[str] = []
            score = entry.get("score")
            trend = entry.get("trend_hint") or entry.get("trend")
            note = entry.get("note")
            probability_pct = entry.get("probability_pct")
            ev_bps = entry.get("ev_bps")
            if entry.get("actionable"):
                bits.append("готов к сделке")
            if isinstance(trend, str) and trend:
                bits.append(trend.lower())
            if isinstance(score, (int, float)):
                bits.append(f"оценка {float(score):.2f}")
            if isinstance(probability_pct, (int, float)):
                bits.append(f"уверенность {float(probability_pct):.1f}%")
            if isinstance(ev_bps, (int, float)) and ev_bps:
                bits.append(f"выгода ~{float(ev_bps):.1f} б.п.")
            if isinstance(note, str) and note:
                bits.append(note)
            detail = ", ".join(bits) if bits else "наблюдаем динамику"
            lines.append(f"{symbol}: {detail}")

        actionable_total = sum(1 for entry in watchlist if entry.get("actionable"))
        if actionable_total:
            lines.append(
                f"Активных идей: {actionable_total} — бот уже проверил вероятность и выгоду."
            )

        if len(watchlist) > 3:
            lines.append(
                f"В очереди наблюдения ещё {len(watchlist) - 3} инструмент(а) — бот отсортировал их по силе сигнала."
            )

        return "Список наблюдения:\n" + "\n".join(lines)

    def _format_health_answer(self) -> str:
        health = self.data_health()
        order = (
            "ai_signal",
            "automation",
            "executions",
            "api_keys",
            "realtime_trading",
        )

        blocks: List[str] = []
        for key in order:
            block = health.get(key)
            if not isinstance(block, dict):
                continue

            title = str(block.get("title") or key.replace("_", " ").title())
            ok = bool(block.get("ok"))
            message = str(block.get("message") or "")
            details = block.get("details")
            icon = "✅" if ok else "⚠️"

            section = [f"{icon} {title}: {message}"]
            if isinstance(details, str) and details.strip():
                section.append(f"    {details.strip()}")
            blocks.append("\n".join(section))

        if not blocks:
            return (
                "Диагностика не дала результата — убедитесь, что данные бота обновляются хотя бы демо-файлами."
            )

        return "\n".join(blocks)

    def _format_automation_answer(self, block: Dict[str, object]) -> str:
        if not isinstance(block, dict) or not block:
            return (
                "Бот работает полностью автоматически — ручных команд больше не требуется."
            )

        message = str(block.get("message") or "Автоматизация активна.").strip()
        details = block.get("details")
        lines = [message]
        if isinstance(details, str) and details.strip():
            lines.append(details.strip())
        if block.get("actionable") is False:
            lines.append("AI пока наблюдает рынок и ждёт подходящего сигнала для входа.")
        return "\n".join(lines)

    def _format_trade_history_answer(
        self,
        trades: List[Dict[str, object]],
        stats: Dict[str, object],
    ) -> str:
        total_trades = int(stats.get("trades", len(trades)) or 0)
        gross_volume = float(stats.get("gross_volume") or 0.0)
        header = (
            f"В журнале {total_trades} сделок на сумму около {gross_volume:.2f} USDT."
        )

        if not trades:
            return (
                header
                + " Пока бот только готовится: новые записи появятся после исполнения первой сделки."
            )

        lines = [header, "Последние операции:"]

        def _format_side(value: object) -> str:
            mapping = {"buy": "покупка", "sell": "продажа"}
            if isinstance(value, str) and value:
                key = value.lower()
                return mapping.get(key, value.lower())
            return "операция"

        preview = trades[:3]
        for entry in preview:
            symbol = str(entry.get("symbol") or "?")
            side = _format_side(entry.get("side"))
            qty = entry.get("qty")
            price = entry.get("price")
            fee = entry.get("fee")
            when = entry.get("when") or "—"

            details: List[str] = []
            if isinstance(qty, (int, float)) and qty:
                details.append(f"{float(qty):.4f} шт")
            if isinstance(price, (int, float)) and price:
                details.append(f"по {float(price):.2f} USDT")
            detail_text = " ".join(details) if details else "без детализации"

            extra_bits: List[str] = []
            if isinstance(fee, (int, float)) and fee:
                extra_bits.append(f"комиссия {float(fee):.4f} USDT")
            if when and when != "—":
                extra_bits.append(f"в {when}")
            extra_text = ", ".join(extra_bits)
            if extra_text:
                extra_text = f" ({extra_text})"

            lines.append(f"- {symbol}: {side} {detail_text}{extra_text}")

        if total_trades > len(preview):
            lines.append(
                f"Показаны последние {len(preview)} записи из {total_trades}. Полный журнал — в разделе исполнений."
            )

        last_trade_at = stats.get("last_trade_at")
        if isinstance(last_trade_at, str) and last_trade_at.strip():
            lines.append(f"Последняя сделка отмечена как {last_trade_at} по UTC.")

        lines.append(
            "Следите, чтобы серия сделок оставалась в рамках дневного лимита потерь и резервов по USDT."
        )

        return "\n".join(lines)

    def _format_fee_activity_answer(self, stats: Dict[str, object]) -> str:
        trades = int(stats.get("trades", 0) or 0)
        if trades <= 0:
            return (
                "Комиссий ещё нет — журнал сделок пуст. "
                "Как только появятся исполнения, покажу расход по комиссиям и активность."
            )

        gross_volume = float(stats.get("gross_volume") or 0.0)
        fees_paid = float(stats.get("fees_paid") or 0.0)
        avg_trade = float(stats.get("avg_trade_value") or 0.0)
        maker_ratio = float(stats.get("maker_ratio") or 0.0) * 100.0
        last_trade_at = str(stats.get("last_trade_at") or "—")

        maker_trade_ratio: Optional[float] = None
        executions = getattr(self._ledger_view, "executions", None)
        if executions:
            maker_trades = sum(1 for record in executions if record.is_maker is True)
            total_trades = len(executions)
            if total_trades > 0:
                maker_trade_ratio = (maker_trades / total_trades) * 100.0

        activity = stats.get("activity") or {}
        recent_15 = int(activity.get("15m") or 0)
        recent_hour = int(activity.get("1h") or 0)
        recent_day = int(activity.get("24h") or 0)

        maker_line = (
            f"Средний размер сделки: {avg_trade:.2f} USDT. "
            f"Доля maker-исполнений: {maker_ratio:.1f}%."
        )
        if (
            maker_trade_ratio is not None
            and abs(maker_trade_ratio - maker_ratio) >= 0.05
        ):
            maker_line = (
                f"Средний размер сделки: {avg_trade:.2f} USDT. "
                f"Доля maker-исполнений: {maker_ratio:.1f}% по объёму, "
                f"{maker_trade_ratio:.1f}% по сделкам."
            )

        lines = [
            (
                f"Совокупные комиссии: {fees_paid:.5f} USDT за {trades} сделк(и) "
                f"с объёмом около {gross_volume:.2f} USDT."
            ),
            maker_line,
            (
                "Активность: за 15 минут {recent_15}, за час {recent_hour}, за сутки {recent_day}. "
                f"Последняя запись: {last_trade_at} по UTC."
            ).format(
                recent_15=recent_15,
                recent_hour=recent_hour,
                recent_day=recent_day,
            ),
        ]

        per_symbol = stats.get("per_symbol") or []
        if per_symbol:
            top = per_symbol[0]
            symbol = str(top.get("symbol") or "—")
            volume = float(top.get("volume") or 0.0)
            buy_share = float(top.get("buy_share") or 0.0) * 100.0
            lines.append(
                (
                    f"Главный символ: {symbol} — {volume:.2f} USDT оборота, "
                    f"покупки занимают {buy_share:.1f}% трафика."
                )
            )

        lines.append(
            "Контролируйте комиссии: чем выше доля maker, тем дешевле обходятся сделки."
        )

        return "\n".join(lines)

    def _default_response(self, brief: GuardianBrief, portfolio: Dict[str, object]) -> str:
        pieces = [
            brief.headline,
            brief.action_text,
            brief.analysis,
            brief.caution,
            brief.confidence_text,
            brief.ev_text,
            brief.updated_text,
        ]
        pieces.append(self._format_profit_answer(portfolio))
        pieces.append(self.risk_summary())
        return "\n\n".join(pieces)

    def initial_message(self) -> str:
        brief = self.generate_brief()
        portfolio = self.portfolio_overview()
        plan = self._format_plan_answer(brief)
        profit = self._format_profit_answer(portfolio)
        message_parts = [
            brief.headline,
            brief.action_text,
            brief.analysis,
            brief.confidence_text,
            brief.ev_text,
            brief.caution,
            brief.updated_text,
            plan,
            profit,
        ]
        staleness = self.staleness_alert(brief)
        if staleness:
            message_parts.append(staleness)
        return "\n\n".join(message_parts)

    def _normalise_question(self, question: object | None) -> str:
        preferred_roles = {"user", "human", "client", "trader", "customer"}
        assistant_like_roles = {
            "assistant",
            "system",
            "bot",
            "ai",
            "tool",
            "function",
        }

        def _mapping_get(mapping: Mapping | object, key: str) -> object:
            if isinstance(mapping, Mapping):
                if hasattr(mapping, "get"):
                    return mapping.get(key)
                try:
                    return mapping[key]  # type: ignore[index]
                except Exception:
                    return None
            return None

        def _as_role(value: object, depth: int = 0) -> str:
            if value is None or depth > 6:
                return ""

            if isinstance(value, (bytes, bytearray, memoryview)):
                value = bytes(value).decode("utf-8", errors="ignore")

            if isinstance(value, str):
                text = value.strip().lower()
                if not text:
                    return ""
                for token in (*preferred_roles, *assistant_like_roles):
                    if token in text:
                        return token
                return text

            if is_dataclass(value):
                try:
                    dataclass_mapping = asdict(value)
                except Exception:  # pragma: no cover - defensive
                    dataclass_mapping = {
                        key: getattr(value, key)
                        for key in getattr(value, "__dataclass_fields__", {})
                    }
                return _as_role(dataclass_mapping, depth + 1)

            for attr_name in ("model_dump", "dict", "to_dict", "as_dict", "asdict"):
                attr = getattr(value, attr_name, None)
                if not callable(attr):
                    continue
                try:
                    mapping_like = attr()
                except TypeError:
                    try:
                        mapping_like = attr(exclude_none=False)
                    except TypeError:
                        continue
                    except Exception:  # pragma: no cover - defensive
                        continue
                except Exception:  # pragma: no cover - defensive
                    continue

                if isinstance(mapping_like, (dict, list, tuple, set, frozenset)):
                    return _as_role(mapping_like, depth + 1)
                if isinstance(mapping_like, str):
                    return _as_role(mapping_like, depth + 1)

            for attr_name in ("role", "type", "kind"):
                if hasattr(value, attr_name):
                    candidate = _as_role(getattr(value, attr_name), depth + 1)
                    if candidate:
                        return candidate

            if isinstance(value, Mapping):
                for key in ("role", "type", "kind", "value", "name", "label"):
                    if key in value:
                        candidate = _as_role(_mapping_get(value, key), depth + 1)
                        if candidate:
                            return candidate
                return ""

            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    candidate = _as_role(item, depth + 1)
                    if candidate:
                        return candidate
                return ""

            for attr in ("value", "name"):
                if hasattr(value, attr):
                    candidate = _as_role(getattr(value, attr), depth + 1)
                    if candidate:
                        return candidate

            if hasattr(value, "__dict__"):
                mapping_like = getattr(value, "__dict__")
                if isinstance(mapping_like, dict):
                    candidate = _as_role(mapping_like, depth + 1)
                    if candidate:
                        return candidate

            try:
                text = str(value)
            except Exception:  # pragma: no cover - defensive
                return ""

            text = text.strip().lower()
            if not text:
                return ""
            for token in (*preferred_roles, *assistant_like_roles):
                if token in text:
                    return token
            return text

        def iter_parts(value: object | None, depth: int = 0) -> Iterable[str]:
            if value is None or depth > 6:
                return []

            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return []

                if len(stripped) <= 2000 and stripped[0] in "{[":
                    try:
                        parsed = json.loads(stripped)
                    except Exception:
                        pass
                    else:
                        parsed_parts = list(iter_parts(parsed, depth + 1))
                        if parsed_parts:
                            return parsed_parts

                return [stripped]

            if isinstance(value, (bytes, bytearray, memoryview)):
                decoded = bytes(value).decode("utf-8", errors="ignore").strip()
                return [decoded] if decoded else []

            if is_dataclass(value):
                try:
                    dataclass_mapping = asdict(value)
                except Exception:  # pragma: no cover - defensive
                    dataclass_mapping = {
                        key: getattr(value, key)
                        for key in getattr(value, "__dataclass_fields__", {})
                    }
                return list(iter_parts(dataclass_mapping, depth + 1))

            for attr_name in ("model_dump", "dict", "to_dict", "as_dict", "asdict"):
                attr = getattr(value, attr_name, None)
                if not callable(attr):
                    continue
                try:
                    mapping_like = attr()
                except TypeError:
                    try:
                        mapping_like = attr(exclude_none=False)
                    except TypeError:
                        continue
                    except Exception:  # pragma: no cover - defensive
                        continue
                except Exception:  # pragma: no cover - defensive
                    continue

                if isinstance(mapping_like, (dict, list, tuple, set, frozenset)):
                    return list(iter_parts(mapping_like, depth + 1))
                if isinstance(mapping_like, str):
                    return list(iter_parts(mapping_like, depth + 1))

            if isinstance(value, Mapping):
                parts: List[str] = []
                role = _as_role(_mapping_get(value, "role")) if "role" in value else ""

                messages = _mapping_get(value, "messages") if "messages" in value else None
                if isinstance(messages, (list, tuple, set, frozenset, Sequence)) and not isinstance(
                    messages, (str, bytes, bytearray, memoryview)
                ):
                    user_parts: List[str] = []
                    other_parts: List[str] = []
                    for message in messages:
                        msg_parts = list(iter_parts(message, depth + 1))
                        msg_role = ""
                        if isinstance(message, Mapping) and "role" in message:
                            msg_role = _as_role(_mapping_get(message, "role"))
                        if msg_role in preferred_roles:
                            user_parts.extend(msg_parts)
                        else:
                            other_parts.extend(msg_parts)
                    if user_parts:
                        return user_parts
                    if other_parts:
                        parts.extend(other_parts)

                choices = _mapping_get(value, "choices") if "choices" in value else None
                if isinstance(choices, (list, tuple, set, frozenset, Sequence)) and not isinstance(
                    choices, (str, bytes, bytearray, memoryview)
                ):
                    for choice in choices:
                        parts.extend(iter_parts(choice, depth + 1))

                content = _mapping_get(value, "content") if "content" in value else None
                if isinstance(content, list):
                    for item in content:
                        parts.extend(iter_parts(item, depth + 1))
                elif isinstance(content, dict):
                    parts.extend(iter_parts(content, depth + 1))

                priority_keys = (
                    "text",
                    "message",
                    "content",
                    "value",
                    "question",
                    "prompt",
                    "input",
                    "body",
                    "delta",
                    "messages",
                    "response",
                    "output",
                    "outputs",
                    "result",
                    "arguments",
                    "args",
                    "data",
                    "payload",
                    "details",
                    "tool_calls",
                    "function_call",
                    "function",
                    "parameters",
                    "params",
                    "inputs",
                    "parts",
                    "segments",
                    "items",
                    "input_text",
                    "output_text",
                    "instruction",
                    "instructions",
                    "query",
                    "request",
                    "task",
                )

                if role in preferred_roles:
                    for key in priority_keys:
                        if key in value:
                            candidate = list(iter_parts(value[key], depth + 1))
                            if candidate:
                                return candidate

                if role and role not in preferred_roles:
                    if role in assistant_like_roles:
                        filtered: List[str] = []
                        for key, item in value.items():
                            if key in {"role", "name"}:
                                continue
                            filtered.extend(iter_parts(item, depth + 1))
                        if filtered:
                            return filtered
                    nested_user_parts: List[str] = []
                    for key, item in value.items():
                        if key in {"role", "name"}:
                            continue
                        nested_user_parts.extend(iter_parts(item, depth + 1))
                    return nested_user_parts

                for key in priority_keys:
                    if key in value:
                        parts.extend(iter_parts(value[key], depth + 1))
                if parts:
                    return parts

                for item in value.values():
                    parts.extend(iter_parts(item, depth + 1))
                return parts

            attribute_priority = (
                "content",
                "message",
                "messages",
                "prompt",
                "question",
                "text",
                "value",
                "input",
                "body",
                "payload",
                "data",
                "details",
                "arguments",
                "args",
                "delta",
                "output",
                "outputs",
                "instruction",
                "instructions",
                "query",
                "request",
                "task",
            )

            collected_parts: List[str] = []
            for attr_name in attribute_priority:
                if hasattr(value, attr_name):
                    collected_parts.extend(iter_parts(getattr(value, attr_name), depth + 1))

            if hasattr(value, "__dict__"):
                mapping_like = getattr(value, "__dict__")
                if isinstance(mapping_like, dict):
                    collected_parts.extend(iter_parts(mapping_like, depth + 1))

            if collected_parts:
                return collected_parts

            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, memoryview)
            ):
                parts: List[str] = []
                for item in value:
                    parts.extend(iter_parts(item, depth + 1))
                return parts

            if isinstance(value, (set, frozenset)):
                parts: List[str] = []
                for item in value:
                    parts.extend(iter_parts(item, depth + 1))
                return parts

            try:
                text = str(value).strip()
            except Exception:  # pragma: no cover - defensive
                return []
            return [text] if text else []

        parts = []
        for part in iter_parts(question):
            if part:
                parts.append(part)

        unique_parts = list(dict.fromkeys(parts))
        return " ".join(unique_parts)

    def answer(self, question: object | None = None) -> str:
        prompt = self._normalise_question(question)
        if not prompt:
            return "Спросите меня о прибыли, риске или плане действий, и я объясню простыми словами."

        brief = self.generate_brief()
        portfolio = self.portfolio_overview()

        if self._contains_any(prompt, ["прибыл", "profit", "доход", "pnl"]):
            return self._format_profit_answer(portfolio)

        if self._contains_any(prompt, ["риск", "risk", "потер", "loss"]):
            return self.risk_summary()

        if self._contains_any(prompt, ["обнов", "свеж", "давно", "last update", "обновил", "age"]):
            summary = self.status_summary()
            return self._format_update_answer(summary)

        if self._contains_any(prompt, ["сигнал", "вероят", "threshold", "порог", "ev", "бп", "б.п."]):
            summary = self.status_summary()
            return self._format_signal_quality_answer(summary)

        if self._contains_any(prompt, ["план", "что делать", "как начать", "инструкц"]):
            return self._format_plan_answer(brief)

        if self._contains_any(prompt, ["экспоз", "загруж", "капитал", "резерв", "exposure", "занято"]):
            return self._format_exposure_answer(portfolio)

        if self._contains_any(prompt, ["настрой", "config", "параметр", "режим работы", "ограничен", "лимит"]):
            return self._format_settings_answer()

        if self._contains_any(prompt, ["портф", "позици", "баланс", "актив", "hold"]):
            return self._format_positions_answer(portfolio)

        if self._contains_any(prompt, ["воч", "watch", "наблюд", "лист", "монитор"]):
            return self._format_watchlist_answer()

        if self._contains_any(prompt, ["здоров", "health", "жив", "данн", "статус"]):
            return self._format_health_answer()

        if self._contains_any(prompt, ["комисс", "fee", "fees", "maker", "taker", "активн"]):
            stats = self.trade_statistics()
            return self._format_fee_activity_answer(stats)

        if self._contains_any(prompt, ["ручн", "manual", "оператор", "вручн", "start", "stop"]):
            automation_block = self.data_health().get("automation")
            return self._format_automation_answer(automation_block)

        if self._contains_any(prompt, ["сделк", "trade", "истор", "журнал", "execut"]):
            trades = self.recent_trades(limit=5)
            stats = self.trade_statistics()
            return self._format_trade_history_answer(trades, stats)

        if self._contains_any(prompt, ["почему", "объясн", "анализ", "что видит", "поясн"]):
            return self.market_story(brief)

        if self._contains_any(prompt, ["куп", "buy", "long"]):
            if brief.mode == "buy":
                return "\n".join([brief.action_text, brief.caution, brief.confidence_text])
            return (
                "Сейчас входить в покупку рано. "
                "Ждём, пока вероятность и выгода станут выше безопасных порогов."
            )

        if self._contains_any(prompt, ["прод", "sell", "short"]):
            if brief.mode == "sell":
                return "\n".join([brief.action_text, brief.caution, brief.confidence_text])
            return "Пока бот не видит сигнала на выход. Держим позицию под контролем и наблюдаем."

        if self._contains_any(prompt, ["привет", "hello", "hi"]):
            return self.initial_message()

        return self._default_response(brief, portfolio)

    def market_story(self, brief: Optional[GuardianBrief] = None) -> str:
        snapshot = self._get_snapshot()
        brief = brief or snapshot.brief
        status = snapshot.status
        narrative = self._narrative_from_status(status, brief.mode, brief.symbol)
        staleness = self.staleness_alert(brief)
        if staleness:
            return "\n\n".join([narrative, staleness])
        return narrative

    def staleness_alert(self, brief: Optional[GuardianBrief] = None) -> Optional[str]:
        brief = brief or self.generate_brief()
        age = brief.status_age
        if age is None:
            return None
        if age >= STALE_SIGNAL_SECONDS:
            return (
                "Сигнал старше 15 минут. Без свежих данных лучше не открывать новые сделки и проверить подключение."
            )
        if age >= WARNING_SIGNAL_SECONDS:
            return "Данные не обновлялись более 5 минут. Убедитесь, что бот подключён и обновляет сигнал."
        return None
