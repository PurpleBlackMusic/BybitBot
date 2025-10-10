from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from .bybit_api import BybitAPI
from .log import log
from .paths import DATA_DIR
from .market_features import build_feature_bundle
from .symbols import ensure_usdt_symbol
from .telegram_notify import send_telegram

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .portfolio_manager import PortfolioManager
    from .symbol_resolver import InstrumentMetadata, SymbolResolver
    from .envs import Settings

SNAPSHOT_FILENAME = "market_snapshot.json"
DEFAULT_CACHE_TTL = 300.0


class MarketScannerError(RuntimeError):
    """Raised when the market snapshot cannot be loaded."""


def _network_snapshot_filename(
    *, testnet: Optional[bool] = None, settings: Optional["Settings"] = None
) -> str:
    """Return the snapshot filename matching the active network."""

    resolved = testnet
    if resolved is None and settings is not None:
        try:
            resolved = bool(getattr(settings, "testnet"))
        except Exception:
            resolved = None

    if resolved:
        if "." in SNAPSHOT_FILENAME:
            stem, ext = SNAPSHOT_FILENAME.rsplit(".", 1)
            return f"{stem}_testnet.{ext}"
        return f"{SNAPSHOT_FILENAME}_testnet"
    return SNAPSHOT_FILENAME


def _snapshot_path(
    data_dir: Path, *, testnet: Optional[bool] = None, settings: Optional["Settings"] = None
) -> Path:
    filename = _network_snapshot_filename(testnet=testnet, settings=settings)
    return Path(data_dir) / "ai" / filename


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
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


def _normalise_percent(value: object) -> Optional[float]:
    pct = _safe_float(value)
    if pct is None:
        return None
    if abs(pct) <= 1.0:
        pct *= 100.0
    return pct


def _spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or ask <= 0:
        return None
    spread = (ask - bid) / ask * 10000.0
    if spread < 0:
        spread = 0.0
    return spread


def _strength_from_change(change_pct: Optional[float]) -> float:
    if change_pct is None:
        return 0.0
    # smooth growth curve that approaches 1 for very strong moves
    return math.tanh(abs(change_pct) / 5.0)


def _probability_from_change(change_pct: Optional[float], trend: str) -> Optional[float]:
    if change_pct is None:
        return None
    strength = _strength_from_change(change_pct)
    if trend == "buy":
        return 0.5 + strength / 2.0
    if trend == "sell":
        return 0.5 - strength / 2.0
    return 0.5


def _score_turnover(turnover: Optional[float]) -> float:
    if turnover is None or turnover <= 0:
        return 0.0
    return math.log10(turnover + 1.0)


def _edge_score(
    turnover: Optional[float],
    change_pct: Optional[float],
    spread_bps: Optional[float],
    *,
    boost: bool = False,
) -> float:
    strength = _strength_from_change(change_pct)
    liquidity = _score_turnover(turnover)
    if spread_bps is not None and spread_bps > 0:
        penalty = math.sqrt(spread_bps)
    else:
        penalty = 1.0
    score = 0.0
    if penalty > 0:
        score = (liquidity * strength) / penalty
    if boost:
        score += 5.0
    return score


def load_market_snapshot(
    data_dir: Path = DATA_DIR,
    *,
    testnet: Optional[bool] = None,
    settings: Optional["Settings"] = None,
) -> Optional[Dict[str, object]]:
    path = _snapshot_path(data_dir, testnet=testnet, settings=settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_market_snapshot(
    snapshot: Dict[str, object],
    data_dir: Path = DATA_DIR,
    *,
    testnet: Optional[bool] = None,
    settings: Optional["Settings"] = None,
) -> None:
    path = _snapshot_path(data_dir, testnet=testnet, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_market_snapshot(api: BybitAPI, category: str = "spot") -> Dict[str, object]:
    response = api.tickers(category=category)
    rows: List[Dict[str, object]] = []
    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, dict):
            rows = result.get("list") or []  # type: ignore[assignment]
        elif isinstance(response.get("list"), list):
            rows = response.get("list")  # type: ignore[assignment]
    snapshot = {
        "ts": time.time(),
        "category": category,
        "rows": rows,
    }
    return snapshot


def scan_market_opportunities(
    api: Optional[BybitAPI],
    *,
    data_dir: Path = DATA_DIR,
    limit: int = 25,
    min_turnover: float = 1_000_000.0,
    min_change_pct: float = 0.5,
    max_spread_bps: float = 60.0,
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
    cache_ttl: float = DEFAULT_CACHE_TTL,
    settings: Optional["Settings"] = None,
    testnet: Optional[bool] = None,
) -> List[Dict[str, object]]:
    """Rank spot symbols by liquidity and momentum to surface opportunities."""

    min_turnover = max(0.0, float(min_turnover))
    effective_change = float(min_change_pct) if min_change_pct is not None else 0.5
    if effective_change < 0.05:
        effective_change = 0.05
    max_spread_bps = float(max_spread_bps)

    snapshot = load_market_snapshot(data_dir, settings=settings, testnet=testnet)
    now = time.time()
    if snapshot is not None and cache_ttl is not None and cache_ttl >= 0:
        ts = _safe_float(snapshot.get("ts"))
        if ts is not None and now - ts > cache_ttl:
            snapshot = None

    if snapshot is None and api is not None:
        try:
            snapshot = fetch_market_snapshot(api)
        except Exception:
            snapshot = None
        else:
            save_market_snapshot(
                snapshot,
                data_dir=data_dir,
                settings=settings,
                testnet=testnet,
            )

    if snapshot is None:
        raise MarketScannerError(
            "Рыночный снапшот недоступен — проверьте подключение к API."
        )

    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        result = snapshot.get("result")
        if isinstance(result, dict):
            rows = result.get("list")  # type: ignore[assignment]
        else:
            rows = []

    def _normalise_symbol_set(symbols: Iterable[object]) -> Set[str]:
        normalised: Set[str] = set()
        for item in symbols:
            candidate, _ = ensure_usdt_symbol(item)
            if candidate:
                normalised.add(candidate)
        return normalised

    entries: List[Dict[str, object]] = []
    wset = _normalise_symbol_set(whitelist or ())
    bset = _normalise_symbol_set(blacklist or ())

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        raw_symbol = raw.get("symbol")
        symbol, quote_source = ensure_usdt_symbol(raw_symbol)
        if not symbol:
            continue
        upper_source = str(raw_symbol).strip().upper() if isinstance(raw_symbol, str) else None
        if bset and symbol in bset:
            continue

        turnover = _safe_float(raw.get("turnover24h"))
        change_pct = _normalise_percent(raw.get("price24hPcnt"))
        volume = _safe_float(raw.get("volume24h"))
        bid = _safe_float(raw.get("bestBidPrice"))
        ask = _safe_float(raw.get("bestAskPrice"))
        spread_bps = _spread_bps(bid, ask)
        feature_bundle = build_feature_bundle(raw)
        blended_change = feature_bundle.get("blended_change_pct")
        volatility_pct = feature_bundle.get("volatility_pct")
        volume_spike_score = feature_bundle.get("volume_spike_score")
        depth_imbalance = feature_bundle.get("depth_imbalance")

        force_include = symbol in wset
        if not force_include and turnover is not None and turnover < min_turnover:
            turnover_ok = False
        else:
            turnover_ok = True

        if change_pct is None:
            trend = "wait"
        elif change_pct >= effective_change:
            trend = "buy"
        elif change_pct <= -effective_change:
            trend = "sell"
        else:
            trend = "wait"

        actionable = False
        spread_ok = True
        if spread_bps is not None and max_spread_bps > 0:
            spread_ok = spread_bps <= max_spread_bps

        if trend in {"buy", "sell"}:
            change_ok = change_pct is not None and abs(change_pct) >= effective_change
            actionable = turnover_ok and spread_ok and change_ok

        if not actionable and not force_include:
            strength = _strength_from_change(change_pct)
            if strength < 0.1:
                continue

        probability = _probability_from_change(change_pct, trend)
        ev_bps = change_pct * 100.0 if change_pct is not None else None
        score = _edge_score(turnover, change_pct, spread_bps, boost=force_include)

        direction = 0
        if trend == "buy":
            direction = 1
        elif trend == "sell":
            direction = -1

        if probability is not None:
            if blended_change is not None and direction != 0:
                probability += 0.1 * direction * math.tanh(blended_change / 10.0)
            if volume_spike_score is not None and direction != 0:
                probability += 0.05 * direction * math.tanh(volume_spike_score)
            if volatility_pct is not None:
                probability -= 0.05 * math.tanh(volatility_pct / 50.0)
            probability = max(0.0, min(1.0, probability))

        feature_score = 0.0
        if blended_change is not None:
            feature_score += abs(blended_change) * 0.05
        if volume_spike_score is not None and volume_spike_score > 0:
            feature_score += volume_spike_score * 0.5
        if depth_imbalance is not None and direction != 0:
            feature_score += depth_imbalance * direction * 2.0
        if volatility_pct is not None:
            feature_score -= math.log1p(max(0.0, volatility_pct)) * 0.1

        score += feature_score

        note_parts: List[str] = []
        if change_pct is not None:
            note_parts.append(f"24ч {change_pct:+.2f}%")
        if turnover is not None and turnover > 0:
            note_parts.append(f"оборот ${turnover / 1_000_000:.2f}M")
        if spread_bps is not None:
            note_parts.append(f"спред {spread_bps:.1f} б.п.")
        if quote_source == "USDC":
            note_parts.append("конвертировано из USDC")
        if volatility_pct is not None:
            note_parts.append(f"волатильность {volatility_pct:.1f}%")
        if volume_spike_score is not None and volume_spike_score > 0:
            note_parts.append(f"всплеск объёма ×{math.exp(volume_spike_score):.2f}")
        if depth_imbalance is not None and direction != 0:
            imbalance_pct = depth_imbalance * 100.0
            side = "покупателей" if imbalance_pct > 0 else "продавцов"
            note_parts.append(f"преимущество {side} {abs(imbalance_pct):.1f}%")

        entry = {
            "symbol": symbol,
            "trend": trend,
            "probability": probability,
            "ev_bps": ev_bps,
            "score": score,
            "note": ", ".join(note_parts) or None,
            "turnover_usd": turnover,
            "change_pct": change_pct,
            "spread_bps": spread_bps,
            "volume": volume,
            "volatility_pct": volatility_pct,
            "volume_spike_score": volume_spike_score,
            "depth_imbalance": depth_imbalance,
            "blended_change_pct": blended_change,
            "source": "market_scanner",
            "actionable": actionable,
        }
        if quote_source == "USDC" and upper_source:
            entry["quote_conversion"] = {"from": "USDC", "to": "USDT", "original": upper_source}
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            0 if item.get("actionable") else 1,
            -(item.get("score") or 0.0),
            -abs(item.get("change_pct") or 0.0),
            item.get("symbol", ""),
        )
    )

    if limit and limit > 0:
        entries = entries[: int(limit)]

    return entries


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_kline_rows(payload: object) -> Sequence[object]:
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, Mapping):
            rows = result.get("list")
            if isinstance(rows, Sequence):
                return rows  # type: ignore[return-value]
        rows = payload.get("list")
        if isinstance(rows, Sequence):
            return rows  # type: ignore[return-value]
    elif isinstance(payload, Sequence):
        return payload
    return []


def _normalise_candles(payload: object) -> List[Dict[str, object]]:
    candles: List[Dict[str, object]] = []
    for entry in _extract_kline_rows(payload):
        start: Optional[int]
        open_: Optional[float]
        high: Optional[float]
        low: Optional[float]
        close: Optional[float]
        volume: Optional[float]
        turnover: Optional[float] = None

        if isinstance(entry, Mapping):
            start = _safe_int(entry.get("start") or entry.get("openTime") or entry.get("timestamp"))
            open_ = _safe_float(entry.get("open"))
            high = _safe_float(entry.get("high"))
            low = _safe_float(entry.get("low"))
            close = _safe_float(entry.get("close"))
            volume = _safe_float(entry.get("volume"))
            turnover = _safe_float(entry.get("turnover"))
        elif isinstance(entry, Sequence):
            sequence = list(entry)
            if not sequence:
                continue
            start = _safe_int(sequence[0])
            open_ = _safe_float(sequence[1]) if len(sequence) > 1 else None
            high = _safe_float(sequence[2]) if len(sequence) > 2 else None
            low = _safe_float(sequence[3]) if len(sequence) > 3 else None
            close = _safe_float(sequence[4]) if len(sequence) > 4 else None
            volume = _safe_float(sequence[5]) if len(sequence) > 5 else None
            turnover = _safe_float(sequence[6]) if len(sequence) > 6 else None
        else:
            continue

        if start is None:
            continue

        record: Dict[str, object] = {"start": start}
        if open_ is not None:
            record["open"] = open_
        if high is not None:
            record["high"] = high
        if low is not None:
            record["low"] = low
        if close is not None:
            record["close"] = close
        if volume is not None:
            record["volume"] = volume
        if turnover is not None:
            record["turnover"] = turnover
        candles.append(record)

    candles.sort(key=lambda row: row.get("start") or 0)
    return candles


class _CandleCache:
    """Small helper that caches recent kline snapshots per symbol."""

    def __init__(
        self,
        api: Optional[BybitAPI],
        *,
        category: str = "spot",
        intervals: Sequence[int] = (1, 5),
        ttl: float = 45.0,
        limit: int = 100,
    ) -> None:
        self.api = api
        self.category = category
        self.intervals = tuple(int(interval) for interval in intervals)
        self.ttl = max(float(ttl), 1.0)
        self.limit = max(int(limit), 1)
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, object]]]] = {}

    def fetch(self, symbol: str, now: float) -> Dict[str, List[Dict[str, object]]]:
        bundle: Dict[str, List[Dict[str, object]]] = {}
        if self.api is None:
            return bundle

        for interval in self.intervals:
            key = (symbol, interval)
            cached = self._cache.get(key)
            if cached is not None:
                ts, candles = cached
                if now - ts <= self.ttl and candles:
                    bundle[f"{interval}m"] = candles
                    continue

            try:
                payload = self.api.kline(
                    category=self.category,
                    symbol=symbol,
                    interval=interval,
                    limit=self.limit,
                )
            except Exception as exc:  # pragma: no cover - network/runtime guard
                log("market_scanner.candles.error", symbol=symbol, interval=interval, err=str(exc))
                continue

            candles = _normalise_candles(payload)
            self._cache[key] = (now, candles)
            bundle[f"{interval}m"] = candles

        return bundle


class MarketScanner:
    """Stateful helper that keeps the multi-asset opportunity universe fresh."""

    def __init__(
        self,
        api: Optional[BybitAPI],
        symbol_resolver: Optional["SymbolResolver"] = None,
        *,
        data_dir: Path = DATA_DIR,
        scanner: Callable[..., List[Dict[str, object]]] = scan_market_opportunities,
        scanner_kwargs: Optional[Mapping[str, object]] = None,
        refresh_interval: Tuple[float, float] = (60.0, 120.0),
        candle_ttl: float = 45.0,
        candle_limit: int = 120,
        mode: str = "breakout",
        portfolio_manager: Optional["PortfolioManager"] = None,
        telegram_sender: Callable[[str], object] = send_telegram,
        category: str = "spot",
    ) -> None:
        self.api = api
        self.symbol_resolver = symbol_resolver
        self.data_dir = Path(data_dir)
        self._scanner = scanner
        self._scanner_kwargs = dict(scanner_kwargs or {})
        self._refresh_interval = (
            max(float(refresh_interval[0]), 5.0),
            max(float(refresh_interval[1]), float(refresh_interval[0])),
        )
        self._next_refresh: float = 0.0
        self._last_update: float = 0.0
        self._top_candidates: List[Dict[str, object]] = []
        self._last_leader_digest: Optional[Tuple[str, ...]] = None
        self.mode = mode
        self.portfolio_manager = portfolio_manager
        self._telegram_sender = telegram_sender
        self._category = category
        self._candle_cache = _CandleCache(
            api,
            category=category,
            intervals=(1, 5),
            ttl=candle_ttl,
            limit=candle_limit,
        )

        # Always ensure the scanner uses the configured data directory
        if "data_dir" not in self._scanner_kwargs:
            self._scanner_kwargs["data_dir"] = self.data_dir

    # ------------------------------------------------------------------
    # public API
    def refresh(self, *, force: bool = False, now: Optional[float] = None) -> List[Dict[str, object]]:
        """Refresh the ranking when the refresh window elapsed."""

        current_ts = now if now is not None else time.time()
        if not force and self._next_refresh and current_ts < self._next_refresh and self._top_candidates:
            return copy.deepcopy(self._top_candidates)

        scan_kwargs = dict(self._scanner_kwargs)
        scan_kwargs.setdefault("data_dir", self.data_dir)
        opportunities = self._scanner(self.api, **scan_kwargs)

        enriched: List[Dict[str, object]] = []
        for entry in opportunities:
            enriched.append(self._enrich_entry(entry, current_ts))

        self._top_candidates = enriched
        self._last_update = current_ts
        self._schedule_next_refresh(current_ts)
        self._notify_leader_change()
        return copy.deepcopy(self._top_candidates)

    def top_candidates(self, limit: int = 5) -> List[Dict[str, object]]:
        entries = self._top_candidates[: max(int(limit), 0)]
        return copy.deepcopy(entries)

    @property
    def last_update(self) -> float:
        return self._last_update

    # ------------------------------------------------------------------
    # internal helpers
    def _schedule_next_refresh(self, now: float) -> None:
        low, high = self._refresh_interval
        if high <= low:
            delay = low
        else:
            delay = random.uniform(low, high)
        self._next_refresh = now + delay

    def _resolve_metadata(self, symbol: str) -> Optional["InstrumentMetadata"]:
        if not self.symbol_resolver:
            return None
        metadata = self.symbol_resolver.metadata(symbol)
        if metadata is None:
            metadata = self.symbol_resolver.resolve_symbol(symbol)
        return metadata

    def _enrich_entry(self, entry: Mapping[str, object], now: float) -> Dict[str, object]:
        enriched = copy.deepcopy(entry)
        symbol = str(entry.get("symbol") or "").strip().upper()
        metadata = self._resolve_metadata(symbol)
        if metadata is not None:
            enriched["instrument"] = metadata.as_dict()
            canonical = metadata.symbol
        else:
            enriched["instrument"] = None
            canonical = symbol

        if canonical:
            enriched["candles"] = self._candle_cache.fetch(canonical, now)
        else:
            enriched["candles"] = {}
        enriched.setdefault("source", "market_scanner")
        return enriched

    def _notify_leader_change(self) -> None:
        if not self._top_candidates:
            return

        symbols = []
        for entry in self._top_candidates[:5]:
            instrument = entry.get("instrument")
            if isinstance(instrument, Mapping):
                base = instrument.get("base")
                symbol = str(base or instrument.get("symbol") or instrument.get("base"))
            else:
                symbol = str(entry.get("symbol") or "")
            if symbol:
                symbols.append(symbol.upper())

        digest = tuple(symbols)
        if not symbols or digest == self._last_leader_digest:
            return

        self._last_leader_digest = digest
        active = 0
        capacity = len(symbols)
        if self.portfolio_manager is not None:
            active = self.portfolio_manager.active_positions
            capacity = self.portfolio_manager.max_positions

        top_line = _format_top_line(symbols)
        message = f"🏁 Scanner: TOP5 → {top_line} | режим {self.mode} | активных: {active}/{capacity}."
        log("market_scanner.leaderboard", top=symbols, mode=self.mode, active=active, capacity=capacity)
        try:
            self._telegram_sender(message)
        except Exception as exc:  # pragma: no cover - safeguard around external IO
            log("market_scanner.telegram.error", err=str(exc))


def _format_top_line(symbols: Sequence[str]) -> str:
    cleaned = [symbol for symbol in symbols if symbol]
    if not cleaned:
        return "—"
    preview = cleaned[:3]
    body = ", ".join(preview)
    if len(cleaned) > len(preview):
        body = f"{body}, …"
    return body
