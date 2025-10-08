"""Symbol resolution helpers with multi-quote support.

This module centralises the logic for turning human friendly trading pairs
into the concrete Bybit instrument symbols.  The resolver keeps a lightweight
in-memory index that maps ``(base, quote)`` tuples to the raw instrument
metadata returned by the ``/v5/market/instruments-info`` endpoint.  The
metadata snapshot is refreshed when the resolver is instantiated and can be
manually refreshed later if the catalogue changes during runtime.

The resolver understands common naming conventions used on the testnet where
synthetic assets often have prefixes such as ``BB`` (e.g. ``BBSOL``).  The
index therefore stores several aliases per instrument so that providing
``SOL/USDT`` resolves to the canonical testnet symbol ``BBSOLUSDT``.  Each
instrument entry exposes the essential trading parameters (tick size, quantity
step, minimum order size and notional) so that other components can build
orders without making extra HTTP calls.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .bybit_api import BybitAPI

COMMON_QUOTES: Tuple[str, ...] = (
    "USDT",
    "USDC",
    "USD",
    "USDD",
    "DAI",
    "BUSD",
    "BTC",
    "ETH",
    "EUR",
    "JPY",
)

_SYMBOL_NORMALISE_SEPARATORS: Tuple[str, ...] = (" ", "-", "_", "/", ":")
_SYMBOL_SPLIT_SEPARATORS: Tuple[str, ...] = ("-", "_", "/", ":")
_SORTED_COMMON_QUOTES: Tuple[str, ...] = tuple(
    sorted(COMMON_QUOTES, key=len, reverse=True)
)


@dataclass(frozen=True)
class InstrumentMetadata:
    """Normalised instrument parameters extracted from Bybit listings."""

    symbol: str
    base: str
    quote: str
    tick_size: Optional[Decimal]
    qty_step: Optional[Decimal]
    min_qty: Optional[Decimal]
    min_notional: Optional[Decimal]
    alias: Optional[str]
    base_synonyms: Tuple[str, ...]
    raw: Mapping[str, object]

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable representation of the metadata."""

        def _fmt(value: Optional[Decimal]) -> Optional[str]:
            if value is None:
                return None
            normalised = format(value.normalize(), "f")
            if "." in normalised:
                normalised = normalised.rstrip("0").rstrip(".")
            return normalised or "0"

        return {
            "symbol": self.symbol,
            "base": self.base,
            "quote": self.quote,
            "tick_size": _fmt(self.tick_size),
            "qty_step": _fmt(self.qty_step),
            "min_qty": _fmt(self.min_qty),
            "min_notional": _fmt(self.min_notional),
            "alias": self.alias,
            "base_synonyms": list(self.base_synonyms),
        }


class SymbolResolver:
    """Resolve human friendly trading pairs to Bybit instrument metadata."""

    def __init__(
        self,
        api: Optional[BybitAPI],
        *,
        category: str = "spot",
        refresh: bool = True,
        bootstrap_rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> None:
        self.api = api
        self.category = category
        self._index: Dict[Tuple[str, str], InstrumentMetadata] = {}
        self._by_symbol: Dict[str, InstrumentMetadata] = {}
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()

        if bootstrap_rows is not None:
            self._build_index(bootstrap_rows)
        elif refresh and api is not None:
            self.refresh()

    # ------------------------------------------------------------------
    # public helpers
    def refresh(self) -> None:
        """Fetch the latest instrument catalogue from the Bybit API."""

        if self.api is None:
            raise RuntimeError("API client not configured for SymbolResolver")

        response = self.api.instruments_info(category=self.category)
        rows = _extract_rows(response)
        self._build_index(rows)

    @property
    def last_refresh(self) -> float:
        return self._last_refresh

    @property
    def is_ready(self) -> bool:
        return bool(self._index)

    def resolve(self, base: object, quote: object) -> Optional[InstrumentMetadata]:
        """Return metadata for ``base``/``quote`` if the pair is listed."""

        base_key = _normalise_asset(base)
        quote_key = _normalise_asset(quote)
        if not base_key or not quote_key:
            return None
        with self._lock:
            return self._index.get((base_key, quote_key))

    def resolve_symbol(
        self,
        symbol: object,
        *,
        default_quote: Optional[str] = "USDT",
    ) -> Optional[InstrumentMetadata]:
        """Resolve a combined symbol string (``"SOLUSDT"``, ``"SOL/USDC"``, ...)."""

        cleaned = _clean_symbol(symbol)
        if not cleaned:
            return None

        with self._lock:
            direct = self._by_symbol.get(cleaned)
            if direct is not None:
                return direct

        if default_quote:
            alias_hit = self.resolve(cleaned, default_quote)
            if alias_hit is not None:
                return alias_hit

        base, quote = _split_symbol(cleaned)
        if quote is None:
            quote = default_quote
        if base is None or quote is None:
            return None
        return self.resolve(base, quote)

    def metadata(self, symbol: object) -> Optional[InstrumentMetadata]:
        """Return cached metadata by canonical Bybit symbol."""

        cleaned = _clean_symbol(symbol)
        if not cleaned:
            return None
        with self._lock:
            return self._by_symbol.get(cleaned)

    def all_metadata(self) -> Tuple[InstrumentMetadata, ...]:
        """Return a snapshot of all cached instruments."""

        with self._lock:
            return tuple(self._by_symbol.values())

    # ------------------------------------------------------------------
    # internal helpers
    def _build_index(self, rows: Sequence[Mapping[str, object]]) -> None:
        index: Dict[Tuple[str, str], InstrumentMetadata] = {}
        by_symbol: Dict[str, InstrumentMetadata] = {}

        for row in rows:
            metadata = _parse_instrument_row(row)
            if metadata is None:
                continue
            by_symbol[metadata.symbol] = metadata

            canonical_key = (metadata.base, metadata.quote)
            index[canonical_key] = metadata

            for base_alias in metadata.base_synonyms:
                if base_alias == metadata.base:
                    continue
                key = (base_alias, metadata.quote)
                # Avoid overriding the canonical mapping; aliases fill in blanks only
                index.setdefault(key, metadata)

        with self._lock:
            self._index = index
            self._by_symbol = by_symbol
            self._last_refresh = time.time()


# ----------------------------------------------------------------------
# utility functions


def _extract_rows(payload: Mapping[str, object] | Sequence[Mapping[str, object]]) -> Sequence[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, Mapping):
            rows = result.get("list")
            if isinstance(rows, Sequence):
                return list(rows)  # type: ignore[return-value]
        rows = payload.get("list")
        if isinstance(rows, Sequence):
            return list(rows)  # type: ignore[return-value]
        return []

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return list(payload)
    return []


def _normalise_asset(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _clean_symbol(value: object) -> Optional[str]:
    text = _normalise_asset(value)
    if not text:
        return None
    for separator in _SYMBOL_NORMALISE_SEPARATORS:
        if separator in text:
            text = text.replace(separator, "")
    return text or None


def _split_symbol(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    for separator in _SYMBOL_SPLIT_SEPARATORS:
        if separator in symbol:
            base, quote = symbol.split(separator, 1)
            base = base.strip().upper() or None
            quote = quote.strip().upper() or None
            return base, quote

    for quote in _SORTED_COMMON_QUOTES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            return base or None, quote

    return symbol or None, None


def _to_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return Decimal(stripped)
        except (InvalidOperation, ValueError):
            return None
    return None


def _parse_instrument_row(row: Mapping[str, object]) -> Optional[InstrumentMetadata]:
    if not isinstance(row, Mapping):
        return None

    symbol = _clean_symbol(row.get("symbol"))
    if not symbol:
        return None

    status = str(row.get("status") or "").strip().lower()
    if status and status not in {"trading", "listed"}:
        return None

    base = _normalise_asset(row.get("baseCoin"))
    quote = _normalise_asset(row.get("quoteCoin"))
    if not base or not quote:
        derived_base, derived_quote = _split_symbol(symbol)
        base = base or derived_base
        quote = quote or derived_quote

    if not base or not quote:
        return None

    alias = _normalise_asset(row.get("alias"))

    lot_filter = row.get("lotSizeFilter")
    if not isinstance(lot_filter, Mapping):
        lot_filter = {}
    price_filter = row.get("priceFilter")
    if not isinstance(price_filter, Mapping):
        price_filter = {}

    tick_size = _to_decimal(price_filter.get("tickSize"))
    if tick_size is None:
        tick_size = _to_decimal(lot_filter.get("tickSize"))

    qty_step = (
        _to_decimal(lot_filter.get("minOrderQtyIncrement"))
        or _to_decimal(lot_filter.get("qtyStep"))
        or _to_decimal(lot_filter.get("basePrecision"))
    )

    min_qty = _to_decimal(lot_filter.get("minOrderQty"))
    if min_qty is None:
        # Some payloads use "minOrderAmt" for the base requirement
        min_qty = _to_decimal(lot_filter.get("minOrderAmt"))

    min_notional = _to_decimal(lot_filter.get("minOrderAmt"))

    base_synonyms = _build_base_synonyms(symbol, base, alias)

    return InstrumentMetadata(
        symbol=symbol,
        base=base,
        quote=quote,
        tick_size=tick_size,
        qty_step=qty_step,
        min_qty=min_qty,
        min_notional=min_notional,
        alias=alias,
        base_synonyms=base_synonyms,
        raw=row,
    )


def _build_base_synonyms(symbol: str, base: str, alias: Optional[str]) -> Tuple[str, ...]:
    synonyms = {base}

    if alias:
        synonyms.add(alias)

    # Add the base derived from the symbol (e.g. BTC from BTCUSDT)
    derived_base, derived_quote = _split_symbol(symbol)
    if derived_base:
        synonyms.add(derived_base)

    # Heuristics for testnet prefixes (BBSOL -> SOL, TESTBTC -> BTC, etc.)
    for prefix in ("BB", "TEST", "T"):
        if base.startswith(prefix) and len(base) > len(prefix) + 1:
            synonyms.add(base[len(prefix) :])

    # Remove empty entries and normalise case
    cleaned = {_normalise_asset(item) for item in synonyms if _normalise_asset(item)}
    ordered = sorted(cleaned, key=lambda item: (len(item), item))
    return tuple(ordered)

