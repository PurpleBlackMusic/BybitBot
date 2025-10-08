"""Symbol normalisation helpers for Bybit spot trading."""

from __future__ import annotations

from typing import Optional, Tuple

_KNOWN_QUOTES: Tuple[str, ...] = (
    "USDT",
    "USDC",
    "USDD",
    "BUSD",
    "DAI",
    "USD",
    "EUR",
    "BTC",
    "ETH",
    "JPY",
)


def _clean_symbol_text(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if not text:
        return ""
    for separator in (" ", "-", "_", "/", ":"):
        if separator in text:
            text = text.replace(separator, "")
    return text.strip()


def ensure_usdt_symbol(symbol: object) -> Tuple[Optional[str], Optional[str]]:
    """Return a USDT-quoted symbol or ``None`` if unsupported.

    The helper maps USDC quotes to their USDT counterpart and rejects
    instruments that settle in any other currency. When only the base asset
    is provided (e.g. ``"BTC"``) the function assumes a USDT quote and
    returns ``"BTCUSDT"``. The second element of the tuple describes the
    original quote: ``"USDC"`` when a conversion happened, ``"BASE"`` when
    the quote was implied from the base asset only and ``None`` otherwise.
    """

    cleaned = _clean_symbol_text(symbol)
    if not cleaned or cleaned == "?":
        return None, None

    for quote in _KNOWN_QUOTES:
        if cleaned.endswith(quote):
            base = cleaned[: -len(quote)] or ""
            if not base:
                return None, quote
            if quote == "USDT":
                return f"{base}USDT", None
            if quote == "USDC":
                return f"{base}USDT", "USDC"
            return None, quote

    # no explicit quote — fall back to USDT only when a base asset exists
    base = cleaned
    if not base:
        return None, None
    return f"{base}USDT", "BASE"

