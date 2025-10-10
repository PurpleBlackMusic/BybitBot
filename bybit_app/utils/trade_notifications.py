from __future__ import annotations

from typing import Optional


def format_sell_close_message(
    *,
    symbol: str,
    qty_text: str,
    base_asset: str,
    price_text: str,
    pnl_text: str,
    sold_text: Optional[str] = None,
    remainder_text: Optional[str] = None,
    position_closed: bool = False,
) -> str:
    """Собирает сообщение о продаже/закрытии позиции в едином формате."""

    message = (
        f"🔴 {symbol}: закрытие {qty_text} {base_asset} по {price_text}, "
        f"PnL сделки {pnl_text}"
    )

    details: list[str] = []
    if sold_text:
        details.append(f"продано: {sold_text}")
    if remainder_text:
        details.append(f"осталось: {remainder_text}")
    elif position_closed:
        details.append("позиция закрыта")

    if details:
        message += " (" + "; ".join(details) + ")"

    return message

