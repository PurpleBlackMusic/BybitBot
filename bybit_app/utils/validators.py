from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Mapping, Sequence


__all__ = ["SpotValidationResult", "validate_spot_rules"]


@dataclass(frozen=True)
class SpotValidationResult:
    """Result of applying exchange spot trading rules to a price/quantity pair."""

    price: Decimal
    qty: Decimal
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def notional(self) -> Decimal:
        return self.price * self.qty

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable view compatible with older callers."""

        price_text = _format_decimal(self.price, self.tick_size)
        qty_text = _format_decimal(self.qty, self.qty_step)

        return {
            "ok": self.ok,
            "price": price_text,
            "qty": qty_text,
            "price_q": price_text,
            "qty_q": qty_text,
            "tick_size": self.tick_size,
            "qty_step": self.qty_step,
            "min_qty": self.min_qty,
            "min_notional": self.min_notional,
            "reasons": list(self.reasons),
        }


def _to_decimal(value: object, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} value for spot validation: {value!r}") from exc


def _instrument_decimal(
    instrument: Mapping[str, object],
    paths: Sequence[Sequence[str]],
    *,
    default: str = "0",
) -> Decimal:
    for path in paths:
        current: object = instrument
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        try:
            return Decimal(str(current))
        except (InvalidOperation, TypeError, ValueError):
            continue
    return Decimal(default)


def _quantize(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    try:
        return value.quantize(step, rounding=ROUND_DOWN)
    except InvalidOperation:
        exponent = step.normalize().as_tuple().exponent
        quantum = Decimal("1").scaleb(exponent)
        return value.quantize(quantum, rounding=ROUND_DOWN)


def _format_decimal(value: Decimal, step: Decimal) -> str:
    if step > 0:
        try:
            quantised = value.quantize(step, rounding=ROUND_DOWN)
        except InvalidOperation:
            exponent = step.normalize().as_tuple().exponent
            quantum = Decimal("1").scaleb(exponent)
            quantised = value.quantize(quantum, rounding=ROUND_DOWN)
        exponent = step.normalize().as_tuple().exponent
        places = abs(exponent) if exponent < 0 else 0
    else:
        quantised = value
        exponent = quantised.normalize().as_tuple().exponent
        places = abs(exponent) if exponent < 0 else 0

    if places > 0:
        text = f"{quantised:.{places}f}"
    else:
        integral = quantised.to_integral_value(rounding=ROUND_DOWN)
        if quantised == integral:
            text = format(integral, "f")
        else:
            text = format(quantised.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def validate_spot_rules(*args, **kwargs) -> SpotValidationResult:
    """Quantise price and quantity according to instrument spot trading rules."""

    instrument: Mapping[str, object] | None = None
    price = kwargs.get("price")
    qty = kwargs.get("qty")

    if len(args) == 1 and isinstance(args[0], Mapping):
        instrument = args[0]
    elif len(args) >= 6:
        price = args[3]
        qty = args[4]
        candidate = args[5]
        if isinstance(candidate, Mapping):
            instrument = candidate
    else:
        candidate = kwargs.get("instrument")
        if isinstance(candidate, Mapping):
            instrument = candidate

    if instrument is None:
        raise ValueError("instrument is required for validation")

    if price is None or qty is None:
        raise ValueError("price and qty are required for validation")

    price_decimal = _to_decimal(price, field="price")
    qty_decimal = _to_decimal(qty, field="qty")

    tick_size = _instrument_decimal(
        instrument,
        ( ("priceFilter", "tickSize"), ("tickSize",), ),
        default="0.00000001",
    )
    qty_step = _instrument_decimal(
        instrument,
        ( ("lotSizeFilter", "qtyStep"), ("qtyStep",), ("lotSize",), ),
        default="0.00000001",
    )
    min_qty = _instrument_decimal(
        instrument,
        (
            ("lotSizeFilter", "minOrderQty"),
            ("lotSizeFilter", "minQty"),
            ("minOrderQty",),
            ("minQty",),
        ),
        default="0",
    )
    min_notional = _instrument_decimal(
        instrument,
        (
            ("lotSizeFilter", "minOrderAmt"),
            ("lotSizeFilter", "minNotional"),
            ("minOrderAmt",),
            ("minNotional",),
        ),
        default="0",
    )

    price_q = _quantize(price_decimal, tick_size)
    qty_q = _quantize(qty_decimal, qty_step)

    reasons: list[str] = []
    notional = price_q * qty_q

    tolerance_components = [Decimal("0.00000001")]
    if tick_size > 0 and qty_q > 0:
        tolerance_components.append((tick_size * qty_q).copy_abs())
    if price_q > 0 and qty_step > 0:
        tolerance_components.append((price_q * qty_step).copy_abs())
    if min_notional > 0:
        tolerance_components.append((min_notional * Decimal("0.000001")).copy_abs())
    notional_tolerance = max(tolerance_components)

    if min_notional > 0:
        gap = min_notional - notional
        if gap > notional_tolerance:
            reasons.append(
                f"notional {notional:.12f} < minNotional {min_notional:.12f}"
            )

    if min_qty > 0:
        qty_tolerance = qty_step if qty_step > 0 else Decimal("0.00000001")
        qty_gap = min_qty - qty_q
        if qty_gap > qty_tolerance:
            reasons.append(f"qty {qty_q:.12f} < minQty {min_qty:.12f}")

    return SpotValidationResult(
        price=price_q,
        qty=qty_q,
        tick_size=tick_size,
        qty_step=qty_step,
        min_qty=min_qty,
        min_notional=min_notional,
        reasons=tuple(reasons),
    )
