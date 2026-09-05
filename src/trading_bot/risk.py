"""Position sizing and order-size constraints."""

from __future__ import annotations

import math


class RiskManager:
    """Computes entry order sizes subject to capital and lot constraints.

    Attributes:
        position_size_pct: fraction of equity allocated to a new position.
        quantity_precision: number of decimal places the quantity is rounded
            *down* to (exchange lot size).
        min_notional: minimum order value (quantity * price) in quote
            currency; smaller orders are rejected.
    """

    def __init__(
        self,
        position_size_pct: float = 0.95,
        quantity_precision: int = 6,
        min_notional: float = 5.0,
    ) -> None:
        if not 0.0 < position_size_pct <= 1.0:
            raise ValueError(f"position_size_pct must be in (0, 1], got {position_size_pct}")
        if quantity_precision < 0:
            raise ValueError(f"quantity_precision must be >= 0, got {quantity_precision}")
        if min_notional < 0:
            raise ValueError(f"min_notional must be >= 0, got {min_notional}")
        self.position_size_pct = position_size_pct
        self.quantity_precision = quantity_precision
        self.min_notional = min_notional

    def position_quantity(self, equity: float, price: float) -> float:
        """Return the entry quantity for the given equity and price.

        The quantity is ``equity * position_size_pct / price`` rounded down to
        ``quantity_precision`` decimals. If the resulting order value is below
        ``min_notional`` (or the inputs are non-positive), returns ``0.0``
        meaning "do not trade".
        """
        if equity <= 0.0 or price <= 0.0:
            return 0.0
        raw = equity * self.position_size_pct / price
        scale = 10**self.quantity_precision
        quantity = math.floor(raw * scale) / scale
        if quantity <= 0.0 or quantity * price < self.min_notional:
            return 0.0
        return quantity
