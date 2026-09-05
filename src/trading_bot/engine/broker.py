"""Simulated market-order execution with fees and slippage."""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.strategy.base import Fill

SIDE_BUY = "buy"
SIDE_SELL = "sell"


class SimulatedBroker:
    """Deterministic fill model for backtests.

    Buys fill above the reference price, sells below it (slippage works
    against us); the fee is a flat rate over the executed notional.
    """

    def __init__(self, fee_rate: float, slippage_bps: float) -> None:
        if fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")
        if slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {slippage_bps}")
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

    def execute_market(
        self,
        side: str,
        quantity: float,
        price_ref: float,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> Fill:
        """Execute a market order against ``price_ref`` (typically an open).

        Args:
            side: ``"buy"`` or ``"sell"``.
            quantity: order size in base currency (must be positive).
            price_ref: reference price before slippage.
            timestamp: execution timestamp.
            reason: reason carried from the originating signal.

        Returns:
            The resulting :class:`Fill` with the slipped price and the fee
            (``notional * fee_rate`` in quote currency).
        """
        if side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(f"side must be {SIDE_BUY!r} or {SIDE_SELL!r}, got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if math.isnan(price_ref) or price_ref <= 0:
            raise ValueError(f"price_ref must be a positive number, got {price_ref}")

        slippage = self.slippage_bps / 10_000.0
        price = price_ref * (1.0 + slippage) if side == SIDE_BUY else price_ref * (1.0 - slippage)
        fee = quantity * price * self.fee_rate
        return Fill(
            side=side, price=price, quantity=quantity, timestamp=timestamp, fee=fee, reason=reason
        )
