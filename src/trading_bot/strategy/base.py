"""Strategy plugin contract.

A strategy does not know who calls it — a backtest engine or a future live
engine. It receives closed candles one by one via :meth:`Strategy.on_candle`
and returns trading *intentions* (:class:`Signal`); execution is fully owned
by the engine and its broker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SignalKind(StrEnum):
    """Kind of trading intention emitted by a strategy."""

    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"


@dataclass(frozen=True)
class Signal:
    """A trading intention computed from closed candles.

    Attributes:
        kind: what the strategy wants to do.
        reason: human-readable explanation (used in trade records and logs).
        stop_loss: optional stop level attached to an entry, expressed
            against the signal candle's close; the engine re-anchors the
            distance to the actual entry fill price and uses it for intrabar
            stop checks while the position is open.
        take_profit: optional take-profit price attached to an entry
            (re-anchored to the fill price the same way as ``stop_loss``).
    """

    kind: SignalKind
    reason: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True)
class Fill:
    """A confirmed execution reported back to the strategy.

    Attributes:
        side: ``"buy"`` or ``"sell"``.
        price: actual execution price (already includes slippage).
        quantity: executed quantity in base currency.
        timestamp: candle timestamp of the execution.
        fee: fee paid for this fill, in quote currency.
        reason: the reason carried from the originating signal.
    """

    side: str
    price: float
    quantity: float
    timestamp: pd.Timestamp
    fee: float
    reason: str


class Strategy(ABC):
    """Base class for trading strategies.

    Contract:
    - ``candles`` passed to :meth:`on_candle` is the whole history up to and
      including the current **closed** candle;
    - the strategy must not mutate the dataframe and must not look ahead;
    - state must be resettable via :meth:`reset` so the same instance can be
      used for repeated runs.
    """

    name: str = "base"

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Number of initial candles the strategy needs before it can signal."""

    @abstractmethod
    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        """Process the closed candles history and return trading intentions."""

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 (intentional optional hook)
        """Hook called by the engine after an order fills (no-op by default).

        Lets the strategy track actual entry prices (e.g. for stop tracking).
        """

    def reset(self) -> None:  # noqa: B027 (intentional optional hook)
        """Reset internal state for a fresh run (no-op by default)."""
