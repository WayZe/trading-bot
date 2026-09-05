"""Учёт портфеля: кэш, одна long-позиция, записи сделок."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Position:
    """Открытая long-позиция."""

    quantity: float
    entry_price: float
    entry_ts: pd.Timestamp


@dataclass(frozen=True)
class TradeRecord:
    """Закрытая круговая сделка."""

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    reason_entry: str
    reason_exit: str


TRADE_RECORD_FIELDS = list(TradeRecord.__dataclass_fields__)


class Portfolio:
    """Учёт кэша и позиции не более чем с одной открытой long-позицией.

    Все цены и денежные суммы — в котируемой валюте (USDT).
    """

    def __init__(self, start_cash: float) -> None:
        if start_cash <= 0:
            raise ValueError(f"start_cash must be positive, got {start_cash}")
        self.start_cash = float(start_cash)
        self.cash = float(start_cash)
        self.position: Position | None = None
        self.realized_pnl = 0.0
        self.trades: list[TradeRecord] = []
        self._entry_fee = 0.0

    def buy(self, quantity: float, price: float, fee: float, ts: pd.Timestamp) -> None:
        """Открыть позицию: ``cash -= quantity * price + fee``."""
        if self.position is not None:
            raise RuntimeError("position already open: pyramiding is not supported in MVP")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self.cash -= quantity * price + fee
        self.position = Position(quantity=quantity, entry_price=price, entry_ts=ts)
        self._entry_fee = fee

    def sell(
        self,
        quantity: float,
        price: float,
        fee: float,
        ts: pd.Timestamp,
        reason_exit: str,
        reason_entry: str,
    ) -> None:
        """Закрыть позицию: ``cash += quantity * price - fee``.

        Записывает :class:`TradeRecord` с чистым pnl (включая комиссию и
        входа, и выхода) и обновляет ``realized_pnl``.
        """
        position = self.position
        if position is None:
            raise RuntimeError("no position to sell")
        if quantity != position.quantity:
            raise ValueError(
                f"partial closes are not supported: got {quantity}, "
                f"position holds {position.quantity}"
            )
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self.cash += quantity * price - fee
        pnl = quantity * (price - position.entry_price) - self._entry_fee - fee
        self.realized_pnl += pnl
        self.trades.append(
            TradeRecord(
                entry_ts=position.entry_ts,
                exit_ts=ts,
                entry_price=position.entry_price,
                exit_price=price,
                quantity=quantity,
                pnl=pnl,
                reason_entry=reason_entry,
                reason_exit=reason_exit,
            )
        )
        self.position = None
        self._entry_fee = 0.0

    def equity(self, price: float) -> float:
        """Эквити по рынку: кэш плюс стоимость позиции по ``price``."""
        if self.position is not None:
            return self.cash + self.position.quantity * price
        return self.cash
