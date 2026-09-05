"""Симуляция исполнения рыночных ордеров с комиссией и проскальзыванием."""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.strategy.base import Fill

SIDE_BUY = "buy"
SIDE_SELL = "sell"


class SimulatedBroker:
    """Детерминированная модель исполнения для бэктестов.

    Покупки исполняются выше опорной цены, продажи ниже неё (проскальзывание
    работает против нас); комиссия — плоская ставка от исполненного объёма.
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
        """Исполнить рыночный ордер по ``price_ref`` (обычно это open).

        Args:
            side: ``"buy"`` или ``"sell"``.
            quantity: размер ордера в базовой валюте (должен быть положительным).
            price_ref: опорная цена до проскальзывания.
            timestamp: метка времени исполнения.
            reason: причина, перенесённая от исходного сигнала.

        Returns:
            Итоговый :class:`Fill` с ценой после проскальзывания и комиссией
            (``notional * fee_rate`` в котируемой валюте).
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
