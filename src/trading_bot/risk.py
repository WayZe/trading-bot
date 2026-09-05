"""Размер позиции и ограничения на объём ордера."""

from __future__ import annotations

import math


class RiskManager:
    """Вычисляет объёмы входных ордеров с учётом капитала и лотных ограничений.

    Attributes:
        position_size_pct: доля капитала, выделяемая под новую позицию.
        quantity_precision: число десятичных знаков, до которых объём
            округляется *вниз* (размер лота на бирже).
        min_notional: минимальная стоимость ордера (quantity * price) в
            котируемой валюте; меньшие ордера отклоняются.
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
        """Вернуть объём входа для заданного капитала и цены.

        Объём равен ``equity * position_size_pct / price``, округлённому вниз
        до ``quantity_precision`` знаков. Если стоимость ордера ниже
        ``min_notional`` (или входы неположительные), возвращается ``0.0`` —
        то есть «не торговать».
        """
        if equity <= 0.0 or price <= 0.0:
            return 0.0
        raw = equity * self.position_size_pct / price
        scale = 10**self.quantity_precision
        quantity = math.floor(raw * scale) / scale
        if quantity <= 0.0 or quantity * price < self.min_notional:
            return 0.0
        return quantity
