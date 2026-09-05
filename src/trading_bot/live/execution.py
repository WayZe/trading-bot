"""Адаптеры исполнения ордеров для live-раннера.

Два режима за одним интерфейсом :class:`ExecutionAdapter`:

- :class:`PaperAdapter` — симуляция: цена берётся из последнего тикера
  (публичные данные), комиссия и проскальзывание считает
  :class:`~trading_bot.engine.broker.SimulatedBroker`. Не требует API-ключей
  и **никогда** не вызывает приватные методы биржи — это инвариант, который
  тест ``tests/live/test_runner.py`` доказывает подменой биржи, где любой
  приватный вызов роняет тест.
- :class:`TestnetAdapter` — реальные рыночные ордера на bybit testnet через
  ccxt: после размещения ордера запрашивается его фактический статус, и в
  результат идут средняя цена исполнения, исполненный объём и комиссия.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading_bot.data.exchange import ExchangeClient
from trading_bot.engine.broker import SimulatedBroker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillResult:
    """Факт исполнения ордера адаптером.

    Attributes:
        side: ``"buy"`` или ``"sell"``.
        quantity: исполненный объём в базовой валюте (в testnet — фактический).
        price: фактическая цена исполнения (уже с проскальзыванием в paper).
        fee: комиссия в котируемой валюте.
        ts: момент исполнения (UTC).
        reason: причина сделки, перенесённая от сигнала.
    """

    side: str
    quantity: float
    price: float
    fee: float
    ts: pd.Timestamp
    reason: str


class ExecutionAdapter(ABC):
    """Интерфейс исполнения: рынок, который ставит ордера, и источник equity."""

    @abstractmethod
    def execute(
        self,
        side: str,
        quantity: float,
        reason: str,
        stop_distance: float | None = None,
        tp_distance: float | None = None,
    ) -> FillResult:
        """Исполнить рыночный ордер и вернуть факт исполнения.

        Args:
            side: ``"buy"`` или ``"sell"``.
            quantity: объём в базовой валюте.
            reason: причина сделки (попадает в state.trades и логи).
            stop_distance: дистанция стопа от close сигнальной свечи —
                только контекст для логов адаптера, уровни считает раннер.
            tp_distance: дистанция тейк-профита, аналогично.
        """

    @abstractmethod
    def fetch_equity(self) -> float:
        """Вернуть доступный капитал в котируемой валюте (для размера позиции)."""


class PaperAdapter(ExecutionAdapter):
    """Paper-исполнение по последнему тикеру через :class:`SimulatedBroker`.

    Приватных вызовов биржи нет вообще: цена приходит из ``price_source``
    (публичный тикер), equity — из ``equity_source`` (state раннера).
    """

    def __init__(
        self,
        broker: SimulatedBroker,
        price_source: Callable[[], float],
        equity_source: Callable[[], float],
    ) -> None:
        self.broker = broker
        self.price_source = price_source
        self.equity_source = equity_source

    def execute(
        self,
        side: str,
        quantity: float,
        reason: str,
        stop_distance: float | None = None,
        tp_distance: float | None = None,
    ) -> FillResult:
        """Симулировать рыночное исполнение по текущему тикеру.

        Комиссия и проскальзывание — те же, что в бэктесте (конфиг
        ``fee_rate``/``slippage_bps``), поэтому paper-результаты сравнимы с
        бэктестом на тех же данных.
        """
        price_ref = float(self.price_source())
        fill = self.broker.execute_market(
            side, quantity, price_ref, pd.Timestamp.now(tz="UTC"), reason
        )
        logger.debug(
            "paper fill: %s %.6f @ %.4f (fee %.4f, reason=%r)",
            fill.side,
            fill.quantity,
            fill.price,
            fill.fee,
            reason,
        )
        return FillResult(
            side=side,
            quantity=fill.quantity,
            price=fill.price,
            fee=fill.fee,
            ts=fill.timestamp,
            reason=reason,
        )

    def fetch_equity(self) -> float:
        """Вернуть equity paper-портфеля (ведёт раннер в state)."""
        return float(self.equity_source())


class TestnetAdapter(ExecutionAdapter):
    """Исполнение рыночными ордерами на bybit testnet через ccxt.

    После размещения ордера запрашивается его статус: в результат идут
    фактическая средняя цена исполнения, исполненный объём и комиссия —
    симуляция здесь заканчивается.
    """

    def __init__(self, exchange: ExchangeClient, symbol: str) -> None:
        self.exchange = exchange
        self.symbol = symbol

    def execute(
        self,
        side: str,
        quantity: float,
        reason: str,
        stop_distance: float | None = None,
        tp_distance: float | None = None,
    ) -> FillResult:
        """Разместить market-ордер и дождаться его статуса через fetch_order."""
        logger.info(
            "testnet order: %s %.6f %s (reason=%r, stop_distance=%s, tp_distance=%s)",
            side,
            quantity,
            self.symbol,
            reason,
            stop_distance,
            tp_distance,
        )
        order = self.exchange.create_market_order(self.symbol, side, quantity)
        order_id = order["id"]
        status = self.exchange.fetch_order(order_id, self.symbol)
        price = float(status.get("average") or status.get("price"))
        filled = float(status.get("filled") or quantity)
        fee_data = status.get("fee") or {}
        fee = float(fee_data.get("cost") or 0.0)
        return FillResult(
            side=side,
            quantity=filled,
            price=price,
            fee=fee,
            ts=pd.Timestamp.now(tz="UTC"),
            reason=reason,
        )

    def fetch_equity(self) -> float:
        """Вернуть свободный USDT с биржи (testnet-эквити)."""
        return float(self.exchange.fetch_free_balance("USDT"))
