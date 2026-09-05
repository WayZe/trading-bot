"""Тесты адаптеров исполнения (paper и testnet)."""

from __future__ import annotations

import pytest

from tests.live.fakes import FakeCcxt, PrivateCcxt
from trading_bot.data.exchange import ExchangeClient
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.live.execution import PaperAdapter
from trading_bot.live.execution import TestnetAdapter as _TestnetAdapter


class TestPaperAdapter:
    def test_buy_fills_above_ticker_by_slippage(self) -> None:
        fake = FakeCcxt([], ticker_price=100.0)
        client = ExchangeClient()
        client.exchange = fake
        adapter = PaperAdapter(
            SimulatedBroker(fee_rate=0.001, slippage_bps=5.0),
            price_source=lambda: client.fetch_ticker_last("BTC/USDT"),
            equity_source=lambda: 5_000.0,
        )

        fill = adapter.execute("buy", 2.0, "donchian breakout up", 2.0, None)

        assert fill.side == "buy"
        assert fill.quantity == 2.0
        assert fill.price == pytest.approx(100.0 * (1 + 5 / 10_000))
        assert fill.fee == pytest.approx(2.0 * fill.price * 0.001)
        assert fill.reason == "donchian breakout up"
        assert fill.ts.tz is not None
        assert fake.private_calls == []  # публичный тикер — можно, приватные — нет

    def test_sell_fills_below_ticker_by_slippage(self) -> None:
        adapter = PaperAdapter(
            SimulatedBroker(fee_rate=0.0, slippage_bps=10.0),
            price_source=lambda: 200.0,
            equity_source=lambda: 0.0,
        )

        fill = adapter.execute("sell", 1.0, "stop loss")

        assert fill.price == pytest.approx(200.0 * (1 - 10 / 10_000))
        assert fill.fee == 0.0

    def test_fetch_equity_uses_source(self) -> None:
        adapter = PaperAdapter(
            SimulatedBroker(fee_rate=0.001, slippage_bps=5.0),
            price_source=lambda: 100.0,
            equity_source=lambda: 4_321.0,
        )

        assert adapter.fetch_equity() == 4_321.0


class TestTestnetAdapter:
    def _adapter(self, order_status: dict, free_usdt: float = 0.0):
        fake = PrivateCcxt(
            [], order_status=order_status, free_usdt=free_usdt
        )
        client = ExchangeClient()
        client.exchange = fake
        return _TestnetAdapter(client, "BTC/USDT"), fake

    def test_execute_uses_actual_fill_price_fee_and_quantity(self) -> None:
        adapter, fake = self._adapter(
            order_status={"average": 121.5, "filled": 0.8, "fee": {"cost": 0.12}},
            free_usdt=9_500.0,
        )

        fill = adapter.execute("buy", 0.8, "donchian breakout up", 4.5, None)

        assert fill.side == "buy"
        assert fill.price == 121.5  # фактическая средняя цена исполнения
        assert fill.quantity == 0.8
        assert fill.fee == 0.12
        assert fill.reason == "donchian breakout up"
        assert fake.created_orders == [("BTC/USDT", "market", "buy", 0.8)]

    def test_execute_falls_back_to_price_and_zero_fee(self) -> None:
        adapter, _ = self._adapter(order_status={"price": 120.0, "filled": 1.0})

        fill = adapter.execute("sell", 1.0, "stop loss")

        assert fill.price == 120.0
        assert fill.quantity == 1.0
        assert fill.fee == 0.0

    def test_fetch_equity_returns_free_usdt(self) -> None:
        adapter, _ = self._adapter(order_status={}, free_usdt=7_777.5)

        assert adapter.fetch_equity() == 7_777.5
