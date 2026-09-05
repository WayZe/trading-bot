"""Tests for portfolio accounting."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.engine.portfolio import Portfolio

TS1 = pd.Timestamp("2025-08-01T00:00:00Z")
TS2 = pd.Timestamp("2025-08-01T04:00:00Z")


class TestBuySellCycle:
    def test_buy_debits_cash_and_opens_position(self) -> None:
        portfolio = Portfolio(start_cash=10_000.0)

        portfolio.buy(quantity=2.0, price=100.0, fee=0.2, ts=TS1)

        assert portfolio.cash == pytest.approx(10_000.0 - 200.0 - 0.2)  # 9799.8
        assert portfolio.position is not None
        assert portfolio.position.quantity == 2.0
        assert portfolio.position.entry_price == 100.0
        assert portfolio.position.entry_ts == TS1
        assert portfolio.realized_pnl == pytest.approx(0.0)
        assert portfolio.trades == []

    def test_sell_credits_cash_and_records_trade(self) -> None:
        portfolio = Portfolio(start_cash=10_000.0)
        portfolio.buy(quantity=2.0, price=100.0, fee=0.2, ts=TS1)

        portfolio.sell(
            quantity=2.0,
            price=110.0,
            fee=0.22,
            ts=TS2,
            reason_exit="sma cross down",
            reason_entry="sma cross up",
        )

        assert portfolio.cash == pytest.approx(10_000.0 - 200.0 - 0.2 + 220.0 - 0.22)
        assert portfolio.position is None
        # pnl = 2 * (110 - 100) - 0.2 - 0.22 = 19.58 (net of both fees)
        assert portfolio.realized_pnl == pytest.approx(19.58)
        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]
        assert trade.entry_ts == TS1
        assert trade.exit_ts == TS2
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.quantity == 2.0
        assert trade.pnl == pytest.approx(19.58)
        assert trade.reason_entry == "sma cross up"
        assert trade.reason_exit == "sma cross down"
        # cash after the round trip equals start plus realized pnl
        assert portfolio.cash == pytest.approx(10_000.0 + portfolio.realized_pnl)

    def test_losing_trade(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)
        portfolio.buy(quantity=1.0, price=100.0, fee=0.1, ts=TS1)

        portfolio.sell(
            quantity=1.0, price=90.0, fee=0.09, ts=TS2, reason_exit="s", reason_entry="b"
        )

        assert portfolio.realized_pnl == pytest.approx(-10.0 - 0.1 - 0.09)
        assert portfolio.cash == pytest.approx(1_000.0 - 10.0 - 0.1 - 0.09)


class TestEquity:
    def test_equity_without_position_is_cash(self) -> None:
        portfolio = Portfolio(start_cash=1_500.0)

        assert portfolio.equity(price=123.0) == pytest.approx(1_500.0)

    def test_equity_with_position_marks_to_market(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)
        portfolio.buy(quantity=2.0, price=100.0, fee=0.2, ts=TS1)

        assert portfolio.equity(price=110.0) == pytest.approx(1_000.0 - 200.0 - 0.2 + 220.0)
        assert portfolio.equity(price=90.0) == pytest.approx(1_000.0 - 200.0 - 0.2 + 180.0)


class TestValidation:
    def test_buy_with_open_position_raises(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)
        portfolio.buy(quantity=1.0, price=100.0, fee=0.1, ts=TS1)

        with pytest.raises(RuntimeError, match="pyramiding"):
            portfolio.buy(quantity=1.0, price=100.0, fee=0.1, ts=TS2)

    def test_sell_without_position_raises(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)

        with pytest.raises(RuntimeError, match="no position"):
            portfolio.sell(
                quantity=1.0, price=100.0, fee=0.1, ts=TS1, reason_exit="", reason_entry=""
            )

    def test_partial_sell_raises(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)
        portfolio.buy(quantity=2.0, price=100.0, fee=0.2, ts=TS1)

        with pytest.raises(ValueError, match="partial"):
            portfolio.sell(
                quantity=1.0, price=100.0, fee=0.1, ts=TS2, reason_exit="", reason_entry=""
            )

    def test_nonpositive_inputs_raise(self) -> None:
        portfolio = Portfolio(start_cash=1_000.0)

        with pytest.raises(ValueError, match="quantity"):
            portfolio.buy(quantity=0.0, price=100.0, fee=0.0, ts=TS1)
        with pytest.raises(ValueError, match="price"):
            portfolio.buy(quantity=1.0, price=0.0, fee=0.0, ts=TS1)

    def test_nonpositive_start_cash_raises(self) -> None:
        with pytest.raises(ValueError, match="start_cash"):
            Portfolio(start_cash=0.0)
