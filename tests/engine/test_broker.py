"""Tests for the simulated broker."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.engine.broker import SimulatedBroker

TS = pd.Timestamp("2025-08-01T00:00:00Z")


class TestExecuteMarket:
    def test_buy_slips_up(self) -> None:
        broker = SimulatedBroker(fee_rate=0.001, slippage_bps=5.0)

        fill = broker.execute_market("buy", quantity=2.0, price_ref=100.0, timestamp=TS, reason="r")

        assert fill.side == "buy"
        assert fill.price == pytest.approx(100.0 * 1.0005)  # 100.05
        assert fill.quantity == 2.0
        assert fill.timestamp == TS
        assert fill.reason == "r"

    def test_sell_slips_down(self) -> None:
        broker = SimulatedBroker(fee_rate=0.001, slippage_bps=5.0)

        fill = broker.execute_market(
            "sell", quantity=2.0, price_ref=100.0, timestamp=TS, reason="r"
        )

        assert fill.side == "sell"
        assert fill.price == pytest.approx(100.0 * 0.9995)  # 99.95

    def test_fee_is_notional_times_rate(self) -> None:
        broker = SimulatedBroker(fee_rate=0.002, slippage_bps=10.0)

        fill = broker.execute_market("buy", quantity=3.0, price_ref=50.0, timestamp=TS, reason="r")

        assert fill.fee == pytest.approx(3.0 * 50.0 * 1.001 * 0.002)

    def test_zero_costs_pass_through(self) -> None:
        broker = SimulatedBroker(fee_rate=0.0, slippage_bps=0.0)

        fill = broker.execute_market("buy", quantity=1.0, price_ref=42.0, timestamp=TS, reason="r")

        assert fill.price == pytest.approx(42.0)
        assert fill.fee == pytest.approx(0.0)

    def test_invalid_side_raises(self) -> None:
        broker = SimulatedBroker(fee_rate=0.001, slippage_bps=5.0)

        with pytest.raises(ValueError, match="side"):
            broker.execute_market("hold", 1.0, 100.0, TS, "r")

    def test_nonpositive_quantity_raises(self) -> None:
        broker = SimulatedBroker(fee_rate=0.001, slippage_bps=5.0)

        with pytest.raises(ValueError, match="quantity"):
            broker.execute_market("buy", 0.0, 100.0, TS, "r")

    def test_negative_costs_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="fee_rate"):
            SimulatedBroker(fee_rate=-0.1, slippage_bps=5.0)
        with pytest.raises(ValueError, match="slippage"):
            SimulatedBroker(fee_rate=0.1, slippage_bps=-1.0)
