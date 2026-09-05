"""Tests for the risk manager / position sizing."""

from __future__ import annotations

import pytest

from trading_bot.risk import RiskManager


class TestPositionQuantity:
    def test_size_is_share_of_equity(self) -> None:
        risk = RiskManager(position_size_pct=0.95)

        assert risk.position_quantity(equity=1_000.0, price=100.0) == pytest.approx(9.5)

    def test_rounds_down_to_precision(self) -> None:
        # 100 * 0.95 / 3 = 31.666666... -> floor at 6 decimals
        risk = RiskManager(position_size_pct=0.95, quantity_precision=6)

        quantity = risk.position_quantity(equity=100.0, price=3.0)

        assert quantity == pytest.approx(31.666666)
        assert quantity * 3.0 <= 95.0  # never overshoots the allocated share

    def test_precision_is_respected(self) -> None:
        risk = RiskManager(position_size_pct=1.0, quantity_precision=2)

        assert risk.position_quantity(equity=100.0, price=3.0) == pytest.approx(33.33)

    def test_min_notional_cuts_small_orders(self) -> None:
        # 10 * 0.95 / 1000 = 0.0095 -> notional 9.5 < min_notional 10
        risk = RiskManager(position_size_pct=0.95, min_notional=10.0)

        assert risk.position_quantity(equity=10.0, price=1_000.0) == 0.0

    def test_min_notional_passes_at_boundary(self) -> None:
        risk = RiskManager(position_size_pct=1.0, min_notional=5.0)

        assert risk.position_quantity(equity=5.0, price=1.0) == pytest.approx(5.0)

    def test_nonpositive_inputs_give_zero(self) -> None:
        risk = RiskManager()

        assert risk.position_quantity(equity=0.0, price=100.0) == 0.0
        assert risk.position_quantity(equity=-5.0, price=100.0) == 0.0
        assert risk.position_quantity(equity=100.0, price=0.0) == 0.0

    def test_dust_quantity_gives_zero(self) -> None:
        # 1 * 1.0 / 1e9 floors to 0 at precision 6
        risk = RiskManager(position_size_pct=1.0, quantity_precision=6, min_notional=0.0)

        assert risk.position_quantity(equity=1.0, price=1e9) == 0.0


class TestValidation:
    def test_bad_position_size_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="position_size_pct"):
            RiskManager(position_size_pct=1.5)

    def test_negative_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_precision"):
            RiskManager(quantity_precision=-1)

    def test_negative_min_notional_raises(self) -> None:
        with pytest.raises(ValueError, match="min_notional"):
            RiskManager(min_notional=-1.0)
