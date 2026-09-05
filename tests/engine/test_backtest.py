"""Tests for the backtest engine event loop (offline, synthetic candles)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, rows_to_df
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Signal, SignalKind, Strategy

FEE = 0.001
SLIP = 5.0  # bps
START_CASH = 1_000.0


class ScriptedStrategy(Strategy):
    """Emits programmed signals at fixed candle indices."""

    name = "scripted"

    def __init__(
        self,
        entry_at: int | None = 2,
        exit_at: int | None = 7,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        self._entry_at = entry_at
        self._exit_at = exit_at
        self._stop_loss = stop_loss
        self._take_profit = take_profit

    @property
    def warmup_period(self) -> int:
        return 0

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        i = len(candles) - 1
        if i == self._entry_at:
            return [
                Signal(
                    SignalKind.LONG_ENTRY,
                    reason="test entry",
                    stop_loss=self._stop_loss,
                    take_profit=self._take_profit,
                )
            ]
        if i == self._exit_at:
            return [Signal(SignalKind.LONG_EXIT, reason="test exit")]
        return []


class PryingStrategy(Strategy):
    """Tries to read past the end of the candle slice at every step."""

    name = "prying"

    def __init__(self) -> None:
        self.seen_lengths: list[int] = []
        self.out_of_bounds_blocked: list[bool] = []

    @property
    def warmup_period(self) -> int:
        return 0

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        n = len(candles)
        self.seen_lengths.append(n)
        try:
            candles.iloc[n]  # one row past the slice must not exist
            self.out_of_bounds_blocked.append(False)
        except IndexError:
            self.out_of_bounds_blocked.append(True)
        return []


def flat_candles(prices: list[float]) -> pd.DataFrame:
    """Candles where o = h = l = c (valid OHLC)."""
    rows = [
        [BASE_MS + i * HOUR_MS, p, p, p, p, 1.0] for i, p in enumerate(prices)
    ]
    return rows_to_df(rows)


def make_engine(strategy: Strategy) -> BacktestEngine:
    return BacktestEngine(
        strategy=strategy,
        risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=5.0),
        broker=SimulatedBroker(fee_rate=FEE, slippage_bps=SLIP),
        start_cash=START_CASH,
    )


class TestFullCycle:
    def test_one_entry_one_exit_exact_math(self) -> None:
        prices = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0]
        prices += [105.0] * 7  # 15 candles total; exit executes at open[8] = 105
        candles = flat_candles(prices)
        engine = make_engine(ScriptedStrategy(entry_at=2, exit_at=7))

        result = engine.run(candles)

        # Sizing at the signal candle (close=100, equity=1000): qty = 9.5.
        qty = 9.5
        entry_price = 100.0 * (1 + SLIP / 10_000)  # 100.05
        entry_fee = qty * entry_price * FEE
        exit_price = 105.0 * (1 - SLIP / 10_000)  # 104.9475
        exit_fee = qty * exit_price * FEE
        expected_pnl = qty * (exit_price - entry_price) - entry_fee - exit_fee
        expected_final_cash = START_CASH - qty * entry_price - entry_fee
        expected_final_cash += qty * exit_price - exit_fee

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["entry_ts"] == candles["timestamp"].iloc[3]
        assert trade["exit_ts"] == candles["timestamp"].iloc[8]
        assert trade["entry_price"] == pytest.approx(entry_price)
        assert trade["exit_price"] == pytest.approx(exit_price)
        assert trade["quantity"] == pytest.approx(qty)
        assert trade["pnl"] == pytest.approx(expected_pnl)
        assert trade["reason_entry"] == "test entry"
        assert trade["reason_exit"] == "test exit"

        assert result.open_position is None
        assert result.n_pending_unfilled == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_cash)

    def test_equity_curve_structure(self) -> None:
        candles = flat_candles([100.0, 100.0, 100.0, 100.0, 110.0] * 3)
        engine = make_engine(ScriptedStrategy(entry_at=2, exit_at=7))

        result = engine.run(candles)

        assert len(result.equity_curve) == len(candles)
        assert list(result.equity_curve.index) == list(candles["timestamp"])
        # Flat before the entry executes at open of candle 3.
        assert result.equity_curve.iloc[:3].to_list() == pytest.approx([START_CASH] * 3)
        # Mark-to-market while the position is open: entry fills at open[3]
        # (close 100), the 110 closes start at candle 4.
        entry_price = 100.0 * (1 + SLIP / 10_000)
        qty = 9.5
        cash_after_entry = START_CASH - qty * entry_price - qty * entry_price * FEE
        assert result.equity_curve.iloc[3] == pytest.approx(cash_after_entry + qty * 100.0)
        assert result.equity_curve.iloc[4] == pytest.approx(cash_after_entry + qty * 110.0)
        assert result.candles_start == candles["timestamp"].iloc[0]
        assert result.candles_end == candles["timestamp"].iloc[-1]

    def test_open_position_is_marked_but_not_a_trade(self) -> None:
        candles = flat_candles([100.0] * 5 + [120.0] * 5)
        engine = make_engine(ScriptedStrategy(entry_at=2, exit_at=None))

        result = engine.run(candles)

        assert result.trades.empty
        assert result.open_position is not None
        assert result.open_position.quantity == pytest.approx(9.5)
        entry_price = 100.0 * (1 + SLIP / 10_000)
        cash = START_CASH - 9.5 * entry_price - 9.5 * entry_price * FEE
        assert result.equity_curve.iloc[-1] == pytest.approx(cash + 9.5 * 120.0)

    def test_exit_signal_on_last_candle_is_not_executed(self) -> None:
        candles = flat_candles([100.0] * 15)
        engine = make_engine(ScriptedStrategy(entry_at=2, exit_at=14))

        result = engine.run(candles)

        assert result.n_pending_unfilled == 1
        assert result.trades.empty
        assert result.open_position is not None

    def test_rerun_resets_strategy_state(self) -> None:
        candles = flat_candles([100.0] * 4 + [110.0] * 4 + [105.0] * 7)
        engine = make_engine(ScriptedStrategy(entry_at=2, exit_at=7))

        first = engine.run(candles)
        second = engine.run(candles)

        pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)
        pd.testing.assert_frame_equal(first.trades, second.trades)


class TestLookAhead:
    def test_engine_exposes_exactly_i_plus_1_rows_at_step_i(self) -> None:
        n = 10
        candles = flat_candles([100.0] * n)
        strategy = PryingStrategy()
        engine = make_engine(strategy)

        engine.run(candles)

        assert strategy.seen_lengths == list(range(1, n + 1))
        # Reading one row past the slice always fails: no future data leaks.
        assert all(strategy.out_of_bounds_blocked)


class TestIntrabarStops:
    def test_stop_gap_down_fills_at_open_not_at_stop(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1.0],
            # Opens at 90, far below the 95 stop: the conservative fill is the open.
            [BASE_MS + 2 * HOUR_MS, 90.0, 92.0, 88.0, 91.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(90.0 * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "stop loss"
        assert result.open_position is None

    def test_stop_hit_inside_bar_fills_at_stop(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 101.0, 94.0, 99.0, 1.0],  # low grazes the 95 stop
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(95.0 * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "stop loss"

    def test_take_profit_fills_at_target(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 106.0, 99.0, 105.0, 1.0],  # high reaches 105
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, take_profit=105.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(105.0 * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "take profit"

    def test_stop_has_priority_over_take_profit(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 106.0, 94.0, 101.0, 1.0],  # both levels hit
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0, take_profit=105.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["reason_exit"] == "stop loss"
        assert trade["exit_price"] == pytest.approx(95.0 * (1 - SLIP / 10_000))


class TestSignalGuards:
    def test_exit_without_position_is_ignored(self) -> None:
        candles = flat_candles([100.0] * 5)
        engine = make_engine(ScriptedStrategy(entry_at=None, exit_at=1))

        result = engine.run(candles)

        assert result.trades.empty
        assert result.open_position is None
        assert result.equity_curve.iloc[-1] == pytest.approx(START_CASH)

    def test_entry_below_min_notional_is_skipped(self) -> None:
        # Sizing puts ~95% of the 1000 equity into the order (notional ~950);
        # with min_notional above that the entry is rejected entirely.
        candles = flat_candles([2_000_000.0] * 5)
        engine = BacktestEngine(
            strategy=ScriptedStrategy(entry_at=1, exit_at=None),
            risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=1_000.0),
            broker=SimulatedBroker(fee_rate=FEE, slippage_bps=SLIP),
            start_cash=START_CASH,
        )

        result = engine.run(candles)

        assert result.open_position is None
        assert result.equity_curve.iloc[-1] == pytest.approx(START_CASH)


class LongWarmupStrategy(ScriptedStrategy):
    @property
    def warmup_period(self) -> int:
        return 10


class TestValidation:
    def test_empty_candles_raise(self) -> None:
        engine = make_engine(ScriptedStrategy())

        with pytest.raises(ValueError, match="empty"):
            engine.run(rows_to_df([]))

    def test_missing_columns_raise(self) -> None:
        engine = make_engine(ScriptedStrategy())
        df = pd.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="columns"):
            engine.run(df)

    def test_warmup_longer_than_data_runs_flat(self) -> None:
        candles = flat_candles([100.0] * 3)
        engine = make_engine(LongWarmupStrategy(entry_at=1))

        result = engine.run(candles)

        assert result.trades.empty
        assert result.equity_curve.iloc[-1] == pytest.approx(START_CASH)
