"""Tests for the backtest engine event loop (offline, synthetic candles)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, rows_to_df
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Signal, SignalKind, Strategy
from trading_bot.strategy.sma_cross import SmaCrossStrategy

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
    """Stop/TP levels are re-anchored to the entry fill price.

    For a signal on candle 0 (close 100, flat candle) the entry executes at
    open[1] slipped: with SLIP=5 bps that is ``100 * 1.0005``. A stop of 95
    (distance 5 from the signal close) becomes ``fill - 5``, a take-profit
    of 105 becomes ``fill + 5``.
    """

    SIGNAL_CLOSE = 100.0
    ENTRY_FILL = 100.0 * (1 + SLIP / 10_000)

    def test_stop_gap_down_fills_at_open_not_at_stop(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1.0],
            # Opens at 90, far below the stop: the conservative fill is the open.
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
            # Low 94 is below the re-anchored stop (fill - 5 = 95.05).
            [BASE_MS + HOUR_MS, 100.0, 101.0, 94.0, 99.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(
            (self.ENTRY_FILL - 5.0) * (1 - SLIP / 10_000)
        )
        assert trade["reason_exit"] == "stop loss"

    def test_take_profit_fills_at_target(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            # High 106 reaches the re-anchored TP (fill + 5 = 105.05).
            [BASE_MS + HOUR_MS, 100.0, 106.0, 99.0, 105.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, take_profit=105.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(
            (self.ENTRY_FILL + 5.0) * (1 - SLIP / 10_000)
        )
        assert trade["reason_exit"] == "take profit"

    def test_take_profit_gap_up_fills_at_open_not_at_target(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1.0],
            # Opens at 110, above the re-anchored TP (105.05): fill at the open.
            [BASE_MS + 2 * HOUR_MS, 110.0, 112.0, 109.0, 111.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, take_profit=105.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["exit_price"] == pytest.approx(110.0 * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "take profit"

    def test_stop_has_priority_over_take_profit(self) -> None:
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            # Both re-anchored levels (95.05 / 105.05) are hit in this candle.
            [BASE_MS + HOUR_MS, 100.0, 106.0, 94.0, 101.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0, take_profit=105.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["reason_exit"] == "stop loss"
        assert trade["exit_price"] == pytest.approx(
            (self.ENTRY_FILL - 5.0) * (1 - SLIP / 10_000)
        )


class TestStopAnchoring:
    """The engine owns the stop and anchors it to the actual fill price."""

    def test_gap_up_entry_stop_is_exactly_fill_minus_distance(self) -> None:
        # Signal candle closes at 100 with stop 95 (distance 5); the next
        # candle opens 103 (gap up), so the fill is 103 * 1.0005 and the
        # active stop is fill - 5, not the strategy's 95. The candle's low
        # exactly equals the re-anchored stop: the <= comparison triggers it.
        entry_fill = 103.0 * (1 + SLIP / 10_000)
        active_stop = entry_fill - 5.0
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 103.0, 104.0, 102.0, 103.5, 1.0],
            [BASE_MS + 2 * HOUR_MS, 98.5, 99.0, active_stop, 97.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        trade = result.trades.iloc[0]
        assert trade["entry_price"] == pytest.approx(entry_fill)
        assert trade["exit_price"] == pytest.approx(active_stop * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "stop loss"

    def test_gap_up_entry_price_below_projected_stop_closes_intrabar(self) -> None:
        # Review scenario: gap-up entry (fill ~103.05) lifts the active stop
        # to ~98.05, above the strategy-projected 95. A dip to 96 stays above
        # the projected stop but breaches the fill-anchored one: the engine
        # must close intrabar instead of holding the position open.
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 103.0, 104.0, 102.5, 103.5, 1.0],
            [BASE_MS + 2 * HOUR_MS, 98.5, 99.0, 96.0, 97.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason_exit"] == "stop loss"
        assert trade["exit_ts"] == candles["timestamp"].iloc[2]
        assert result.open_position is None

    def test_stop_on_the_entry_candle_itself(self) -> None:
        # Signal on candle 0 -> entry at open[1]; the same candle 1 dips
        # below the re-anchored stop, so the exit happens within candle 1.
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            # Fill at 101 * 1.0005 -> active stop 101.0505 - 5 = 96.0505;
            # low 95 is below it.
            [BASE_MS + HOUR_MS, 101.0, 101.5, 95.0, 96.5, 1.0],
            [BASE_MS + 2 * HOUR_MS, 96.0, 96.5, 95.5, 96.0, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, stop_loss=95.0))

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["entry_ts"] == candles["timestamp"].iloc[1]
        assert trade["exit_ts"] == candles["timestamp"].iloc[1]
        active_stop = 101.0 * (1 + SLIP / 10_000) - 5.0
        assert trade["exit_price"] == pytest.approx(active_stop * (1 - SLIP / 10_000))
        assert trade["reason_exit"] == "stop loss"
        assert result.open_position is None

    def test_pending_exit_at_open_beats_intrabar_stop_same_candle(self) -> None:
        # The strategy's exit (queued on candle 1) executes at open[2]; the
        # same candle also breaches the stop, but the position is already
        # closed — no double sell is possible.
        rows = [
            [BASE_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [BASE_MS + HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1.0],  # entry, no stop hit
            [BASE_MS + 2 * HOUR_MS, 96.0, 97.0, 94.0, 95.0, 1.0],  # exit + stop breach
        ]
        candles = rows_to_df(rows)
        engine = make_engine(ScriptedStrategy(entry_at=0, exit_at=1, stop_loss=95.0))

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason_exit"] == "test exit"
        assert trade["exit_price"] == pytest.approx(96.0 * (1 - SLIP / 10_000))
        assert result.open_position is None


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

    def test_nan_in_required_columns_raises(self) -> None:
        candles = flat_candles([100.0] * 5)
        candles.loc[2, "close"] = float("nan")
        engine = make_engine(ScriptedStrategy())

        with pytest.raises(ValueError, match="NaN.*'close'"):
            engine.run(candles)

    def test_high_below_low_raises(self) -> None:
        candles = flat_candles([100.0] * 3)
        candles.loc[1, "high"] = 90.0
        candles.loc[1, "low"] = 100.0
        engine = make_engine(ScriptedStrategy())

        with pytest.raises(ValueError, match="high < low"):
            engine.run(candles)

    def test_warmup_longer_than_data_runs_flat(self) -> None:
        candles = flat_candles([100.0] * 3)
        engine = make_engine(LongWarmupStrategy(entry_at=1))

        result = engine.run(candles)

        assert result.trades.empty
        assert result.equity_curve.iloc[-1] == pytest.approx(START_CASH)


class TestSmaCrossIntegration:
    def test_full_cycle_one_trade_and_money_converge(self) -> None:
        # Rise (cross up) then fall (cross down) with a 2.0 ATR spread;
        # extra tail candles let the queued exit execute at the next open.
        closes = [
            100.0, 99.5, 99.0, 98.5, 98.0, 97.5,  # decline: fast below slow
            98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0,  # rise: cross up
            100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0,  # fall: cross down
            96.5, 96.0, 96.0, 96.0,  # tail: exit executes, no re-entry
        ]
        rows = [
            [BASE_MS + i * HOUR_MS, c, c + 1.0, c - 1.0, c, 1.0]
            for i, c in enumerate(closes)
        ]
        candles = rows_to_df(rows)
        engine = BacktestEngine(
            strategy=SmaCrossStrategy(fast=3, slow=5, atr_period=3, atr_mult=2.0),
            risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=5.0),
            broker=SimulatedBroker(fee_rate=FEE, slippage_bps=SLIP),
            start_cash=START_CASH,
        )

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason_entry"] == "sma cross up"
        assert trade["reason_exit"] == "sma cross down"
        assert result.open_position is None
        assert result.n_pending_unfilled == 0
        # Flat at the end: final equity is exactly start cash plus the pnl.
        assert result.equity_curve.iloc[-1] == pytest.approx(
            START_CASH + float(trade["pnl"])
        )
