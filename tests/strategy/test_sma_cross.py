"""Tests for the SMA cross strategy on synthetic candle series."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, rows_to_df
from trading_bot.strategy import create_strategy
from trading_bot.strategy.base import Fill, Signal, SignalKind
from trading_bot.strategy.sma_cross import SmaCrossStrategy

# Candles are built so that ATR is exactly 2.0 after its warmup:
# high = close + 1, low = close - 1, and close steps of 0.5 keep every
# True Range equal to 2. With atr_mult=2 the stop distance is exactly 4.0.
SPREAD = 1.0
ATR_VALUE = 2.0

UP_CROSS_CLOSES = [
    100.0, 99.5, 99.0, 98.5, 98.0, 97.5,  # decline: fast below slow
    98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0,  # rise: fast crosses up
]
DOWN_CLOSES = [
    100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0,  # decline: cross down
]


def candles_from_closes(closes: list[float]) -> pd.DataFrame:
    rows = [
        [BASE_MS + i * HOUR_MS, c, c + SPREAD, c - SPREAD, c, 1.0]
        for i, c in enumerate(closes)
    ]
    return rows_to_df(rows)


def feed(strategy: SmaCrossStrategy, candles: pd.DataFrame) -> list[tuple[int, object]]:
    """Feed growing slices; return [(candle_index, signal), ...]."""
    emitted: list[tuple[int, object]] = []
    for i in range(len(candles)):
        for signal in strategy.on_candle(candles.iloc[: i + 1]):
            emitted.append((i, signal))
    return emitted


def make_fill(side: str, price: float, timestamp: pd.Timestamp) -> Fill:
    return Fill(side=side, price=price, quantity=1.0, timestamp=timestamp, fee=0.0, reason="")


def make_strategy() -> SmaCrossStrategy:
    return SmaCrossStrategy(fast=3, slow=5, atr_period=3, atr_mult=2.0)


class TestContract:
    def test_name_and_warmup(self) -> None:
        strategy = make_strategy()

        assert strategy.name == "sma_cross"
        assert strategy.warmup_period == 5 + 3

    def test_fast_must_be_below_slow(self) -> None:
        with pytest.raises(ValueError, match="fast"):
            SmaCrossStrategy(fast=50, slow=20)

    def test_registry_creates_strategy(self) -> None:
        strategy = create_strategy("sma_cross", {"fast": 3, "slow": 5, "atr_period": 3})

        assert isinstance(strategy, SmaCrossStrategy)
        assert strategy.warmup_period == 8

    def test_registry_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            create_strategy("nope", {})

    def test_registry_bad_params(self) -> None:
        with pytest.raises(ValueError, match="invalid strategy_params"):
            create_strategy("sma_cross", {"unexpected": 1})


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fast": "3"}, "must be an integer"),
            ({"slow": 6.0}, "must be an integer"),
            ({"atr_period": None}, "must be an integer"),
            ({"fast": 0}, "must be >= 1"),
            ({"atr_mult": float("nan")}, "must be finite"),
            ({"atr_mult": float("inf")}, "must be finite"),
            ({"atr_mult": "2"}, "must be a number"),
            ({"atr_mult": 0.0}, "must be positive"),
            ({"atr_mult": -1.0}, "must be positive"),
        ],
    )
    def test_bad_params_raise_value_error(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SmaCrossStrategy(**kwargs)


class TestEntry:
    def test_single_cross_up_emits_one_long_entry(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES)

        emitted = feed(strategy, candles)

        entries = [s for _, s in emitted if s.kind is SignalKind.LONG_ENTRY]
        assert len(entries) == 1
        assert emitted[0][1].kind is SignalKind.LONG_ENTRY

    def test_entry_signal_fields(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES)

        emitted = feed(strategy, candles)

        index, signal = emitted[0]
        close_at_signal = candles["close"].iloc[index]
        assert signal.kind is SignalKind.LONG_ENTRY
        assert signal.reason == "sma cross up"
        assert signal.take_profit is None
        assert signal.stop_loss == pytest.approx(close_at_signal - 2.0 * ATR_VALUE)

    def test_entry_matches_independent_sma_oracle(self) -> None:
        # Independent oracle: rolling means computed directly in the test.
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES)
        close = candles["close"]
        fast = close.rolling(3).mean()
        slow = close.rolling(5).mean()

        emitted = feed(strategy, candles)

        index, _ = emitted[0]
        assert fast.iloc[index] > slow.iloc[index]
        assert fast.iloc[index - 1] <= slow.iloc[index - 1]
        # No earlier index satisfies the cross condition.
        for j in range(index):
            assert not (fast.iloc[j] > slow.iloc[j] and fast.iloc[j - 1] <= slow.iloc[j - 1])

    def test_no_entry_during_warmup(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

        assert feed(strategy, candles) == []


class TestExit:
    def test_cross_down_emits_long_exit_after_fill(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES + DOWN_CLOSES)

        emitted = feed(strategy, candles)
        entry_index, entry = next(
            (i, s) for i, s in emitted if s.kind is SignalKind.LONG_ENTRY
        )
        entry_price = float(candles["close"].iloc[entry_index])
        strategy.on_fill(make_fill("buy", entry_price, candles["timestamp"].iloc[entry_index]))

        exits = [
            (i, s)
            for i, s in feed(strategy, candles.iloc[entry_index + 1 :])
            if s.kind is SignalKind.LONG_EXIT
        ]
        # Rebase exit indices (feed() restarts from a sliced dataframe).
        exits = [(i + entry_index + 1, s) for i, s in exits]

        assert len(exits) == 1
        assert exits[0][1].reason == "sma cross down"

    def test_no_strategy_stop_breach_exit_engine_owns_the_stop(self) -> None:
        # The stop-breach exit path is gone: even a close below the would-be
        # fill-based stop (entry close 99 - 2*ATR = 95; the crash closes at
        # 94.5) can only exit through the SMA cross down — the engine's
        # intrabar stop is the only stop path left.
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES + [94.5, 94.0, 93.5])

        exits: list[Signal] = []
        entry_close: float | None = None
        in_position = False
        for i in range(len(candles)):
            if entry_close is not None and not in_position:
                # The engine fills the queued entry at this candle's open.
                strategy.on_fill(
                    make_fill("buy", entry_close, candles["timestamp"].iloc[i])
                )
                in_position = True
            for signal in strategy.on_candle(candles.iloc[: i + 1]):
                if signal.kind is SignalKind.LONG_ENTRY:
                    entry_close = float(candles["close"].iloc[i])
                elif signal.kind is SignalKind.LONG_EXIT:
                    exits.append(signal)

        assert entry_close is not None
        assert entry_close - 2.0 * ATR_VALUE > 94.5  # the crash is below the stop
        assert len(exits) == 1
        assert exits[0].reason == "sma cross down"

    def test_no_exit_without_position(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES + DOWN_CLOSES)

        emitted = feed(strategy, candles)  # no on_fill -> strategy stays flat

        assert all(s.kind is SignalKind.LONG_ENTRY for _, s in emitted)

    def test_sell_fill_clears_position_state(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES)
        emitted = feed(strategy, candles)
        entry_index, _ = emitted[0]
        entry_price = float(candles["close"].iloc[entry_index])
        strategy.on_fill(make_fill("buy", entry_price, candles["timestamp"].iloc[entry_index]))
        strategy.on_fill(make_fill("sell", entry_price, candles["timestamp"].iloc[entry_index]))

        # Flat again: the same cross-up re-fires on a fresh identical rise.
        assert strategy._in_position is False


class TestReset:
    def test_reset_clears_state_for_a_rerun(self) -> None:
        strategy = make_strategy()
        candles = candles_from_closes(UP_CROSS_CLOSES + DOWN_CLOSES)

        first_run = feed(strategy, candles)
        entry_index, _ = first_run[0]
        strategy.on_fill(
            make_fill("buy", 99.95, candles["timestamp"].iloc[entry_index])
        )
        assert strategy._in_position is True

        strategy.reset()

        assert strategy._in_position is False
        second_run = feed(strategy, candles)
        assert [(i, s.kind, s.reason) for i, s in second_run] == [
            (i, s.kind, s.reason) for i, s in first_run
        ]
