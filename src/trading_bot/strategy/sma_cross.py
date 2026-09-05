"""SMA crossover strategy with an ATR-based stop."""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.indicators import atr as atr_indicator
from trading_bot.indicators import sma
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy

REASON_CROSS_UP = "sma cross up"
REASON_CROSS_DOWN = "sma cross down"
REASON_STOP_BREACH = "stop level breach"


class SmaCrossStrategy(Strategy):
    """Long-only strategy: enter when the fast SMA crosses above the slow SMA.

    Entry: ``fast[t] > slow[t]`` and ``fast[t-1] <= slow[t-1]``. The emitted
    signal carries ``stop_loss = close[t] - atr_mult * ATR[t]`` computed on
    the signal candle's close.

    Exit: the fast SMA crosses back below the slow one, OR the candle closes
    below the stop level tracked from the *actual* fill price:
    ``stop = entry_fill_price - atr_mult * ATR[signal candle]`` (the ATR of
    the signal candle is stored when the entry signal is emitted). The stop
    breach takes priority over the cross-down reason.

    Indicators are recomputed on the growing candle slice and cached by the
    slice length, which is cheap for the candle counts involved.
    """

    name = "sma_cross"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        atr_period: int = 14,
        atr_mult: float = 2.0,
    ) -> None:
        if fast < 1 or slow < 1 or atr_period < 1:
            raise ValueError("fast, slow and atr_period must be >= 1")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be positive, got {atr_mult}")
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self._reset_state()

    @property
    def warmup_period(self) -> int:
        return self.slow + self.atr_period

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        self._update_indicators(candles)
        i = len(candles) - 1
        if i < 1:
            return []

        fast_now = self._fast_line.iloc[i]
        slow_now = self._slow_line.iloc[i]
        fast_prev = self._fast_line.iloc[i - 1]
        slow_prev = self._slow_line.iloc[i - 1]
        if (
            math.isnan(fast_now)
            or math.isnan(slow_now)
            or math.isnan(fast_prev)
            or math.isnan(slow_prev)
        ):
            return []

        close_now = float(candles["close"].iloc[i])
        cross_up = fast_now > slow_now and fast_prev <= slow_prev
        cross_down = fast_now < slow_now and fast_prev >= slow_prev

        if self._in_position:
            if self._stop_level is not None and close_now < self._stop_level:
                return [Signal(SignalKind.LONG_EXIT, reason=REASON_STOP_BREACH)]
            if cross_down:
                return [Signal(SignalKind.LONG_EXIT, reason=REASON_CROSS_DOWN)]
        elif cross_up:
            atr_now = float(self._atr_line.iloc[i])
            if math.isnan(atr_now):
                return []
            self._atr_at_entry = atr_now
            stop_loss = close_now - self.atr_mult * atr_now
            return [Signal(SignalKind.LONG_ENTRY, reason=REASON_CROSS_UP, stop_loss=stop_loss)]
        return []

    def on_fill(self, fill: Fill) -> None:
        if fill.side == "buy":
            self._in_position = True
            if self._atr_at_entry is not None:
                self._stop_level = fill.price - self.atr_mult * self._atr_at_entry
        elif fill.side == "sell":
            self._in_position = False
            self._stop_level = None
            self._atr_at_entry = None

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._cache_len = -1
        self._fast_line: pd.Series | None = None
        self._slow_line: pd.Series | None = None
        self._atr_line: pd.Series | None = None
        self._in_position = False
        self._atr_at_entry: float | None = None
        self._stop_level: float | None = None

    def _update_indicators(self, candles: pd.DataFrame) -> None:
        n = len(candles)
        if n == self._cache_len:
            return
        close = candles["close"]
        self._fast_line = sma(close, self.fast)
        self._slow_line = sma(close, self.slow)
        self._atr_line = atr_indicator(
            candles["high"], candles["low"], close, self.atr_period
        )
        self._cache_len = n
