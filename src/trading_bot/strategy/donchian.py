"""Стратегия пробоя канала Дончиана со стопом на основе ATR."""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.indicators import atr as atr_indicator
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy

REASON_BREAKOUT_UP = "donchian breakout up"
REASON_BREAKDOWN = "donchian breakdown"


class DonchianBreakoutStrategy(Strategy):
    """Long-only стратегия: вход на пробое максимума, выход на пробое минимума.

    Классика Черепах: канал входа длиннее канала выхода, позиция держится,
    пока не пробит короткий канал в противоположную сторону.

    Вход: ``close[t] > max(high[t-entry_period .. t-1])`` — верхняя граница
    канала считается по ``entry_period`` **предыдущим** свечам, без текущей
    (rolling max по high со сдвигом 1). Учёт текущей свечи сделал бы пробой
    невозможным в принципе: close не бывает выше собственного high. Сигнал
    несёт ``stop_loss = close[t] - atr_mult * ATR[t]`` — *дистанцию* стопа,
    заданную относительно close сигнальной свечи. Стоп принадлежит движку:
    после исполнения входа тот переносит дистанцию на фактическую цену
    исполнения и проверяет уровень внутри свечи.

    Выход: ``close[t] < min(low[t-exit_period .. t-1])`` — нижняя граница
    тоже по предыдущим свечам (тот же сдвиг 1). ``exit_period`` обязан быть
    меньше ``entry_period``: короткий выходной канал — суть системы Черепах
    (быстрая фиксация против медленного входа), равные/обратные периоды
    вырождают систему в чистое следование без защиты.

    Отдельного выхода по пробою стопа здесь нет — интрабарный стоп движка
    единственный стоп-путь.

    Индикаторы пересчитываются на растущем срезе свечей и кешируются по длине
    среза, что дёшево при данных объёмах свечей.
    """

    name = "donchian"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_mult: float = 2.0,
    ) -> None:
        for param_name, value in (
            ("entry_period", entry_period),
            ("exit_period", exit_period),
            ("atr_period", atr_period),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{param_name} must be an integer, got {value!r}")
        # entry_period >= 2: канал входа строится по предыдущим свечам, окно
        # из одной свечи превратило бы пробой в тривиальное "выше вчерашнего
        # high" без фильтрации шума.
        if entry_period < 2:
            raise ValueError(f"entry_period must be >= 2, got {entry_period}")
        if exit_period < 1:
            raise ValueError(f"exit_period must be >= 1, got {exit_period}")
        if exit_period >= entry_period:
            raise ValueError(
                f"exit_period ({exit_period}) must be smaller than "
                f"entry_period ({entry_period}): the shorter exit channel is "
                "the point of the Turtle system"
            )
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if isinstance(atr_mult, bool) or not isinstance(atr_mult, (int, float)):
            raise ValueError(f"atr_mult must be a number, got {atr_mult!r}")
        if not math.isfinite(atr_mult):
            raise ValueError(f"atr_mult must be finite, got {atr_mult!r}")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be positive, got {atr_mult}")
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self._reset_state()

    @property
    def warmup_period(self) -> int:
        return max(self.entry_period, self.exit_period) + self.atr_period

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        self._update_indicators(candles)
        i = len(candles) - 1

        close_now = float(candles["close"].iloc[i])
        channel_high = self._entry_channel.iloc[i]
        channel_low = self._exit_channel.iloc[i]
        if math.isnan(channel_high) or math.isnan(channel_low):
            return []

        if self._in_position:
            if close_now < channel_low:
                return [Signal(SignalKind.LONG_EXIT, reason=REASON_BREAKDOWN)]
        elif close_now > channel_high:
            atr_now = float(self._atr_line.iloc[i])
            if math.isnan(atr_now):
                return []
            stop_loss = close_now - self.atr_mult * atr_now
            return [
                Signal(
                    SignalKind.LONG_ENTRY,
                    reason=REASON_BREAKOUT_UP,
                    stop_loss=stop_loss,
                )
            ]
        return []

    def on_fill(self, fill: Fill) -> None:
        """Отслеживать, открыта ли позиция (используется для фильтра выходных сигналов)."""
        if fill.side == "buy":
            self._in_position = True
        elif fill.side == "sell":
            self._in_position = False

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._cache_len = -1
        self._entry_channel: pd.Series | None = None
        self._exit_channel: pd.Series | None = None
        self._atr_line: pd.Series | None = None
        self._in_position = False

    def _update_indicators(self, candles: pd.DataFrame) -> None:
        n = len(candles)
        if n == self._cache_len:
            return
        high = candles["high"]
        low = candles["low"]
        # Сдвиг 1: канал по предыдущим свечам, текущая в канал не входит.
        self._entry_channel = high.rolling(self.entry_period).max().shift(1)
        self._exit_channel = low.rolling(self.exit_period).min().shift(1)
        self._atr_line = atr_indicator(
            high, low, candles["close"], self.atr_period
        )
        self._cache_len = n
