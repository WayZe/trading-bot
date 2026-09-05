"""Обёртка-композитор: тренд-фильтр поверх внутренней стратегии."""

from __future__ import annotations

import math

import pandas as pd

from trading_bot.indicators import sma
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy


class TrendFiltered(Strategy):
    """Гейт входов по тренду: вход разрешён только выше долгой SMA.

    Обёртка вокруг другой стратегии (``inner``) и не содержит собственной
    логики входа/выхода: сигналы ``inner`` проходят через фильтр.

    Семантика фильтра:

    - гейтится **только** ``LONG_ENTRY``: если close последней закрытой свечи
      не выше ``sma(trend_source, trend_period)`` (медвежий тренд), входные
      сигналы ``inner`` отбрасываются, и позиция просто не открывается;
    - ``LONG_EXIT`` и приложенные к сигналам стопы/тейки **не фильтруются** —
      выходить из позиции в медвежьем тренде легитимно, а фильтрация выхода
      удерживала бы убыточную позицию; ``stop_loss``/``take_profit``
      входного сигнала передаются движку как есть;
    - трендовая SMA ещё NaN (свечей меньше ``trend_period``) — вход также
      запрещён: фильтр консервативен, пока тренд не определён.

    ``name`` — составной, из имени ``inner`` и периода тренда
    (например ``"sma_cross_trend200"``): в таблицах walk-forward и meta-файлах
    видно, какой период фильтра дал результат. ``warmup_period`` — максимум из
    прогрева ``inner`` и ``trend_period``. ``on_fill``/``reset`` делегируются
    ``inner``; собственный кеш трендовой SMA сбрасывается вместе с ним.

    Индикаторы пересчитываются на растущем срезе свечей и кешируются по длине
    среза (тот же приём, что в ``SmaCrossStrategy``).
    """

    def __init__(self, inner: Strategy, trend_period: int, trend_source: str = "close") -> None:
        if not isinstance(inner, Strategy):
            raise TypeError(f"inner must be a Strategy instance, got {type(inner).__name__}")
        if isinstance(trend_period, bool) or not isinstance(trend_period, int):
            raise ValueError(f"trend_period must be an integer, got {trend_period!r}")
        if trend_period < 1:
            raise ValueError(f"trend_period must be >= 1, got {trend_period}")
        if not isinstance(trend_source, str) or not trend_source:
            raise ValueError(f"trend_source must be a non-empty string, got {trend_source!r}")
        self.inner = inner
        self.trend_period = trend_period
        self.trend_source = trend_source
        self._reset_state()

    @property
    def name(self) -> str:
        """Составное имя: ``<inner.name>_trend<trend_period>``."""
        return f"{self.inner.name}_trend{self.trend_period}"

    @property
    def warmup_period(self) -> int:
        return max(self.inner.warmup_period, self.trend_period)

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        self._update_indicators(candles)
        signals = self.inner.on_candle(candles)
        if not signals:
            return []
        trend_now = self._trend_line.iloc[-1]
        close_now = float(candles[self.trend_source].iloc[-1])
        if math.isnan(trend_now) or close_now <= float(trend_now):
            # Медвежий тренд (или тренд ещё не определён): входы запрещены,
            # выходы проходят всегда.
            return [s for s in signals if s.kind is not SignalKind.LONG_ENTRY]
        return signals

    def on_fill(self, fill: Fill) -> None:
        self.inner.on_fill(fill)

    def reset(self) -> None:
        self.inner.reset()
        self._reset_state()

    def _reset_state(self) -> None:
        self._cache_len = -1
        self._trend_line: pd.Series | None = None

    def _update_indicators(self, candles: pd.DataFrame) -> None:
        n = len(candles)
        if n == self._cache_len:
            return
        self._trend_line = sma(candles[self.trend_source], self.trend_period)
        self._cache_len = n
