"""Vector technical indicators (pandas-based, written in-house).

Shared NaN/warmup semantics: every indicator returns a ``pd.Series`` of the
same length and index as the input, with leading ``NaN`` values until the
indicator is "ready" (has seen enough input candles):

- ``sma`` / ``ema``: the first valid value is at index ``period - 1``
  (i.e. after ``period`` closes);
- ``rsi`` / ``atr`` (Wilder-style): the first valid value is at index
  ``period`` (they consume ``period`` price deltas / true ranges).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` closes.

    Leading ``period - 1`` values are NaN; the first valid value sits at
    index ``period - 1``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with ``span=period`` (``adjust=False``).

    ``ewm`` naturally produces values from the first element; to keep the
    warmup semantics uniform with :func:`sma`, values before index
    ``period - 1`` are replaced with NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = close.ewm(span=period, adjust=False).mean()
    result.iloc[: period - 1] = np.nan
    return result


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, classic Wilder smoothing.

    Gains and losses are smoothed with ``ewm(alpha=1/period, adjust=False)``
    (the recursive form of Wilder's averaging). Equivalent to
    ``100 - 100 / (1 + rs)`` but written as ``100 * ag / (ag + al)`` to avoid
    infinities when one side is exactly zero. Leading ``period`` values are
    NaN; the first valid value sits at index ``period``.

    On a flat market (every delta zero after warmup) both smoothed averages
    are zero, so the result is NaN rather than a neutral 50.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    result = 100.0 * avg_gain / (avg_gain + avg_loss)
    result.iloc[:period] = np.nan
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing.

    True Range uses the previous close (``max(high - low,
    |high - prev_close|, |low - prev_close|)``) and is smoothed with
    ``ewm(alpha=1/period, adjust=False)``. Leading ``period`` values are NaN;
    the first valid value sits at index ``period``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = np.nan  # True Range is undefined for the first candle
    result = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    result.iloc[:period] = np.nan
    return result
