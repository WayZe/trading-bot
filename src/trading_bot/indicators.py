"""Векторные технические индикаторы (на pandas, написаны самостоятельно).

Общая семантика NaN/прогрева: каждый индикатор возвращает ``pd.Series`` той же
длины и с тем же индексом, что и вход, с ведущими ``NaN`` до тех пор, пока
индикатор не «готов» (не увидел достаточно входных свечей):

- ``sma`` / ``ema``: первое валидное значение на индексе ``period - 1``
  (то есть после ``period`` закрытий);
- ``rsi`` / ``atr`` (сглаживание по Уайлдеру): первое валидное значение на
  индексе ``period`` (они потребляют ``period`` ценовых дельт / истинных
  диапазонов).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Простая скользящая средняя по ``period`` закрытий.

    Первые ``period - 1`` значений — NaN; первое валидное значение стоит
    на индексе ``period - 1``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Экспоненциальная скользящая средняя с ``span=period`` (``adjust=False``).

    ``ewm`` естественно даёт значения начиная с первого элемента; чтобы
    сохранить семантику прогрева единой с :func:`sma`, значения до индекса
    ``period - 1`` заменяются на NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = close.ewm(span=period, adjust=False).mean()
    result.iloc[: period - 1] = np.nan
    return result


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Индекс относительной силы, классическое сглаживание по Уайлдеру.

    Прибыли и убытки сглаживаются через ``ewm(alpha=1/period, adjust=False)``
    (рекурсивная форма усреднения Уайлдера). Эквивалентно
    ``100 - 100 / (1 + rs)``, но записано как ``100 * ag / (ag + al)``, чтобы
    избежать бесконечностей, когда одна из сторон в точности нулевая. Первые
    ``period`` значений — NaN; первое валидное значение стоит на индексе
    ``period``.

    На плоском рынке (все дельты нулевые после прогрева) оба сглаженных
    средних равны нулю, поэтому результат — NaN, а не нейтральные 50.
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
    """Средний истинный диапазон со сглаживанием по Уайлдеру.

    Истинный диапазон использует предыдущее закрытие (``max(high - low,
    |high - prev_close|, |low - prev_close|)``) и сглаживается через
    ``ewm(alpha=1/period, adjust=False)``. Первые ``period`` значений — NaN;
    первое валидное значение стоит на индексе ``period``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = np.nan  # истинный диапазон не определён для первой свечи
    result = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    result.iloc[:period] = np.nan
    return result
