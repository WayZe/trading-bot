"""Тесты векторных индикаторов на посчитанных вручную значениях."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_bot.indicators import atr, ema, rsi, sma


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


class TestSma:
    def test_known_values(self) -> None:
        result = sma(series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)

        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_warmup_is_exactly_period_candles(self) -> None:
        result = sma(series([10.0] * 10), 5)

        # Значение готово только после `period` свечей: NaN на 0..period-2,
        # первое валидное значение на индексе period-1.
        assert result.iloc[:4].isna().all()
        assert not math.isnan(result.iloc[4])
        assert result.iloc[4] == pytest.approx(10.0)

    def test_preserves_length_and_index(self) -> None:
        close = pd.Series([1.0, 5.0, 3.0, 7.0], index=[10, 20, 30, 40])

        result = sma(close, 2)

        assert len(result) == len(close)
        assert list(result.index) == [10, 20, 30, 40]

    def test_constant_series_gives_constant(self) -> None:
        result = sma(series([42.0] * 6), 3)

        assert result.iloc[2:].eq(42.0).all()

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            sma(series([1.0, 2.0]), 0)


class TestEma:
    def test_known_values(self) -> None:
        # span=3 -> alpha=0.5; рекурсия adjust=False, посчитано вручную:
        # 1, 1.5, 2.25, 3.125, 4.0625
        result = ema(series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)

        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == pytest.approx(2.25)
        assert result.iloc[3] == pytest.approx(3.125)
        assert result.iloc[4] == pytest.approx(4.0625)

    def test_warmup_matches_sma_semantics(self) -> None:
        result = ema(series([10.0] * 8), 4)

        assert result.iloc[:3].isna().all()
        assert result.iloc[3] == pytest.approx(10.0)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            ema(series([1.0, 2.0]), -1)


class TestRsi:
    def test_known_values(self) -> None:
        # RSI по Уайлдеру с period=3 (alpha=1/3), посчитано вручную; закрытия
        # [10, 11, 10, 11, 12]: ряды avg_gain/avg_loss дают
        # rsi[3] = 100 * (7/9) / (7/9 + 2/9) = 700/9,
        # rsi[4] = 100 * (23/27) / (23/27 + 4/27) = 2300/27.
        result = rsi(series([10.0, 11.0, 10.0, 11.0, 12.0]), 3)

        assert result.iloc[:3].isna().all()
        assert result.iloc[3] == pytest.approx(700.0 / 9.0)
        assert result.iloc[4] == pytest.approx(2300.0 / 27.0)

    def test_monotonic_up_is_100(self) -> None:
        result = rsi(series([float(i) for i in range(20)]), 14)

        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].to_list() == pytest.approx([100.0] * 6)

    def test_monotonic_down_is_0(self) -> None:
        result = rsi(series([float(20 - i) for i in range(20)]), 14)

        assert result.iloc[14:].to_list() == pytest.approx([0.0] * 6)

    def test_default_period_is_14(self) -> None:
        result = rsi(series([float(i) for i in range(20)]))

        assert result.iloc[:14].isna().all()
        assert result.iloc[14] == pytest.approx(100.0)


class TestAtr:
    def test_known_values(self) -> None:
        # Посчитано вручную: TR = [nan, 3, 2, 3, 2]; ewm по Уайлдеру (alpha=1/3):
        # atr[1]=3, atr[2]=2/3*3+2/3=8/3, atr[3]=2/3*8/3+1=25/9,
        # atr[4]=2/3*25/9+2/3=68/27.
        high = series([11.0, 13.0, 12.0, 14.0, 13.0])
        low = series([9.0, 11.0, 10.0, 12.0, 11.0])
        close = series([10.0, 12.0, 11.0, 13.0, 12.0])

        result = atr(high, low, close, 3)

        assert result.iloc[:3].isna().all()
        assert result.iloc[3] == pytest.approx(25.0 / 9.0)
        assert result.iloc[4] == pytest.approx(68.0 / 27.0)

    def test_constant_range_gives_constant_atr(self) -> None:
        # Свечи с high=close+1, low=close-1 и малыми шагами close всегда дают
        # TR = 2, так что ATR после прогрева равен ровно 2.
        n = 10
        close = series([100.0 + 0.5 * i for i in range(n)])
        high = close + 1.0
        low = close - 1.0

        result = atr(high, low, close, 3)

        assert result.iloc[:3].isna().all()
        assert result.iloc[3:].to_list() == pytest.approx([2.0] * (n - 3))

    def test_default_period_is_14(self) -> None:
        n = 20
        close = series([100.0 + 0.5 * i for i in range(n)])

        result = atr(close + 1.0, close - 1.0, close)

        assert result.iloc[:14].isna().all()
        assert result.iloc[14:].to_list() == pytest.approx([2.0] * (n - 14))

    def test_first_true_range_is_nan_not_high_minus_low(self) -> None:
        result = atr(series([12.0]), series([8.0]), series([10.0]), 1)

        assert result.isna().all()


class TestSharedContract:
    @pytest.mark.parametrize("period", [1, 2, 5])
    def test_all_indicators_same_length(self, period: int) -> None:
        n = 12
        close = series([100.0 + i for i in range(n)])
        high = close + 1.0
        low = close - 1.0

        assert len(sma(close, period)) == n
        assert len(ema(close, period)) == n
        assert len(rsi(close, period)) == n
        assert len(atr(high, low, close, period)) == n

    def test_no_nan_after_warmup(self) -> None:
        n = 30
        close = series([100.0 + 0.3 * i for i in range(n)])

        assert sma(close, 5).iloc[4:].notna().all()
        assert ema(close, 5).iloc[4:].notna().all()
        assert rsi(close, 5).iloc[5:].notna().all()
        assert atr(close + 1.0, close - 1.0, close, 5).iloc[5:].notna().all()

    def test_all_nan_series_input(self) -> None:
        close = series([np.nan, np.nan, np.nan, 1.0, 2.0])

        sma_result = sma(close, 2)
        assert sma_result.iloc[:3].isna().all()
