"""Тесты тренд-фильтра TrendFiltered и фабрики sma_cross_trend."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, rows_to_df
from trading_bot.strategy import STRATEGY_REGISTRY, create_strategy
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy
from trading_bot.strategy.sma_cross import SmaCrossStrategy
from trading_bot.strategy.trend_filter import TrendFiltered

# Тот же синтетический кросс, что в тестах sma_cross: вход происходит ровно
# один раз (на индексе 7, close=98.5, стоп close - 2*ATR = 94.5).
SPREAD = 1.0
UP_CROSS_CLOSES = [
    100.0, 99.5, 99.0, 98.5, 98.0, 97.5,  # падение: fast ниже slow
    98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0,  # рост: fast пересекает вверх
]
# Ступенька: восемь плоских закрытий, затем резкий скачок — кросс вверх
# на индексе 8 при close=104 выше трендовой SMA(5)=100.8.
STEP_UP_CLOSES = [100.0] * 8 + [104.0]
# V-разворот: спад, затем слабое восстановление — кросс вверх на индексе 8
# при close=101 ниже трендовой SMA(9)≈101.11.
DOWN_V_CLOSES = [104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 100.0, 101.0]


def candles_from_closes(closes: list[float]) -> pd.DataFrame:
    rows = [
        [BASE_MS + i * HOUR_MS, c, c + SPREAD, c - SPREAD, c, 1.0]
        for i, c in enumerate(closes)
    ]
    return rows_to_df(rows)


def feed(strategy: Strategy, candles: pd.DataFrame) -> list[tuple[int, Signal]]:
    """Скормить растущие срезы; вернуть [(индекс свечи, сигнал), ...]."""
    emitted: list[tuple[int, Signal]] = []
    for i in range(len(candles)):
        for signal in strategy.on_candle(candles.iloc[: i + 1]):
            emitted.append((i, signal))
    return emitted


class StubInner(Strategy):
    """Заглушка: отдаёт заранее заданные сигналы и считает делегированные вызовы."""

    name = "stub"

    def __init__(self, warmup: int = 1) -> None:
        self._warmup = warmup
        self.signals_to_emit: list[Signal] = []
        self.fills: list[Fill] = []
        self.reset_count = 0

    @property
    def warmup_period(self) -> int:
        return self._warmup

    def on_candle(self, candles: pd.DataFrame) -> list[Signal]:
        return list(self.signals_to_emit)

    def on_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

    def reset(self) -> None:
        self.reset_count += 1


def make_entry(stop_loss: float | None = None) -> Signal:
    return Signal(SignalKind.LONG_ENTRY, reason="stub entry", stop_loss=stop_loss)


def make_exit() -> Signal:
    return Signal(SignalKind.LONG_EXIT, reason="stub exit")


def make_fill() -> Fill:
    return Fill(side="buy", price=100.0, quantity=1.0, timestamp=pd.Timestamp(0, tz="UTC"),
                fee=0.0, reason="")


class TestGating:
    def test_entry_blocked_in_downtrend(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([110.0, 109.0, 108.0, 107.0, 106.0, 105.0])
        inner.signals_to_emit = [make_entry()]

        assert wrapper.on_candle(candles) == []

    def test_entry_blocked_when_close_not_above_sma(self) -> None:
        # Плоский ряд: close == SMA — вход не разрешается (нужно строго выше).
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([100.0] * 6)
        inner.signals_to_emit = [make_entry()]

        assert wrapper.on_candle(candles) == []

    def test_entry_passes_in_uptrend(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([101.0, 102.0, 103.0, 104.0, 105.0])
        entry = make_entry()
        inner.signals_to_emit = [entry]

        assert wrapper.on_candle(candles) == [entry]

    def test_exit_passes_in_downtrend(self) -> None:
        # Выходы не фильтруются: медвежий тренд не удерживает позицию.
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([110.0, 109.0, 108.0, 107.0, 106.0])
        exit_signal = make_exit()
        inner.signals_to_emit = [exit_signal]

        assert wrapper.on_candle(candles) == [exit_signal]

    def test_only_entry_dropped_from_mixed_signals(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([110.0, 109.0, 108.0, 107.0, 106.0])
        exit_signal = make_exit()
        inner.signals_to_emit = [exit_signal, make_entry(stop_loss=90.0)]

        assert wrapper.on_candle(candles) == [exit_signal]

    def test_entry_blocked_while_trend_sma_is_nan(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=5)
        inner.signals_to_emit = [make_entry()]

        # 4 свечи < trend_period: SMA ещё NaN, вход запрещён даже в росте.
        warmup_candles = candles_from_closes([100.0, 101.0, 102.0, 103.0])
        assert wrapper.on_candle(warmup_candles) == []

        # На пятой свече SMA определена и растущий тренд пропускает вход.
        full_candles = candles_from_closes([100.0, 101.0, 102.0, 103.0, 104.0])
        assert wrapper.on_candle(full_candles) == [inner.signals_to_emit[0]]

    def test_stop_loss_and_take_profit_passed_through(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([101.0, 102.0, 103.0, 104.0, 105.0])
        entry = Signal(
            SignalKind.LONG_ENTRY, reason="stub entry", stop_loss=95.0, take_profit=110.0
        )
        inner.signals_to_emit = [entry]

        result = wrapper.on_candle(candles)

        assert result == [entry]
        assert result[0].stop_loss == 95.0
        assert result[0].take_profit == 110.0


class TestContract:
    def test_composite_name(self) -> None:
        wrapper = TrendFiltered(StubInner(), trend_period=7)

        assert wrapper.name == "stub_trend7"

    def test_warmup_is_max_of_inner_and_trend(self) -> None:
        assert TrendFiltered(StubInner(warmup=3), trend_period=10).warmup_period == 10
        assert TrendFiltered(StubInner(warmup=12), trend_period=5).warmup_period == 12

    def test_on_fill_delegates(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        fill = make_fill()

        wrapper.on_fill(fill)

        assert inner.fills == [fill]

    def test_reset_delegates_and_clears_cache(self) -> None:
        inner = StubInner()
        wrapper = TrendFiltered(inner, trend_period=3)
        candles = candles_from_closes([101.0, 102.0, 103.0, 104.0, 105.0])
        inner.signals_to_emit = [make_entry()]
        first_run = feed(wrapper, candles)

        wrapper.reset()

        assert inner.reset_count == 1
        # После сброса (включая кеш SMA) прогон повторяется с тем же результатом.
        second_run = feed(wrapper, candles)
        assert [(i, s.kind, s.reason) for i, s in second_run] == [
            (i, s.kind, s.reason) for i, s in first_run
        ]

    def test_inner_must_be_strategy(self) -> None:
        with pytest.raises(TypeError, match="inner must be a Strategy"):
            TrendFiltered("not a strategy", trend_period=3)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"trend_period": "5"}, "must be an integer"),
            ({"trend_period": True}, "must be an integer"),
            ({"trend_period": 5.0}, "must be an integer"),
            ({"trend_period": 0}, "must be >= 1"),
            ({"trend_period": -3}, "must be >= 1"),
            ({"trend_period": 3, "trend_source": ""}, "non-empty string"),
        ],
    )
    def test_bad_params_raise_value_error(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TrendFiltered(StubInner(), **kwargs)


class TestWithRealInner:
    def make_wrapper(self, trend_period: int) -> TrendFiltered:
        inner = SmaCrossStrategy(fast=3, slow=5, atr_period=3, atr_mult=2.0)
        return TrendFiltered(inner, trend_period=trend_period)

    def test_entry_blocked_below_trend_sma(self) -> None:
        candles = candles_from_closes(DOWN_V_CLOSES)

        # Предусловие-оракул: без фильтра вход есть ровно один (на последней
        # свече), и в точке входа close ниже трендовой SMA(9) — фильтр обязан
        # его снять.
        bare = SmaCrossStrategy(fast=3, slow=5, atr_period=3, atr_mult=2.0)
        emitted = feed(bare, candles)
        assert len(emitted) == 1
        entry_index = emitted[0][0]
        trend_sma = candles["close"].rolling(9).mean()
        assert candles["close"].iloc[entry_index] < trend_sma.iloc[entry_index]

        wrapped = feed(self.make_wrapper(trend_period=9), candles)

        assert wrapped == []

    def test_entry_blocked_during_trend_warmup(self) -> None:
        # trend_period=20 > число свечей: трендовая SMA NaN на всём срезе,
        # поэтому вход внутренней стратегии не проходит ни разу.
        wrapper = self.make_wrapper(trend_period=20)
        candles = candles_from_closes(UP_CROSS_CLOSES)

        assert wrapper.warmup_period == 20
        assert feed(wrapper, candles) == []

    def test_entry_passes_with_stop_loss_above_trend(self) -> None:
        wrapper = self.make_wrapper(trend_period=5)
        candles = candles_from_closes(STEP_UP_CLOSES)

        emitted = feed(wrapper, candles)

        assert len(emitted) == 1
        index, signal = emitted[0]
        assert index == 8
        assert signal.kind is SignalKind.LONG_ENTRY
        assert signal.reason == "sma cross up"
        # Скачок свечи (close 104, prev close 100) обновляет Wilder-ATR(3)
        # до 3.0, дистанция стопа 2*ATR от close сигнальной свечи.
        assert signal.stop_loss == pytest.approx(104.0 - 2.0 * 3.0)


class TestRegistry:
    def test_sma_cross_trend_registered(self) -> None:
        assert "sma_cross_trend" in STRATEGY_REGISTRY

    def test_create_sma_cross_trend(self) -> None:
        strategy = create_strategy(
            "sma_cross_trend",
            {"fast": 3, "slow": 5, "atr_period": 3, "atr_mult": 2.0, "trend_period": 100},
        )

        assert isinstance(strategy, TrendFiltered)
        assert isinstance(strategy.inner, SmaCrossStrategy)
        assert strategy.inner.fast == 3
        assert strategy.trend_period == 100
        assert strategy.name == "sma_cross_trend100"
        assert strategy.warmup_period == 100

    def test_create_sma_cross_trend_inner_defaults(self) -> None:
        strategy = create_strategy("sma_cross_trend", {"trend_period": 200})

        assert isinstance(strategy, TrendFiltered)
        assert isinstance(strategy.inner, SmaCrossStrategy)
        assert (strategy.inner.fast, strategy.inner.slow) == (20, 50)
        assert strategy.warmup_period == 200

    def test_create_sma_cross_trend_unknown_param(self) -> None:
        with pytest.raises(ValueError, match="invalid strategy_params"):
            create_strategy("sma_cross_trend", {"trend_peryod": 5})
