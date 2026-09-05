"""Тесты стратегии пробоя Дончиана на синтетических рядах свечей."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, rows_to_df
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.indicators import atr as atr_indicator
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Fill, Signal, SignalKind
from trading_bot.strategy.donchian import DonchianBreakoutStrategy

# Базовая свеча: close 100, high 101, low 99. При шагах, не выводящих
# истинный диапазон за 2.0, Wilder-ATR(3) после прогрева равен ровно 2.0.
FLAT = (101.0, 99.0, 100.0)


def candles_from_hlc(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Перевести строки ``(high, low, close)`` во фрейм свечей (open = close)."""
    out = [
        [BASE_MS + i * HOUR_MS, c, h, lo, c, 1.0]
        for i, (h, lo, c) in enumerate(rows)
    ]
    return rows_to_df(out)


def feed(strategy: DonchianBreakoutStrategy, candles: pd.DataFrame) -> list[tuple[int, Signal]]:
    """Скормить растущие срезы; вернуть [(индекс свечи, сигнал), ...]."""
    emitted: list[tuple[int, Signal]] = []
    for i in range(len(candles)):
        for signal in strategy.on_candle(candles.iloc[: i + 1]):
            emitted.append((i, signal))
    return emitted


def feed_with_fills(
    strategy: DonchianBreakoutStrategy, candles: pd.DataFrame
) -> list[tuple[int, Signal]]:
    """Как :func:`feed`, но исполненные входы/выходы сообщаются стратегии через ``on_fill``."""
    emitted: list[tuple[int, Signal]] = []
    for i in range(len(candles)):
        for signal in strategy.on_candle(candles.iloc[: i + 1]):
            emitted.append((i, signal))
            if signal.kind is SignalKind.LONG_ENTRY:
                strategy.on_fill(
                    Fill(
                        side="buy",
                        price=float(candles["close"].iloc[i]),
                        quantity=1.0,
                        timestamp=candles["timestamp"].iloc[i],
                        fee=0.0,
                        reason=signal.reason,
                    )
                )
            elif signal.kind is SignalKind.LONG_EXIT:
                strategy.on_fill(
                    Fill(
                        side="sell",
                        price=float(candles["close"].iloc[i]),
                        quantity=1.0,
                        timestamp=candles["timestamp"].iloc[i],
                        fee=0.0,
                        reason=signal.reason,
                    )
                )
    return emitted


def make_fill(side: str, price: float, timestamp: pd.Timestamp) -> Fill:
    return Fill(side=side, price=price, quantity=1.0, timestamp=timestamp, fee=0.0, reason="")


def make_strategy() -> DonchianBreakoutStrategy:
    return DonchianBreakoutStrategy(entry_period=3, exit_period=2, atr_period=3, atr_mult=2.0)


class TestContract:
    def test_name_and_warmup(self) -> None:
        strategy = make_strategy()

        assert strategy.name == "donchian"
        assert strategy.warmup_period == max(3, 2) + 3

    def test_default_warmup(self) -> None:
        strategy = DonchianBreakoutStrategy()

        assert (strategy.entry_period, strategy.exit_period) == (20, 10)
        assert strategy.warmup_period == 20 + 14

    def test_custom_params_stick(self) -> None:
        strategy = DonchianBreakoutStrategy(entry_period=5, exit_period=2, atr_period=3)

        assert strategy.entry_period == 5
        assert strategy.exit_period == 2
        assert strategy.warmup_period == 5 + 3


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"entry_period": "3"}, "must be an integer"),
            ({"entry_period": True}, "must be an integer"),
            ({"exit_period": 2.0}, "must be an integer"),
            ({"atr_period": None}, "must be an integer"),
            ({"entry_period": 1}, "entry_period must be >= 2"),
            ({"entry_period": 0}, "entry_period must be >= 2"),
            ({"exit_period": 0}, "exit_period must be >= 1"),
            ({"atr_period": 0}, "atr_period must be >= 1"),
            # Классика Черепах: канал выхода короче канала входа.
            ({"entry_period": 3, "exit_period": 3}, "must be smaller than entry_period"),
            ({"entry_period": 3, "exit_period": 5}, "must be smaller than entry_period"),
            ({"atr_mult": "2"}, "must be a number"),
            ({"atr_mult": float("nan")}, "must be finite"),
            ({"atr_mult": float("inf")}, "must be finite"),
            ({"atr_mult": 0.0}, "must be positive"),
            ({"atr_mult": -1.0}, "must be positive"),
        ],
    )
    def test_bad_params_raise_value_error(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DonchianBreakoutStrategy(**kwargs)


class TestEntry:
    def test_breakout_above_previous_high_emits_one_entry(self) -> None:
        strategy = make_strategy()
        # Пять плоских свечей (канал max(high[2..4]) = 101), затем свеча с
        # close 102.5 выше канала: единственный вход на индексе 5.
        candles = candles_from_hlc([FLAT] * 5 + [(103.0, 101.0, 102.5)])

        emitted = feed(strategy, candles)

        assert [(i, s.kind) for i, s in emitted] == [(5, SignalKind.LONG_ENTRY)]
        assert emitted[0][1].reason == "donchian breakout up"

    def test_entry_matches_independent_channel_oracle(self) -> None:
        # Независимый оракул: каналы считаются прямо в тесте через
        # rolling(...).max/min().shift(1), состояние позиции — конечный автомат.
        rows = [
            FLAT,  # 0
            FLAT,  # 1
            FLAT,  # 2
            (103.0, 101.0, 102.5),  # 3 пробой вверх -> вход
            (104.5, 102.5, 104.0),  # 4
            (104.5, 101.5, 102.0),  # 5
            (102.5, 100.5, 101.0),  # 6 пробой канала выхода вниз -> выход
            (101.5, 99.5, 100.0),  # 7
            FLAT,  # 8
            (103.5, 101.5, 103.0),  # 9 пробой вверх -> вход
            (105.0, 103.0, 104.5),  # 10
            (105.0, 100.0, 100.5),  # 11 пробой вниз -> выход
            FLAT,  # 12
            FLAT,  # 13
        ]
        candles = candles_from_hlc(rows)
        high, low, close = candles["high"], candles["low"], candles["close"]
        entry_channel = high.rolling(3).max().shift(1)
        exit_channel = low.rolling(2).min().shift(1)

        expected: list[tuple[int, SignalKind]] = []
        in_position = False
        for i in range(len(candles)):
            ch_high, ch_low = entry_channel.iloc[i], exit_channel.iloc[i]
            if pd.isna(ch_high) or pd.isna(ch_low):
                continue
            if not in_position and close.iloc[i] > ch_high:
                expected.append((i, SignalKind.LONG_ENTRY))
                in_position = True
            elif in_position and close.iloc[i] < ch_low:
                expected.append((i, SignalKind.LONG_EXIT))
                in_position = False

        strategy = make_strategy()
        emitted = feed_with_fills(strategy, candles)

        assert [(i, s.kind) for i, s in emitted] == expected
        assert expected == [(3, SignalKind.LONG_ENTRY), (6, SignalKind.LONG_EXIT),
                            (9, SignalKind.LONG_ENTRY), (11, SignalKind.LONG_EXIT)]

    def test_no_entry_inside_channel(self) -> None:
        # Закрытие строго внутри канала (100.5 < 101) — входа нет.
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 5 + [(101.5, 99.5, 100.5)])

        assert feed(strategy, candles) == []

    def test_no_entry_when_close_equals_channel_max(self) -> None:
        # close == max предыдущих high ровно: неравенство строгое, входа нет.
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 5 + [(101.0, 99.0, 101.0)])

        assert feed(strategy, candles) == []

    def test_channel_excludes_current_candle_no_look_ahead(self) -> None:
        # Свеча с высоким собственным wick (high 106) и close 105 выше
        # максимума ПРЕДЫДУЩИХ свечей (101): вход обязан случиться. Реализация,
        # включающая текущую свечу в канал, получила бы max = 106 > close и
        # входа бы не дала, — сам факт входа доказывает сдвиг канала.
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 3 + [(106.0, 104.0, 105.0)])

        emitted = feed(strategy, candles)

        assert [(i, s.kind) for i, s in emitted] == [(3, SignalKind.LONG_ENTRY)]

    def test_no_entry_during_warmup(self) -> None:
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 5)

        assert feed(strategy, candles) == []

    def test_entry_signal_fields(self) -> None:
        # Пробойная свеча (close 102, high 102, low 100) держит истинный
        # диапазон равным 2.0: ATR(3) не меняется, стоп = close - 2*ATR = 98.
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 3 + [(102.0, 100.0, 102.0)])

        emitted = feed(strategy, candles)

        index, signal = emitted[0]
        assert index == 3
        assert signal.kind is SignalKind.LONG_ENTRY
        assert signal.reason == "donchian breakout up"
        assert signal.take_profit is None
        atr_oracle = atr_indicator(
            candles["high"], candles["low"], candles["close"], 3
        )
        expected_stop = float(candles["close"].iloc[3]) - 2.0 * float(atr_oracle.iloc[3])
        assert signal.stop_loss == pytest.approx(expected_stop)
        assert signal.stop_loss == pytest.approx(98.0)


class TestExit:
    def make_strategy_in_position(self) -> tuple[DonchianBreakoutStrategy, pd.DataFrame, int]:
        """Прогон до входа (индекс 3) и сообщение о buy-fill; вернуть контекст."""
        strategy = make_strategy()
        candles = candles_from_hlc([FLAT] * 3 + [(102.0, 100.0, 102.0)])
        emitted = feed(strategy, candles)
        entry_index, entry = emitted[0]
        strategy.on_fill(
            make_fill(
                "buy", float(candles["close"].iloc[entry_index]),
                candles["timestamp"].iloc[entry_index],
            )
        )
        return strategy, candles, entry_index

    def test_breakdown_below_previous_low_emits_exit(self) -> None:
        strategy, candles, entry_index = self.make_strategy_in_position()
        # Канал выхода = min(low[3], low[4]) = min(100, 101) = 100; close 99
        # ниже канала -> выход. Свеча с close ниже собственного low невозможна,
        # поэтому здесь close 99 при low 98.5: собственный low тоже в канале
        # не участвует.
        tail = candles_from_hlc([(103.0, 101.0, 102.0), (101.0, 98.5, 99.0)])

        exits: list[tuple[int, Signal]] = []
        for i in range(len(tail)):
            extended = pd.concat([candles, tail.iloc[: i + 1]], ignore_index=True)
            for signal in strategy.on_candle(extended):
                exits.append((len(candles) + i, signal))

        assert [(i, s.kind) for i, s in exits] == [(entry_index + 2, SignalKind.LONG_EXIT)]
        assert exits[0][1].reason == "donchian breakdown"
        assert exits[0][1].stop_loss is None

    def test_exit_channel_excludes_current_candle(self) -> None:
        # Свеча с длинным нижним wick (low 99.5), но close 102 — выше канала
        # выхода (100): выхода нет. Реализация, включающая текущую свечу в
        # канал, получила бы min = 99.5 < close и вышла бы ошибочно.
        strategy, candles, _ = self.make_strategy_in_position()
        tail = candles_from_hlc([(103.0, 101.0, 102.0), (102.0, 99.5, 102.0)])

        for i in range(len(tail)):
            extended = pd.concat([candles, tail.iloc[: i + 1]], ignore_index=True)
            assert strategy.on_candle(extended) == []

    def test_no_exit_without_position(self) -> None:
        strategy = make_strategy()
        candles = candles_from_hlc(
            [FLAT] * 3
            + [(102.0, 100.0, 102.0), (103.0, 101.0, 102.0), (101.0, 98.5, 99.0)]
        )

        emitted = feed(strategy, candles)  # без on_fill -> стратегия остаётся без позиции

        assert all(s.kind is SignalKind.LONG_ENTRY for _, s in emitted)


class TestReset:
    def test_reset_clears_state_for_a_rerun(self) -> None:
        strategy = make_strategy()
        candles = candles_from_hlc(
            [FLAT] * 3
            + [(102.0, 100.0, 102.0), (103.0, 101.0, 102.0), (101.0, 98.5, 99.0)]
        )

        first_run = feed(strategy, candles)
        entry_index, _ = first_run[0]
        strategy.on_fill(
            make_fill("buy", 102.0, candles["timestamp"].iloc[entry_index])
        )
        assert strategy._in_position is True

        strategy.reset()

        assert strategy._in_position is False
        second_run = feed(strategy, candles)
        assert [(i, s.kind, s.reason) for i, s in second_run] == [
            (i, s.kind, s.reason) for i, s in first_run
        ]


FEE = 0.001
SLIP = 5.0  # bps
START_CASH = 1_000.0


class TestEngineIntegration:
    """Пара вход/выход внутри полного цикла BacktestEngine."""

    def test_full_cycle_one_trade_entry_and_exit_reasons(self) -> None:
        # Рост до пробоя (вход исполняется на open[8]), затем пробой короткого
        # канала вниз (выход исполняется на open[12]). Переякоренный стоп
        # (fill - 4 = 98.05) остаётся ниже low всех свечей позиции (99+),
        # поэтому выход — именно сигналом стратегии, а не стопом движка.
        rows = [
            [BASE_MS + i * HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1.0]
            for i in range(7)  # 0..6: плоско, канал входа = 101
        ]
        rows += [
            # 7: пробой вверх (close 102 > 101); TR = 2, стоп = 102 - 2*2 = 98.
            [BASE_MS + 7 * HOUR_MS, 100.0, 102.0, 100.0, 102.0, 1.0],
        ]
        rows += [
            [BASE_MS + i * HOUR_MS, 102.0, 103.0, 101.0, 102.0, 1.0]
            for i in range(8, 11)  # 8..10: вход исполнен, плоско выше
        ]
        rows += [
            # 11: пробой канала выхода (min(low[9..10]) = 101) вниз; low 99
            # выше переякоренного стопа 98.05 — стоп не мешает.
            [BASE_MS + 11 * HOUR_MS, 102.0, 102.0, 99.0, 99.5, 1.0],
            [BASE_MS + 12 * HOUR_MS, 99.5, 100.0, 99.0, 99.5, 1.0],  # выход по open
            [BASE_MS + 13 * HOUR_MS, 99.5, 100.0, 99.0, 99.5, 1.0],
        ]
        candles = rows_to_df(rows)
        engine = BacktestEngine(
            strategy=DonchianBreakoutStrategy(
                entry_period=3, exit_period=2, atr_period=3, atr_mult=2.0
            ),
            risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=5.0),
            broker=SimulatedBroker(fee_rate=FEE, slippage_bps=SLIP),
            start_cash=START_CASH,
        )

        result = engine.run(candles)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        # Сигнал на свече 7 -> вход по open свечи 8; сигнал на свече 11 ->
        # выход по open свечи 12.
        assert trade["entry_ts"] == candles["timestamp"].iloc[8]
        assert trade["exit_ts"] == candles["timestamp"].iloc[12]
        assert trade["reason_entry"] == "donchian breakout up"
        assert trade["reason_exit"] == "donchian breakdown"
        assert result.open_position is None
        assert result.n_pending_unfilled == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(
            START_CASH + float(trade["pnl"])
        )
