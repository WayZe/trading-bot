"""Событийный бэктест-движок.

Цикл по свечам (для свечи ``i``):

1. Исполнить ордер, поставленный в очередь на свече ``i - 1``, по ``open[i]``
   через брокера (сигнал по закрытой свече, исполнение по следующему open —
   без заглядывания в будущее).
2. Проверить интрабарные стоп-лосс/тейк-профит открытой позиции на свече
   ``i``: гэп через уровень исполняется по open (консервативно); при
   одновременном касании обоих уровней приоритет у стопа.
3. Переоценить эквити по рынку на ``close[i]``.
4. Вызвать ``strategy.on_candle(candles[:i+1])`` и поставить итоговые сигналы
   в очередь как ордера на следующий open.

Движок — единственный владелец стоп-лосс/тейк-профит: уровни, переносимые
входным сигналом, — это *дистанции*, заданные относительно close сигнальной
свечи; после исполнения входа они переякориваются на фактическую цену
исполнения (``active_stop = fill.price - (stop_ref_close - stop_loss)`` и
``active_take_profit = fill.price + (take_profit - stop_ref_close)``).
Неположительная дистанция после переякоривания отключает уровень.

Ордера, оставшиеся в очереди после последней свечи, никогда не исполняются и
отражаются в ``BacktestResult.n_pending_unfilled``; открытая позиция остаётся
в ``open_position`` (переоценивается по рынку в кривой эквити) и не
записывается как сделка.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig
from trading_bot.engine.broker import SIDE_BUY, SIDE_SELL, SimulatedBroker
from trading_bot.engine.portfolio import (
    TRADE_RECORD_FIELDS,
    Portfolio,
    Position,
)
from trading_bot.engine.stops import (
    REASON_STOP_LOSS,
    REASON_TAKE_PROFIT,
)
from trading_bot.engine.stops import (
    reanchor_stop_below as _reanchor_below,
)
from trading_bot.engine.stops import (
    reanchor_take_profit_above as _reanchor_above,
)
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Signal, SignalKind, Strategy

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")


def validate_candles(candles: pd.DataFrame) -> None:
    """Проверить фрейм свечей для бэктест-движка.

    Общая для :meth:`BacktestEngine.run` и research sweep, который
    валидирует свечи один раз перед прогоном комбинаций.

    Raises:
        ValueError: если свечи пусты, не хватает обязательных колонок, есть
            NaN в обязательных колонках или нарушен инвариант ``high >= low``.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in candles.columns]
    if missing:
        raise ValueError(f"candles miss required columns: {missing}")
    if candles.empty:
        raise ValueError("candles are empty: nothing to backtest")
    nan_columns = [col for col in REQUIRED_COLUMNS if candles[col].isna().any()]
    if nan_columns:
        raise ValueError(f"candles contain NaN values in columns: {nan_columns}")
    inverted = candles["high"] < candles["low"]
    if bool(inverted.any()):
        raise ValueError(
            f"candles contain {int(inverted.sum())} candle(s) where high < low"
        )


@dataclass
class _PendingOrder:
    """Ордер, поставленный сигналом в очередь и ждущий open следующей свечи.

    ``stop_loss`` / ``take_profit`` — уровни, вычисленные стратегией от
    close сигнальной свечи (``stop_ref_close``); после исполнения движок
    переносит их дистанции на фактическую цену исполнения.
    """

    side: str
    quantity: float
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    stop_ref_close: float | None = None


@dataclass
class BacktestResult:
    """Артефакты одного прогона бэктеста."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    candles_start: pd.Timestamp | None
    candles_end: pd.Timestamp | None
    open_position: Position | None
    n_pending_unfilled: int
    config: BacktestConfig | None = None


class BacktestEngine:
    """Прогоняет :class:`Strategy` по историческим свечам с симуляцией исполнения."""

    def __init__(
        self,
        strategy: Strategy,
        risk: RiskManager,
        broker: SimulatedBroker,
        start_cash: float,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk
        self.broker = broker
        self.start_cash = float(start_cash)
        self.config = config

    def run(self, candles: pd.DataFrame) -> BacktestResult:
        """Прогнать событийный цикл по ``candles`` и вернуть результат.

        Raises:
            ValueError: если свечи не проходят :func:`validate_candles`.
        """
        validate_candles(candles)

        n = len(candles)
        warmup = self.strategy.warmup_period
        if n <= warmup:
            logger.warning(
                "only %d candles available, strategy warmup is %d: no signals will fire",
                n,
                warmup,
            )

        self.strategy.reset()
        portfolio = Portfolio(self.start_cash)

        timestamps = candles["timestamp"]
        opens = candles["open"].to_numpy(dtype="float64")
        highs = candles["high"].to_numpy(dtype="float64")
        lows = candles["low"].to_numpy(dtype="float64")
        closes = candles["close"].to_numpy(dtype="float64")
        equity_values = np.empty(n, dtype="float64")

        pending: _PendingOrder | None = None
        active_stop: float | None = None
        active_tp: float | None = None
        entry_reason = ""

        for i in range(n):
            ts = timestamps.iloc[i]

            # 1. Исполнить ордер, поставленный на предыдущей свече, по этому open.
            if pending is not None:
                order, pending = pending, None
                if order.side == SIDE_BUY:
                    fill = self.broker.execute_market(
                        SIDE_BUY, order.quantity, opens[i], ts, order.reason
                    )
                    portfolio.buy(order.quantity, fill.price, fill.fee, ts)
                    active_stop = _reanchor_below(
                        order.stop_loss, order.stop_ref_close, fill.price
                    )
                    active_tp = _reanchor_above(
                        order.take_profit, order.stop_ref_close, fill.price
                    )
                    entry_reason = order.reason
                    self.strategy.on_fill(fill)
                    logger.debug(
                        "entry filled: %.6f @ %.4f (fee %.4f) at %s",
                        fill.quantity,
                        fill.price,
                        fill.fee,
                        ts,
                    )
                else:
                    fill = self.broker.execute_market(
                        SIDE_SELL, order.quantity, opens[i], ts, order.reason
                    )
                    portfolio.sell(
                        order.quantity,
                        fill.price,
                        fill.fee,
                        ts,
                        reason_exit=order.reason,
                        reason_entry=entry_reason,
                    )
                    active_stop = active_tp = None
                    entry_reason = ""
                    self.strategy.on_fill(fill)
                    logger.debug(
                        "exit filled: %.6f @ %.4f (fee %.4f) at %s",
                        fill.quantity,
                        fill.price,
                        fill.fee,
                        ts,
                    )

            # 2. Интрабарные стоп-лосс / тейк-профит на этой свече.
            if portfolio.position is not None:
                position = portfolio.position
                exit_price: float | None = None
                exit_reason = ""
                if active_stop is not None and lows[i] <= active_stop:
                    exit_price = min(opens[i], active_stop)  # гэп вниз -> исполнение по open
                    exit_reason = REASON_STOP_LOSS
                elif active_tp is not None and highs[i] >= active_tp:
                    exit_price = max(opens[i], active_tp)  # гэп вверх -> исполнение по open
                    exit_reason = REASON_TAKE_PROFIT
                if exit_price is not None:
                    fill = self.broker.execute_market(
                        SIDE_SELL, position.quantity, exit_price, ts, exit_reason
                    )
                    portfolio.sell(
                        position.quantity,
                        fill.price,
                        fill.fee,
                        ts,
                        reason_exit=exit_reason,
                        reason_entry=entry_reason,
                    )
                    active_stop = active_tp = None
                    entry_reason = ""
                    self.strategy.on_fill(fill)
                    logger.debug(
                        "%s filled: %.6f @ %.4f at %s",
                        exit_reason,
                        fill.quantity,
                        fill.price,
                        ts,
                    )

            # 3. Переоценить эквити по рынку на close.
            equity_values[i] = portfolio.equity(closes[i])

            # 4. Дать стратегии увидеть закрытую свечу и поставить ордера в очередь.
            if i >= warmup:
                for signal in self.strategy.on_candle(candles.iloc[: i + 1]):
                    pending = self._queue_signal(signal, pending, portfolio, closes[i], i)

        n_pending_unfilled = 0
        if pending is not None:
            n_pending_unfilled = 1
            logger.warning(
                "pending %s order (reason=%r) is left unexecuted: "
                "no candle remains after the last signal",
                pending.side,
                pending.reason,
            )

        equity_curve = pd.Series(equity_values, index=timestamps, name="equity")
        trades = pd.DataFrame(
            [asdict(trade) for trade in portfolio.trades], columns=TRADE_RECORD_FIELDS
        )
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            candles_start=pd.Timestamp(timestamps.iloc[0]),
            candles_end=pd.Timestamp(timestamps.iloc[-1]),
            open_position=portfolio.position,
            n_pending_unfilled=n_pending_unfilled,
            config=self.config,
        )

    def _queue_signal(
        self,
        signal: Signal,
        pending: _PendingOrder | None,
        portfolio: Portfolio,
        close: float,
        index: int,
    ) -> _PendingOrder | None:
        """Проверить сигнал на соответствие состоянию портфеля и поставить ордер в очередь."""
        if signal.kind is SignalKind.LONG_ENTRY:
            if portfolio.position is not None:
                logger.debug("LONG_ENTRY at candle %d ignored: position already open", index)
                return pending
            if pending is not None:
                logger.debug("LONG_ENTRY at candle %d ignored: order already pending", index)
                return pending
            quantity = self.risk.position_quantity(portfolio.equity(close), close)
            if quantity <= 0.0:
                logger.warning(
                    "LONG_ENTRY at candle %d skipped: quantity below min_notional "
                    "(equity %.2f, close %.2f)",
                    index,
                    portfolio.equity(close),
                    close,
                )
                return pending
            return _PendingOrder(
                side=SIDE_BUY,
                quantity=quantity,
                reason=signal.reason,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                stop_ref_close=close,
            )

        if signal.kind is SignalKind.LONG_EXIT:
            if portfolio.position is None:
                logger.debug("LONG_EXIT at candle %d ignored: no open position", index)
                return pending
            if pending is not None:
                logger.debug("LONG_EXIT at candle %d ignored: order already pending", index)
                return pending
            return _PendingOrder(
                side=SIDE_SELL, quantity=portfolio.position.quantity, reason=signal.reason
            )

        logger.warning("unknown signal kind %r ignored", signal.kind)
        return pending


def build_engine(cfg: BacktestConfig, strategy: Strategy) -> BacktestEngine:
    """Создать :class:`BacktestEngine`, полностью сконфигурированный из ``cfg``.

    Единственное место, где исполнительный стек (лимиты риска, комиссии и
    проскальзывание брокера, стартовый капитал) собирается из
    :class:`BacktestConfig`; общее для CLI backtest и research sweep. Конфиг
    прикрепляется к движку, так что итоговый :class:`BacktestResult` несёт
    его обратно.
    """
    return BacktestEngine(
        strategy=strategy,
        risk=RiskManager(
            position_size_pct=cfg.position_size_pct,
            quantity_precision=cfg.quantity_precision,
            min_notional=cfg.min_notional,
        ),
        broker=SimulatedBroker(fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps),
        start_cash=cfg.start_cash,
        config=cfg,
    )
