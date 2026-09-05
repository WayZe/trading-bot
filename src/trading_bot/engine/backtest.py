"""Event-driven backtest engine.

Per-candle loop (for candle ``i``):

1. Execute the order queued on candle ``i - 1`` at ``open[i]`` via the broker
   (signal on a closed candle, execution on the next open — no look-ahead).
2. Check intrabar stop-loss / take-profit of the open position on candle
   ``i``: a gap through the level fills at the open (conservative), the stop
   has priority if both levels are hit in the same candle.
3. Mark equity to market at ``close[i]``.
4. Call ``strategy.on_candle(candles[:i+1])`` and queue resulting signals as
   orders for the next open.

Orders still pending after the last candle are never executed and are
reported in ``BacktestResult.n_pending_unfilled``; an open position stays in
``open_position`` (marked to market in the equity curve) and is not recorded
as a trade.
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
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Signal, SignalKind, Strategy

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")

REASON_STOP_LOSS = "stop loss"
REASON_TAKE_PROFIT = "take profit"


@dataclass
class _PendingOrder:
    """An order queued by a signal, waiting for the next candle's open."""

    side: str
    quantity: float
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class BacktestResult:
    """Artifacts of a single backtest run."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    candles_start: pd.Timestamp | None
    candles_end: pd.Timestamp | None
    open_position: Position | None
    n_pending_unfilled: int
    config: BacktestConfig | None = None


class BacktestEngine:
    """Runs a :class:`Strategy` over historical candles with simulated fills."""

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
        """Run the event loop over ``candles`` and return the result.

        Raises:
            ValueError: if the candles are empty or miss required columns.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in candles.columns]
        if missing:
            raise ValueError(f"candles miss required columns: {missing}")
        if candles.empty:
            raise ValueError("candles are empty: nothing to backtest")

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

            # 1. Execute the order queued on the previous candle at this open.
            if pending is not None:
                order, pending = pending, None
                if order.side == SIDE_BUY:
                    fill = self.broker.execute_market(
                        SIDE_BUY, order.quantity, opens[i], ts, order.reason
                    )
                    portfolio.buy(order.quantity, fill.price, fill.fee, ts)
                    active_stop = order.stop_loss
                    active_tp = order.take_profit
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

            # 2. Intrabar stop-loss / take-profit on this candle.
            if portfolio.position is not None:
                position = portfolio.position
                exit_price: float | None = None
                exit_reason = ""
                if active_stop is not None and lows[i] <= active_stop:
                    exit_price = min(opens[i], active_stop)  # gap down -> fill at open
                    exit_reason = REASON_STOP_LOSS
                elif active_tp is not None and highs[i] >= active_tp:
                    exit_price = max(opens[i], active_tp)  # gap up -> fill at open
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

            # 3. Mark equity to market at the close.
            equity_values[i] = portfolio.equity(closes[i])

            # 4. Let the strategy see the closed candle and queue orders.
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
        """Validate a signal against the portfolio state and queue an order."""
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
