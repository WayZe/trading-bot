"""Performance metrics for backtest runs (pure functions, no I/O)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.data.exchange import timeframe_to_ms

# A year in milliseconds (365.25 days): the shared time base for CAGR years
# and Sharpe annualization.
YEAR_MS = 365.25 * 24 * 60 * 60 * 1000
MS_PER_DAY = 24 * 60 * 60 * 1000

_DAY = pd.Timedelta(days=1)
_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class MetricsReport:
    """Performance metrics of a single backtest run.

    Equity metrics:
        total_return_pct: ``(final / first - 1) * 100``.
        final_equity: last equity value, in quote currency.
        cagr_pct: annualized total return over ``span_days / 365.25`` years;
            ``None`` when the curve has no time span (a single point).
        sharpe: annualized Sharpe ratio of per-candle equity returns
            (risk-free rate 0); ``None`` when return dispersion is zero
            or undefined.
        max_drawdown_pct: deepest peak-to-trough decline, in percent (<= 0).
        max_drawdown_days: days from the peak of the deepest drawdown to its
            recovery, or to the end of the curve if never recovered; ``None``
            when the curve never drew down.
        span_days: time span of the equity curve, in days.

    Trade metrics (all ``None`` when the run closed no trades):
        n_trades: number of closed round-trip trades (always an int).
        winrate_pct: share of trades with ``pnl > 0``.
        profit_factor: gross profit / gross loss; ``inf`` when there are
            no losing trades.
        avg_trade_pnl, avg_win, avg_loss, best_trade, worst_trade: pnl
            statistics in quote currency (``avg_win``/``avg_loss`` are
            ``None`` when there are no wins/losses respectively).
        avg_holding_hours: mean ``exit_ts - entry_ts`` in hours.
        total_fees: estimated fees if a ``fee_rate`` was supplied (see
            :func:`compute_metrics`), else ``None``.
    """

    total_return_pct: float
    final_equity: float
    cagr_pct: float | None
    sharpe: float | None
    max_drawdown_pct: float
    max_drawdown_days: float | None
    span_days: float

    n_trades: int
    winrate_pct: float | None
    profit_factor: float | None
    avg_trade_pnl: float | None
    avg_win: float | None
    avg_loss: float | None
    best_trade: float | None
    worst_trade: float | None
    avg_holding_hours: float | None
    total_fees: float | None

    @property
    def short_span(self) -> bool:
        """True when the run is shorter than a year (CAGR is extrapolated)."""
        return self.span_days * MS_PER_DAY < YEAR_MS


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Buy & hold benchmark metrics over the same period as a backtest.

    Attributes:
        total_return_pct: ``(final / first - 1) * 100``.
        cagr_pct: annualized total return; ``None`` for a single-point curve.
        sharpe: annualized Sharpe ratio of per-candle returns (risk-free
            rate 0); ``None`` when the dispersion is zero or undefined.
        max_drawdown_pct: deepest peak-to-trough decline, in percent (<= 0).
    """

    total_return_pct: float
    cagr_pct: float | None
    sharpe: float | None
    max_drawdown_pct: float


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    timeframe: str,
    *,
    fee_rate: float | None = None,
) -> MetricsReport:
    """Compute performance metrics from backtest artifacts.

    Args:
        equity: mark-to-market equity curve indexed by candle timestamps.
        trades: closed trades with the ``TradeRecord`` schema (``entry_ts``,
            ``exit_ts``, ``entry_price``, ``exit_price``, ``quantity``,
            ``pnl``, ...); may be empty.
        timeframe: candle timeframe (``"15m"``, ``"4h"``, ``"1d"`` ...) used
            to annualize Sharpe: periods per year = ``YEAR_MS / timeframe_ms``.
        fee_rate: optional taker fee per side; when given, ``total_fees`` is
            computed as ``fee_rate * sum(quantity * (entry_price +
            exit_price))`` over the recorded fill prices (which already
            include slippage, matching the broker's fee model).

    Raises:
        ValueError: if the equity curve is empty or the timeframe is invalid.
    """
    if equity.empty:
        raise ValueError("equity curve is empty: nothing to report")

    equity_metrics = _equity_metrics(equity, timeframe)
    trade_metrics = _trade_metrics(trades, fee_rate=fee_rate)

    return MetricsReport(**equity_metrics, **trade_metrics)


def _equity_metrics(equity: pd.Series, timeframe: str) -> dict[str, float | None]:
    """Core equity-curve metrics shared by strategy and benchmark reports.

    Returns a dict with ``total_return_pct``, ``final_equity``, ``cagr_pct``,
    ``sharpe``, ``max_drawdown_pct``, ``max_drawdown_days`` and ``span_days``.
    """
    values = equity.to_numpy(dtype="float64")
    start_equity = float(values[0])
    final_equity = float(values[-1])
    total_return_pct = (final_equity / start_equity - 1.0) * 100.0

    span = equity.index[-1] - equity.index[0]
    span_days = float(span / _DAY)
    if span_days > 0.0:
        years = span_days * MS_PER_DAY / YEAR_MS
        cagr_pct = ((final_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = None

    returns = equity.astype("float64").pct_change().dropna()
    sharpe: float | None = None
    if len(returns) >= 2:
        std = float(returns.std())  # ddof=1
        if not math.isnan(std) and std > 0.0:
            periods_per_year = YEAR_MS / timeframe_to_ms(timeframe)
            sharpe = float(returns.mean()) / std * math.sqrt(periods_per_year)

    max_drawdown_pct, max_drawdown_days = _drawdown(equity, values)

    return {
        "total_return_pct": total_return_pct,
        "final_equity": final_equity,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_days": max_drawdown_days,
        "span_days": span_days,
    }


def benchmark_equity(close: pd.Series, start_cash: float) -> pd.Series:
    """Build the buy & hold equity curve from candle closes.

    The full ``start_cash`` is invested at the first close; the curve is
    ``start_cash * close / close[0]`` over the same index. Pure function.

    Raises:
        ValueError: if ``close`` is empty or starts at zero.
    """
    if close.empty:
        raise ValueError("close series is empty: nothing to benchmark")
    first_close = float(close.iloc[0])
    if first_close == 0.0:
        raise ValueError("first close is zero: cannot build the benchmark curve")
    return start_cash * close.astype("float64") / first_close


def compute_benchmark_metrics(equity: pd.Series, timeframe: str) -> BenchmarkMetrics:
    """Compute buy & hold metrics from a benchmark equity curve.

    Reuses the same formulas as :func:`compute_metrics` (via the shared
    equity helpers), without trade metrics.

    Raises:
        ValueError: if the equity curve is empty or the timeframe is invalid.
    """
    if equity.empty:
        raise ValueError("equity curve is empty: nothing to report")
    fields = _equity_metrics(equity, timeframe)
    return BenchmarkMetrics(
        total_return_pct=fields["total_return_pct"],
        cagr_pct=fields["cagr_pct"],
        sharpe=fields["sharpe"],
        max_drawdown_pct=fields["max_drawdown_pct"],
    )


def _drawdown(equity: pd.Series, values: np.ndarray) -> tuple[float, float | None]:
    """Return ``(max drawdown %, duration in days)`` of the deepest drawdown.

    The duration spans from the peak preceding the deepest trough to the
    first recovery to that peak (or to the end of the curve when the equity
    never recovers).
    """
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_dd_pct = float(drawdown.min() * 100.0)
    if max_dd_pct >= 0.0:
        return 0.0, None

    trough_pos = int(drawdown.argmin())
    peak_pos = int(values[: trough_pos + 1].argmax())
    peak_value = values[peak_pos]
    recovered = np.nonzero(values[trough_pos:] >= peak_value)[0]
    end_pos = trough_pos + int(recovered[0]) if recovered.size else len(values) - 1
    duration = float((equity.index[end_pos] - equity.index[peak_pos]) / _DAY)
    return max_dd_pct, duration


def _trade_metrics(
    trades: pd.DataFrame, *, fee_rate: float | None
) -> dict[str, float | int | None]:
    """Compute trade-level metrics; every metric is ``None`` for 0 trades."""
    n_trades = int(len(trades))
    if n_trades == 0:
        return {
            "n_trades": 0,
            "winrate_pct": None,
            "profit_factor": None,
            "avg_trade_pnl": None,
            "avg_win": None,
            "avg_loss": None,
            "best_trade": None,
            "worst_trade": None,
            "avg_holding_hours": None,
            "total_fees": None,
        }

    pnl = trades["pnl"].astype("float64")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    entry_ts = pd.to_datetime(trades["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(trades["exit_ts"], utc=True)
    avg_holding_hours = float((exit_ts - entry_ts).mean() / _HOUR)

    total_fees: float | None = None
    if fee_rate is not None and fee_rate >= 0:
        notional = trades["quantity"].astype("float64") * (
            trades["entry_price"].astype("float64")
            + trades["exit_price"].astype("float64")
        )
        total_fees = float(fee_rate * notional.sum())

    return {
        "n_trades": n_trades,
        "winrate_pct": len(wins) / n_trades * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else math.inf,
        "avg_trade_pnl": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_holding_hours": avg_holding_hours,
        "total_fees": total_fees,
    }
