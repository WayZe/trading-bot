"""Walk-forward analysis: re-optimize on in-sample, verify on out-of-sample.

The history is split into consecutive windows of an in-sample (IS) span
followed by an out-of-sample (OOS) span. Per window:

1. the parameter grid is swept over the IS span (reusing :func:`run_sweep`);
2. the best valid combination by ``objective`` is selected;
3. that combination is run once over the OOS span, preceded by a *lead-in*
   of ``strategy.warmup_period`` candles so indicators warm up on data
   before the OOS start (the engine starts every window with an empty
   portfolio, so a position can only be *opened* during the lead-in);
4. only the equity part at/after ``oos_start`` counts towards OOS metrics
   and the stitched curve — PnL earned before ``oos_start`` is discarded.
   A trade opened during the lead-in but closed inside the OOS span still
   contributes its OOS part to the equity/return, yet is excluded from the
   trade metrics — hence a window can show a nonzero return with zero
   recorded trades.

The stitched curve is the concatenation of the per-window OOS equity parts,
each normalized to its own first value and multiplied by the accumulated
growth: an estimate of what the strategy would have returned with periodic
re-optimization, free of look-ahead bias. The calendar gaps between windows
become zero returns in ``pct_change``, which slightly distorts the
annualized Sharpe of the stitched curve — a deliberate simplification.

Pure functions over candles; all I/O (artifacts, plots) lives in the CLI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig
from trading_bot.data.exchange import timeframe_to_ms
from trading_bot.engine.backtest import build_engine
from trading_bot.engine.portfolio import TRADE_RECORD_FIELDS
from trading_bot.report.metrics import compute_metrics
from trading_bot.research.sweep import MAX_COMBINATIONS, run_sweep
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

# Safety cap: walk-forward runs a full sweep per window; too many windows
# means the user probably asked for OOS days too small for the dataset.
MAX_WINDOWS = 50

MODES = ("rolling", "anchored")
OBJECTIVES = ("sharpe", "total_return_pct")

# Columns of a walk-forward results row (after the parameter columns):
# grid parameter names that collide with these would corrupt results.csv,
# so they are rejected at the ``--param`` validation point.
WINDOW_COLUMNS = (
    "is_start",
    "is_end",
    "oos_start",
    "oos_end",
    "mode",
    "is_objective",
    "oos_return_pct",
    "oos_sharpe",
    "oos_max_dd_pct",
    "oos_trades",
    "error",
)

_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Window:
    """One walk-forward window; ``is_end == oos_start`` (no gap, no overlap)."""

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass
class WindowResult:
    """Outcome of a single walk-forward window.

    ``best_params``/``is_objective`` are ``None`` and ``error`` carries the
    reason when the window failed (no valid IS combination or an OOS run
    error); ``oos_*`` metrics are ``None`` for failed windows.
    """

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    mode: str
    best_params: dict | None = None
    is_objective: float | None = None
    oos_return_pct: float | None = None
    oos_sharpe: float | None = None
    oos_max_dd_pct: float | None = None
    oos_trades: int | None = None
    error: str | None = None


@dataclass
class WalkForwardResult:
    """Artifacts of a walk-forward run.

    Attributes:
        windows: one :class:`WindowResult` per planned window, chronological,
            failed windows included (they simply contribute no OOS data).
        stitched_equity: concatenated OOS equity parts, each normalized to
            its first value and multiplied by the accumulated growth; starts
            at 1.0. Empty when no window succeeded.
        trades: all OOS trades across windows (``TradeRecord`` schema).
        meta: JSON-safe run parameters (mode, days, objective, grid, ...).
    """

    windows: list[WindowResult] = field(default_factory=list)
    stitched_equity: pd.Series = field(
        default_factory=lambda: pd.Series(
            dtype="float64", name="equity", index=pd.DatetimeIndex([], tz="UTC")
        )
    )
    trades: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_RECORD_FIELDS)
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def to_frame(self, param_grid: dict[str, list]) -> pd.DataFrame:
        """Window records as a dataframe (grid params, dates, metrics, error)."""
        rows = []
        for window in self.windows:
            row: dict[str, Any] = dict.fromkeys(param_grid, math.nan)
            for name, value in (window.best_params or {}).items():
                row[name] = value
            row.update(
                is_start=window.is_start.isoformat(),
                is_end=window.is_end.isoformat(),
                oos_start=window.oos_start.isoformat(),
                oos_end=window.oos_end.isoformat(),
                mode=window.mode,
                is_objective=window.is_objective,
                oos_return_pct=window.oos_return_pct,
                oos_sharpe=window.oos_sharpe,
                oos_max_dd_pct=window.oos_max_dd_pct,
                oos_trades=window.oos_trades,
                error=window.error,
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=[*param_grid, *WINDOW_COLUMNS])


def plan_windows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    is_days: int,
    oos_days: int,
    mode: str = "rolling",
) -> list[Window]:
    """Plan the walk-forward windows over ``[start_ts, end_ts)``.

    Boundaries are in calendar days.

    ``rolling``: fixed-length IS/OOS blocks tile the timeline back-to-back —
    a new IS starts where the previous OOS ended, so the optimizer never
    sees a previous window's OOS data (the OOS spans are ``is_days`` apart).

    ``anchored``: ``is_start`` stays at the overall start and the IS span
    grows by ``oos_days`` each step; the OOS spans tile the tail without
    overlap.

    A window is planned only if its OOS span ends at or before ``end_ts``.

    Args:
        start_ts: overall start (inclusive), tz-aware UTC.
        end_ts: overall end (exclusive), tz-aware UTC.
        is_days: in-sample span in days; must be positive.
        oos_days: out-of-sample span in days; must be positive.
        mode: ``"rolling"`` or ``"anchored"``.

    Returns:
        Chronological windows; every window satisfies ``is_end == oos_start``.

    Raises:
        ValueError: if ``is_days``/``oos_days`` are not positive, ``mode`` is
            unknown, the data span is shorter than ``is_days + oos_days`` (no
            window fits), or more than :data:`MAX_WINDOWS` windows would be
            planned.
    """
    if is_days <= 0:
        raise ValueError(f"is_days must be positive, got {is_days}")
    if oos_days <= 0:
        raise ValueError(f"oos_days must be positive, got {oos_days}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {', '.join(MODES)}")

    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    is_delta = pd.Timedelta(days=is_days)
    oos_delta = pd.Timedelta(days=oos_days)

    windows: list[Window] = []
    step = 0
    while True:
        if mode == "rolling":
            is_start = start_ts + step * (is_delta + oos_delta)
            is_end = is_start + is_delta
        else:  # anchored
            is_start = start_ts
            is_end = start_ts + is_delta + step * oos_delta
        oos_start = is_end
        oos_end = oos_start + oos_delta
        if oos_end > end_ts:
            break
        windows.append(Window(is_start, is_end, oos_start, oos_end))
        step += 1

    if not windows:
        available = (end_ts - start_ts) / _DAY
        raise ValueError(
            f"data span {available:.1f} days is shorter than the walk-forward "
            f"requirement of {is_days + oos_days} days "
            f"(is_days={is_days} + oos_days={oos_days}); "
            "download more history or reduce --is-days/--oos-days"
        )
    if len(windows) > MAX_WINDOWS:
        raise ValueError(
            f"{len(windows)} walk-forward windows exceed the limit of "
            f"{MAX_WINDOWS}; increase --oos-days or narrow the data range"
        )
    return windows


def run_walkforward(
    candles: pd.DataFrame,
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    *,
    is_days: int,
    oos_days: int,
    mode: str = "rolling",
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Run walk-forward analysis: per-window IS optimization -> OOS verification.

    Windows come from :func:`plan_windows` over the available candle range.
    The IS sweep reuses :func:`run_sweep` (including its error-row semantics)
    via a period-adjusted copy of ``base_config``; the best valid row by
    ``objective`` (NaN/None objective values are not rankable) is then run
    once over the OOS span with a lead-in of ``strategy.warmup_period``
    candles before ``oos_start`` (fewer if the data does not reach back that
    far). OOS metrics are computed on the equity part at/after ``oos_start``
    and trades entering at/after ``oos_start``, so PnL earned before
    ``oos_start`` is deliberately discarded; the engine starts each window
    with an empty portfolio, so the lead-in can only open positions, and a
    lead-in trade closed inside OOS contributes to the equity but not to
    the trade metrics.

    A window whose IS sweep fails (empty IS slice due to a data gap, invalid
    candles, ...) becomes an error row; the remaining windows still run.

    Raises:
        ValueError: if ``candles`` is empty, ``objective`` is unknown, the
            grid exceeds :data:`~trading_bot.research.sweep.MAX_COMBINATIONS`
            (the cap is the same for every window, so it is checked once up
            front), or window planning fails (see :func:`plan_windows`).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; available: {', '.join(OBJECTIVES)}")
    if candles.empty:
        raise ValueError("no candles: the dataset is empty")
    n_combinations = math.prod(len(values) for values in param_grid.values()) or 1
    if n_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"parameter grid expands to {n_combinations} combinations, "
            f"above the limit of {MAX_COMBINATIONS}; narrow the grid"
        )

    step_delta = pd.Timedelta(milliseconds=timeframe_to_ms(base_config.timeframe))
    start_ts = pd.Timestamp(candles["timestamp"].iloc[0])
    end_ts = pd.Timestamp(candles["timestamp"].iloc[-1]) + step_delta
    windows = plan_windows(start_ts, end_ts, is_days, oos_days, mode)
    logger.info(
        "walk-forward: %d %s window(s), IS %dd / OOS %dd, objective=%s",
        len(windows),
        mode,
        is_days,
        oos_days,
        objective,
    )

    results = WalkForwardResult(
        meta={
            "symbol": base_config.symbol,
            "timeframe": base_config.timeframe,
            "strategy": base_config.strategy,
            "mode": mode,
            "is_days": is_days,
            "oos_days": oos_days,
            "objective": objective,
            "param_grid": {name: list(values) for name, values in param_grid.items()},
            "n_windows": len(windows),
            "data_start": start_ts.isoformat(),
            "data_end": end_ts.isoformat(),
        }
    )

    segments: list[pd.Series] = []
    accumulated = 1.0
    for window in windows:
        window_result, oos_part, oos_trades = _run_window(
            candles, base_config, param_grid, window, mode, objective
        )
        results.windows.append(window_result)
        if window_result.error is not None:
            continue
        segments.append(_stitch_segment(oos_part, accumulated))
        accumulated *= float(oos_part.iloc[-1]) / float(oos_part.iloc[0])
        results.trades = (
            oos_trades if results.trades.empty
            else pd.concat([results.trades, oos_trades], ignore_index=True)
        )

    if segments:
        results.stitched_equity = pd.concat(segments)
    return results


def _run_window(
    candles: pd.DataFrame,
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    window: Window,
    mode: str,
    objective: str,
) -> tuple[WindowResult, pd.Series | None, pd.DataFrame | None]:
    """Run one window: IS sweep, best-combo selection, OOS verification.

    Returns ``(result, oos_equity_part, oos_trades)``; the last two are
    ``None`` when the window failed.
    """
    result = WindowResult(
        is_start=window.is_start,
        is_end=window.is_end,
        oos_start=window.oos_start,
        oos_end=window.oos_end,
        mode=mode,
    )
    # Same slice run_sweep would build from the config period; checked here
    # so a data gap (e.g. a missing chunk of history) fails only this window.
    if _slice_range(candles, window.is_start, window.is_end).empty:
        result.error = "no candles in IS window"
        return result, None, None
    try:
        is_frame = run_sweep(
            base_config.model_copy(
                update={
                    "start": window.is_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": window.is_end.strftime("%Y-%m-%d %H:%M:%S"),
                }
            ),
            param_grid,
            candles,
        )
    except ValueError as error:  # one bad IS window must not kill the run
        # the combinations cap is checked up front in run_walkforward, so a
        # cap ValueError can never reach this point
        logger.debug("walk-forward window %s: IS sweep failed", window, exc_info=True)
        result.error = str(error) or type(error).__name__
        return result, None, None
    valid = is_frame[is_frame["error"].isna()]
    if valid.empty:
        result.error = "no valid combos in IS"
        return result, None, None
    scores = pd.to_numeric(valid[objective], errors="coerce")
    if not bool(scores.notna().any()):
        result.error = f"no valid combos in IS with a finite {objective}"
        return result, None, None

    best = valid.loc[scores.idxmax()]
    # dataframe cells come back as numpy scalars; the strategy constructors
    # (and json.dumps) expect plain Python int/float
    result.best_params = {name: _pythonize(best[name]) for name in param_grid}
    result.is_objective = float(best[objective])

    try:
        strategy = create_strategy(
            base_config.strategy, {**base_config.strategy_params, **result.best_params}
        )
        lead_in = pd.Timedelta(
            milliseconds=strategy.warmup_period * timeframe_to_ms(base_config.timeframe)
        )
        oos_slice = _slice_range(candles, window.oos_start - lead_in, window.oos_end)
        engine_result = build_engine(base_config, strategy).run(oos_slice)
    except Exception as error:  # noqa: BLE001 - one bad window must not kill the run
        logger.debug("walk-forward window %s failed", window, exc_info=True)
        result.error = str(error) or type(error).__name__
        return result, None, None

    oos_part = engine_result.equity_curve.loc[engine_result.equity_curve.index >= window.oos_start]
    if oos_part.empty:
        result.error = "no OOS candles at/after oos_start"
        return result, None, None
    oos_trades = _filter_trades_by_entry(engine_result.trades, window.oos_start)
    metrics = compute_metrics(oos_part, oos_trades, base_config.timeframe)
    result.oos_return_pct = metrics.total_return_pct
    result.oos_sharpe = metrics.sharpe
    result.oos_max_dd_pct = metrics.max_drawdown_pct
    result.oos_trades = metrics.n_trades
    return result, oos_part, oos_trades


def _pythonize(value: Any) -> Any:
    """Convert a numpy scalar from a dataframe cell to a plain Python scalar."""
    return value.item() if isinstance(value, np.generic) else value


def _slice_range(candles: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep candles with ``start <= timestamp < end`` (``start`` may predate data)."""
    ts = candles["timestamp"]
    return candles.loc[(ts >= start) & (ts < end)].reset_index(drop=True)


def _filter_trades_by_entry(trades: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    """Keep trades that entered at/after ``oos_start``."""
    if trades.empty:
        return trades
    entry_ts = pd.to_datetime(trades["entry_ts"], utc=True)
    return trades.loc[entry_ts >= oos_start].reset_index(drop=True)


def _stitch_segment(oos_part: pd.Series, accumulated: float) -> pd.Series:
    """Normalize one OOS equity part to its first value and scale by ``accumulated``."""
    return oos_part.astype("float64") / float(oos_part.iloc[0]) * accumulated
