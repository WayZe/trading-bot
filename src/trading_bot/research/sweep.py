"""Parameter-grid sweeps over the backtest engine (pure functions, no I/O)."""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import pandas as pd

from trading_bot.config import BacktestConfig
from trading_bot.engine.backtest import build_engine, validate_candles
from trading_bot.report.metrics import compute_metrics
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

# Safety cap: a sweep is an interactive research tool, not a batch farm.
MAX_COMBINATIONS = 200

# Metric columns of a sweep result row (after the parameter columns).
METRIC_COLUMNS = (
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "max_drawdown_pct",
    "n_trades",
    "winrate_pct",
    "profit_factor",
    "final_equity",
)


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Expand a parameter grid into all combinations (Cartesian product).

    An empty grid yields a single empty combination ``[{}]`` — one run with
    the base parameters.
    """
    if not param_grid:
        return [{}]
    names = list(param_grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(param_grid[name] for name in names))
    ]


def slice_candles(candles: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Keep candles with ``cfg.start <= timestamp < cfg.end``.

    Shared by the CLI backtest and research sweeps so both run over exactly
    the same period for a given config.

    Raises:
        ValueError: if ``candles`` is empty.
    """
    if candles.empty:
        raise ValueError("no candles to slice: the dataset is empty")
    ts = candles["timestamp"]
    start_ts = pd.Timestamp(cfg.start, tz="UTC")
    if ts.iloc[0] > start_ts:
        logger.warning(
            "данные начинаются с %s, позже запрошенного start=%s",
            ts.iloc[0],
            cfg.start,
        )
    mask = ts >= start_ts
    if cfg.end is not None:
        mask &= ts < pd.Timestamp(cfg.end, tz="UTC")
    return candles.loc[mask].reset_index(drop=True)


def run_sweep(
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Run a backtest for every parameter combination in ``param_grid``.

    Each combination overrides ``base_config.strategy_params``; everything
    else (period, fees, slippage, risk limits, start cash) comes from
    ``base_config`` via :func:`build_engine`. The candles are sliced to the
    config period first (same semantics as the CLI backtest) and validated
    once before the combination loop.

    A combination that fails to build or run (e.g. ``fast >= slow``) does not
    abort the sweep: any exception is caught, logged with its traceback at
    debug level, and the row carries a brief error text in ``error`` with NaN
    metrics.

    Returns:
        One row per combination: the combination parameters as separate
        columns, then :data:`METRIC_COLUMNS`, then ``error`` (``None`` when
        the run succeeded).

    Raises:
        ValueError: if the grid expands to more than :data:`MAX_COMBINATIONS`
            combinations, ``candles`` is empty, or no candles remain after
            the period slice (including invalid candle data).
    """
    n_combinations = math.prod(len(values) for values in param_grid.values()) or 1
    if n_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"parameter grid expands to {n_combinations} combinations, "
            f"above the limit of {MAX_COMBINATIONS}; narrow the grid"
        )
    combos = expand_grid(param_grid)
    sliced = slice_candles(candles, base_config)
    if sliced.empty:
        raise ValueError(
            f"no candles in the requested period "
            f"{base_config.start} .. {base_config.end or 'latest available'}"
        )
    validate_candles(sliced)
    rows = [_run_combination(base_config, combo, sliced) for combo in combos]
    return pd.DataFrame(rows, columns=[*param_grid, *METRIC_COLUMNS, "error"])


def _run_combination(
    base_config: BacktestConfig, combo: dict, candles: pd.DataFrame
) -> dict[str, Any]:
    """Run one sweep combination; any failure becomes an ``error`` row."""
    row: dict[str, Any] = dict(combo)
    params = {**base_config.strategy_params, **combo}
    try:
        strategy = create_strategy(base_config.strategy, params)
        engine = build_engine(base_config, strategy)
        result = engine.run(candles)
        metrics = compute_metrics(result.equity_curve, result.trades, base_config.timeframe)
    except Exception as error:  # noqa: BLE001 - one bad combo must not kill the sweep
        logger.debug("sweep combination %s failed", combo, exc_info=True)
        row.update(dict.fromkeys(METRIC_COLUMNS, math.nan))
        row["error"] = str(error) or type(error).__name__
        return row
    row.update(
        total_return_pct=metrics.total_return_pct,
        cagr_pct=metrics.cagr_pct,
        sharpe=metrics.sharpe,
        max_drawdown_pct=metrics.max_drawdown_pct,
        n_trades=metrics.n_trades,
        winrate_pct=metrics.winrate_pct,
        profit_factor=metrics.profit_factor,
        final_equity=metrics.final_equity,
        error=None,
    )
    return row
