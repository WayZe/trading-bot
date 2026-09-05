"""Parameter-grid sweeps over the backtest engine (pure functions, no I/O)."""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import pandas as pd

from trading_bot.config import BacktestConfig
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.report.metrics import compute_metrics
from trading_bot.risk import RiskManager
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
    """
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
    ``base_config``. The candles are sliced to the config period first (same
    semantics as the CLI backtest).

    A combination that fails to build or run (e.g. ``fast >= slow``) does not
    abort the sweep: its row carries the exception text in ``error`` and NaN
    metrics.

    Returns:
        One row per combination: the combination parameters as separate
        columns, then :data:`METRIC_COLUMNS`, then ``error`` (``None`` when
        the run succeeded).

    Raises:
        ValueError: if the grid expands to more than :data:`MAX_COMBINATIONS`
            combinations or no candles remain after the period slice.
    """
    combos = expand_grid(param_grid)
    if len(combos) > MAX_COMBINATIONS:
        raise ValueError(
            f"parameter grid expands to {len(combos)} combinations, "
            f"above the limit of {MAX_COMBINATIONS}; narrow the grid"
        )
    sliced = slice_candles(candles, base_config)
    if sliced.empty:
        raise ValueError(
            f"no candles in the requested period "
            f"{base_config.start} .. {base_config.end or 'latest available'}"
        )
    rows = [_run_combination(base_config, combo, sliced) for combo in combos]
    return pd.DataFrame(rows, columns=[*param_grid, *METRIC_COLUMNS, "error"])


def _run_combination(
    base_config: BacktestConfig, combo: dict, candles: pd.DataFrame
) -> dict[str, Any]:
    """Run one sweep combination; a failure becomes an ``error`` row."""
    row: dict[str, Any] = dict(combo)
    params = {**base_config.strategy_params, **combo}
    try:
        strategy = create_strategy(base_config.strategy, params)
        engine = BacktestEngine(
            strategy=strategy,
            risk=RiskManager(
                position_size_pct=base_config.position_size_pct,
                quantity_precision=base_config.quantity_precision,
                min_notional=base_config.min_notional,
            ),
            broker=SimulatedBroker(
                fee_rate=base_config.fee_rate, slippage_bps=base_config.slippage_bps
            ),
            start_cash=base_config.start_cash,
        )
        result = engine.run(candles)
        metrics = compute_metrics(result.equity_curve, result.trades, base_config.timeframe)
    except (TypeError, ValueError) as error:
        logger.warning("sweep combination %s failed: %s", combo, error)
        row.update(dict.fromkeys(METRIC_COLUMNS, math.nan))
        row["error"] = str(error)
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
