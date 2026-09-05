"""Sweep по сетке параметров поверх бэктест-движка (чистые функции, без I/O)."""

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

# Предохранительный потолок: sweep — интерактивный исследовательский
# инструмент, а не батч-ферма.
MAX_COMBINATIONS = 200

# Колонки метрик в строке результата sweep (после колонок параметров).
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
    """Развернуть сетку параметров во все комбинации (декартово произведение).

    Пустая сетка даёт единственную пустую комбинацию ``[{}]`` — один прогон
    с базовыми параметрами.
    """
    if not param_grid:
        return [{}]
    names = list(param_grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(param_grid[name] for name in names))
    ]


def slice_candles(candles: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Оставить свечи с ``cfg.start <= timestamp < cfg.end``.

    Общая для CLI backtest и research sweep, чтобы оба гонялись ровно по
    одному периоду при одном и том же конфиге.

    Raises:
        ValueError: если ``candles`` пуст.
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
    """Прогнать бэктест для каждой комбинации параметров из ``param_grid``.

    Каждая комбинация переопределяет ``base_config.strategy_params``; всё
    остальное (период, комиссии, проскальзывание, лимиты риска, стартовый
    капитал) берётся из ``base_config`` через :func:`build_engine`. Свечи
    сначала срезаются до периода конфига (та же семантика, что у CLI
    backtest) и валидируются один раз перед циклом по комбинациям.

    Комбинация, которая не собирается или не выполняется (например,
    ``fast >= slow``), не прерывает sweep: любое исключение ловится,
    логируется с трейсбеком на уровне debug, а строка несёт краткий текст
    ошибки в ``error`` и NaN в метриках.

    Returns:
        Одна строка на комбинацию: параметры комбинации отдельными колонками,
        затем :data:`METRIC_COLUMNS`, затем ``error`` (``None`` при успехе).

    Raises:
        ValueError: если сетка разворачивается больше чем в
            :data:`MAX_COMBINATIONS` комбинаций, ``candles`` пуст или после
            среза по периоду не осталось свечей (включая невалидные данные).
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
    """Прогнать одну комбинацию sweep; любой сбой становится строкой с ``error``."""
    row: dict[str, Any] = dict(combo)
    params = {**base_config.strategy_params, **combo}
    try:
        strategy = create_strategy(base_config.strategy, params)
        engine = build_engine(base_config, strategy)
        result = engine.run(candles)
        metrics = compute_metrics(result.equity_curve, result.trades, base_config.timeframe)
    except Exception as error:  # noqa: BLE001 - одна плохая комбинация не должна убить sweep
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
