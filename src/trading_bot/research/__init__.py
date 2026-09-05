"""Research layer: parameter-grid sweeps over the backtest engine."""

from trading_bot.research.sweep import (
    MAX_COMBINATIONS,
    expand_grid,
    run_sweep,
    slice_candles,
)

__all__ = [
    "MAX_COMBINATIONS",
    "expand_grid",
    "run_sweep",
    "slice_candles",
]
