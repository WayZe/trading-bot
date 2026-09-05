"""Research layer: parameter-grid sweeps and walk-forward analysis."""

from trading_bot.research.sweep import (
    MAX_COMBINATIONS,
    expand_grid,
    run_sweep,
    slice_candles,
)
from trading_bot.research.walkforward import (
    MAX_WINDOWS,
    WINDOW_COLUMNS,
    WalkForwardResult,
    Window,
    WindowResult,
    plan_windows,
    run_walkforward,
)

__all__ = [
    "MAX_COMBINATIONS",
    "MAX_WINDOWS",
    "WINDOW_COLUMNS",
    "Window",
    "WindowResult",
    "WalkForwardResult",
    "expand_grid",
    "plan_windows",
    "run_sweep",
    "run_walkforward",
    "slice_candles",
]
