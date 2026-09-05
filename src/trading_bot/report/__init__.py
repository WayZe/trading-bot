"""Backtest reports: performance metrics and matplotlib plots."""

from trading_bot.report.metrics import MetricsReport, compute_metrics
from trading_bot.report.plots import plot_equity, plot_trades

__all__ = [
    "MetricsReport",
    "compute_metrics",
    "plot_equity",
    "plot_trades",
]
