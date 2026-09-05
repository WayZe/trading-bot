"""Отчёты бэктеста: метрики производительности и графики matplotlib."""

from trading_bot.report.metrics import (
    BenchmarkMetrics,
    MetricsReport,
    benchmark_equity,
    compute_benchmark_metrics,
    compute_metrics,
)
from trading_bot.report.plots import plot_equity, plot_trades

__all__ = [
    "BenchmarkMetrics",
    "MetricsReport",
    "benchmark_equity",
    "compute_benchmark_metrics",
    "compute_metrics",
    "plot_equity",
    "plot_trades",
]
