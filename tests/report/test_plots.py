"""Tests for report plots (Agg backend, no display required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.report import plots
from trading_bot.report.plots import plot_equity, plot_stitched_equity, plot_trades


def make_equity(n: int = 50) -> pd.Series:
    index = pd.date_range("2025-08-01", periods=n, freq="4h", tz="UTC")
    values = [10_000.0 + 50.0 * i - (30.0 * (i % 7)) for i in range(n)]
    return pd.Series(values, index=index, name="equity", dtype="float64")


def make_candles(n: int = 50) -> pd.DataFrame:
    index = pd.date_range("2025-08-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series([100.0 + i for i in range(n)], dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


def make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2025-08-02", "2025-08-04"], utc=True),
            "exit_ts": pd.to_datetime(["2025-08-03", "2025-08-05"], utc=True),
            "entry_price": [101.0, 103.0],
            "exit_price": [102.0, 104.0],
            "quantity": [1.0, 1.0],
            "pnl": [10.0, -5.0],
            "reason_entry": ["t", "t"],
            "reason_exit": ["t", "t"],
        }
    )


def test_plot_equity_creates_non_empty_png(tmp_path: Path) -> None:
    path = tmp_path / "equity.png"

    plot_equity(make_equity(), path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_equity_with_benchmark_overlay(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "equity.png"
    index = pd.date_range("2025-08-01", periods=50, freq="4h", tz="UTC")
    benchmark = pd.Series(
        [10_000.0 + 20.0 * i for i in range(50)], index=index, name="equity"
    )

    # plot_equity closes its figure; keep it open to inspect the drawn lines.
    monkeypatch.setattr(plots.plt, "close", lambda *args, **kwargs: None)
    try:
        plot_equity(make_equity(), path, benchmark=benchmark)
        ax = plots.plt.gcf().axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
    finally:
        plots.plt.close("all")

    assert path.exists()
    assert path.stat().st_size > 0
    assert len(ax.get_lines()) == 2
    assert "Стратегия" in labels
    assert "Buy & hold" in labels


def test_plot_stitched_equity_creates_non_empty_png(tmp_path: Path) -> None:
    path = tmp_path / "walkforward.png"

    plot_stitched_equity(make_equity() / 10_000.0, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_stitched_equity_with_benchmark_overlay(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "walkforward.png"
    index = pd.date_range("2025-08-01", periods=50, freq="4h", tz="UTC")
    benchmark = pd.Series([1.0 + 0.01 * i for i in range(50)], index=index, name="equity")

    monkeypatch.setattr(plots.plt, "close", lambda *args, **kwargs: None)
    try:
        plot_stitched_equity(make_equity() / 10_000.0, path, benchmark=benchmark)
        ax = plots.plt.gcf().axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
    finally:
        plots.plt.close("all")

    assert path.exists()
    assert path.stat().st_size > 0
    assert len(ax.get_lines()) == 2
    assert "Walk-forward" in labels[0]
    assert "Buy & hold" in labels


def test_plot_trades_creates_non_empty_png(tmp_path: Path) -> None:
    path = tmp_path / "trades.png"

    plot_trades(make_candles(), make_trades(), path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_trades_with_empty_trades_still_draws_price(tmp_path: Path) -> None:
    path = tmp_path / "trades.png"
    empty = pd.DataFrame(
        columns=["entry_ts", "exit_ts", "entry_price", "exit_price", "quantity", "pnl"]
    )

    plot_trades(make_candles(), empty, path)

    assert path.exists()
    assert path.stat().st_size > 0
