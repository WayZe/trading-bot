"""Matplotlib plots for backtest reports (Agg backend, no display needed).

Labels are in Russian; the default matplotlib font (DejaVu Sans) covers
Cyrillic, so no extra font setup is required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150


def plot_equity(
    equity: pd.Series, path: Path | str, benchmark: pd.Series | None = None
) -> None:
    """Save the equity curve (top) with its drawdown in percent (bottom).

    When ``benchmark`` is given, it is overlaid on the equity subplot as a
    gray dashed line ("Buy & hold") and a legend is added.
    """
    equity = equity.sort_index().astype("float64")
    drawdown = (equity / equity.cummax() - 1.0) * 100.0
    start, end = equity.index[0], equity.index[-1]

    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, height_ratios=(3, 1)
    )
    ax_equity.plot(
        equity.index, equity.to_numpy(), color="#1f77b4", linewidth=1.6, label="Стратегия"
    )
    if benchmark is not None:
        benchmark = benchmark.sort_index().astype("float64")
        ax_equity.plot(
            benchmark.index,
            benchmark.to_numpy(),
            color="#7f7f7f",
            linewidth=1.2,
            linestyle="--",
            label="Buy & hold",
        )
        ax_equity.legend(loc="best")
    ax_equity.set_title(f"Эквити бэктеста, {start:%Y-%m-%d} — {end:%Y-%m-%d}")
    ax_equity.set_ylabel("Эквити, USDT")
    ax_equity.grid(True, alpha=0.3)

    ax_dd.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color="#d62728", alpha=0.4
    )
    ax_dd.set_ylabel("Просадка, %")
    ax_dd.set_xlabel("Дата")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_trades(candles: pd.DataFrame, trades: pd.DataFrame, path: Path | str) -> None:
    """Save the close price with entry/exit markers over it.

    Entries are green up triangles, exits are red down triangles. An empty
    ``trades`` dataframe produces a price-only chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        candles["timestamp"],
        candles["close"].astype("float64"),
        color="#1f77b4",
        linewidth=1.2,
        label="Цена закрытия",
    )
    if not trades.empty:
        ax.scatter(
            trades["entry_ts"],
            trades["entry_price"].astype("float64"),
            marker="^",
            color="#2ca02c",
            s=64,
            label="Вход (long)",
            zorder=3,
        )
        ax.scatter(
            trades["exit_ts"],
            trades["exit_price"].astype("float64"),
            marker="v",
            color="#d62728",
            s=64,
            label="Выход",
            zorder=3,
        )
    ax.set_title("Сделки на графике цены")
    ax.set_ylabel("Цена, USDT")
    ax.set_xlabel("Дата")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
