"""Графики matplotlib для отчётов бэктеста (backend Agg, дисплей не нужен).

Подписи на русском; шрифт matplotlib по умолчанию (DejaVu Sans) покрывает
кириллицу, так что дополнительная настройка шрифтов не требуется.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # должно выполняться до импорта pyplot

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150


def plot_equity(
    equity: pd.Series, path: Path | str, benchmark: pd.Series | None = None
) -> None:
    """Сохранить кривую эквити (сверху) с просадкой в процентах (снизу).

    Если передан ``benchmark``, он накладывается на график эквити серой
    пунктирной линией ("Buy & hold") и добавляется легенда.
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


def plot_stitched_equity(
    stitched: pd.Series, path: Path | str, benchmark: pd.Series | None = None
) -> None:
    """Сохранить сшитую walk-forward кривую (сверху) с её просадкой (снизу).

    Обе кривые — коэффициенты роста, начинающиеся с 1.0 (сшитая OOS-кривая
    эквити и бенчмарк buy & hold за тот же период, нормированные одинаково),
    поэтому подпись оси Y — кратность стартового капитала, а не USDT.
    Тот же визуальный язык, что у :func:`plot_equity`.
    """
    stitched = stitched.sort_index().astype("float64")
    drawdown = (stitched / stitched.cummax() - 1.0) * 100.0
    start, end = stitched.index[0], stitched.index[-1]

    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, height_ratios=(3, 1)
    )
    ax_equity.plot(
        stitched.index,
        stitched.to_numpy(),
        color="#1f77b4",
        linewidth=1.6,
        label="Walk-forward (stitched)",
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
    ax_equity.set_title(f"Walk-forward: сшитая OOS-кривая, {start:%Y-%m-%d} — {end:%Y-%m-%d}")
    ax_equity.set_ylabel("Рост капитала (1.0 = старт)")
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
    """Сохранить график цены закрытия с маркерами входов/выходов поверх.

    Входы — зелёные треугольники вверх, выходы — красные треугольники вниз.
    Пустой фрейм ``trades`` даёт график одной лишь цены.
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
