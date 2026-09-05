"""Tests for backtest metrics (hand-computed literals, no I/O)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_bot.report.metrics import compute_metrics


def make_equity(values: list[float], start: str = "2024-01-01", freq: str = "D"):
    index = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index, name="equity", dtype="float64")


def make_trades(pnls: list[float], holding_hours: list[float] | None = None):
    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i, pnl in enumerate(pnls):
        holding = holding_hours[i] if holding_hours else 24.0
        rows.append(
            {
                "entry_ts": base + pd.Timedelta(days=30 * i),
                "exit_ts": base + pd.Timedelta(days=30 * i, hours=holding),
                "entry_price": 100.0 + i,
                "exit_price": 101.0 + i,
                "quantity": 1.0,
                "pnl": pnl,
                "reason_entry": "test",
                "reason_exit": "test",
            }
        )
    return pd.DataFrame(rows)


def empty_trades() -> pd.DataFrame:
    columns = [
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "quantity",
        "pnl",
        "reason_entry",
        "reason_exit",
    ]
    return pd.DataFrame(columns=columns)


class TestEquityMetrics:
    def test_total_return_and_max_drawdown_with_recovery(self) -> None:
        # Peak 120 (day 1), trough 90 (day 2, dd -25%), recovered on day 4 (130).
        equity = make_equity([100.0, 120.0, 90.0, 100.0, 130.0])

        metrics = compute_metrics(equity, empty_trades(), "1d")

        assert metrics.total_return_pct == pytest.approx(30.0)
        assert metrics.final_equity == pytest.approx(130.0)
        assert metrics.max_drawdown_pct == pytest.approx(-25.0)
        assert metrics.max_drawdown_days == pytest.approx(3.0)  # day 1 -> day 4
        assert metrics.span_days == pytest.approx(4.0)

    def test_max_drawdown_without_recovery_runs_to_the_end(self) -> None:
        equity = make_equity([100.0, 120.0, 90.0])

        metrics = compute_metrics(equity, empty_trades(), "1d")

        assert metrics.max_drawdown_pct == pytest.approx(-25.0)
        assert metrics.max_drawdown_days == pytest.approx(1.0)  # day 1 -> day 2

    def test_flat_equity_has_no_drawdown(self) -> None:
        metrics = compute_metrics(make_equity([100.0, 100.0, 100.0]), empty_trades(), "1d")

        assert metrics.max_drawdown_pct == 0.0
        assert metrics.max_drawdown_days is None

    def test_sharpe_hand_computed(self) -> None:
        # Per-day returns: +0.1, -0.05, +0.1 -> mean 0.05, std(ddof=1)=sqrt(0.0075).
        equity = make_equity([100.0, 110.0, 104.5, 114.95])

        metrics = compute_metrics(equity, empty_trades(), "1d")

        expected = 0.05 / math.sqrt(0.0075) * math.sqrt(365.25)
        assert metrics.sharpe == pytest.approx(expected)
        assert metrics.sharpe > 0.0
        assert metrics.max_drawdown_pct == pytest.approx(-5.0)
        assert metrics.max_drawdown_days == pytest.approx(2.0)

    def test_sharpe_is_none_for_constant_returns(self) -> None:
        metrics = compute_metrics(
            make_equity([100.0, 110.0, 121.0]), empty_trades(), "1d"
        )

        assert metrics.sharpe is None  # zero dispersion

    def test_cagr_over_exactly_one_year(self) -> None:
        # 2024-01-01 00:00 -> 2024-12-31 06:00 is exactly 365.25 days.
        index = pd.DatetimeIndex(
            ["2024-01-01 00:00:00+00:00", "2024-12-31 06:00:00+00:00"], name="timestamp"
        )
        equity = pd.Series([100.0, 121.0], index=index, name="equity")

        metrics = compute_metrics(equity, empty_trades(), "1d")

        assert metrics.span_days == pytest.approx(365.25)
        assert metrics.cagr_pct == pytest.approx(21.0)
        assert not metrics.short_span

    def test_cagr_extrapolated_for_short_span(self) -> None:
        equity = make_equity([100.0, 110.0, 120.0])  # 2 days

        metrics = compute_metrics(equity, empty_trades(), "1d")

        assert metrics.short_span
        assert metrics.cagr_pct is not None
        assert metrics.cagr_pct > metrics.total_return_pct  # 20% over 2 days

    def test_single_point_equity(self) -> None:
        equity = pd.Series(
            [100.0],
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC", name="timestamp"),
        )

        metrics = compute_metrics(equity, empty_trades(), "1d")

        assert metrics.total_return_pct == 0.0
        assert metrics.cagr_pct is None
        assert metrics.sharpe is None
        assert metrics.span_days == 0.0

    def test_empty_equity_raises(self) -> None:
        equity = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))

        with pytest.raises(ValueError, match="empty"):
            compute_metrics(equity, empty_trades(), "1d")


class TestTradeMetrics:
    def test_full_trade_statistics(self) -> None:
        trades = make_trades(
            pnls=[100.0, 50.0, -30.0],
            holding_hours=[24.0, 48.0, 12.0],
        )
        # notionals: (100+101) + (101+102) + (102+103) = 609
        equity = make_equity([1000.0] * 10)

        metrics = compute_metrics(equity, trades, "1d", fee_rate=0.001)

        assert metrics.n_trades == 3
        assert metrics.winrate_pct == pytest.approx(200.0 / 3.0)
        assert metrics.profit_factor == pytest.approx(150.0 / 30.0)
        assert metrics.avg_trade_pnl == pytest.approx(40.0)
        assert metrics.avg_win == pytest.approx(75.0)
        assert metrics.avg_loss == pytest.approx(-30.0)
        assert metrics.best_trade == pytest.approx(100.0)
        assert metrics.worst_trade == pytest.approx(-30.0)
        assert metrics.avg_holding_hours == pytest.approx(28.0)
        assert metrics.total_fees == pytest.approx(0.609)

    def test_profit_factor_infinite_without_losses(self) -> None:
        trades = make_trades(pnls=[10.0, 20.0])

        metrics = compute_metrics(make_equity([1000.0] * 10), trades, "1d")

        assert metrics.profit_factor == math.inf
        assert metrics.winrate_pct == pytest.approx(100.0)
        assert metrics.avg_loss is None
        assert metrics.worst_trade == pytest.approx(10.0)  # min pnl is a win

    def test_profit_factor_when_only_losses(self) -> None:
        trades = make_trades(pnls=[-10.0, -20.0])

        metrics = compute_metrics(make_equity([1000.0] * 10), trades, "1d")

        assert metrics.profit_factor == pytest.approx(0.0)
        assert metrics.avg_win is None

    def test_zero_trades_yield_none_metrics(self) -> None:
        metrics = compute_metrics(make_equity([100.0, 105.0]), empty_trades(), "1d")

        assert metrics.n_trades == 0
        assert metrics.winrate_pct is None
        assert metrics.profit_factor is None
        assert metrics.avg_trade_pnl is None
        assert metrics.avg_holding_hours is None
        assert metrics.total_fees is None
        # Equity metrics are still computed.
        assert metrics.total_return_pct == pytest.approx(5.0)

    def test_total_fees_requires_fee_rate(self) -> None:
        trades = make_trades(pnls=[10.0])

        metrics = compute_metrics(make_equity([1000.0] * 5), trades, "1d")

        assert metrics.total_fees is None

    def test_sharpe_annualization_depends_on_timeframe(self) -> None:
        # Same returns, finer timeframe -> more periods per year -> larger sharpe.
        trades = empty_trades()
        returns = [0.1, -0.05, 0.1]
        values = [100.0]
        for r in returns:
            values.append(values[-1] * (1.0 + r))

        daily = compute_metrics(make_equity(values, freq="D"), trades, "1d")
        four_hour = compute_metrics(make_equity(values, freq="4h"), trades, "4h")

        assert daily.sharpe is not None and four_hour.sharpe is not None
        assert four_hour.sharpe == pytest.approx(daily.sharpe * math.sqrt(6.0))
