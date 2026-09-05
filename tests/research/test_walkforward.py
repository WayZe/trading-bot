"""Tests for walk-forward analysis (pure functions, no I/O)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import BacktestConfig
from trading_bot.data.exchange import timeframe_to_ms
from trading_bot.engine.backtest import build_engine
from trading_bot.research import plan_windows, run_walkforward
from trading_bot.research import walkforward as walkforward_module
from trading_bot.research.sweep import run_sweep
from trading_bot.strategy import create_strategy

PARAMS = {"fast": 3, "slow": 6, "atr_period": 3, "atr_mult": 2.0}
DAY = pd.Timedelta(days=1)


def make_config(**overrides) -> BacktestConfig:
    base: dict = {
        "start": "2025-08-01",
        "strategy": "sma_cross",
        "strategy_params": dict(PARAMS),
    }
    base.update(overrides)
    return BacktestConfig(**base)


def make_cyclic_candles(n: int = 480) -> pd.DataFrame:
    """Candles with a sine close price so SMA crosses fire repeatedly.

    480 candles at 4h = 80 days; the sine period is 48 candles (8 days).
    """
    index = pd.date_range("2025-08-01", periods=n, freq="4h", tz="UTC")
    close = 100.0 + 10.0 * np.sin(np.arange(n) * 2.0 * math.pi / 48.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


class TestPlanWindows:
    START = pd.Timestamp("2025-01-01", tz="UTC")

    def test_rolling_blocks_tile_back_to_back(self) -> None:
        windows = plan_windows(self.START, self.START + 120 * DAY, 30, 10)

        assert len(windows) == 3
        assert windows[0].is_start == self.START
        assert windows[0].is_end == self.START + 30 * DAY
        assert windows[0].oos_start == windows[0].is_end
        assert windows[0].oos_end == self.START + 40 * DAY
        for step, window in enumerate(windows):
            assert window.is_start == self.START + (40 * step) * DAY
            assert window.oos_end == self.START + (40 * (step + 1)) * DAY
            assert window.is_end == window.oos_start
        for previous, current in zip(windows, windows[1:], strict=False):
            # the next IS starts where the previous OOS ended: the optimizer
            # never sees a previous window's OOS data
            assert current.is_start == previous.oos_end

    def test_anchored_windows_share_the_first_start(self) -> None:
        windows = plan_windows(self.START, self.START + 100 * DAY, 30, 10, mode="anchored")

        assert len(windows) == 7
        assert windows[0].oos_end == self.START + 40 * DAY
        for step, window in enumerate(windows):
            assert window.is_start == self.START
            assert window.is_end == self.START + (30 + 10 * step) * DAY
            assert window.oos_start == window.is_end
            assert window.oos_end == self.START + (40 + 10 * step) * DAY

    def test_insufficient_data_raises_with_both_spans_in_message(self) -> None:
        with pytest.raises(ValueError, match=r"20\.0 days.*is_days=30 \+ oos_days=10"):
            plan_windows(self.START, self.START + 20 * DAY, 30, 10)

    def test_exactly_is_plus_oos_days_fits_one_window(self) -> None:
        windows = plan_windows(self.START, self.START + 40 * DAY, 30, 10)

        assert len(windows) == 1
        assert windows[0].oos_end == self.START + 40 * DAY

    def test_window_count_above_limit_raises(self) -> None:
        # 2-day rolling blocks over 103 days = 51 windows, above the cap of 50
        with pytest.raises(ValueError, match="50"):
            plan_windows(self.START, self.START + 103 * DAY, 1, 1)

    @pytest.mark.parametrize("is_days, oos_days", [(0, 10), (-5, 10), (30, 0), (30, -1)])
    def test_non_positive_days_raise(self, is_days: int, oos_days: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            plan_windows(self.START, self.START + 100 * DAY, is_days, oos_days)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            plan_windows(self.START, self.START + 100 * DAY, 30, 10, mode="walk")


class TestRunWalkforward:
    def test_picks_the_expected_best_combo_per_window(self) -> None:
        candles = make_cyclic_candles()  # 80 days
        cfg = make_config()
        grid = {"fast": [3, 4], "slow": [6]}

        wf = run_walkforward(candles, cfg, grid, is_days=10, oos_days=10)

        assert len(wf.windows) == 4  # 20-day rolling blocks over 80 days
        expected = self._expected_best(candles, cfg, grid, wf.windows[0], "sharpe")
        assert wf.windows[0].best_params == {name: expected[name] for name in grid}
        assert isinstance(wf.windows[0].best_params["fast"], int)  # plain scalars
        assert wf.windows[0].is_objective == pytest.approx(float(expected["sharpe"]))
        # every window must have produced OOS numbers
        assert all(w.error is None for w in wf.windows)
        assert all(w.oos_return_pct is not None for w in wf.windows)

    def test_objective_total_return_changes_the_pick(self) -> None:
        candles = make_cyclic_candles()
        cfg = make_config()
        grid = {"fast": [3, 4, 5], "slow": [6, 8]}

        wf = run_walkforward(
            candles, cfg, grid, is_days=10, oos_days=10, objective="total_return_pct"
        )

        expected = self._expected_best(candles, cfg, grid, wf.windows[0], "total_return_pct")
        assert wf.windows[0].best_params == {name: expected[name] for name in grid}
        assert wf.windows[0].is_objective == pytest.approx(
            float(expected["total_return_pct"])
        )

    @staticmethod
    def _expected_best(
        candles: pd.DataFrame,
        cfg: BacktestConfig,
        grid: dict[str, list],
        window,
        objective: str,
    ) -> pd.Series:
        """Mirror the IS selection: best valid row of the IS slice by objective."""
        frame = run_sweep(
            cfg.model_copy(
                update={
                    "start": window.is_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": window.is_end.strftime("%Y-%m-%d %H:%M:%S"),
                }
            ),
            grid,
            candles,
        )
        valid = frame[frame["error"].isna()]
        scores = pd.to_numeric(valid[objective], errors="coerce")
        return valid.loc[scores.idxmax()]

    def test_stitched_curve_is_the_product_of_window_returns(self) -> None:
        candles = make_cyclic_candles()
        cfg = make_config()
        grid = {"fast": [3, 4], "slow": [6]}

        wf = run_walkforward(candles, cfg, grid, is_days=10, oos_days=10)

        stitched = wf.stitched_equity
        assert float(stitched.iloc[0]) == pytest.approx(1.0)
        growth = math.prod(1.0 + w.oos_return_pct / 100.0 for w in wf.windows)
        assert float(stitched.iloc[-1]) == pytest.approx(growth)
        # concatenated OOS parts: monotonic unique index (gaps between blocks ok)
        assert stitched.index.is_monotonic_increasing
        assert stitched.index.is_unique

    def test_single_window_single_combo_matches_direct_engine_run(self) -> None:
        candles = make_cyclic_candles(48)  # 8 days at 4h
        cfg = make_config()

        wf = run_walkforward(
            candles, cfg, {"fast": [3]}, is_days=4, oos_days=4, objective="total_return_pct"
        )

        assert len(wf.windows) == 1
        window = wf.windows[0]
        assert window.error is None

        # direct OOS run with the same calls: lead-in slice, engine, OOS part
        strategy = create_strategy(cfg.strategy, {**PARAMS, "fast": 3})
        lead_in = pd.Timedelta(
            milliseconds=strategy.warmup_period * timeframe_to_ms(cfg.timeframe)
        )
        ts = candles["timestamp"]
        oos_slice = candles.loc[
            (ts >= window.oos_start - lead_in) & (ts < window.oos_end)
        ].reset_index(drop=True)
        result = build_engine(cfg, strategy).run(oos_slice)
        part = result.equity_curve.loc[result.equity_curve.index >= window.oos_start]

        direct_return = (float(part.iloc[-1]) / float(part.iloc[0]) - 1.0) * 100.0
        assert window.oos_return_pct == pytest.approx(direct_return)
        assert window.oos_start == part.index[0]

    def test_lead_in_does_not_leak_into_the_oos_part(self) -> None:
        candles = make_cyclic_candles(48)
        cfg = make_config()

        wf = run_walkforward(
            candles,
            cfg,
            {"fast": [3]},
            is_days=4,
            oos_days=4,
            objective="total_return_pct",
        )

        window = wf.windows[0]
        # stitched curve starts exactly at the first OOS candle: the warm-up
        # part of the OOS slice is excluded from both the index and the value.
        first_oos_ts = candles["timestamp"][
            candles["timestamp"] >= window.oos_start
        ].iloc[0]
        assert wf.stitched_equity.index[0] == first_oos_ts
        assert float(wf.stitched_equity.iloc[0]) == pytest.approx(1.0)
        # trades only from the OOS part, none from the lead-in
        if not wf.trades.empty:
            assert wf.trades["entry_ts"].min() >= window.oos_start

    def test_all_invalid_combos_yield_error_window_without_oos(self) -> None:
        candles = make_cyclic_candles(48)
        cfg = make_config(strategy_params={"fast": 10, "slow": 6, "atr_period": 3})

        wf = run_walkforward(candles, cfg, {"fast": [10], "slow": [6]}, is_days=4, oos_days=4)

        assert len(wf.windows) == 1
        assert "no valid combos in IS" in wf.windows[0].error
        assert wf.windows[0].best_params is None
        assert wf.windows[0].oos_return_pct is None
        assert wf.stitched_equity.empty
        assert wf.trades.empty

    def test_data_gap_in_is_window_fails_only_that_window(self) -> None:
        candles = make_cyclic_candles()  # 80 days -> 4 rolling 10+10 windows
        # drop the IS span of window 1 (days 20..30): a hole in the data
        ts = candles["timestamp"]
        start = ts.iloc[0]
        hole = candles.loc[
            (ts >= start + 20 * DAY) & (ts < start + 30 * DAY)
        ].index
        candles = candles.drop(hole).reset_index(drop=True)

        wf = run_walkforward(
            candles,
            make_config(),
            {"fast": [3], "slow": [6]},
            is_days=10,
            oos_days=10,
        )

        assert len(wf.windows) == 4
        assert wf.windows[1].error == "no candles in IS window"
        assert wf.windows[1].best_params is None
        assert wf.windows[1].oos_return_pct is None
        # every other window still ran to completion
        assert all(w.error is None for i, w in enumerate(wf.windows) if i != 1)
        assert all(w.oos_return_pct is not None for i, w in enumerate(wf.windows) if i != 1)
        assert not wf.stitched_equity.empty

    def test_is_sweep_value_error_fails_only_that_window(self, monkeypatch) -> None:
        candles = make_cyclic_candles()
        cfg = make_config()
        grid = {"fast": [3], "slow": [6]}
        # plan once to learn the window whose IS sweep will blow up
        planned = run_walkforward(candles, cfg, grid, is_days=10, oos_days=10)
        hole = planned.windows[1]
        real_run_sweep = run_sweep

        def fake_run_sweep(config, grid_arg, candles_arg):
            if config.start == hole.is_start.strftime("%Y-%m-%d %H:%M:%S"):
                raise ValueError("boom IS")
            return real_run_sweep(config, grid_arg, candles_arg)

        monkeypatch.setattr(walkforward_module, "run_sweep", fake_run_sweep)

        wf = run_walkforward(candles, cfg, grid, is_days=10, oos_days=10)

        assert len(wf.windows) == 4
        assert wf.windows[1].error == "boom IS"
        assert wf.windows[1].oos_return_pct is None
        assert all(w.error is None for i, w in enumerate(wf.windows) if i != 1)

    def test_combination_cap_raises_instead_of_error_rows(self) -> None:
        candles = make_cyclic_candles()  # planning fits, the cap check is up front
        grid = {"fast": list(range(15)), "slow": list(range(15))}  # 225 > 200

        with pytest.raises(ValueError, match="combinations"):
            run_walkforward(candles, make_config(), grid, is_days=10, oos_days=10)

    def test_window_count_above_limit_raises(self) -> None:
        candles = make_cyclic_candles(1500)  # 250 days; planning only, no runs

        with pytest.raises(ValueError, match="50"):
            run_walkforward(candles, make_config(), {"fast": [3]}, is_days=1, oos_days=1)

    def test_insufficient_data_raises(self) -> None:
        candles = make_cyclic_candles(48)  # 8 days

        with pytest.raises(ValueError, match="is_days=30"):
            run_walkforward(candles, make_config(), {"fast": [3]}, is_days=30, oos_days=30)

    def test_unknown_objective_raises(self) -> None:
        candles = make_cyclic_candles(48)

        with pytest.raises(ValueError, match="objective"):
            run_walkforward(
                candles, make_config(), {"fast": [3]}, is_days=4, oos_days=4, objective="calmar"
            )

    def test_empty_candles_raise(self) -> None:
        with pytest.raises(ValueError, match="no candles"):
            run_walkforward(
                make_cyclic_candles(48).head(0),
                make_config(),
                {"fast": [3]},
                is_days=4,
                oos_days=4,
            )

    def test_meta_carries_the_wfo_parameters(self) -> None:
        candles = make_cyclic_candles(48)
        grid = {"fast": [3, 4], "slow": [6]}

        wf = run_walkforward(
            candles,
            make_config(),
            grid,
            is_days=4,
            oos_days=4,
            mode="anchored",
            objective="total_return_pct",
        )

        assert wf.meta["mode"] == "anchored"
        assert wf.meta["is_days"] == 4
        assert wf.meta["oos_days"] == 4
        assert wf.meta["objective"] == "total_return_pct"
        assert wf.meta["param_grid"] == {"fast": [3, 4], "slow": [6]}
        assert wf.meta["symbol"] == "BTC/USDT"
        assert wf.meta["timeframe"] == "4h"


class TestWalkForwardResultFrame:
    def test_to_frame_has_grid_and_window_columns(self) -> None:
        candles = make_cyclic_candles(48)
        grid = {"fast": [3], "slow": [6]}

        wf = run_walkforward(
            candles,
            make_config(),
            grid,
            is_days=4,
            oos_days=4,
            objective="total_return_pct",
        )
        frame = wf.to_frame(grid)

        assert len(frame) == len(wf.windows) == 1
        for name in grid:
            assert name in frame.columns
        for name in (
            "is_start",
            "oos_start",
            "oos_end",
            "mode",
            "is_objective",
            "oos_return_pct",
            "oos_sharpe",
            "oos_max_dd_pct",
            "oos_trades",
            "error",
        ):
            assert name in frame.columns
        assert pd.isna(frame.iloc[0]["error"])
        assert frame.iloc[0]["fast"] == 3

    def test_to_frame_error_row_has_nan_params(self) -> None:
        candles = make_cyclic_candles(48)
        cfg = make_config(strategy_params={"fast": 10, "slow": 6, "atr_period": 3})

        wf = run_walkforward(candles, cfg, {"fast": [10], "slow": [6]}, is_days=4, oos_days=4)
        frame = wf.to_frame({"fast": [10], "slow": [6]})

        assert math.isnan(frame.iloc[0]["fast"])
        assert "no valid combos in IS" in str(frame.iloc[0]["error"])
