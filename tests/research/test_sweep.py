"""Тесты sweep по сетке параметров (чистые функции, без I/O)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_candles, rows_to_df
from trading_bot.config import BacktestConfig
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.report.metrics import compute_metrics
from trading_bot.research import expand_grid, run_sweep, slice_candles
from trading_bot.research import sweep as sweep_module
from trading_bot.risk import RiskManager
from trading_bot.strategy import create_strategy

PARAMS = {"fast": 3, "slow": 6, "atr_period": 3, "atr_mult": 2.0}


def make_config(**overrides) -> BacktestConfig:
    base: dict = {
        "start": "2025-08-01",
        "strategy": "sma_cross",
        "strategy_params": dict(PARAMS),
    }
    base.update(overrides)
    return BacktestConfig(**base)


def make_candles_df(n: int = 60) -> pd.DataFrame:
    return rows_to_df(make_candles(n, tf_ms=4 * 3_600_000))


def make_cyclic_candles(n: int = 240) -> pd.DataFrame:
    """Свечи с синусоидальным close, чтобы SMA-пересечения срабатывали в обе стороны."""
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


class TestExpandGrid:
    def test_cartesian_product_preserves_key_order(self) -> None:
        grid = {"fast": [5, 10], "slow": [20, 30, 40]}

        combos = expand_grid(grid)

        assert combos == [
            {"fast": 5, "slow": 20},
            {"fast": 5, "slow": 30},
            {"fast": 5, "slow": 40},
            {"fast": 10, "slow": 20},
            {"fast": 10, "slow": 30},
            {"fast": 10, "slow": 40},
        ]

    def test_empty_grid_yields_single_empty_combo(self) -> None:
        assert expand_grid({}) == [{}]

    def test_single_dimension(self) -> None:
        assert expand_grid({"atr_mult": [1.5, 2.0]}) == [
            {"atr_mult": 1.5},
            {"atr_mult": 2.0},
        ]


class TestSliceCandles:
    def test_respects_start_and_end(self) -> None:
        candles = make_candles_df(48)  # 48 x 4h с 2025-08-01
        cfg = BacktestConfig(start="2025-08-01", end="2025-08-04")

        sliced = slice_candles(candles, cfg)

        assert len(sliced) == 18  # 1 авг 00:00 .. 3 авг 20:00, end не включается
        assert sliced["timestamp"].iloc[0] == pd.Timestamp("2025-08-01", tz="UTC")
        assert sliced["timestamp"].iloc[-1] == pd.Timestamp(
            "2025-08-03 20:00", tz="UTC"
        )

    def test_empty_candles_raise(self) -> None:
        empty = rows_to_df([])

        with pytest.raises(ValueError, match="no candles"):
            slice_candles(empty, BacktestConfig(start="2025-08-01"))


class TestRunSweep:
    @staticmethod
    def _direct_run(cfg: BacktestConfig, candles: pd.DataFrame, params: dict):
        strategy = create_strategy("sma_cross", params)
        engine = BacktestEngine(
            strategy=strategy,
            risk=RiskManager(),
            broker=SimulatedBroker(fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps),
            start_cash=cfg.start_cash,
        )
        result = engine.run(slice_candles(candles, cfg))
        return compute_metrics(result.equity_curve, result.trades, cfg.timeframe)

    def test_single_combination_matches_direct_engine_run(self) -> None:
        cfg = make_config()
        candles = make_cyclic_candles()

        results = run_sweep(cfg, {"fast": [3]}, candles)

        assert len(results) == 1
        direct = self._direct_run(cfg, candles, dict(PARAMS))

        assert results.iloc[0]["error"] is None
        assert direct.n_trades > 0  # данные должны давать реальные сделки
        assert results.iloc[0]["total_return_pct"] == pytest.approx(direct.total_return_pct)
        assert results.iloc[0]["n_trades"] == direct.n_trades
        assert results.iloc[0]["sharpe"] == pytest.approx(direct.sharpe)
        assert results.iloc[0]["final_equity"] == pytest.approx(direct.final_equity)

    def test_combo_overrides_base_params(self) -> None:
        cfg = make_config()
        candles = make_cyclic_candles()

        results = run_sweep(cfg, {"slow": [6, 12]}, candles)

        assert len(results) == 2
        # Каждая строка должна совпадать с прямым прогоном своего значения
        # комбинации (комбинация побеждает базовый slow=6), и два прогона
        # должны реально различаться.
        direct_slow6 = self._direct_run(cfg, candles, dict(PARAMS))
        direct_slow12 = self._direct_run(cfg, candles, {**PARAMS, "slow": 12})
        assert results.iloc[0]["total_return_pct"] == pytest.approx(
            direct_slow6.total_return_pct
        )
        assert results.iloc[1]["total_return_pct"] == pytest.approx(
            direct_slow12.total_return_pct
        )
        assert direct_slow6.total_return_pct != pytest.approx(direct_slow12.total_return_pct)

    def test_invalid_combo_yields_error_row_not_exception(self) -> None:
        cfg = make_config(strategy_params={"fast": 3, "slow": 10})
        candles = make_candles_df(120)

        results = run_sweep(cfg, {"fast": [5, 15]}, candles)  # fast=15 >= slow=10

        assert len(results) == 2
        ok = results[results["error"].isna()]
        failed = results[results["error"].notna()]
        assert len(ok) == 1 and ok.iloc[0]["fast"] == 5
        assert len(failed) == 1 and failed.iloc[0]["fast"] == 15
        assert "fast (15) must be smaller than slow (10)" in failed.iloc[0]["error"]
        assert math.isnan(failed.iloc[0]["total_return_pct"])
        assert math.isnan(failed.iloc[0]["final_equity"])

    def test_unknown_param_yields_error_row(self) -> None:
        cfg = make_config()
        candles = make_candles_df(120)

        results = run_sweep(cfg, {"nope": [1]}, candles)

        assert len(results) == 1
        assert "nope" in str(results.iloc[0]["error"])

    def test_grid_above_limit_raises(self) -> None:
        cfg = make_config()

        with pytest.raises(ValueError, match="200"):
            run_sweep(cfg, {"fast": list(range(201))}, make_candles_df(10))

    def test_grid_product_above_limit_raises_before_expansion(self) -> None:
        cfg = make_config()
        grid = {"fast": list(range(15)), "slow": list(range(15))}  # 15 * 15 = 225

        with pytest.raises(ValueError, match="225"):
            run_sweep(cfg, grid, make_candles_df(10))

    def test_runtime_error_combo_becomes_error_row(self, monkeypatch) -> None:
        cfg = make_config()
        candles = make_candles_df(120)
        real_create = sweep_module.create_strategy

        def flaky_create(name, params):
            if params.get("slow") == 12:
                raise RuntimeError("portfolio exploded")
            return real_create(name, params)

        monkeypatch.setattr(sweep_module, "create_strategy", flaky_create)

        results = run_sweep(cfg, {"slow": [6, 12]}, candles)

        assert len(results) == 2
        ok = results[results["error"].isna()]
        failed = results[results["error"].notna()]
        assert len(ok) == 1 and ok.iloc[0]["slow"] == 6
        assert len(failed) == 1 and failed.iloc[0]["slow"] == 12
        assert "portfolio exploded" in failed.iloc[0]["error"]
        assert math.isnan(failed.iloc[0]["total_return_pct"])

    def test_invalid_candles_raise_before_the_combination_loop(self) -> None:
        cfg = make_config()
        candles = make_candles_df(120)
        candles.loc[0, "high"] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            run_sweep(cfg, {"fast": [3, 5]}, candles)

    def test_empty_candles_raise(self) -> None:
        cfg = make_config()

        with pytest.raises(ValueError, match="no candles"):
            run_sweep(cfg, {"fast": [3]}, rows_to_df([]))

    def test_empty_period_raises(self) -> None:
        cfg = make_config(start="2030-01-01")

        with pytest.raises(ValueError, match="no candles"):
            run_sweep(cfg, {"fast": [3]}, make_candles_df(10))

    def test_result_columns_include_params_metrics_and_error(self) -> None:
        cfg = make_config()
        candles = make_candles_df(120)

        results = run_sweep(cfg, {"fast": [3], "atr_mult": [2.0]}, candles)

        assert list(results.columns) == [
            "fast",
            "atr_mult",
            "total_return_pct",
            "cagr_pct",
            "sharpe",
            "max_drawdown_pct",
            "n_trades",
            "winrate_pct",
            "profit_factor",
            "final_equity",
            "error",
        ]
