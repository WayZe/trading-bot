"""Tests for the CLI application (offline; download needs the network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.conftest import make_candles, rows_to_df
from trading_bot.cli import _parse_grid, app
from trading_bot.config import BacktestConfig
from trading_bot.data.storage import CandleStorage

runner = CliRunner()


def _make_run_dir(tmp_path: Path, with_candles: bool = True, with_benchmark: bool = False) -> Path:
    """Create a mini run directory (equity + trades + meta) in tmp_path."""
    run = tmp_path / "reports" / "last_run"
    run.mkdir(parents=True, exist_ok=True)

    index = pd.date_range("2025-08-01", periods=8, freq="4h", tz="UTC")
    equity = pd.Series(
        [10_000, 10_100, 9_900, 10_050, 10_200, 10_150, 10_300, 10_400],
        index=index,
        name="equity",
    )
    equity.to_frame().to_parquet(run / "equity.parquet", engine="pyarrow")

    if with_benchmark:
        benchmark = pd.Series(
            [10_000, 10_050, 10_100, 10_080, 10_150, 10_200, 10_180, 10_250],
            index=index,
            name="equity",
        )
        benchmark.to_frame().to_parquet(run / "benchmark.parquet", engine="pyarrow")

    trades = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2025-08-01 04:00", "2025-08-01 12:00"], utc=True),
            "exit_ts": pd.to_datetime(["2025-08-01 08:00", "2025-08-01 16:00"], utc=True),
            "entry_price": [100.0, 102.0],
            "exit_price": [101.0, 100.0],
            "quantity": [1.0, 1.0],
            "pnl": [10.0, -8.0],
            "reason_entry": ["test", "test"],
            "reason_exit": ["test", "test"],
        }
    )
    trades.to_csv(run / "trades.csv", index=False)

    meta = {
        "config": BacktestConfig(start="2025-08-01").model_dump(mode="json"),
        "summary": {"open_position": None, "n_pending_unfilled": 0},
    }
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if with_candles:
        storage = CandleStorage(tmp_path / "data")
        storage.save("bybit", "BTC/USDT", "4h", rows_to_df(make_candles(12)))
    return run


def test_report_builds_table_and_plots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run_dir(tmp_path)

    result = runner.invoke(app, ["report", "--run-dir", str(run)])

    assert result.exit_code == 0, result.output
    assert "Отчёт бэктеста" in result.output
    assert "sma_cross · BTC/USDT · 4h" in result.output
    assert "Доходность" in result.output
    assert "Profit factor" in result.output
    assert "Buy & hold" not in result.output  # no benchmark artifact
    equity_png = run / "equity.png"
    trades_png = run / "trades.png"
    assert equity_png.exists() and equity_png.stat().st_size > 0
    assert trades_png.exists() and trades_png.stat().st_size > 0


def test_report_with_benchmark_shows_comparison(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run_dir(tmp_path, with_benchmark=True)

    result = runner.invoke(app, ["report", "--run-dir", str(run)])

    assert result.exit_code == 0, result.output
    assert "Стратегия vs Buy & hold" in result.output
    assert "Максимальная просадка" in result.output
    assert "Коэффициент Шарпа" in result.output
    assert (run / "equity.png").exists()
    assert (run / "equity.png").stat().st_size > 0


def test_report_accepts_explicit_candles_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no data/ here on purpose
    run = _make_run_dir(tmp_path, with_candles=False)
    candles = tmp_path / "candles.parquet"
    rows_to_df(make_candles(12)).to_parquet(candles, index=False)

    result = runner.invoke(
        app, ["report", "--run-dir", str(run), "--candles", str(candles)]
    )

    assert result.exit_code == 0, result.output
    assert (run / "trades.png").exists()


def test_report_without_candles_skips_trades_plot_with_hint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run_dir(tmp_path, with_candles=False)

    result = runner.invoke(app, ["report", "--run-dir", str(run)])

    assert result.exit_code == 0, result.output
    assert "график сделок пропущен" in result.output
    assert "--candles" in result.output
    assert (run / "equity.png").exists()
    assert not (run / "trades.png").exists()


def test_report_with_missing_explicit_candles_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run_dir(tmp_path, with_candles=False)

    result = runner.invoke(
        app, ["report", "--run-dir", str(run), "--candles", str(tmp_path / "nope.parquet")]
    )

    assert result.exit_code == 1
    assert "не найден" in result.output


def test_report_without_run_dir_fails_with_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 1
    assert "артефактов" in result.output
    assert "backtest" in result.output


def _prepare_data(tmp_path: Path, n_candles: int = 60) -> Path:
    """Save synthetic rising candles into tmp_path/data (the CLI data root)."""
    storage = CandleStorage(tmp_path / "data")
    storage.save("bybit", "BTC/USDT", "4h", rows_to_df(make_candles(n_candles)))
    return storage.path_for("bybit", "BTC/USDT", "4h")


def test_backtest_runs_and_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = _prepare_data(tmp_path)
    config = tmp_path / "backtest.yaml"
    config.write_text(
        'start: "2025-07-01"\n'
        "strategy: sma_cross\n"
        "strategy_params:\n"
        "  fast: 3\n"
        "  slow: 6\n"
        "  atr_period: 3\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["backtest", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert data_path.exists()
    assert "Бэктест" in result.output
    assert "свечей" in result.output
    assert "сделок" in result.output
    last_run = tmp_path / "reports" / "last_run"
    assert (last_run / "equity.parquet").exists()
    assert (last_run / "benchmark.parquet").exists()
    assert (last_run / "trades.csv").exists()
    assert (last_run / "meta.json").exists()
    meta = json.loads((last_run / "meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["symbol"] == "BTC/USDT"
    assert meta["summary"]["n_candles"] == 60
    assert meta["summary"]["start_cash"] == 10_000.0
    assert meta["summary"]["n_trades"] >= 0
    benchmark = pd.read_parquet(last_run / "benchmark.parquet", engine="pyarrow")
    assert benchmark["equity"].iloc[0] == 10_000.0  # start_cash at the first close


def test_backtest_without_data_fails_with_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "backtest.yaml"
    config.write_text('start: "2025-07-01"\n', encoding="utf-8")

    result = runner.invoke(app, ["backtest", "--config", str(config)])

    assert result.exit_code == 1
    assert "download" in result.output


def test_backtest_with_empty_range_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_data(tmp_path)
    config = tmp_path / "backtest.yaml"
    config.write_text('start: "2030-01-01"\n', encoding="utf-8")

    result = runner.invoke(app, ["backtest", "--config", str(config)])

    assert result.exit_code == 1
    assert "нет свечей" in result.output


def test_backtest_with_unknown_strategy_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_data(tmp_path)
    config = tmp_path / "backtest.yaml"
    config.write_text('start: "2025-07-01"\nstrategy: nope\n', encoding="utf-8")

    result = runner.invoke(app, ["backtest", "--config", str(config)])

    assert result.exit_code == 1
    assert "стратегии" in result.output


def test_backtest_with_missing_config_fails() -> None:
    result = runner.invoke(app, ["backtest", "--config", "/nonexistent/config.yaml"])

    assert result.exit_code != 0


SWEEP_CONFIG = (
    'start: "2025-07-01"\n'
    "strategy: sma_cross\n"
    "strategy_params:\n"
    "  fast: 3\n"
    "  slow: 6\n"
    "  atr_period: 3\n"
)


class TestSweep:
    def test_runs_and_writes_artifacts(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _prepare_data(tmp_path, n_candles=120)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app,
            ["sweep", "--config", str(config), "--param", "fast=3,4", "--param", "slow=6"],
        )

        assert result.exit_code == 0, result.output
        sweep_dir = tmp_path / "reports" / "sweep" / "last"
        results = pd.read_csv(sweep_dir / "results.csv")
        assert len(results) == 2
        assert set(results["fast"]) == {3, 4}
        assert "total_return_pct" in results.columns
        assert "error" in results.columns
        meta = json.loads((sweep_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["grid"] == {"fast": [3, 4], "slow": [6]}
        assert meta["n_combinations"] == 2
        assert meta["config"]["symbol"] == "BTC/USDT"
        assert "Лучшая комбинация" in result.output
        assert "Доходность" in result.output

    def test_invalid_combo_is_reported_not_fatal(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _prepare_data(tmp_path, n_candles=120)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app,
            ["sweep", "--config", str(config), "--param", "fast=3,10", "--param", "slow=6"],
        )

        assert result.exit_code == 0, result.output
        assert "must be smaller than slow" in result.output

    def test_without_param_fails_with_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _prepare_data(tmp_path)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(app, ["sweep", "--config", str(config)])

        assert result.exit_code == 1
        assert "--param" in result.output

    def test_without_data_fails_with_download_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(app, ["sweep", "--config", str(config), "--param", "fast=3"])

        assert result.exit_code == 1
        assert "download" in result.output

    def test_bad_param_spec_fails(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _prepare_data(tmp_path)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(app, ["sweep", "--config", str(config), "--param", "fast"])

        assert result.exit_code != 0
        assert "name=v1,v2,..." in result.output

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_non_finite_param_fails_cleanly(
        self, tmp_path: Path, monkeypatch, value: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "backtest.yaml"
        config.write_text(SWEEP_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app, ["sweep", "--config", str(config), "--param", f"atr_mult={value}"]
        )

        assert result.exit_code != 0
        assert "конечным" in result.output
        assert "Traceback" not in result.output


class TestParseGrid:
    def test_coerces_int_float_and_str(self) -> None:
        assert _parse_grid(["fast=10,20", "atr_mult=1.5,2", "mode=agg"]) == {
            "fast": [10, 20],
            "atr_mult": [1.5, 2.0],
            "mode": ["agg"],
        }

    def test_duplicate_param_fails(self) -> None:
        from typer import BadParameter

        with pytest.raises(BadParameter):
            _parse_grid(["fast=1", "fast=2"])

    @pytest.mark.parametrize("spec", ["atr_mult=nan", "atr_mult=inf", "fast=Infinity"])
    def test_non_finite_value_fails(self, spec: str) -> None:
        from typer import BadParameter

        with pytest.raises(BadParameter, match="конечным"):
            _parse_grid([spec])


class TestSymbolTimeframeOverrides:
    ETH_CONFIG = 'start: "2025-07-01"\nsymbol: BTC/USDT\ntimeframe: 4h\n'

    def test_backtest_symbol_override_uses_other_dataset(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        CandleStorage(tmp_path / "data").save(
            "bybit", "ETH/USDT", "4h", rows_to_df(make_candles(60))
        )
        config = tmp_path / "backtest.yaml"
        config.write_text(self.ETH_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app, ["backtest", "--config", str(config), "--symbol", "ETH/USDT"]
        )

        assert result.exit_code == 0, result.output
        meta = json.loads(
            (tmp_path / "reports" / "last_run" / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["config"]["symbol"] == "ETH/USDT"
        assert meta["config"]["timeframe"] == "4h"
        assert meta["summary"]["n_candles"] == 60

    def test_backtest_timeframe_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        CandleStorage(tmp_path / "data").save(
            "bybit", "BTC/USDT", "1h", rows_to_df(make_candles(60))
        )
        config = tmp_path / "backtest.yaml"
        config.write_text(self.ETH_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app, ["backtest", "--config", str(config), "--timeframe", "1h"]
        )

        assert result.exit_code == 0, result.output
        meta = json.loads(
            (tmp_path / "reports" / "last_run" / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["config"]["timeframe"] == "1h"

    def test_backtest_invalid_timeframe_override_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _prepare_data(tmp_path)
        config = tmp_path / "backtest.yaml"
        config.write_text(self.ETH_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app, ["backtest", "--config", str(config), "--timeframe", "7x"]
        )

        assert result.exit_code == 1
        assert "переопределения" in result.output

    def test_sweep_symbol_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        CandleStorage(tmp_path / "data").save(
            "bybit", "SOL/USDT", "4h", rows_to_df(make_candles(120))
        )
        config = tmp_path / "backtest.yaml"
        config.write_text(self.ETH_CONFIG, encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "sweep",
                "--config",
                str(config),
                "--symbol",
                "SOL/USDT",
                "--param",
                "fast=3",
            ],
        )

        assert result.exit_code == 0, result.output
        meta = json.loads(
            (tmp_path / "reports" / "sweep" / "last" / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        assert meta["config"]["symbol"] == "SOL/USDT"


class TestDownloadSince:
    def test_download_without_since_and_without_update_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["download", "--symbol", "BTC/USDT", "--timeframe", "1h"])

        assert result.exit_code == 1
        assert "--since" in result.output

    def test_download_update_without_since_and_no_dataset_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app, ["download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--update"]
        )

        assert result.exit_code == 1
        assert "--since" in result.output

    def test_download_update_without_since_uses_existing_dataset(
        self, tmp_path: Path, monkeypatch, mocker
    ) -> None:
        monkeypatch.chdir(tmp_path)
        df = rows_to_df(make_candles(5))
        CandleStorage(tmp_path / "data").save("bybit", "BTC/USDT", "1h", df)
        downloader_cls = mocker.patch("trading_bot.cli.HistoryDownloader")
        downloader_cls.return_value.update.return_value = df
        mocker.patch("trading_bot.cli.ExchangeClient")

        result = runner.invoke(
            app, ["download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--update"]
        )

        assert result.exit_code == 0, result.output
        assert "Updating" in result.output
        downloader_cls.return_value.update.assert_called_once()
        downloader_cls.return_value.download.assert_not_called()
