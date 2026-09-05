"""Tests for the CLI application (offline; download needs the network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from tests.conftest import make_candles, rows_to_df
from trading_bot.cli import app
from trading_bot.config import BacktestConfig
from trading_bot.data.storage import CandleStorage

runner = CliRunner()


def _make_run_dir(tmp_path: Path, with_candles: bool = True) -> Path:
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
    equity_png = run / "equity.png"
    trades_png = run / "trades.png"
    assert equity_png.exists() and equity_png.stat().st_size > 0
    assert trades_png.exists() and trades_png.stat().st_size > 0


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
    assert (last_run / "trades.csv").exists()
    assert (last_run / "meta.json").exists()
    meta = json.loads((last_run / "meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["symbol"] == "BTC/USDT"
    assert meta["summary"]["n_candles"] == 60
    assert meta["summary"]["start_cash"] == 10_000.0
    assert meta["summary"]["n_trades"] >= 0


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
