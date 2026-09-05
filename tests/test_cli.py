"""Tests for the CLI application (offline; download needs the network)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import make_candles, rows_to_df
from trading_bot.cli import app
from trading_bot.data.storage import CandleStorage

runner = CliRunner()


def test_report_stub() -> None:
    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0
    assert "этапе 4" in result.output


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
