"""Tests for the CLI application (stubs only; download needs the network)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from trading_bot.cli import app

runner = CliRunner()


def test_report_stub() -> None:
    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0
    assert "этапе 4" in result.output


def test_backtest_stub(tmp_path: Path) -> None:
    config = tmp_path / "backtest.yaml"
    config.write_text('start: "2024-01-01"\n', encoding="utf-8")

    result = runner.invoke(app, ["backtest", "--config", str(config)])

    assert result.exit_code == 0
    assert "этапе 3" in result.output
    assert "BTC/USDT" in result.output


def test_backtest_stub_with_missing_config_fails() -> None:
    result = runner.invoke(app, ["backtest", "--config", "/nonexistent/config.yaml"])

    assert result.exit_code != 0
