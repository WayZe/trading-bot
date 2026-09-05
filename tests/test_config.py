"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.config import BacktestConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "backtest.yaml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_full_config(self, tmp_path) -> None:
        path = write_config(
            tmp_path,
            """
symbol: ETH/USDT
timeframe: 1d
start: "2023-01-01"
end: "2024-01-01"
fee_rate: 0.0005
slippage_bps: 10
position_size_pct: 0.5
strategy: my_strategy
strategy_params:
  period: 14
  threshold: 0.5
""",
        )

        cfg = load_config(path)

        assert cfg.symbol == "ETH/USDT"
        assert cfg.timeframe == "1d"
        assert cfg.start == "2023-01-01"
        assert cfg.end == "2024-01-01"
        assert cfg.fee_rate == 0.0005
        assert cfg.slippage_bps == 10
        assert cfg.position_size_pct == 0.5
        assert cfg.strategy == "my_strategy"
        assert cfg.strategy_params == {"period": 14, "threshold": 0.5}

    def test_defaults(self, tmp_path) -> None:
        path = write_config(tmp_path, 'start: "2024-01-01"\n')

        cfg = load_config(path)

        assert cfg.symbol == "BTC/USDT"
        assert cfg.timeframe == "4h"
        assert cfg.end is None
        assert cfg.start_cash == 10_000.0
        assert cfg.fee_rate == 0.001
        assert cfg.slippage_bps == 5.0
        assert cfg.position_size_pct == 0.95
        assert cfg.quantity_precision == 6
        assert cfg.min_notional == 5.0
        assert cfg.strategy == "sma_cross"
        assert cfg.strategy_params == {}

    def test_shipped_example_config_loads(self) -> None:
        cfg = load_config(PROJECT_ROOT / "config" / "backtest.yaml")

        assert cfg.symbol == "BTC/USDT"
        assert cfg.timeframe == "4h"
        assert cfg.strategy == "sma_cross"
        assert cfg.strategy_params["fast"] == 20
        assert cfg.strategy_params["slow"] == 50
        assert cfg.start_cash == 10_000.0
        assert cfg.quantity_precision == 6
        assert cfg.min_notional == 5.0

    def test_missing_start_raises(self, tmp_path) -> None:
        path = write_config(tmp_path, "symbol: BTC/USDT\n")

        with pytest.raises(ValidationError):
            load_config(path)

    def test_bad_date_format_raises(self, tmp_path) -> None:
        path = write_config(tmp_path, 'start: "01/02/2024"\n')

        with pytest.raises(ValidationError):
            load_config(path)

    def test_end_before_start_raises(self, tmp_path) -> None:
        path = write_config(
            tmp_path, 'start: "2024-06-01"\nend: "2024-01-01"\n'
        )

        with pytest.raises(ValidationError, match="end .* must not be before start"):
            load_config(path)

    def test_non_mapping_config_raises(self, tmp_path) -> None:
        path = write_config(tmp_path, "- just\n- a\n- list\n")

        with pytest.raises(ValueError, match="mapping"):
            load_config(path)


class TestBacktestConfig:
    def test_invalid_position_size_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(start="2024-01-01", position_size_pct=1.5)

    def test_negative_fee_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(start="2024-01-01", fee_rate=-0.1)

    def test_nonpositive_start_cash_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(start="2024-01-01", start_cash=0.0)

    def test_bad_quantity_precision_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(start="2024-01-01", quantity_precision=-1)

    def test_negative_min_notional_raises(self) -> None:
        with pytest.raises(ValidationError):
            BacktestConfig(start="2024-01-01", min_notional=-1.0)

    def test_valid_timeframes_pass(self) -> None:
        for timeframe in ("15m", "1h", "4h", "12h", "1d"):
            assert BacktestConfig(start="2024-01-01", timeframe=timeframe).timeframe == (
                timeframe
            )

    def test_bad_timeframe_format_raises(self) -> None:
        for timeframe in ("4H", "hour", "m15", "1w", "1y", "4", "m", ""):
            with pytest.raises(ValidationError, match="timeframe"):
                BacktestConfig(start="2024-01-01", timeframe=timeframe)

    def test_timeframe_above_one_day_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed 1d"):
            BacktestConfig(start="2024-01-01", timeframe="2d")
