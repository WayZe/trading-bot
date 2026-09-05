"""Pydantic configuration models and YAML loading."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ccxt-style timeframe: one or more digits followed by m/h/d (e.g. 15m, 4h, 1d).
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhd])$")
_TIMEFRAME_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
_MAX_TIMEFRAME_MS = 86_400_000  # 1d


class BacktestConfig(BaseModel):
    """Configuration of a single backtest run.

    Attributes:
        symbol: ccxt trading pair, e.g. ``"BTC/USDT"``.
        timeframe: candle timeframe, e.g. ``"15m"``, ``"4h"``, ``"1d"``.
        start: backtest start date, ``YYYY-MM-DD`` (inclusive).
        end: backtest end date, ``YYYY-MM-DD`` (exclusive); ``None`` means
            "up to the latest available data".
        start_cash: initial cash in quote currency.
        fee_rate: exchange taker fee per trade side, as a fraction (0.001 = 0.1%).
        slippage_bps: execution slippage in basis points (5 bps = 0.05%).
        position_size_pct: fraction of equity allocated to a new position.
        quantity_precision: decimal places entry quantities are rounded down to.
        min_notional: minimum order value in quote currency.
        strategy: strategy plugin name registered in ``trading_bot.strategy``.
        strategy_params: free-form parameters passed to the strategy plugin.
    """

    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    start: str
    end: str | None = None
    start_cash: float = Field(default=10_000.0, gt=0.0)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    position_size_pct: float = Field(default=0.95, gt=0.0, le=1.0)
    quantity_precision: int = Field(default=6, ge=0, le=10)
    min_notional: float = Field(default=5.0, ge=0.0)
    strategy: str = "sma_cross"
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        """Ensure the timeframe is a ccxt-style ``<count><m|h|d>`` of at most 1d."""
        match = _TIMEFRAME_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                f"timeframe must be '<count><m|h|d>' in ccxt style, "
                f"e.g. '15m', '4h', '1d'; got {value!r}"
            )
        count, unit = int(match.group(1)), match.group(2)
        if count * _TIMEFRAME_UNIT_MS[unit] > _MAX_TIMEFRAME_MS:
            raise ValueError(f"timeframe must not exceed 1d; got {value!r}")
        return value

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, value: str | None) -> str | None:
        """Ensure date fields use the ``YYYY-MM-DD`` format."""
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_period(self) -> BacktestConfig:
        """Ensure the end date is not before the start date."""
        if self.end is not None and self.end < self.start:
            raise ValueError(f"end ({self.end}) must not be before start ({self.start})")
        return self


def load_config(path: Path | str) -> BacktestConfig:
    """Load a :class:`BacktestConfig` from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return BacktestConfig.model_validate(raw)
